# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import cv2
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mmdet3d.models import HEADS, build_loss
from .petrv2_head import PETRv2Head
from mmdet.models.utils.transformer import inverse_sigmoid
import math
from mmcv.runner import force_fp32
from mmdet.core import (MlvlPointGenerator, bbox_overlaps, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply, reduce_mean)
from mmcv.cnn import Conv2d, Linear, ConvModule, bias_init_with_prob
from mmdet.models.utils import NormedLinear, build_transformer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from ..utils.depthnet import build_depthnet
from mmdet3d.models import builder


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Args:
        pos: (N_query, 3)
        num_pos_feats:
        temperature:
    Returns:
        posemb: (N_query, num_feats * 3)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)     # (num_feats, )
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
    pos_x = pos[..., 0, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
    pos_y = pos[..., 1, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
    pos_z = pos[..., 2, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_z/10000^(0/128), pos_z/10000^(0/128), pos_z/10000^(2/128), pos_z/10000^(2/128), ...]

    # (N_query, num_feats/2, 2) --> (N_query, num_feats)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_x/10000^(0/128)), cos(pos_x/10000^(0/128)), sin(pos_x/10000^(2/128)), cos(pos_x/10000^(2/128)), ...]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_y/10000^(0/128)), cos(pos_y/10000^(0/128)), sin(pos_y/10000^(2/128)), cos(pos_y/10000^(2/128)), ...]
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_z/10000^(0/128)), cos(pos_z/10000^(0/128)), sin(pos_z/10000^(2/128)), cos(pos_z/10000^(2/128)), ...]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # (N_query, num_feats * 3)
    return posemb


class SELayer(nn.Module):
    def __init__(self, depth_channel, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(depth_channel, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: coords_position_embeding  (B*N, embed_dims, H, W)
            x_se: depth distribution     (B*N, depth_num, H, W)
        Returns:
            3D PE:  (B*N, embed_dims, H, W)
        """
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)   # (B*N, embed_dims, H, W)
        return x * self.gate(x_se)


class RegLayer(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 shared_reg_fcs=2,
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 act_layer=nn.ReLU,
                 drop=0.0
                 ):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                Linear(embed_dims, embed_dims),
                act_layer(),
                Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        """
        Args:
            x: (B, N_query, C=embed_dims)
        Returns:

        """
        reg_feat = self.reg_branch(x)   # (B, N_query, C=embed_dims)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())   # (B, N_query, reg_dim)
            outs.append(out)
        outs = torch.cat(outs, -1)  # (B, N_query, code_size)
        return outs


class Far3DStageA2DHead(nn.Module):
    """Far3D StageA proposal head with a shared P3 depth map.

    Objectness, class, box and center predictions use every configured FPN
    level. Depth is predicted once from P3 and sampled at each decoded proposal
    center, matching the official Far3D query-construction path.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 depth_bins,
                 num_feature_levels=4,
                 feat_channels=256,
                 stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.depth_bins = int(depth_bins)
        self.depth_out_channels = self.depth_bins + 1
        self.num_feature_levels = int(num_feature_levels)
        conv_cfg = None
        norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg = dict(type='Swish')

        def make_tower():
            layers = []
            chn = in_channels
            for _ in range(stacked_convs):
                layers.append(ConvModule(
                    chn, feat_channels, 3, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
                chn = feat_channels
            return nn.Sequential(*layers)

        self.cls_towers = nn.ModuleList(
            [make_tower() for _ in range(self.num_feature_levels)])
        self.reg_towers = nn.ModuleList(
            [make_tower() for _ in range(self.num_feature_levels)])
        self.depth_tower = make_tower()
        self.conv_objs = nn.ModuleList([
            nn.Conv2d(feat_channels, 1, 1)
            for _ in range(self.num_feature_levels)])
        self.conv_clss = nn.ModuleList([
            nn.Conv2d(feat_channels, num_classes, 1)
            for _ in range(self.num_feature_levels)])
        self.conv_bboxes = nn.ModuleList([
            nn.Conv2d(feat_channels, 4, 1)
            for _ in range(self.num_feature_levels)])
        self.conv_centers = nn.ModuleList([
            nn.Conv2d(feat_channels, 2, 1)
            for _ in range(self.num_feature_levels)])
        self.conv_depth = nn.Conv2d(
            feat_channels, self.depth_out_channels, 1)
        self.init_weights()

    def init_weights(self):
        prior_bias = bias_init_with_prob(0.01)
        for conv_obj, conv_cls in zip(self.conv_objs, self.conv_clss):
            nn.init.constant_(conv_obj.bias, prior_bias)
            nn.init.constant_(conv_cls.bias, prior_bias)

    def forward_single(self, x, level_idx):
        cls_feat = self.cls_towers[level_idx](x)
        reg_feat = self.reg_towers[level_idx](x)
        return dict(
            objectness=self.conv_objs[level_idx](reg_feat),
            cls_logits=self.conv_clss[level_idx](cls_feat),
            bbox_raw=self.conv_bboxes[level_idx](reg_feat),
            center_raw=self.conv_centers[level_idx](reg_feat))

    def forward(self, feats):
        if torch.is_tensor(feats):
            feats = [feats]
        if len(feats) > self.num_feature_levels:
            raise ValueError(
                f'Received {len(feats)} StageA levels, but only '
                f'{self.num_feature_levels} heads were initialized')
        if not feats:
            raise ValueError('StageA requires at least the P3 feature level')
        outputs = [self.forward_single(x, level_idx)
                   for level_idx, x in enumerate(feats)]
        depth_feat = self.depth_tower(feats[0])
        outputs[0]['depth_logits'] = self.conv_depth(depth_feat)
        return outputs


@HEADS.register_module()
class PETRV2DepthHead(PETRv2Head):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 with_depth_supervision=True,
                 depthnet=dict(
                     type='VanillaDepthNet',
                     in_channels=2048,
                     context_channels=256,
                     depth_channels=64,
                     mid_channels=512,
                 ),
                 with_filter=False,     # 取topK个3D坐标来生成3D PE
                 num_keep=5,
                 with_dpe=False,    # Distribution-guide position encoder
                 use_sigmoid=True,
                 use_detach=False,
                 loss_depth=dict(type='BinaryCrossEntropyLoss', with_logits=False,
                                 reduction='mean', loss_weight=3.0),
                 **kwargs):
        self.with_dpe = with_dpe
        self.with_depth_supervision = with_depth_supervision
        self.with_filter = with_filter
        self.use_sigmoid = use_sigmoid
        self.use_detach = use_detach

        if self.with_depth_supervision:
            kwargs['in_channels'] = depthnet['context_channels']
            if with_filter:
                kwargs['depth_num'] = num_keep
                self.num_keep = num_keep
        super(PETRV2DepthHead, self).__init__(**kwargs)

        if self.with_depth_supervision:
            self.depth_net = build_depthnet(depthnet)
            self.depth_num = self.depth_net.depth_channels
            self.loss_depth = builder.build_loss(loss_depth)
        else:
            self.depth_net = None

        if self.with_dpe:
            self.dpe = SELayer(self.depth_num, self.embed_dims)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        if self.with_multi:
            reg_branch = RegLayer(self.embed_dims, self.num_reg_fcs, self.group_reg_dims)
        else:
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)
        # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        if self.with_fpe:
            self.fpe = SELayer(self.embed_dims, self.embed_dims)

    def position_embeding(self, img_feats, img_metas, masks=None, depth_score=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_score: (B, N_view, D, H, W)
        Returns:
            coords_position_embeding: (B, N_view, embed_dims, H, W)
            coords_mask: (B, N_view, H, W)
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H      # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W      # (W, )

        if self.LID:
            # (D, )
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        # (3, W, H, D)  --> (W, H, D, 3)    3: (u, v, d)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0).contiguous()
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)      # (W, H, D, 4)    4: (u, v, d, 1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)      # (W, H, D, 4)    4: (du, dv, d, 1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)      # (B, N_view, 4, 4)

        # (1, 1, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)

        # (B, N_view, K, H, W),  (B, N_view, K, H, W)
        if depth_score is not None and self.with_filter:
            _, keep_indices = torch.topk(depth_score, k=self.num_keep, dim=2)
            keep_indices = keep_indices.permute(0, 1, 4, 3, 2).contiguous()     # (B, N_view, W, H, K)
            keep_indices = keep_indices[..., None, None].repeat(1, 1, 1, 1, 1, 4, 1)
            coords = torch.gather(coords, dim=4, index=keep_indices)
            D = self.num_keep

        # (B, N_view, 1, 1, 1, 4, 4) --> (B, N_view, W, H, D, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)

        # 图像中每个像素对应的frustum points，借助img2lidars投影到lidar系中.
        # (B, N_view, W, H, D, 4, 4) @ (B, N_view, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        # --> (B, N_view, W, H, D, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # 借助position_range，对3D坐标进行归一化.
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)     # (B, N_view, W, H, D, 3), 超出range的points mask
        # (B, N_view, W, H), 若该像素对应的frustum points 有过多的点超出range， 则对应的coords_mask=1
        # 在后续attention过程中， 会消除这些像素的影响.
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)   # (B, N_view, H, W)
        # (B, N_view, W, H, D, 3) --> (B, N_view, D, 3, H, W) --> (B*N_view, D*3, H, W)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, D*3, H, W)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(coords3d)      # (B*N_view, embed_dims, H, W)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        x = mlvl_feats[self.position_level]   # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))    # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)    # (B*N_view, C, H, W)
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']    # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']    # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)    # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)    # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()     # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()        # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W)
            depth, x = self.depth_net(x, intrinsics, extrinsics)
            if self.use_sigmoid:
                depth_score = depth.sigmoid()
            else:
                depth_score = depth.softmax(dim=1)

            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            self.depth_score = depth_score
            depth_score = depth_score.view(batch_size, num_cams, -1, fH, fW)
        else:
            depth_score = None

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])     # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)    # (B, N_view, H, W)

        if self.with_position:
            # 额, 但是这里没有使用coords_mask.
            # 3D PE: (B, N_view, embed_dims, H, W)
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, depth_score)
            if self.with_dpe:
                if self.use_detach:
                    coords_position_embeding = self.dpe(
                        coords_position_embeding.flatten(0, 1), depth_score.detach().flatten(0, 1)).view(
                        x.size())   # (B, N, embed_dims, H, W)
                else:
                    coords_position_embeding = self.dpe(
                        coords_position_embeding.flatten(0, 1), depth_score.flatten(0, 1)).view(x.size())  # (B, N, embed_dims, H, W)

            if self.with_fpe:
                coords_position_embeding = self.fpe(
                    coords_position_embeding.flatten(0, 1), x.flatten(0, 1)).view(
                    x.size())  # (B, N, embed_dims, H, W)

            pos_embed = coords_position_embeding

            if self.with_multiview:
                # 加入 2D PE 和 multi-view prior
                sin_embed = self.positional_encoding(masks)    # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed   # (B, N_view, embed_dims, H, W)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        # import time
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs_dec, _ = self.transformer(x,   # (B, N_view, embed_dim, H, W)
                                       masks,   # (B, N_view, H, W)
                                       query_embeds,    # (N_query, embed_dims)
                                       pos_embed,       # (B, N_view, embed_dims, H, W)
                                       self.reg_branches    # 没有进行box_refine, 因此没有用到reg_branches.
                                       )
        outs_dec = torch.nan_to_num(outs_dec)       # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        if self.with_time:
            time_stamps = []
            for img_meta in img_metas:
                time_stamps.append(np.asarray(img_meta['timestamp']))
            time_stamp = x.new_tensor(time_stamps)
            time_stamp = time_stamp.view(batch_size, -1, 6)     # (B, N_frame=2, N_view=6)
            # (B, N_view) - (B, N_view) --> (B, N_view) --> (B, )
            mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())   # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])   # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()     # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()     # normalized_cz

            if self.with_time:
                # (vx, vy) = (distance_x, distance_y) / tx
                tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)   # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)    # (num_layers, B, N_query, code_size)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,   # (num_layers, B, N_query, n_cls)
            'all_bbox_preds': all_bbox_preds,   # (num_layers, B, N_query, code_size)
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        if self.depth_net is not None:
            outs['depth'] = depth_score.view(batch_size*num_cams, -1, fH, fW)
        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             depth_map,
             depth_map_mask,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            depth_map:  # (B*N_views, fH, fW)
            depth_map_mask:  # (B*N_views, fH, fW)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']      # (num_layers, B, N_query, n_cls)
        all_bbox_preds = preds_dicts['all_bbox_preds']      # (num_layers, B, N_query, code_size)   code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 分别计算每层decoder layer的loss
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if self.with_depth_supervision:
            depth = preds_dicts['depth']    # (B*N_view, D, H, W)
            depth_loss = self.depth_loss(depth, depth_map, depth_map_mask)
            loss_dict['depth_loss'] = depth_loss

        return loss_dict

    def mask_points_by_dist(self, depth_map, depth_map_mask, min_dist, max_dist):
        mask = depth_map.new_ones(depth_map.shape, dtype=torch.bool)
        mask = torch.logical_and(mask, depth_map >= min_dist)
        mask = torch.logical_and(mask, depth_map < max_dist)
        depth_map_mask[~mask] = 0
        depth_map[~mask] = 0
        return depth_map, depth_map_mask

    def depth_loss(self, depth, depth_map, depth_map_mask):
        """
        Args:
            depth: (B*N_view, D, H, W)
            depth_map: (B*N_view, H, W)
            depth_map_mask: (B*N_view, H, W)
        Returns:

        """
        depth = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)      # (B*N_view*H*W, D)
        depth_map = depth_map.view(-1)                          # (B*N_view*H*W, )
        depth_map_mask = depth_map_mask.view(-1).float()                # (B*N_view*H*W, )

        min_dist = self.depth_start
        if self.LID:
            bin_size = 2 * (self.position_range[3] - min_dist) / (self.depth_num * (1 + self.depth_num))
            depth_label = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - min_dist) / bin_size)   # (H, W)
        else:
            bin_size = (self.position_range[3] - min_dist) / self.depth_num
            depth_label = (depth_map - min_dist) / bin_size

        depth_label = depth_label.long()
        depth_label, depth_map_mask = self.mask_points_by_dist(depth_label, depth_map_mask, 0, self.depth_num)

        # for vis
        # depth_gt = depth_label.float().detach().cpu().numpy().reshape((6, 16, 44))
        # depth_mask = depth_map_mask.float().detach().cpu().numpy().reshape((6, 16, 44))
        # depth_gt = depth_gt / self.depth_num * 255
        # depth_gt = depth_gt.astype(np.uint8)
        # depth_mask *= 255
        # depth_mask = depth_mask.astype(np.uint8)
        # for i in range(6):
        #     depth_gt_vis = cv2.applyColorMap(depth_gt[i], colormap=cv2.COLORMAP_RAINBOW)
        #     cv2.imshow(f'depth {i}', depth_gt_vis)
        #     cv2.imshow(f'depth_mask{i}', depth_mask[i])
        #     cv2.waitKey(0)

        loss_depth = self.loss_depth(depth, depth_label, depth_map_mask,
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))
        return loss_depth


# --------------------------利用预测的depth map将pixel映射到3D point， 然后生成3D PE ----------------------------------------
class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def compute_fg_mask(self, gt_boxes3d, shape, lidar2imgs, device):
        """
        Args:
            gt_boxes3d: List[(N0, 9), (N1, 9), ...]
            shape: (B, N_view, H, W)
            lidar2imgs: (B, N_view, 4, 4)
            img_metas: List[img_meta, ...]
        Returns:
            fg_mask: (B, N_view, H, W)
        """
        batch_size, n_view, fH, fW = shape
        fg_mask = torch.zeros((batch_size, n_view, fH, fW), dtype=torch.bool, device=device)  # (B, N_view, H, W)
        for batch_idx in range(batch_size):
            cur_gt_boxes3d = gt_boxes3d[batch_idx]  # (N, 9)
            cur_gt_boxes3d = LiDARInstance3DBoxes(cur_gt_boxes3d, box_dim=9, origin=(0.5, 0.5, 0.5))
            cur_corners3d = cur_gt_boxes3d.corners      # (N, 8, 3)
            cur_corners3d = torch.cat([cur_corners3d, torch.ones_like(cur_corners3d[..., 0:1])],
                                      dim=-1)  # (N, 8, 4)
            cur_corners3d = cur_corners3d.view(-1, 4)  # (N*8, 4)

            cur_lidar2imgs = lidar2imgs[batch_idx]      # (N_view, 4, 4)

            for view_id in range(n_view):
                cur_lidar2img = cur_lidar2imgs[view_id]  # (4, 4)
                cur_corners3d_proj = cur_corners3d @ cur_lidar2img.T  # (N*8, 4)
                cur_corners2ds = cur_corners3d_proj[:, :2] / cur_corners3d_proj[:, 2:3]  # (N*8, 2)
                cur_corners_depths = cur_corners3d_proj[:, 2].view(-1, 8)    # (N, 8)
                cur_corners2ds = cur_corners2ds.view(-1, 8, 2)  # (N, 8, 2)

                for box_id in range(len(cur_corners2ds)):
                    cur_corners_depth = cur_corners_depths[box_id]  # (8, )
                    cur_corners2d = cur_corners2ds[box_id]      # (8, 2)
                    in_front = cur_corners_depth > 1e-5     # (8, )
                    if in_front.sum() == 0:
                        continue

                    cur_valid_corners2d = cur_corners2d[in_front]   # (valid, 2)
                    cur_valid_corners2d /= self.downsample_factor

                    min_uv, _ = torch.min(cur_valid_corners2d, dim=0)  # (2, )
                    max_uv, _ = torch.max(cur_valid_corners2d, dim=0)  # (2, )
                    cur_boxes2d = torch.cat([min_uv, max_uv], dim=0)  # (4, ) 4: (u1, v1, u2, v2)

                    cur_boxes2d[:2] = torch.floor(cur_boxes2d[:2])
                    cur_boxes2d[2:] = torch.ceil(cur_boxes2d[2:])

                    cur_boxes2d[0] = torch.clamp(cur_boxes2d[0], min=0, max=fW-1)
                    cur_boxes2d[1] = torch.clamp(cur_boxes2d[1], min=0, max=fH-1)
                    cur_boxes2d[2] = torch.clamp(cur_boxes2d[2], min=0, max=fW-1)
                    cur_boxes2d[3] = torch.clamp(cur_boxes2d[3], min=0, max=fH-1)

                    cur_boxes2d = cur_boxes2d.long()
                    u1, v1, u2, v2 = cur_boxes2d
                    fg_mask[batch_idx, view_id, v1:v2, u1:u2] = True

        # for vis
        # fg_mask = fg_mask[0]    # (N_view, H, W)
        # for view_id in range(n_view):
        #     cur_fg_mask = fg_mask[view_id]  # (H, W)
        #     cur_fg_mask = cur_fg_mask.float().cpu().detach().numpy()
        #     cur_fg_mask *= 255
        #     cur_fg_mask = cur_fg_mask.astype(np.uint8)
        #     cv2.imshow("img", cur_fg_mask)
        #     cv2.waitKey(0)

        return fg_mask

    def forward(self, loss, shape, gt_boxes3d, depth_map_mask, img_metas):
        """
        Forward pass
        Args:
            loss: (B*N_view*H*W, ), Pixel-wise loss
            shape: (B, N_view, H, W)
            gt_boxes2d: List[(N0, 9), (N1, 9), ...]
            depth_map_mask: (B*N_view*H*W, )
            img_metas: List[img_meta0, img_meta1, ...]

        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        lidar2imgs = []
        for img_meta in img_metas:
            lidar2img = []
            for i in range(len(img_meta['lidar2img'])):
                lidar2img.append(img_meta['lidar2img'][i])
            lidar2imgs.append(np.asarray(lidar2img))
        lidar2img = np.asarray(lidar2imgs)
        lidar2img = loss.new_tensor(lidar2img)      # (B, N_view, 4, 4)

        # Compute masks
        fg_mask = self.compute_fg_mask(gt_boxes3d=gt_boxes3d,
                                       shape=shape,
                                       lidar2imgs=lidar2img,
                                       device=loss.device)    # (B, N_view, H, W)
        fg_mask = fg_mask.view(-1)
        bg_mask = ~fg_mask

        depth_map_mask = depth_map_mask.bool()
        fg_mask = fg_mask[depth_map_mask]
        bg_mask = bg_mask[depth_map_mask]

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        # num_pixels = fg_mask.sum() + bg_mask.sum()
        num_pixels = weights.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        return fg_loss, bg_loss


@HEADS.register_module()
class PETRV2DepthHeadV2(PETRV2DepthHead):
    def __init__(self,
                 use_dfl=False,
                 use_detach=False,
                 share_pe_encoder=True,
                 with_2dpe_only=False,
                 use_prob_depth=True,
                 use_balancer=False,
                 loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=1.0),
                 balancer_cfg=dict(fg_weight=5.0, bg_weight=1.0, downsample_factor=16),
                 with_pos_info=False,
                 use_far3d_stagea=False,
                 append_far3d_adaptive_queries=None,
                 use_sparse_multiscale_decoder=None,
                 use_global_sparse_surface_pe=False,
                 far3d_stagea_cfg=None,
                 far3d_transformer=None,
                 **kwargs):
        self.share_pe_encoder = share_pe_encoder
        self.with_2dpe_only = with_2dpe_only
        self.with_pos_info = with_pos_info
        self.use_far3d_stagea = use_far3d_stagea
        self.append_far3d_adaptive_queries = (
            self.use_far3d_stagea
            if append_far3d_adaptive_queries is None
            else bool(append_far3d_adaptive_queries))
        if (self.append_far3d_adaptive_queries
                and not self.use_far3d_stagea):
            raise ValueError(
                'Adaptive Query injection requires use_far3d_stagea=True')
        # None preserves the legacy StageA behavior; an explicit False allows
        # training-time proposal supervision with the original P4 decoder.
        self.use_sparse_multiscale_decoder = (
            bool(use_far3d_stagea)
            if use_sparse_multiscale_decoder is None
            else bool(use_sparse_multiscale_decoder))
        if (self.append_far3d_adaptive_queries
                and not self.use_sparse_multiscale_decoder):
            raise ValueError(
                'Adaptive Query injection requires the sparse multi-scale '
                'decoder')
        self.use_global_sparse_surface_pe = bool(
            use_global_sparse_surface_pe)
        self.far3d_stagea_cfg = dict(far3d_stagea_cfg or {})
        self.far3d_range_dn_cfg = dict(
            self.far3d_stagea_cfg.get('range_dn', {}))
        self.use_far3d_range_dn = bool(
            self.far3d_range_dn_cfg.get('enabled', False))
        self.far3d_range_dn_scalar = int(
            self.far3d_range_dn_cfg.get('scalar', 10))
        self.far3d_range_dn_noise_scale = float(
            self.far3d_range_dn_cfg.get('noise_scale', 1.0))
        self.far3d_range_dn_noise_trans = float(
            self.far3d_range_dn_cfg.get('noise_trans', 0.0))
        self.far3d_range_dn_weight = float(
            self.far3d_range_dn_cfg.get('dn_weight', 1.0))
        self.far3d_range_dn_offset = float(
            self.far3d_range_dn_cfg.get('offset', 0.5))
        self.far3d_range_dn_offset_p = float(
            self.far3d_range_dn_cfg.get('offset_p', 0.0))
        self.far3d_range_dn_samples_per_gt = int(
            self.far3d_range_dn_cfg.get('num_smp_per_gt', 3))
        self.far3d_range_dn_query_num = int(
            self.far3d_range_dn_cfg.get('query_num_dn', 600))
        if self.use_far3d_range_dn:
            if not (self.use_far3d_stagea
                    and self.use_sparse_multiscale_decoder):
                raise ValueError(
                    'Range-modulated DN requires StageA and the sparse '
                    'multi-scale decoder')
            if self.far3d_range_dn_scalar <= 0:
                raise ValueError('Range-DN scalar must be positive')
            if self.far3d_range_dn_samples_per_gt < 2:
                raise ValueError(
                    'Range-DN requires one positive and at least one '
                    'negative sample per GT')
            if self.far3d_range_dn_query_num <= 0:
                raise ValueError('Range-DN query_num_dn must be positive')
        self.far3d_transformer_cfg = copy.deepcopy(far3d_transformer)
        self.far3d_stagea_depth_bins = int(
            self.far3d_stagea_cfg.get('depth_num_bins', 16))
        self.far3d_stagea_depth_out_channels = (
            self.far3d_stagea_depth_bins + 1)
        self.far3d_stagea_use_localmax = bool(
            self.far3d_stagea_cfg.get('use_2d_score_localmax', False))
        self.far3d_stagea_localmax_kernel = int(
            self.far3d_stagea_cfg.get('score_localmax_kernel', 3))
        if self.far3d_stagea_localmax_kernel % 2 == 0:
            raise ValueError('score_localmax_kernel must be odd')
        self.far3d_stagea_depth_range_min = float(
            self.far3d_stagea_cfg.get('depth_range_min', -1.0))
        super(PETRV2DepthHeadV2, self).__init__(**kwargs)
        if (self.use_global_sparse_surface_pe
                and not self.use_sparse_multiscale_decoder):
            raise ValueError(
                'Global sparse Surface PE requires '
                'use_sparse_multiscale_decoder=True')
        if (self.use_global_sparse_surface_pe
                and not self.with_depth_supervision):
            raise ValueError(
                'Global sparse Surface PE requires depth supervision')
        self.use_dfl = use_dfl
        self.use_detach = use_detach
        self.use_balancer = use_balancer
        if self.use_sparse_multiscale_decoder:
            if self.far3d_transformer_cfg is None:
                raise ValueError(
                    'The sparse multi-scale decoder requires '
                    'far3d_transformer')
            # PETRv2Head builds the inherited P4 transformer before invoking
            # this constructor. Replace it so only the selected decoder remains
            # registered and optimized.
            self.transformer = build_transformer(
                self.far3d_transformer_cfg)
            if not getattr(
                    self.transformer, 'uses_multilevel_features', False):
                raise TypeError(
                    'far3d_transformer must consume multi-level FPN features')
            transformer_uses_surface_pe = bool(getattr(
                self.transformer, 'uses_global_sparse_surface_pe', False))
            if (transformer_uses_surface_pe
                    != self.use_global_sparse_surface_pe):
                raise ValueError(
                    'Head and far3d_transformer Global sparse Surface PE '
                    'settings must match')
        else:
            if self.far3d_transformer_cfg is not None:
                raise ValueError(
                    'far3d_transformer requires '
                    'use_sparse_multiscale_decoder=True')

        if self.with_depth_supervision:
            self.with_pgd = getattr(self.depth_net, 'with_pgd', False)
            if self.use_dfl:
                self.loss_dfl = build_loss(loss_dfl)

            self.use_prob_depth = use_prob_depth
            if self.use_prob_depth:
                index = torch.arange(start=0, end=self.depth_num, step=1).float()  # (D, )
                bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num - 1)
                depth_bin = self.depth_start + bin_size * index  # (D, )
                self.register_buffer('project', depth_bin)  # (D, )
            if not self.use_prob_depth:
                assert self.depth_num == 1, 'depth_num setting is wrong'
                assert self.with_pgd is False, 'direct depth prediction cannot be combined with pgd'
                assert self.use_dfl is False, 'direct depth prediction cannot be combined with dfl'

            if self.use_balancer:
                assert self.loss_depth.reduction == 'none', 'reduction must be none when use_balancer is True'
                self.balancer = Balancer(**balancer_cfg)

        if self.use_far3d_stagea:
            self.register_buffer(
                'far3d_stagea_depth_centers',
                self._make_far3d_stagea_depth_centers(),
                persistent=False)
            self.register_buffer(
                'far3d_stagea_iter',
                torch.zeros(1, dtype=torch.long))
            strides = self.far3d_stagea_cfg.get('strides', [8, 16, 32, 64])
            self.far3d_stagea_prior_generator = MlvlPointGenerator(strides, offset=0)
            assigner_cfg = self.far3d_stagea_cfg.get(
                'assigner', dict(type='SimOTAAssigner', center_radius=2.5))
            self.far3d_stagea_assigner = build_assigner(assigner_cfg)
            self.far3d_stagea_sampler = build_sampler(dict(type='PseudoSampler'), context=self)

    def init_weights(self):
        """Initialize the active DenseGT or joint Far3D decoder."""
        super().init_weights()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        use_dense_global_path = not self.use_sparse_multiscale_decoder
        if use_dense_global_path:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        if self.with_multi:
            reg_branch = RegLayer(self.embed_dims, self.num_reg_fcs, self.group_reg_dims)
        else:
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])

        if use_dense_global_path:
            if self.with_multiview:
                self.adapt_pos3d = nn.Sequential(
                    nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
                )
            elif self.with_2dpe_only:
                self.adapt_pos3d = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                )

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)

        if self.share_pe_encoder:
            position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            if self.with_position and use_dense_global_path:
                self.position_encoder = position_encoder

            # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
            self.query_embedding = position_encoder
        else:
            if self.with_position and use_dense_global_path:
                # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
                self.position_encoder = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_embedding = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )

        if self.with_pos_info:
            self.extra_position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

        if self.with_fpe and use_dense_global_path:
            self.fpe = SELayer(self.embed_dims, self.embed_dims)

        if getattr(self, 'use_far3d_stagea', False):
            num_stagea_levels = int(
                self.far3d_stagea_cfg.get('num_feature_levels', 4))
            self.far3d_stagea_head = Far3DStageA2DHead(
                in_channels=self.embed_dims,
                num_classes=self.num_classes,
                depth_bins=self.far3d_stagea_depth_bins,
                num_feature_levels=num_stagea_levels,
                feat_channels=self.far3d_stagea_cfg.get(
                    'feat_channels', self.embed_dims),
                stacked_convs=self.far3d_stagea_cfg.get('stacked_convs', 2))
            self.far3d_stagea_context_embed = nn.Sequential(
                nn.Linear(self.embed_dims + 1, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims))

    def integral(self, depth_pred):
        """
        Args:
            depth_pred: (N, D)
        Returns:
            depth_val: (N, )
        """
        depth_score = F.softmax(depth_pred, dim=-1)     # (N, D)
        depth_val = F.linear(depth_score, self.project.type_as(depth_score))  # (N, D) * (D, )  --> (N, )
        return depth_val


    def forward_depth_only(self, mlvl_feats, img_metas):
        """Forward only the depth branch for depth pretraining/diagnosis."""
        assert self.with_depth_supervision and self.depth_net is not None
        x = mlvl_feats[self.position_level]
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        x = x.flatten(0, 1)

        intrinsics_list = []
        extrinsics_list = []
        for batch_id in range(len(img_metas)):
            cur_intrinsics = x.new_tensor(img_metas[batch_id]['intrinsics'])
            cur_extrinsics = x.new_tensor(img_metas[batch_id]['extrinsics'])
            intrinsics_list.append(cur_intrinsics)
            extrinsics_list.append(cur_extrinsics)
        intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()
        extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()

        if self.with_pgd:
            depth, _, depth_direct = self.depth_net(x, intrinsics, extrinsics)
        else:
            depth, _ = self.depth_net(x, intrinsics, extrinsics)

        outs = {}
        if self.use_prob_depth:
            self.depth_score = depth
            depth_prob = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)
            depth_prob_val = self.integral(depth_prob)
            depth_map_pred = depth_prob_val
            outs['depth_prob'] = depth_prob
            if self.with_pgd:
                sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                depth_direct_val = depth_direct.view(-1)
                depth_map_pred = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
                outs['depth_prob_val'] = depth_prob_val
                outs['depth_direct_val'] = depth_direct_val
        else:
            depth_map_pred = depth.exp().view(-1)

        self.depth_map = depth_map_pred.view(batch_size, num_cams, fH, fW)
        outs['depth_map_pred'] = depth_map_pred
        return outs

    def _stagea_get(self, key, default):
        return self.far3d_stagea_cfg.get(key, default)

    def _prepare_far3d_range_dn_queries(
            self, query_pos, query_content, reference_points,
            query_key_padding_mask, gt_bboxes_3d, gt_labels_3d):
        """Prepend official-style range-modulated 3D denoising queries."""
        if not (self.training and self.use_far3d_range_dn):
            return (
                query_pos, query_content, reference_points,
                query_key_padding_mask, None, None)

        if gt_bboxes_3d is None or gt_labels_3d is None:
            raise ValueError(
                'Range-DN training requires gt_bboxes_3d and gt_labels_3d')
        batch_size = reference_points.size(0)
        if (len(gt_bboxes_3d) != batch_size
                or len(gt_labels_3d) != batch_size):
            raise ValueError('Range-DN GT batch size does not match queries')

        device = reference_points.device
        dtype = reference_points.dtype
        targets = []
        labels = []
        known_num = []
        for boxes_3d, labels_3d in zip(gt_bboxes_3d, gt_labels_3d):
            target = torch.cat(
                (boxes_3d.gravity_center, boxes_3d.tensor[:, 3:]),
                dim=1).to(device=device, dtype=dtype)
            targets.append(target)
            labels.append(labels_3d.to(device=device, dtype=torch.long))
            known_num.append(int(target.size(0)))

        max_gt = max(known_num) if known_num else 0
        if max_gt == 0:
            return (
                query_pos, query_content, reference_points,
                query_key_padding_mask, None, None)

        groups = min(
            self.far3d_range_dn_scalar,
            self.far3d_range_dn_query_num // max_gt)
        if groups <= 0:
            return (
                query_pos, query_content, reference_points,
                query_key_padding_mask, None, None)

        known_bboxs = torch.cat(targets, dim=0)
        known_labels = torch.cat(labels, dim=0)
        target_batch = torch.cat([
            torch.full(
                (num_gt,), batch_id, dtype=torch.long, device=device)
            for batch_id, num_gt in enumerate(known_num)
        ])
        target_local = torch.cat([
            torch.arange(num_gt, dtype=torch.long, device=device)
            for num_gt in known_num
        ])
        known_centers = known_bboxs[:, :3]
        known_scales = known_bboxs[:, 3:6]
        num_samples = self.far3d_range_dn_samples_per_gt
        num_negative = num_samples - 1
        single_pad = max_gt * num_samples
        pad_size = single_pad * groups

        pc_range = reference_points.new_tensor(self.pc_range)
        pc_min, pc_max = pc_range[:3], pc_range[3:6]
        padded_reference_points = reference_points.new_zeros(
            batch_size, pad_size, 3)
        dn_padding_mask = torch.ones(
            batch_size, pad_size, dtype=torch.bool, device=device)

        sample_batch_indices = []
        sample_slot_indices = []
        costs = []
        candidate_batch = target_batch[:, None].expand(
            -1, num_samples).reshape(-1)
        candidate_local = (
            target_local[:, None] * num_samples
            + torch.arange(
                num_samples, dtype=torch.long, device=device)[None]
        ).reshape(-1)

        for group_id in range(groups):
            positive_extent = (
                known_scales * 0.5 + self.far3d_range_dn_noise_trans)
            positive_delta = (
                torch.rand_like(known_centers)
                + self.far3d_range_dn_offset_p)
            positive_delta = (
                positive_delta * positive_extent
                * self.far3d_range_dn_noise_scale)
            positive_sign = torch.randint(
                0, 2, known_centers.shape, device=device
            ).to(dtype=dtype).mul_(2.0).sub_(1.0)
            positive_centers = (
                known_centers + positive_sign * positive_delta)

            negative_base = torch.log(known_centers.abs() + 1.0)
            negative_base = negative_base[:, None].expand(
                -1, num_negative, -1)
            negative_delta = (
                torch.rand_like(negative_base)
                + self.far3d_range_dn_offset) * negative_base
            negative_sign = torch.randint(
                0, 2, negative_base.shape, device=device
            ).to(dtype=dtype).mul_(2.0).sub_(1.0)
            negative_centers = (
                known_centers[:, None]
                + negative_sign * negative_delta)

            candidate_centers = torch.cat([
                positive_centers[:, None], negative_centers
            ], dim=1).reshape(-1, 3)
            cost = torch.cdist(
                candidate_centers.float(), known_centers.float(), p=1)
            batch_mismatch = (
                candidate_batch[:, None] != target_batch[None]).float()
            cost = cost + batch_mismatch * 1e5
            costs.append(torch.nan_to_num(
                cost.detach().cpu(), nan=1e5, posinf=1e5, neginf=1e5))

            normalized_centers = (
                (candidate_centers - pc_min) / (pc_max - pc_min)
            ).clamp(0.0, 1.0)
            slots = group_id * single_pad + candidate_local
            padded_reference_points[
                candidate_batch, slots] = normalized_centers
            dn_padding_mask[candidate_batch, slots] = False
            sample_batch_indices.append(candidate_batch)
            sample_slot_indices.append(slots)

        dn_query_pos = self.query_embedding(
            pos2posemb3d(padded_reference_points))
        dn_query_content = torch.zeros_like(dn_query_pos)
        if query_content is None:
            query_content = torch.zeros_like(query_pos)

        query_pos = torch.cat([dn_query_pos, query_pos], dim=1)
        query_content = torch.cat(
            [dn_query_content, query_content], dim=1)
        reference_points = torch.cat(
            [padded_reference_points, reference_points], dim=1)
        query_key_padding_mask = torch.cat(
            [dn_padding_mask, query_key_padding_mask], dim=1)

        total_queries = query_pos.size(1)
        self_attn_mask = torch.zeros(
            total_queries, total_queries, dtype=torch.bool, device=device)
        self_attn_mask[pad_size:, :pad_size] = True
        for group_id in range(groups):
            start = group_id * single_pad
            end = start + single_pad
            self_attn_mask[start:end, :start] = True
            self_attn_mask[start:end, end:pad_size] = True

        dn_meta = dict(
            pad_size=pad_size,
            groups=groups,
            num_samples=num_samples,
            sample_batch_indices=torch.cat(
                sample_batch_indices, dim=0),
            sample_slot_indices=torch.cat(
                sample_slot_indices, dim=0),
            known_labels=known_labels.unsqueeze(0).repeat(groups, 1),
            known_bboxs=known_bboxs.unsqueeze(0).repeat(groups, 1, 1),
            costs=costs,
            num_targets=groups * int(known_bboxs.size(0)))
        return (
            query_pos, query_content, reference_points,
            query_key_padding_mask, self_attn_mask, dn_meta)

    def _prepare_far3d_range_dn_loss_targets(self, dn_meta):
        output_known_class, output_known_coord = (
            dn_meta['output_known_lbs_bboxes'])
        batch_indices = dn_meta['sample_batch_indices'].long()
        slot_indices = dn_meta['sample_slot_indices'].long()
        output_known_class = output_known_class.permute(
            1, 2, 0, 3)[batch_indices, slot_indices].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(
            1, 2, 0, 3)[batch_indices, slot_indices].permute(1, 0, 2)

        known_labels = dn_meta['known_labels']
        known_bboxs = dn_meta['known_bboxs']
        num_samples = int(dn_meta['num_samples'])
        num_gt = known_labels.size(1)
        num_candidates = num_gt * num_samples
        labels = []
        bbox_targets = []
        for group_id, cost in enumerate(dn_meta['costs']):
            cls_target = output_known_class.new_full(
                (num_candidates,), self.num_classes, dtype=torch.long)
            bbox_target = known_bboxs.new_zeros(
                num_candidates, known_bboxs.size(-1))
            matched_rows, matched_cols = linear_sum_assignment(
                cost.numpy())
            matched_rows = torch.as_tensor(
                matched_rows, dtype=torch.long, device=known_bboxs.device)
            matched_cols = torch.as_tensor(
                matched_cols, dtype=torch.long, device=known_bboxs.device)
            cls_target[matched_rows] = known_labels[
                group_id, matched_cols]
            bbox_target[matched_rows] = known_bboxs[
                group_id, matched_cols]
            labels.append(cls_target)
            bbox_targets.append(bbox_target)

        return (
            torch.cat(labels, dim=0),
            torch.cat(bbox_targets, dim=0),
            output_known_class,
            output_known_coord,
            int(dn_meta['num_targets']))

    def _far3d_range_dn_loss_single(
            self, cls_scores, bbox_preds, known_bboxs, known_labels,
            num_total_pos):
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(float(num_total_pos), 1.0)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights,
            avg_factor=cls_avg_factor)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        bbox_weights = torch.ones_like(bbox_preds)
        bbox_weights[known_labels == self.num_classes] = 0
        normalized_bbox_targets = normalize_bbox(
            known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        reduced_num_pos = torch.clamp(
            reduce_mean(loss_cls.new_tensor([num_total_pos])),
            min=1).item()
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=reduced_num_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return (
            self.far3d_range_dn_weight * loss_cls,
            self.far3d_range_dn_weight * loss_bbox)


    def _make_far3d_stagea_depth_centers(self):
        min_depth = float(self.far3d_stagea_cfg.get('depth_min', self.depth_start))
        max_depth = float(self.far3d_stagea_cfg.get('depth_max', self.position_range[3]))
        depth_bins = int(self.far3d_stagea_depth_bins)
        mode = self.far3d_stagea_cfg.get('depth_bin_mode', 'lid')
        idx = torch.arange(depth_bins, dtype=torch.float32)
        if mode == 'lid':
            bin_size = 2 * (max_depth - min_depth) / (depth_bins * (1 + depth_bins))
            centers = min_depth + bin_size / 8.0 * (torch.square(idx / 0.5 + 1.0) - 1.0)
        else:
            bin_size = (max_depth - min_depth) / depth_bins
            centers = min_depth + (idx + 0.5) * bin_size
        centers = centers.clamp(min=min_depth, max=max_depth)
        return torch.cat([centers, centers.new_tensor([max_depth])], dim=0)

    def _far3d_stagea_depth_to_bin(self, depth):
        min_depth = float(self._stagea_get('depth_min', self.depth_start))
        max_depth = float(self._stagea_get('depth_max', self.position_range[3]))
        depth_bins = int(self.far3d_stagea_depth_bins)
        if self._stagea_get('depth_bin_mode', 'lid') == 'lid':
            bin_size = 2 * (max_depth - min_depth) / (depth_bins * (1 + depth_bins))
            label = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth - min_depth) / bin_size)
        else:
            label = (depth - min_depth) / ((max_depth - min_depth) / depth_bins)
        invalid = ((label < 0) | (label > depth_bins) |
                   (~torch.isfinite(label)))
        label = torch.where(invalid, label.new_full(label.shape, depth_bins),
                            label)
        return label.long().clamp(0, depth_bins)

    def _stagea_bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        tl_x = xys[..., 0] - whs[..., 0] * 0.5
        tl_y = xys[..., 1] - whs[..., 1] * 0.5
        br_x = xys[..., 0] + whs[..., 0] * 0.5
        br_y = xys[..., 1] + whs[..., 1] * 0.5
        return torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)

    def _flatten_far3d_stagea_preds(self, preds_list):
        featmap_sizes = [
            preds['objectness'].shape[-2:] for preds in preds_list]
        priors = self.far3d_stagea_prior_generator.grid_priors(
            featmap_sizes,
            dtype=preds_list[0]['objectness'].dtype,
            device=preds_list[0]['objectness'].device,
            with_stride=True)
        flatten_priors = torch.cat(priors, dim=0)
        flatten_obj = torch.cat([
            preds['objectness'].permute(0, 2, 3, 1).reshape(
                preds['objectness'].shape[0], -1, 1)
            for preds in preds_list], dim=1)
        flatten_cls = torch.cat([
            preds['cls_logits'].permute(0, 2, 3, 1).reshape(
                preds['cls_logits'].shape[0], -1, self.num_classes)
            for preds in preds_list], dim=1)
        flatten_bbox_raw = torch.cat([
            preds['bbox_raw'].permute(0, 2, 3, 1).reshape(
                preds['bbox_raw'].shape[0], -1, 4)
            for preds in preds_list], dim=1)
        flatten_center_raw = torch.cat([
            preds['center_raw'].permute(0, 2, 3, 1).reshape(
                preds['center_raw'].shape[0], -1, 2)
            for preds in preds_list], dim=1)
        flatten_bboxes = self._stagea_bbox_decode(
            flatten_priors, flatten_bbox_raw)
        proposal_centers = (
            flatten_bboxes[..., :2] + flatten_bboxes[..., 2:]) * 0.5

        p3_depth_logits = preds_list[0].get('depth_logits')
        if p3_depth_logits is None:
            raise KeyError('StageA P3 prediction is missing depth_logits')
        if p3_depth_logits.shape[:2] != (
                flatten_obj.shape[0], self.far3d_stagea_depth_out_channels):
            raise ValueError(
                'Unexpected P3 depth shape: '
                f'{tuple(p3_depth_logits.shape)}')
        depth_height, depth_width = p3_depth_logits.shape[-2:]
        p3_stride = priors[0][0, 2:]
        depth_x = torch.round(
            proposal_centers[..., 0] / p3_stride[0]).long()
        depth_y = torch.round(
            proposal_centers[..., 1] / p3_stride[1]).long()
        depth_x = depth_x.clamp(0, depth_width - 1)
        depth_y = depth_y.clamp(0, depth_height - 1)
        depth_sample_indices = depth_y * depth_width + depth_x
        depth_indices = depth_sample_indices.unsqueeze(-1).expand(
            -1, -1, self.far3d_stagea_depth_out_channels)
        p3_depth_flat = p3_depth_logits.permute(0, 2, 3, 1).reshape(
            p3_depth_logits.shape[0], -1,
            self.far3d_stagea_depth_out_channels)
        flatten_depth_logits = torch.gather(
            p3_depth_flat, dim=1, index=depth_indices)

        flatten_level_ids = torch.cat([
            torch.full(
                (preds['objectness'].shape[-2]
                 * preds['objectness'].shape[-1],),
                level_id, dtype=torch.long,
                device=preds['objectness'].device)
            for level_id, preds in enumerate(preds_list)], dim=0)
        return dict(
            priors=flatten_priors,
            objectness=flatten_obj,
            cls_logits=flatten_cls,
            bbox_raw=flatten_bbox_raw,
            decoded_bboxes=flatten_bboxes,
            proposal_centers=proposal_centers,
            center_raw=flatten_center_raw,
            depth_logits=flatten_depth_logits,
            depth_sample_indices=depth_sample_indices,
            p3_stride=p3_stride,
            level_ids=flatten_level_ids)

    def _get_far3d_stagea_gt_for_image(self, img_metas, img_idx, num_cams, device):
        batch_id = img_idx // num_cams
        cam_id = img_idx % num_cams
        meta = img_metas[batch_id]
        gt_boxes = meta.get('gt2d_boxes', [])
        gt_labels = meta.get('gt2d_labels', [])
        gt_depths = meta.get('gt2d_depths', [])
        if cam_id >= len(gt_boxes):
            return (torch.zeros((0, 4), dtype=torch.float32, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                    torch.zeros((0, 2), dtype=torch.float32, device=device),
                    torch.zeros((0,), dtype=torch.float32, device=device))
        boxes = torch.as_tensor(gt_boxes[cam_id], dtype=torch.float32, device=device).reshape(-1, 4)
        labels = torch.as_tensor(gt_labels[cam_id], dtype=torch.long, device=device).reshape(-1)
        depths = torch.as_tensor(gt_depths[cam_id], dtype=torch.float32, device=device).reshape(-1)
        num = min(boxes.shape[0], labels.shape[0], depths.shape[0])
        boxes, labels, depths = boxes[:num], labels[:num], depths[:num]
        if num > 0:
            centers = (boxes[:, 0:2] + boxes[:, 2:4]) * 0.5
            valid = (
                torch.isfinite(boxes).all(dim=1) & torch.isfinite(depths)
                & (depths > 0) & (labels >= 0) & (labels < self.num_classes)
                & ((boxes[:, 2] - boxes[:, 0]) > 1.0)
                & ((boxes[:, 3] - boxes[:, 1]) > 1.0))
            boxes, labels, centers, depths = boxes[valid], labels[valid], centers[valid], depths[valid]
        else:
            centers = torch.zeros((0, 2), dtype=torch.float32, device=device)
        return boxes, labels, centers, depths

    @torch.no_grad()
    def _build_far3d_stagea_dense_depth_targets(
            self, flat, preds_list, img_metas):
        """Build Far3D's P3 instance-center depth map and background mask."""
        depth_logits = preds_list[0]['depth_logits']
        num_imgs, _, depth_height, depth_width = depth_logits.shape
        num_cams = preds_list[0]['num_cams']
        device = depth_logits.device
        background_bin = self.far3d_stagea_depth_bins
        depth_targets = torch.full(
            (num_imgs, depth_height, depth_width), background_bin,
            dtype=torch.long, device=device)
        foreground_mask = torch.zeros_like(depth_targets, dtype=torch.bool)
        stride_x, stride_y = flat['p3_stride']
        grid_x = torch.arange(
            depth_width, device=device).view(1, 1, depth_width)
        grid_y = torch.arange(
            depth_height, device=device).view(1, depth_height, 1)

        for img_idx in range(num_imgs):
            boxes, _, _, depths = self._get_far3d_stagea_gt_for_image(
                img_metas, img_idx, num_cams, device)
            if boxes.numel() == 0:
                continue

            x1 = torch.floor(boxes[:, 0] / stride_x).long().clamp(
                0, depth_width)
            y1 = torch.floor(boxes[:, 1] / stride_y).long().clamp(
                0, depth_height)
            x2 = torch.ceil(boxes[:, 2] / stride_x).long().clamp(
                0, depth_width)
            y2 = torch.ceil(boxes[:, 3] / stride_y).long().clamp(
                0, depth_height)
            valid = (x2 > x1) & (y2 > y1)
            if not valid.any():
                continue
            x1, y1, x2, y2, depths = (
                value[valid] for value in (x1, y1, x2, y2, depths))

            inside = (
                (grid_x >= x1[:, None, None])
                & (grid_x < x2[:, None, None])
                & (grid_y >= y1[:, None, None])
                & (grid_y < y2[:, None, None]))
            nearest_depth, nearest_box = torch.where(
                inside,
                depths[:, None, None],
                depths.new_full((), float('inf'))).min(dim=0)
            current_foreground = torch.isfinite(nearest_depth)
            box_depth_bins = self._far3d_stagea_depth_to_bin(depths)
            current_targets = box_depth_bins[nearest_box]
            depth_targets[img_idx] = torch.where(
                current_foreground, current_targets,
                depth_targets[img_idx])
            foreground_mask[img_idx] = current_foreground

        sampled_depth_bins = torch.gather(
            depth_targets.flatten(1), dim=1,
            index=flat['depth_sample_indices'])
        return depth_targets, foreground_mask, sampled_depth_bins

    def _build_far3d_stagea_targets(self, preds_list, img_metas):
        flat = self._flatten_far3d_stagea_preds(preds_list)
        priors = flat['priors']
        cls_preds = flat['cls_logits']
        objectness = flat['objectness'].squeeze(-1)
        decoded_bboxes = flat['decoded_bboxes']
        num_imgs, num_priors = objectness.shape
        num_cams = preds_list[0]['num_cams']
        device = objectness.device

        pos_masks = torch.zeros(num_imgs, num_priors, dtype=torch.bool, device=device)
        obj_targets = torch.zeros(num_imgs, num_priors, 1, device=device)
        cls_targets = torch.zeros(num_imgs, num_priors, self.num_classes, device=device)
        bbox_targets = torch.zeros(num_imgs, num_priors, 4, device=device)
        bbox_l1_targets = torch.zeros(num_imgs, num_priors, 4, device=device)
        center_targets = torch.zeros(num_imgs, num_priors, 2, device=device)
        p3_depth_targets, p3_depth_foreground_mask, sampled_depth_bins = (
            self._build_far3d_stagea_dense_depth_targets(
                flat, preds_list, img_metas))
        num_fg = []
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        for img_idx in range(num_imgs):
            gt_boxes, gt_labels, centers2d, depths = self._get_far3d_stagea_gt_for_image(
                img_metas, img_idx, num_cams, device)
            if gt_labels.numel() == 0:
                num_fg.append(0)
                continue
            assign_result = self.far3d_stagea_assigner.assign(
                cls_preds[img_idx].detach().sigmoid() * objectness[img_idx].detach().sigmoid().unsqueeze(1),
                offset_priors,
                decoded_bboxes[img_idx].detach(),
                gt_boxes,
                gt_labels)
            sampling_result = self.far3d_stagea_sampler.sample(assign_result, priors, gt_boxes)
            pos_inds = sampling_result.pos_inds
            num_pos = int(pos_inds.numel())
            num_fg.append(num_pos)
            if num_pos == 0:
                continue
            assigned = sampling_result.pos_assigned_gt_inds.long()
            pos_masks[img_idx, pos_inds] = True
            obj_targets[img_idx, pos_inds, 0] = 1.0
            pos_ious = assign_result.max_overlaps[pos_inds].clamp_min(0)
            cls_targets[img_idx, pos_inds] = F.one_hot(
                sampling_result.pos_gt_labels, self.num_classes).float() * pos_ious.unsqueeze(-1)
            bbox_targets[img_idx, pos_inds] = sampling_result.pos_gt_bboxes
            gt_cxcywh = bbox_xyxy_to_cxcywh(
                sampling_result.pos_gt_bboxes)
            bbox_l1_targets[img_idx, pos_inds, :2] = (
                gt_cxcywh[:, :2] - priors[pos_inds, :2]
            ) / priors[pos_inds, 2:]
            bbox_l1_targets[img_idx, pos_inds, 2:] = torch.log(
                gt_cxcywh[:, 2:] / priors[pos_inds, 2:] + 1e-8)
            center_targets[img_idx, pos_inds] = (centers2d[assigned] - priors[pos_inds, :2]) / priors[pos_inds, 2:]

        num_pos_tensor = objectness.new_tensor(float(sum(num_fg)))
        num_total_samples = max(reduce_mean(num_pos_tensor), 1.0)
        return dict(
            flat=flat,
            pos_masks=pos_masks,
            obj_targets=obj_targets,
            cls_targets=cls_targets,
            bbox_targets=bbox_targets,
            bbox_l1_targets=bbox_l1_targets,
            center_targets=center_targets,
            p3_depth_targets=p3_depth_targets,
            p3_depth_foreground_mask=p3_depth_foreground_mask,
            sampled_depth_bins=sampled_depth_bins,
            num_total_samples=num_total_samples)

    def _stagea_localmax_keep(self, preds_list):
        keep_list = []
        kernel = int(self.far3d_stagea_localmax_kernel)
        for preds in preds_list:
            obj = preds['objectness'].sigmoid().squeeze(1)
            cls_score = preds['cls_logits'].sigmoid().amax(dim=1)
            score = obj * cls_score
            pooled = F.max_pool2d(
                score.unsqueeze(1), kernel_size=kernel, stride=1,
                padding=kernel // 2).squeeze(1)
            keep_list.append((score == pooled).flatten(1))
        return torch.cat(keep_list, dim=1)

    def _decode_far3d_stagea_depth_hypotheses(self, depth_prob, depth_centers,
                                              depth_topk=1):
        depth_topk = max(1, min(int(depth_topk), depth_prob.shape[-1]))
        depth_conf, depth_idx = torch.topk(depth_prob, k=depth_topk, dim=-1)
        depth_window = int(self._stagea_get('depth_window', 1))
        depth_aggregation = self._stagea_get('depth_aggregation', 'bin')
        if depth_aggregation == 'window' and depth_window > 1:
            half = depth_window // 2
            offsets = torch.arange(
                -half, half + 1, device=depth_prob.device, dtype=torch.long)
            win_idx = (depth_idx.unsqueeze(-1) + offsets).clamp(
                0, depth_prob.shape[-1] - 1)
            gather_prob = depth_prob.unsqueeze(2).expand(
                -1, -1, depth_topk, -1)
            win_prob = torch.gather(gather_prob, dim=-1, index=win_idx)
            win_depth = depth_centers[win_idx]
            depth_value = ((win_prob * win_depth).sum(dim=-1) /
                           win_prob.sum(dim=-1).clamp_min(1e-6))
            return depth_value, depth_conf, depth_idx
        return depth_centers[depth_idx], depth_conf, depth_idx

    def _decode_far3d_stagea_queries(
            self, preds_list, context, img_metas, input_img_h, input_img_w,
            targets=None):
        flat = (targets['flat'] if targets is not None
                else self._flatten_far3d_stagea_preds(preds_list))
        priors = flat['priors']
        objectness = flat['objectness'].squeeze(-1)
        cls_logits = flat['cls_logits']
        depth_logits = flat['depth_logits']
        proposal_centers = flat['proposal_centers']
        num_imgs, num_priors = objectness.shape
        batch_size = preds_list[0]['batch_size']
        num_cams = preds_list[0]['num_cams']
        device = objectness.device
        if context.shape != (num_imgs, num_priors, self.embed_dims):
            raise ValueError(
                'StageA context must align with flattened FPN priors, got '
                f'{tuple(context.shape)} for {(num_imgs, num_priors)}')

        sample_max_per_cam = int(self._stagea_get(
            'sample_max_per_cam', 16))
        per_cam = int(self._stagea_get(
            'topk_per_cam', sample_max_per_cam))
        per_cam = max(1, min(per_cam, sample_max_per_cam))
        max_queries = self._stagea_get('max_adaptive_queries', None)
        max_queries = (int(max_queries) if max_queries is not None
                       else sample_max_per_cam * num_cams)
        score_thr = float(self._stagea_get('score_thr', 0.1))
        log_odds_thr = float(self._stagea_get('score_thr', 0.1))
        depth_topk = max(1, int(self._stagea_get('depth_topk', 1)))
        min_depth = float(self._stagea_get(
            'depth_min', self.depth_start))
        max_depth = float(self._stagea_get(
            'depth_max', self.position_range[3]))

        cls_score = cls_logits.sigmoid().max(dim=-1).values
        proposal_score = objectness.sigmoid() * cls_score
        depth_prob = F.softmax(depth_logits, dim=-1)
        depth_centers = self.far3d_stagea_depth_centers.to(
            device=device, dtype=depth_prob.dtype)

        warmup_iters = int(self._stagea_get(
            'gt_depth_warmup_iters', 0))
        use_gt_depth = (
            self.training
            and bool(self._stagea_get('train_use_gt_depth', True))
            and targets is not None
            and warmup_iters > 0
            and int(self.far3d_stagea_iter.item()) < warmup_iters)
        depth_range_min_bin = None
        if use_gt_depth:
            depth_indices = targets['sampled_depth_bins'].unsqueeze(-1)
            depth_values = depth_centers[depth_indices]
            depth_confs = depth_values.new_ones(depth_values.shape)
        else:
            depth_values, depth_confs, depth_indices = (
                self._decode_far3d_stagea_depth_hypotheses(
                    depth_prob, depth_centers, depth_topk))
            if depth_topk > 1 and self.far3d_stagea_depth_range_min > 0:
                range_min = depth_centers.new_tensor(
                    self.far3d_stagea_depth_range_min)
                depth_range_min_bin = int(torch.searchsorted(
                    depth_centers, range_min).clamp(
                        0, depth_centers.numel() - 1).item())

        localmax_keep = (
            self._stagea_localmax_keep(preds_list)
            if self.far3d_stagea_use_localmax else None)
        query_content = context.new_zeros(
            batch_size, max_queries, self.embed_dims)
        query_pos = context.new_zeros(
            batch_size, max_queries, self.embed_dims)
        reference_points = context.new_full(
            (batch_size, max_queries, 3), 0.5)
        query_padding_mask = torch.ones(
            batch_size, max_queries, dtype=torch.bool, device=device)
        pc_range = context.new_tensor(self.pc_range)
        pc_min, pc_max = pc_range[:3], pc_range[3:]

        for batch_id in range(batch_size):
            image_indices = torch.arange(
                batch_id * num_cams, (batch_id + 1) * num_cams,
                device=device)
            candidates = []
            for image_id in image_indices.tolist():
                score = proposal_score[image_id]
                valid = torch.isfinite(score) & (score > score_thr)
                if localmax_keep is not None:
                    valid = valid & localmax_keep[image_id]
                valid_indices = torch.nonzero(
                    valid, as_tuple=False).squeeze(1)
                if valid_indices.numel() == 0:
                    continue
                num_selected = min(per_cam, int(valid_indices.numel()))
                _, order = torch.topk(
                    score[valid_indices], k=num_selected,
                    largest=True, sorted=True)
                selected_priors = valid_indices[order]
                candidates.append((
                    torch.full_like(selected_priors, image_id),
                    selected_priors,
                    score[selected_priors]))

            if not candidates:
                continue
            image_ids = torch.cat([item[0] for item in candidates], dim=0)
            prior_ids = torch.cat([item[1] for item in candidates], dim=0)
            scores = torch.cat([item[2] for item in candidates], dim=0)
            num_base_queries = int(image_ids.numel())
            hyp_k = depth_values.shape[-1]

            if hyp_k > 1:
                hyp_confs = depth_confs[image_ids, prior_ids]
                hyp_values = depth_values[image_ids, prior_ids]
                hyp_indices = depth_indices[image_ids, prior_ids]
                base_scores = proposal_score[image_ids, prior_ids]
                hyp_scores = base_scores.unsqueeze(-1).expand_as(hyp_confs)
                hyp_keep = torch.ones_like(hyp_confs, dtype=torch.bool)
                if depth_range_min_bin is not None:
                    far_seed = hyp_indices[:, 0] >= depth_range_min_bin
                    hyp_keep[:, 1:] = far_seed.unsqueeze(-1)
                depth_ratio = (
                    hyp_confs / hyp_confs[:, :1].clamp_min(1e-6))

                base_image_ids = image_ids
                base_prior_ids = prior_ids
                extra_rows = []
                extra_hypotheses = []
                for cam_id in range(num_cams):
                    camera_base = (
                        base_image_ids % num_cams) == cam_id
                    num_empty = (
                        sample_max_per_cam
                        - int(camera_base.sum().item()))
                    if num_empty <= 0:
                        continue
                    extra_keep = (
                        hyp_keep[:, 1:]
                        & camera_base.unsqueeze(-1))
                    extra_pairs = torch.nonzero(
                        extra_keep, as_tuple=False)
                    if extra_pairs.numel() == 0:
                        continue
                    extra_scores = hyp_scores[
                        extra_pairs[:, 0], extra_pairs[:, 1] + 1]
                    extra_k = min(
                        num_empty, int(extra_pairs.shape[0]))
                    _, extra_order = torch.topk(
                        extra_scores, k=extra_k,
                        largest=True, sorted=True)
                    selected_pairs = extra_pairs[extra_order]
                    extra_rows.append(selected_pairs[:, 0])
                    extra_hypotheses.append(
                        selected_pairs[:, 1] + 1)

                base_hypotheses = torch.zeros(
                    num_base_queries, dtype=torch.long, device=device)
                if extra_rows:
                    selected_rows = torch.cat([
                        torch.arange(num_base_queries, device=device),
                        torch.cat(extra_rows, dim=0)])
                    selected_hypotheses = torch.cat([
                        base_hypotheses,
                        torch.cat(extra_hypotheses, dim=0)])
                else:
                    selected_rows = torch.arange(
                        num_base_queries, device=device)
                    selected_hypotheses = base_hypotheses

                image_ids = base_image_ids[selected_rows]
                prior_ids = base_prior_ids[selected_rows]
                scores = hyp_scores[
                    selected_rows, selected_hypotheses]
                selected_depth_values = hyp_values[
                    selected_rows, selected_hypotheses]
                selected_proposal_scores = base_scores[selected_rows]
                selected_depth_ratio = depth_ratio[
                    selected_rows, selected_hypotheses]
            else:
                selected_depth_values = depth_values[
                    image_ids, prior_ids, 0]
                selected_proposal_scores = proposal_score[
                    image_ids, prior_ids]
                selected_depth_ratio = torch.ones_like(
                    selected_proposal_scores)


            if scores.numel() > max_queries:
                if num_base_queries > max_queries:
                    raise ValueError(
                        'max_adaptive_queries must cover every selected '
                        'Top1 proposal when fill-only Top2 is enabled')
                num_extra = max_queries - num_base_queries
                base_order = torch.arange(
                    num_base_queries, device=device)
                if num_extra > 0:
                    _, extra_order = torch.topk(
                        scores[num_base_queries:], k=num_extra,
                        largest=True, sorted=True)
                    order = torch.cat([
                        base_order,
                        extra_order + num_base_queries])
                else:
                    order = base_order
                image_ids = image_ids[order]
                prior_ids = prior_ids[order]
                scores = scores[order]
                selected_depth_values = selected_depth_values[order]
                selected_proposal_scores = (
                    selected_proposal_scores[order])
                selected_depth_ratio = selected_depth_ratio[order]

            cameras = image_ids % num_cams
            centers = proposal_centers[image_ids, prior_ids]
            u = centers[:, 0].clamp(0, input_img_w - 1)
            v = centers[:, 1].clamp(0, input_img_h - 1)
            depth = selected_depth_values.clamp(min_depth, max_depth)
            lidar2img = context.new_tensor(
                img_metas[batch_id]['lidar2img'])[:num_cams]
            img2lidar = torch.inverse(lidar2img[cameras])
            points_image = torch.stack([
                u * depth, v * depth, depth, torch.ones_like(depth)
            ], dim=-1)
            xyz = torch.matmul(
                img2lidar, points_image.unsqueeze(-1)
            ).squeeze(-1)[..., :3]
            reference = ((xyz - pc_min) / (pc_max - pc_min)).detach()

            context_input = context[image_ids, prior_ids].detach()
            score_for_log = selected_proposal_scores.detach().clamp(
                1e-4, 1.0 - 1e-4)
            threshold = min(
                max(log_odds_thr, 1e-4), 1.0 - 1e-4)
            threshold_logit = math.log(
                threshold / (1.0 - threshold))
            score_log_odds = (
                torch.log(score_for_log / (1.0 - score_for_log))
                - threshold_logit)
            score_log_odds = (
                score_log_odds
                * selected_depth_ratio.detach())
            adaptive_content = self.far3d_stagea_context_embed(
                torch.cat([
                    context_input, score_log_odds.unsqueeze(-1)
                ], dim=-1))
            adaptive_pos = self.query_embedding(
                pos2posemb3d(reference))

            num_queries = scores.numel()
            query_content[batch_id, :num_queries] = adaptive_content
            query_pos[batch_id, :num_queries] = adaptive_pos
            reference_points[batch_id, :num_queries] = reference
            query_padding_mask[batch_id, :num_queries] = False

        max_used_queries = int(
            (~query_padding_mask).sum(dim=1).max().item())
        return dict(
            query_content=query_content[:, :max_used_queries],
            query_pos=query_pos[:, :max_used_queries],
            reference_points=reference_points[:, :max_used_queries],
            query_padding_mask=query_padding_mask[:, :max_used_queries])

    def _far3d_stagea_dense_depth_loss(
            self, depth_logits, depth_targets, foreground_mask):
        """Compute Far3D DDN focal loss over foreground and background."""
        logits = depth_logits.float()
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_index = depth_targets.unsqueeze(1)
        target_log_prob = torch.gather(
            log_prob, dim=1, index=target_index).squeeze(1)
        target_prob = torch.gather(
            prob, dim=1, index=target_index).squeeze(1)
        alpha = float(self._stagea_get('ddn_focal_alpha', 0.25))
        gamma = float(self._stagea_get('ddn_focal_gamma', 2.0))
        focal_loss = (
            -alpha
            * (1.0 - target_prob).pow(gamma)
            * target_log_prob)
        foreground_weight = float(
            self._stagea_get('ddn_foreground_weight', 13.0))
        background_weight = float(
            self._stagea_get('ddn_background_weight', 1.0))
        pixel_weights = torch.where(
            foreground_mask,
            focal_loss.new_full((), foreground_weight),
            focal_loss.new_full((), background_weight))
        loss = (focal_loss * pixel_weights).sum() / max(
            depth_targets.numel(), 1)

        with torch.no_grad():
            predicted_bins = logits.argmax(dim=1)
            valid_foreground = (
                foreground_mask
                & (depth_targets < self.far3d_stagea_depth_bins))
            if valid_foreground.any():
                depth_centers = self.far3d_stagea_depth_centers.to(
                    device=logits.device, dtype=logits.dtype)
                predicted_depth = depth_centers[
                    predicted_bins[valid_foreground]]
                target_depth = depth_centers[
                    depth_targets[valid_foreground]].clamp_min(1e-3)
                depth_error = predicted_depth - target_depth
                depth_ratio = torch.maximum(
                    predicted_depth.clamp_min(1e-3) / target_depth,
                    target_depth / predicted_depth.clamp_min(1e-3))
                metrics = dict(
                    mae=depth_error.abs().mean(),
                    rmse=depth_error.square().mean().sqrt(),
                    bias=depth_error.mean(),
                    delta1=(depth_ratio < 1.25).float().mean())
            else:
                zero = loss.detach() * 0.0
                metrics = dict(
                    mae=zero, rmse=zero, bias=zero, delta1=zero)
        return loss, metrics

    def _far3d_stagea_loss(self, preds_list, targets):
        flat = targets['flat']
        pos_masks = targets['pos_masks']
        num_total = targets['num_total_samples']
        loss_depth, depth_metrics = self._far3d_stagea_dense_depth_loss(
            preds_list[0]['depth_logits'],
            targets['p3_depth_targets'],
            targets['p3_depth_foreground_mask'])
        obj_logits = flat['objectness'].reshape(-1, 1)
        obj_targets = targets['obj_targets'].reshape(-1, 1)
        loss_obj = F.binary_cross_entropy_with_logits(
            obj_logits, obj_targets, reduction='sum') / num_total
        # The context MLP is skipped when no proposal reaches the threshold;
        # retain it in the DDP graph without changing the loss value.
        graph_zero = (
            flat['cls_logits'].sum() +
            flat['bbox_raw'].sum() +
            flat['center_raw'].sum() +
            flat['depth_logits'].sum()) * 0.0
        graph_zero = graph_zero + sum(
            parameter.sum() * 0.0
            for parameter in self.far3d_stagea_context_embed.parameters())
        loss_obj = loss_obj + graph_zero
        if pos_masks.sum() == 0:
            zero = loss_obj * 0.0
            return dict(
                far3d_2d_loss_score=loss_obj * float(
                    self._stagea_get('loss_score_weight', 1.0)),
                far3d_2d_loss_cls=zero,
                far3d_2d_loss_iou=zero,
                far3d_2d_loss_bbox=zero,
                far3d_2d_loss_centers2d=zero,
                far3d_2d_loss_depth=loss_depth * float(
                    self._stagea_get('loss_depth_weight', 1.0)),
                far3d_2d_depth_mae=depth_metrics['mae'],
                far3d_2d_depth_rmse=depth_metrics['rmse'],
                far3d_2d_depth_bias=depth_metrics['bias'],
                far3d_2d_depth_delta1=depth_metrics['delta1'])

        loss_cls = F.binary_cross_entropy_with_logits(
            flat['cls_logits'][pos_masks], targets['cls_targets'][pos_masks],
            reduction='sum') / num_total
        ious = bbox_overlaps(
            flat['decoded_bboxes'][pos_masks],
            targets['bbox_targets'][pos_masks], is_aligned=True).clamp(
                min=0.0, max=1.0)
        loss_iou = (1.0 - ious.square()).sum() / num_total
        loss_bbox = F.l1_loss(
            flat['bbox_raw'][pos_masks],
            targets['bbox_l1_targets'][pos_masks],
            reduction='sum') / num_total
        loss_center = F.l1_loss(
            flat['center_raw'][pos_masks], targets['center_targets'][pos_masks],
            reduction='sum') / num_total
        return dict(
            far3d_2d_loss_score=loss_obj * float(
                self._stagea_get('loss_score_weight', 1.0)),
            far3d_2d_loss_cls=loss_cls * float(
                self._stagea_get('loss_cls_weight', 1.0)),
            far3d_2d_loss_iou=loss_iou * float(
                self._stagea_get('loss_iou_weight', 1.0)),
            far3d_2d_loss_bbox=loss_bbox * float(
                self._stagea_get('loss_bbox_weight', 1.0)),
            far3d_2d_loss_centers2d=loss_center * float(
                self._stagea_get('loss_center_weight', 1.0)),
            far3d_2d_loss_depth=loss_depth * float(
                self._stagea_get('loss_depth_weight', 1.0)),
            far3d_2d_depth_mae=depth_metrics['mae'],
            far3d_2d_depth_rmse=depth_metrics['rmse'],
            far3d_2d_depth_bias=depth_metrics['bias'],
            far3d_2d_depth_delta1=depth_metrics['delta1'])

    def position_embeding(self, img_feats, img_metas, masks=None, depth_map=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_map: (B, N_view, H, W)
            depth_map_mask: (B, N_view, H, W)
        Returns:
            coords_position_embeding: (B, N_view, embed_dims, H, W)
            coords_mask: (B, N_view, H, W)
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H  # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W  # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, W, H, 2).repeat(B, N, 1, 1, 1)       # (B, N_view, W, H, 2)
        self.coords2d = coords

        depth_map = depth_map.permute(0, 1, 3, 2).contiguous()      # (B, N_view, W, H)

        # inplace
        # coords = torch.cat((coords, depth_map.unsqueeze(dim=-1)), dim=-1)       # (B, N_view, W, H, 3)   3:(u, v, d)
        # coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)    # (B, N_view, W, H, 4)    4: (u, v, d, 1)
        # coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(
        #     coords[..., 2:3]) * eps)  # (B, N_view, W, H, 4)    4: (du, dv, d, 1)

        depth_map = depth_map.unsqueeze(dim=-1)     # (B, N_view, W, H, 1)
        coords = coords * torch.maximum(depth_map, torch.ones_like(depth_map) * eps)  # (B, N_view, W, H, 2)    (du, dv)
        coords = torch.cat([coords, depth_map], dim=-1)     # (B, N_view, W, H, 3)   (du, dv, d)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)  # (B, N_view, W, H, 4)   (du, dv, d, 1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N_view, 4, 4)

        coords = coords.unsqueeze(dim=-1)       # (B, N_view, W, H, 4, 1)
        # (B, N_view, 1, 1, 4, 4) --> (B, N_view, W, H, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)


        # 图像中每个像素对应的frustum points，借助img2lidars投影到lidar系中.
        # (B, N_view, W, H, 4, 4) @ (B, N_view, H, D, 4, 1) --> (B, N_view, W, H, 4, 1)
        # --> (B, N_view, W, H, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        self.coords3d = coords3d

        # 借助position_range，对3D坐标进行归一化.
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                    self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                    self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                    self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (B, N_view, W, H, 3), 超出range的points mask
        coords_mask = coords_mask.sum(dim=-1) > 0       # (B, N_view, W, H)
        # 在后续attention过程中， 会消除这些像素的影响.
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # (B, N_view, H, W)

        coords3d = coords3d.permute(0, 1, 3, 2, 4).contiguous().view(B*N, H, W, 3)      # (B*N_view, H, W, 3)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, H, W, 3)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(pos2posemb3d(coords3d))  # (B*N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(0, 3, 1, 2).contiguous()    # (B*N_view, embed_dims, H, W)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas, gt_bboxes_3d=None,
                gt_labels_3d=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        # Dense depth is predicted from position_level (P4); proposal
        # supervision and the selected decoder path are configured separately.
        x = mlvl_feats[self.position_level]
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]

        x = x.flatten(0, 1)  # (B*N_view, C, H, W)
        surface_confidence = None
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']  # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']  # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)  # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)  # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()  # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()  # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W), (B*N_view, 1, H, W)
            if self.with_pgd:
                depth, x, depth_direct = self.depth_net(
                    x, intrinsics, extrinsics)
            else:
                # (B * N_view, D/1, H, W),  (B*N_view, C, H, W)
                depth, x = self.depth_net(
                    x, intrinsics, extrinsics)
            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            if self.use_prob_depth:
                self.depth_score = depth   # 未经过softmax
                depth_prob = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B*N_view, H, W, D) --> (B*N_view*H*W, D)
                depth_prob_val = self.integral(depth_prob)      # (B*N_view*H*W, )
                depth_map_pred = depth_prob_val
                if self.use_global_sparse_surface_pe:
                    surface_confidence = F.softmax(
                        depth.float(), dim=1).amax(dim=1).reshape(
                            batch_size, num_cams, fH, fW)

                if self.with_pgd:
                    sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                    depth_direct_val = depth_direct.view(-1)      # (B*N_view*H*W, )
                    depth_pgd_fuse = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
                    depth_map_pred = depth_pgd_fuse
            else:
                # direct depth
                depth_map_pred = depth.exp().view(-1)     # (B*N_view*H*W, )
        else:
            depth_map_pred = None

        surface_depth = None
        if self.use_global_sparse_surface_pe:
            if depth_map_pred is None:
                raise RuntimeError(
                    'Global sparse Surface PE requires predicted depth')
            surface_depth = depth_map_pred.reshape(
                batch_size, num_cams, fH, fW)
            if surface_confidence is None:
                surface_confidence = torch.ones_like(surface_depth)

        base_reference_points = self.reference_points.weight
        if self.use_sparse_multiscale_decoder:
            # The sparse decoder embeds normalized references directly.
            base_query_pos = self.query_embedding(
                pos2posemb3d(base_reference_points))
        else:
            # The original DenseGT decoder keeps inverse-sigmoid references.
            base_query_pos = self.query_embedding(
                pos2posemb3d(inverse_sigmoid(base_reference_points)))
        reference_points = base_reference_points.unsqueeze(0).expand(
            batch_size, -1, -1)
        query_key_padding_mask = None
        far3d_stagea_preds = None
        far3d_stagea_targets = None
        far3d_range_dn_meta = None
        stagea_num_cams = int(self._stagea_get(
            'supervised_num_cams', num_cams))
        if not 0 < stagea_num_cams <= num_cams:
            raise ValueError(
                'StageA supervised_num_cams must be in '
                f'[1, {num_cams}], got {stagea_num_cams}')

        if (self.use_far3d_stagea
                and not self.use_sparse_multiscale_decoder
                and self.training):
            num_feature_levels = min(
                int(self._stagea_get(
                    'num_feature_levels', len(mlvl_feats))),
                len(mlvl_feats))
            stagea_feats = [
                feat[:, :stagea_num_cams].flatten(0, 1)
                for feat in mlvl_feats[:num_feature_levels]]
            far3d_stagea_preds = self.far3d_stagea_head(stagea_feats)
            for preds in far3d_stagea_preds:
                preds['batch_size'] = batch_size
                preds['num_cams'] = stagea_num_cams
            far3d_stagea_targets = self._build_far3d_stagea_targets(
                far3d_stagea_preds, img_metas)

        if (self.use_far3d_stagea
                and self.use_sparse_multiscale_decoder):
            prepared_features = self.transformer.prepare_features(
                mlvl_feats, img_metas, base_reference_points.dtype)

            num_feature_levels = min(
                int(self._stagea_get(
                    'num_feature_levels', len(mlvl_feats))),
                len(mlvl_feats))
            stagea_feats = [
                feat[:, :stagea_num_cams].flatten(0, 1)
                for feat in mlvl_feats[:num_feature_levels]]
            far3d_stagea_preds = self.far3d_stagea_head(stagea_feats)
            for preds in far3d_stagea_preds:
                preds['batch_size'] = batch_size
                preds['num_cams'] = stagea_num_cams
            if self.training:
                far3d_stagea_targets = self._build_far3d_stagea_targets(
                    far3d_stagea_preds, img_metas)

            if self.append_far3d_adaptive_queries:
                stagea_context = prepared_features[0].view(
                    batch_size, num_cams, -1, self.embed_dims
                )[:, :stagea_num_cams].reshape(
                    batch_size * stagea_num_cams, -1,
                    self.embed_dims)
                stagea_query = self._decode_far3d_stagea_queries(
                    far3d_stagea_preds, stagea_context, img_metas,
                    input_img_h, input_img_w, far3d_stagea_targets)
                if self.training:
                    self.far3d_stagea_iter.add_(1)

                global_query_pos = base_query_pos.unsqueeze(0).expand(
                    batch_size, -1, -1)
                global_query_content = torch.zeros_like(global_query_pos)
                if stagea_query['query_content'].size(1) == 0:
                    # Keep the Adaptive content MLP in the DDP graph when no
                    # proposal survives filtering for the whole local batch.
                    for parameter in self.far3d_stagea_context_embed.parameters():
                        global_query_content = (
                            global_query_content + parameter.sum() * 0.0)
                global_padding_mask = torch.zeros(
                    batch_size, self.num_query, dtype=torch.bool,
                    device=global_query_pos.device)

                joint_query_pos = torch.cat(
                    [global_query_pos, stagea_query['query_pos']], dim=1)
                joint_query_content = torch.cat(
                    [global_query_content,
                     stagea_query['query_content']], dim=1)
                reference_points = torch.cat(
                    [reference_points,
                     stagea_query['reference_points']], dim=1)
                query_key_padding_mask = torch.cat(
                    [global_padding_mask,
                     stagea_query['query_padding_mask']], dim=1)
                (
                    joint_query_pos, joint_query_content,
                    reference_points, query_key_padding_mask,
                    self_attn_mask, far3d_range_dn_meta
                ) = self._prepare_far3d_range_dn_queries(
                    joint_query_pos, joint_query_content,
                    reference_points, query_key_padding_mask,
                    gt_bboxes_3d, gt_labels_3d)


                outs_dec, _ = self.transformer(
                    mlvl_feats=mlvl_feats,
                    query_embed=joint_query_pos,
                    reference_points=reference_points,
                    img_metas=img_metas,
                    pc_range=self.pc_range,
                    query_content=joint_query_content,
                    query_key_padding_mask=query_key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    prepared_features=prepared_features,
                    surface_depth=surface_depth,
                    surface_confidence=surface_confidence)
            else:
                # Auxiliary-only ablation: StageA still learns from its 2D
                # losses, while the sparse decoder sees only Global Queries.
                query_key_padding_mask = torch.zeros(
                    batch_size, self.num_query, dtype=torch.bool,
                    device=base_reference_points.device)
                if self.training and self.use_far3d_range_dn:
                    # Decisive control: Range-DN regularizes the same sparse
                    # decoder, but normal detection uses Global Queries only.
                    global_query_pos = base_query_pos.unsqueeze(0).expand(
                        batch_size, -1, -1)
                    global_query_content = torch.zeros_like(
                        global_query_pos)
                    (
                        global_query_pos, global_query_content,
                        reference_points, query_key_padding_mask,
                        self_attn_mask, far3d_range_dn_meta
                    ) = self._prepare_far3d_range_dn_queries(
                        global_query_pos, global_query_content,
                        reference_points, query_key_padding_mask,
                        gt_bboxes_3d, gt_labels_3d)
                    outs_dec, _ = self.transformer(
                        mlvl_feats=mlvl_feats,
                        query_embed=global_query_pos,
                        reference_points=reference_points,
                        img_metas=img_metas,
                        pc_range=self.pc_range,
                        query_content=global_query_content,
                        query_key_padding_mask=query_key_padding_mask,
                        self_attn_mask=self_attn_mask,
                        prepared_features=prepared_features,
                        surface_depth=surface_depth,
                        surface_confidence=surface_confidence)
                else:
                    outs_dec, _ = self.transformer(
                        mlvl_feats=mlvl_feats,
                        query_embed=base_query_pos,
                        reference_points=reference_points,
                        img_metas=img_metas,
                        pc_range=self.pc_range,
                        query_content=None,
                        query_key_padding_mask=query_key_padding_mask,
                        prepared_features=prepared_features,
                        surface_depth=surface_depth,
                        surface_confidence=surface_confidence)
        elif self.use_sparse_multiscale_decoder:
            prepared_features = self.transformer.prepare_features(
                mlvl_feats, img_metas, base_reference_points.dtype)
            query_key_padding_mask = torch.zeros(
                batch_size, self.num_query, dtype=torch.bool,
                device=base_reference_points.device)

            outs_dec, _ = self.transformer(
                mlvl_feats=mlvl_feats,
                query_embed=base_query_pos,
                reference_points=reference_points,
                img_metas=img_metas,
                pc_range=self.pc_range,
                query_content=None,
                query_key_padding_mask=query_key_padding_mask,
                prepared_features=prepared_features,
                surface_depth=surface_depth,
                surface_confidence=surface_confidence)
        else:
            masks = x.new_ones(
                (batch_size, num_cams, input_img_h, input_img_w))
            for img_id in range(batch_size):
                for cam_id in range(num_cams):
                    img_h, img_w, _ = img_metas[
                        img_id]['img_shape'][cam_id]
                    masks[img_id, cam_id, :img_h, :img_w] = 0

            x = self.input_proj(x)
            x = x.view(
                batch_size, num_cams, *x.shape[-3:])
            masks = F.interpolate(
                masks, size=x.shape[-2:]).to(torch.bool)

            if self.with_position:
                depth_map = (
                    depth_map_pred.detach()
                    if self.use_detach else depth_map_pred)
                depth_map = depth_map.view(
                    batch_size, num_cams, fH, fW)
                self.depth_map = depth_map
                coords_position_embeding, _ = self.position_embeding(
                    mlvl_feats, img_metas, masks, depth_map)

                if self.with_fpe:
                    coords_position_embeding = self.fpe(
                        coords_position_embeding.flatten(0, 1),
                        x.flatten(0, 1)).view(x.size())

                pos_embed = coords_position_embeding
                if self.with_multiview:
                    sin_embed = self.positional_encoding(masks)
                    sin_embed = self.adapt_pos3d(
                        sin_embed.flatten(0, 1)).view(x.size())
                    pos_embed = pos_embed + sin_embed
                elif self.with_2dpe_only:
                    pos_embeds = []
                    for cam_id in range(num_cams):
                        xy_embed = self.positional_encoding(
                            masks[:, cam_id, :, :])
                        pos_embeds.append(xy_embed.unsqueeze(1))
                    sin_embed = torch.cat(pos_embeds, 1)
                    sin_embed = self.adapt_pos3d(
                        sin_embed.flatten(0, 1)).view(x.size())
                    pos_embed = pos_embed + sin_embed
            else:
                if self.with_multiview:
                    pos_embed = self.positional_encoding(masks)
                    pos_embed = self.adapt_pos3d(
                        pos_embed.flatten(0, 1)).view(x.size())
                elif self.with_2dpe_only:
                    pos_embeds = []
                    for cam_id in range(num_cams):
                        pos_embed = self.positional_encoding(
                            masks[:, cam_id, :, :])
                        pos_embeds.append(pos_embed.unsqueeze(1))
                    pos_embed = torch.cat(pos_embeds, 1)
                else:
                    pos_embed = x.new_zeros(x.size())

            outs_dec, _ = self.transformer(
                x,
                masks,
                base_query_pos,
                pos_embed,
                self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)  # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        if self.with_time:
            time_stamps = []
            for img_meta in img_metas:
                time_stamps.append(np.asarray(img_meta['timestamp']))
            time_stamp = reference_points.new_tensor(time_stamps)
            time_stamp = time_stamp.view(batch_size, -1, 6)     # (B, N_frame=2, N_view=6)
            # (B, N_view) - (B, N_view) --> (B, N_view) --> (B, )
            mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

        if self.with_pos_info:
            pos_feat = self.extra_position_encoder(inverse_sigmoid(reference_points))   # (B, N_query, C)
            outs_dec = outs_dec + pos_feat[None, ...]   # (num_layers, B, N_query, C=embed_dims)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())  # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # normalized_cz

            if self.with_time:
                # (vx, vy) = (distance_x, distance_y) / tx
                tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)  # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)  # (num_layers, B, N_query, code_size)
        if query_key_padding_mask is not None:
            all_cls_scores = all_cls_scores.masked_fill(
                query_key_padding_mask[None, :, :, None], -20.0)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        if far3d_range_dn_meta is not None:
            pad_size = int(far3d_range_dn_meta['pad_size'])
            far3d_range_dn_meta['output_known_lbs_bboxes'] = (
                all_cls_scores[:, :, :pad_size],
                all_bbox_preds[:, :, :pad_size])
            all_cls_scores = all_cls_scores[:, :, pad_size:]
            all_bbox_preds = all_bbox_preds[:, :, pad_size:]
            query_key_padding_mask = query_key_padding_mask[:, pad_size:]


        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        if self.use_far3d_range_dn:
            outs['far3d_range_dn_meta'] = far3d_range_dn_meta
        if self.use_far3d_stagea:
            outs['far3d_stagea_preds'] = far3d_stagea_preds
            outs['far3d_stagea_targets'] = far3d_stagea_targets
            outs['far3d_stagea_query_padding_mask'] = query_key_padding_mask

        if self.depth_net is not None:
            outs['depth_map_pred'] = depth_map_pred    # (B*N_view*H*W, )
            if self.use_prob_depth:
                outs['depth_prob'] = depth_prob     # (B*N_view*H*W, D)
                if self.with_pgd:
                    outs['depth_prob_val'] = depth_prob_val  # (B*N_view*H*W, )
                    outs['depth_direct_val'] = depth_direct_val  # (B*N_view*H*W, )

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             depth_map,
             depth_map_mask,
             img_metas=None,
             gt_bboxes_ignore=None
             ):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            depth_map:  # (B*N_views, fH, fW)
            depth_map_mask:  # (B*N_views, fH, fW)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            img_metas:
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']      # (num_layers, B, N_query, n_cls)
        all_bbox_preds = preds_dicts['all_bbox_preds']      # (num_layers, B, N_query, code_size)   code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 分别计算每层decoder layer的loss
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        if self.use_far3d_range_dn:
            dn_meta = preds_dicts.get('far3d_range_dn_meta', None)
            if dn_meta is not None:
                (
                    known_labels, known_bboxs,
                    output_known_class, output_known_coord,
                    num_dn_targets
                ) = self._prepare_far3d_range_dn_loss_targets(dn_meta)
                dn_losses_cls, dn_losses_bbox = multi_apply(
                    self._far3d_range_dn_loss_single,
                    output_known_class,
                    output_known_coord,
                    [known_bboxs for _ in range(num_dec_layers)],
                    [known_labels for _ in range(num_dec_layers)],
                    [num_dn_targets for _ in range(num_dec_layers)])
            else:
                dn_losses_cls = [
                    all_cls_scores[layer_id].sum() * 0.0
                    for layer_id in range(num_dec_layers)]
                dn_losses_bbox = [
                    all_bbox_preds[layer_id].sum() * 0.0
                    for layer_id in range(num_dec_layers)]

            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            for layer_id, (loss_cls_i, loss_bbox_i) in enumerate(zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1])):
                loss_dict[
                    f'd{layer_id}.dn_loss_cls'] = loss_cls_i
                loss_dict[
                    f'd{layer_id}.dn_loss_bbox'] = loss_bbox_i


        if self.with_depth_supervision:
            depth_map_pred = preds_dicts['depth_map_pred']    # (B*N_view*H*W, )
            depth_prob = preds_dicts.get('depth_prob', None)    # (B*N_view*H*W, D)

            depth_loss_dict = self.depth_loss(depth_map_pred, depth_map, depth_map_mask, depth_prob,
                                              gt_boxes3d=gt_bboxes_list, img_metas=img_metas)
            loss_dict.update(depth_loss_dict)

        if self.use_far3d_stagea and preds_dicts.get('far3d_stagea_targets', None) is not None:
            stagea_loss_dict = self._far3d_stagea_loss(
                preds_dicts['far3d_stagea_preds'],
                preds_dicts['far3d_stagea_targets'])
            loss_dict.update(stagea_loss_dict)
        self.loss_dict = loss_dict
        return loss_dict

    def _add_depth_error_metrics(self, loss_dict, depth_pred, depth_tgt, valid, prefix='depth_map'):
        """Add logging-only depth error metrics.

        The returned tensors do not contain ``loss`` in their names, so MMDet
        logs them but does not add them into the optimized training loss.
        """
        zero = depth_pred.new_zeros(())
        with torch.no_grad():
            valid = valid & torch.isfinite(depth_pred) & torch.isfinite(depth_tgt)
            valid = valid & (depth_tgt > 0)
            if valid.sum() == 0:
                loss_dict[f'{prefix}_mae'] = zero
                loss_dict[f'{prefix}_rmse'] = zero
                loss_dict[f'{prefix}_absrel'] = zero
                loss_dict[f'{prefix}_delta1'] = zero
                loss_dict[f'{prefix}_bias'] = zero
                loss_dict[f'{prefix}_scale'] = zero
                return

            pred = depth_pred[valid].float()
            tgt = depth_tgt[valid].float()
            pred_safe = pred.clamp_min(1e-3)
            tgt_safe = tgt.clamp_min(1e-3)
            diff = pred - tgt
            abs_diff = diff.abs()
            ratio = torch.max(pred_safe / tgt_safe, tgt_safe / pred_safe)

            loss_dict[f'{prefix}_mae'] = abs_diff.mean()
            loss_dict[f'{prefix}_rmse'] = torch.sqrt((diff ** 2).mean().clamp_min(1e-12))
            loss_dict[f'{prefix}_absrel'] = (abs_diff / tgt_safe).mean()
            loss_dict[f'{prefix}_delta1'] = (ratio < 1.25).float().mean()
            loss_dict[f'{prefix}_bias'] = diff.mean()
            loss_dict[f'{prefix}_scale'] = pred_safe.mean() / tgt_safe.mean().clamp_min(1e-3)

    def depth_loss(self, depth_map_pred, depth_map_tgt, depth_map_mask, depth_prob=None, gt_boxes3d=None,
                   img_metas=None):
        """
        Args:
            depth_map_pred: (B*N_view*H*W, )
            gt_boxes3d: (B*N_view, H, W)
            depth_map_mask: (B*N_view, H, W)
            depth_prob: (B*N_view*H*W, D)
            gt_boxes3d: List[(N0, 9), (N1, 9), ...]
            img_metas: List[img_meta0, img_meta1, ...]
        Returns:

        """
        batch_size = len(gt_boxes3d)
        n_view = depth_map_tgt.shape[0] // batch_size
        fH, fW = depth_map_tgt.shape[1], depth_map_tgt.shape[2]

        depth_map_mask = depth_map_mask.view(-1).float()                # (B*N_view*H*W, )
        depth_map_tgt = depth_map_tgt.view(-1)                          # (B*N_view*H*W, )

        min_dist = self.depth_start
        depth_map_tgt, depth_map_mask = self.mask_points_by_dist(depth_map_tgt, depth_map_mask,
                                                                 min_dist=min_dist,
                                                                 max_dist=self.position_range[3])

        valid = depth_map_mask > 0
        valid_depth_pred = depth_map_pred[valid]      # (N_valid, )

        loss_dict = {}
        self._add_depth_error_metrics(
            loss_dict, depth_map_pred, depth_map_tgt, valid, prefix='depth_map')
        loss_depth = self.loss_depth(pred=valid_depth_pred, target=depth_map_tgt[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))

        if self.use_balancer:
            shape = (batch_size, n_view, fH, fW)
            fg_loss_depth, bg_loss_depth = self.balancer(loss_depth, shape, gt_boxes3d, depth_map_mask, img_metas)
            loss_dict['fg_loss_depth'] = fg_loss_depth
            loss_dict['bg_loss_depth'] = bg_loss_depth
        else:
            loss_dict['loss_depth'] = loss_depth

        if self.use_dfl and depth_prob is not None:
            bin_size = (self.position_range[3] - min_dist) / (self.depth_num - 1)
            depth_label_clip = (depth_map_tgt - min_dist) / bin_size
            depth_map_clip, depth_map_mask = self.mask_points_by_dist(depth_label_clip, depth_map_mask, 0,
                                                                      self.depth_num - 1)      # (B*N_view*H*W, )

            valid = depth_map_mask > 0      # (B*N_view*H*W, )
            valid_depth_prob = depth_prob[valid]    # (N_valid, )
            loss_dfl = self.loss_dfl(pred=valid_depth_prob, target=depth_map_clip[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))

            if self.use_balancer:
                shape = (batch_size, n_view, fH, fW)
                fg_loss_dfl, bg_loss_dfl = self.balancer(loss_dfl, shape, gt_boxes3d, depth_map_mask, img_metas)
                loss_dict['fg_loss_dfl'] = fg_loss_dfl
                loss_dict['bg_loss_dfl'] = bg_loss_dfl
            else:
                loss_dict['loss_dfl'] = loss_dfl

        return loss_dict
