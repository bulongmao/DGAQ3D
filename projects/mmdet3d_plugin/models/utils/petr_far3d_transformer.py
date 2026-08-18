# ------------------------------------------------------------------------
# Adapted from the official Far3D deformable feature aggregation decoder.
# ------------------------------------------------------------------------
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, xavier_init
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (
    TransformerLayerSequence, build_attention, build_feedforward_network,
    build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """Encode normalized xyz coordinates with the 3DPPE sine/cosine PE."""
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=-1).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        dim=-1).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()),
        dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


class CameraGeometryModulation(nn.Module):
    """Modulate flattened image features with camera calibration."""

    def __init__(self, condition_dims=14, embed_dims=256, use_ln=False):
        super().__init__()
        self.use_ln = use_ln
        self.reduce = nn.Sequential(
            nn.Linear(condition_dims, embed_dims),
            nn.ReLU(inplace=True))
        self.gamma = nn.Linear(embed_dims, embed_dims)
        self.beta = nn.Linear(embed_dims, embed_dims)
        if use_ln:
            self.norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feature, condition):
        if self.use_ln:
            feature = self.norm(feature)
        condition = self.reduce(condition)
        return self.gamma(condition) * feature + self.beta(condition)


@TRANSFORMER.register_module()
class PETRFar3DTransformer(BaseModule):
    """Far3D-style sparse multi-level decoder for the 3DPPE head."""

    uses_multilevel_features = True

    def __init__(self,
                 decoder,
                 num_feature_levels=4,
                 num_cams=6,
                 use_spatial_alignment=True,
                 intrinsic_scale=1000.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_spatial_alignment = use_spatial_alignment
        self.intrinsic_scale = intrinsic_scale
        if use_spatial_alignment:
            self.spatial_alignment = CameraGeometryModulation(
                condition_dims=14, embed_dims=self.embed_dims, use_ln=False)
        surface_pe_flags = [
            bool(getattr(layer.cross_attn, 'use_global_surface_pe', False))
            for layer in self.decoder.layers
        ]
        if any(surface_pe_flags) and not all(surface_pe_flags):
            raise ValueError(
                'Global sparse Surface PE must be enabled for every decoder '
                'layer or disabled for every decoder layer')
        self.uses_global_sparse_surface_pe = bool(
            surface_pe_flags and all(surface_pe_flags))

    def init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if module is not self and hasattr(module, 'init_weight'):
                module.init_weight()
        self._is_init = True

    @staticmethod
    def _stack_meta_tensor(img_metas, key, reference):
        if any(key not in meta for meta in img_metas):
            raise KeyError(
                f'PETRFar3DTransformer requires "{key}" in img_metas')
        return torch.stack(
            [reference.new_tensor(meta[key]) for meta in img_metas], dim=0)

    def _camera_geometry(self, img_metas, reference, num_cams):
        intrinsics = self._stack_meta_tensor(
            img_metas, 'intrinsics', reference)[:, :num_cams]
        extrinsics = self._stack_meta_tensor(
            img_metas, 'extrinsics', reference)[:, :num_cams]
        condition = torch.cat([
            intrinsics[..., 0, 0:1] / self.intrinsic_scale,
            intrinsics[..., 1, 1:2] / self.intrinsic_scale,
            extrinsics[..., :3, :].flatten(-2),
        ], dim=-1)
        return condition.flatten(0, 1).unsqueeze(1)

    @staticmethod
    def _pad_shapes(img_metas, reference, num_cams):
        shapes = []
        for meta in img_metas:
            if 'pad_shape' not in meta or len(meta['pad_shape']) < num_cams:
                raise KeyError(
                    'PETRFar3DTransformer requires per-camera pad_shape')
            shapes.append([
                [shape[0], shape[1]]
                for shape in meta['pad_shape'][:num_cams]
            ])
        return reference.new_tensor(shapes)

    def _flatten_features(
            self, mlvl_feats, img_metas, output_dtype=None):
        if len(mlvl_feats) < self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} FPN levels, got '
                f'{len(mlvl_feats)}')
        batch_size, num_cams = mlvl_feats[0].shape[:2]
        if num_cams != self.num_cams:
            raise ValueError(
                f'Configured for {self.num_cams} cameras, got {num_cams}')

        camera_condition = None
        if self.use_spatial_alignment:
            camera_condition = self._camera_geometry(
                img_metas, mlvl_feats[0], num_cams)

        flattened = []
        spatial_shapes = []
        for feature in mlvl_feats[:self.num_feature_levels]:
            if feature.shape[:2] != (batch_size, num_cams):
                raise ValueError(
                    'All FPN levels must share batch/camera axes')
            if feature.size(2) != self.embed_dims:
                raise ValueError(
                    f'FPN channels must be {self.embed_dims}, got '
                    f'{feature.size(2)}')
            height, width = feature.shape[-2:]
            feature = feature.reshape(
                batch_size * num_cams, self.embed_dims, -1
            ).transpose(1, 2)
            if self.use_spatial_alignment:
                feature = self.spatial_alignment(
                    feature, camera_condition)
            if output_dtype is not None:
                feature = feature.to(dtype=output_dtype)
            flattened.append(feature)
            spatial_shapes.append((height, width))

        flattened = torch.cat(flattened, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=flattened.device)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])
        return flattened, spatial_shapes, level_start_index

    def prepare_features(self, mlvl_feats, img_metas, output_dtype):
        """Flatten and geometry-modulate FPN maps once for Query and decoder."""
        return self._flatten_features(
            mlvl_feats, img_metas, output_dtype=output_dtype)

    def forward(self,
                mlvl_feats,
                query_embed,
                reference_points,
                img_metas,
                pc_range,
                query_content=None,
                query_key_padding_mask=None,
                self_attn_mask=None,
                prepared_features=None,
                surface_depth=None,
                surface_confidence=None,
                **kwargs):
        batch_size = mlvl_feats[0].size(0)
        num_cams = mlvl_feats[0].size(1)
        if prepared_features is None:
            feature, spatial_shapes, level_start_index = (
                self.prepare_features(
                    mlvl_feats, img_metas, query_embed.dtype))
        else:
            if len(prepared_features) != 3:
                raise ValueError(
                    'prepared_features must contain feature, spatial_shapes '
                    'and level_start_index')
            feature, spatial_shapes, level_start_index = prepared_features
            feature = feature.to(dtype=query_embed.dtype)

        if query_embed.dim() == 2:
            query_pos = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        elif query_embed.dim() == 3:
            query_pos = query_embed
        else:
            raise ValueError(f'Unsupported query_embed shape {query_embed.shape}')

        if query_content is None:
            query = torch.zeros_like(query_pos)
        elif query_content.dim() == 2:
            query = query_content.unsqueeze(0).expand(batch_size, -1, -1)
        elif query_content.dim() == 3:
            query = query_content
        else:
            raise ValueError(
                f'Unsupported query_content shape {query_content.shape}')

        if reference_points.shape[:2] != query.shape[:2]:
            raise ValueError(
                'reference_points and query must have matching batch/query '
                'dimensions')
        if query_key_padding_mask is None:
            query_key_padding_mask = torch.zeros(
                query.shape[:2], dtype=torch.bool, device=query.device)

        lidar2img = self._stack_meta_tensor(
            img_metas, 'lidar2img', feature)[:, :num_cams]
        pad_shapes = self._pad_shapes(img_metas, feature, num_cams)
        pc_range = feature.new_tensor(pc_range)

        output = self.decoder(
            query=query,
            query_pos=query_pos,
            mlvl_feats=feature,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            pc_range=pc_range,
            lidar2img=lidar2img,
            pad_shapes=pad_shapes,
            query_key_padding_mask=query_key_padding_mask,
            self_attn_mask=self_attn_mask,
            surface_depth=surface_depth,
            surface_confidence=surface_confidence)
        return output, feature


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRFar3DTransformerDecoder(TransformerLayerSequence):
    """Decoder sequence returning all intermediate query features."""

    def __init__(self, *args, return_intermediate=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                reference_points,
                spatial_shapes,
                level_start_index,
                pc_range,
                lidar2img,
                pad_shapes,
                query_key_padding_mask,
                self_attn_mask=None,
                surface_depth=None,
                surface_confidence=None):
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                mlvl_feats=mlvl_feats,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                pc_range=pc_range,
                lidar2img=lidar2img,
                pad_shapes=pad_shapes,
                query_key_padding_mask=query_key_padding_mask,
                self_attn_mask=self_attn_mask,
                surface_depth=surface_depth,
                surface_confidence=surface_confidence)
            query = query.masked_fill(
                query_key_padding_mask.unsqueeze(-1), 0.0)
            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return query.unsqueeze(0)


@TRANSFORMER_LAYER.register_module()
class PETRFar3DDecoderLayer(BaseModule):
    """Self-attention, sparse Far3D cross-attention, then FFN."""

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=2048,
                 ffn_dropout=0.1,
                 ffn_num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                  'ffn', 'norm'),
                 batch_first=True,
                 with_cp=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        if operation_order != ('self_attn', 'norm', 'cross_attn', 'norm',
                               'ffn', 'norm'):
            raise ValueError('PETRFar3DDecoderLayer requires post-norm order')
        if not batch_first:
            raise ValueError('PETRFar3DDecoderLayer requires batch_first=True')
        if len(attn_cfgs) != 2:
            raise ValueError('Expected self-attention and cross-attention')
        self.operation_order = operation_order
        self.pre_norm = False

        attn_cfgs = copy.deepcopy(attn_cfgs)
        for cfg in attn_cfgs:
            cfg.setdefault('batch_first', True)
        self.self_attn = build_attention(attn_cfgs[0])
        self.cross_attn = build_attention(attn_cfgs[1])
        self.embed_dims = self.self_attn.embed_dims
        if self.cross_attn.embed_dims != self.embed_dims:
            raise ValueError('Self/cross attention embed_dims must match')

        self.norms = nn.ModuleList([
            build_norm_layer(norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ])
        self.ffn = build_feedforward_network(dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=ffn_num_fcs,
            ffn_drop=ffn_dropout,
            act_cfg=act_cfg))
        self.use_checkpoint = with_cp

    def _forward(self,
                 query,
                 query_pos,
                 mlvl_feats,
                 reference_points,
                 spatial_shapes,
                 level_start_index,
                 pc_range,
                 lidar2img,
                 pad_shapes,
                 query_key_padding_mask,
                 self_attn_mask=None,
                 surface_depth=None,
                 surface_confidence=None):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            identity=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=query_key_padding_mask)
        query = self.norms[0](query)
        query = self.cross_attn(
            query,
            query_pos,
            mlvl_feats,
            reference_points,
            spatial_shapes,
            level_start_index,
            pc_range,
            lidar2img,
            pad_shapes,
            query_key_padding_mask,
            surface_depth=surface_depth,
            surface_confidence=surface_confidence)
        query = self.norms[1](query)
        query = self.ffn(query, identity=query)
        return self.norms[2](query)

    def forward(self, **kwargs):
        query = kwargs['query']
        args = (
            query,
            kwargs['query_pos'],
            kwargs['mlvl_feats'],
            kwargs['reference_points'],
            kwargs['spatial_shapes'],
            kwargs['level_start_index'],
            kwargs['pc_range'],
            kwargs['lidar2img'],
            kwargs['pad_shapes'],
            kwargs['query_key_padding_mask'],
            kwargs.get('self_attn_mask', None),
        )
        surface_depth = kwargs.get('surface_depth', None)
        surface_confidence = kwargs.get('surface_confidence', None)
        if surface_depth is not None or surface_confidence is not None:
            args = args + (surface_depth, surface_confidence)
        if self.use_checkpoint and self.training and query.requires_grad:
            return cp.checkpoint(self._forward, *args)
        return self._forward(*args)


@ATTENTION.register_module()
class Far3DDeformableFeatureAggregationCuda(BaseModule):
    """Official Far3D sparse 3D-to-multiview feature aggregation."""

    def __init__(self,
                 embed_dims=256,
                 num_groups=8,
                 num_levels=4,
                 num_cams=6,
                 dropout=0.1,
                 num_pts=13,
                 im2col_step=64,
                 batch_first=True,
                 bias=2.0,
                 surface_pe_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if not batch_first:
            raise ValueError(
                'Far3DDeformableFeatureAggregationCuda requires '
                'batch_first=True')
        if embed_dims % num_groups != 0:
            raise ValueError('embed_dims must be divisible by num_groups')
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.im2col_step = im2col_step
        self.bias = bias
        self.surface_pe_cfg = dict(surface_pe_cfg or {})
        self.use_global_surface_pe = bool(
            self.surface_pe_cfg.get('enabled', False))
        self.num_global_queries = int(
            self.surface_pe_cfg.get('num_global_queries', 900))
        self.surface_eps = float(
            self.surface_pe_cfg.get('eps', 1e-5))
        self.surface_max_depth = float(
            self.surface_pe_cfg.get('max_depth', 61.2))
        self.surface_relation_clip = float(
            self.surface_pe_cfg.get('relation_clip', 2.0))
        self.surface_log_ratio_clip = float(
            self.surface_pe_cfg.get('log_ratio_clip', 4.0))
        self.surface_detach_depth = bool(
            self.surface_pe_cfg.get('detach_depth', True))
        self.surface_gate_bias = float(
            self.surface_pe_cfg.get('gate_bias', -2.0))
        if self.use_global_surface_pe:
            if self.num_global_queries <= 0:
                raise ValueError('num_global_queries must be positive')
            if self.surface_eps <= 0 or self.surface_max_depth <= self.surface_eps:
                raise ValueError(
                    'Surface PE requires 0 < eps < max_depth')
            if self.embed_dims % 2 != 0:
                raise ValueError('Surface PE requires an even embed_dims')

        self.weights_fc = nn.Linear(
            embed_dims, num_groups * num_levels * num_pts)
        self.learnable_fc = nn.Linear(embed_dims, num_pts * 3)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims))
        self.drop = nn.Dropout(dropout)

        if self.use_global_surface_pe:
            self.surface_position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims))
            self.surface_relation_encoder = nn.Sequential(
                nn.Linear(6, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims))
            self.surface_gate = nn.Linear(6, self.embed_dims)
            self.surface_output_proj = nn.Linear(
                self.embed_dims, self.embed_dims)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution='uniform', bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias, -self.bias, self.bias)
        if self.use_global_surface_pe:
            for encoder in (
                    self.surface_position_encoder,
                    self.surface_relation_encoder):
                for module in encoder.modules():
                    if isinstance(module, nn.Linear):
                        xavier_init(
                            module, distribution='uniform', bias=0.0)
            constant_init(
                self.surface_gate, val=0.0, bias=self.surface_gate_bias)
            # This makes the complete detector exactly equal to the validated
            # StageA function before the first optimization step.
            constant_init(self.surface_output_proj, val=0.0, bias=0.0)

    @staticmethod
    def _global_reference_points(reference_points, pc_range):
        return (reference_points * (pc_range[3:6] - pc_range[:3])
                + pc_range[:3])

    def _get_weights(self,
                     instance_feature,
                     query_pos,
                     lidar2img,
                     query_key_padding_mask):
        batch_size, num_queries = instance_feature.shape[:2]
        camera_embed = self.cam_embed(
            lidar2img[..., :3, :].flatten(-2))
        feature = ((instance_feature + query_pos).unsqueeze(2)
                   + camera_embed.unsqueeze(1))
        logits = self.weights_fc(feature).view(
            batch_size, num_queries, self.num_cams, self.num_levels,
            self.num_pts, self.num_groups)
        weights = logits.view(
            batch_size, num_queries, -1, self.num_groups).softmax(dim=2)
        weights = weights.view(
            batch_size, num_queries, self.num_cams, self.num_levels,
            self.num_pts, self.num_groups)
        weights = weights.permute(0, 2, 1, 5, 3, 4).contiguous()
        weights = weights.view(
            batch_size * self.num_cams, num_queries, self.num_groups,
            self.num_levels, self.num_pts)
        padding = query_key_padding_mask[:, None, :, None, None, None]
        padding = padding.expand(
            -1, self.num_cams, -1, self.num_groups, self.num_levels,
            self.num_pts).reshape_as(weights)
        return weights.masked_fill(padding, 0.0)

    def _feature_sampling(self,
                          feat_flatten,
                          spatial_shapes,
                          level_start_index,
                          key_points,
                          weights,
                          lidar2img,
                          pad_shapes):
        batch_size, num_queries, num_pts = key_points.shape[:3]
        points = torch.cat([
            key_points, torch.ones_like(key_points[..., :1])
        ], dim=-1)
        points_2d = torch.matmul(
            lidar2img[:, :, None, None],
            points[:, None, :, :, :, None]).squeeze(-1)
        points_2d = points_2d[..., :2] / points_2d[..., 2:3].clamp_min(1e-5)
        normalizer = torch.stack([
            pad_shapes[..., 1], pad_shapes[..., 0]
        ], dim=-1)[:, :, None, None]
        points_2d = points_2d / normalizer
        points_2d = points_2d[:, :, :, None, None, :, :].expand(
            -1, -1, -1, self.num_groups, self.num_levels, -1, -1)
        points_2d = points_2d.reshape(
            batch_size * self.num_cams, num_queries, self.num_groups,
            self.num_levels, num_pts, 2).contiguous()

        num_camera_batches, num_values = feat_flatten.shape[:2]
        value = feat_flatten.reshape(
            num_camera_batches, num_values, self.num_groups,
            self.group_dims)
        if value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, points_2d,
                weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, points_2d, weights)
        output = output.reshape(
            batch_size, self.num_cams, num_queries, self.embed_dims)
        return output.sum(dim=1)

    def _surface_geometry_aggregation(
            self, key_points, weights, surface_depth, surface_confidence,
            lidar2img, pad_shapes, pc_range):
        """Build key-side surface geometry at the sparse sample points."""
        batch_size, num_queries, num_pts = key_points.shape[:3]
        num_global = min(self.num_global_queries, num_queries)
        if num_global <= 0:
            return key_points.new_zeros(
                batch_size, 0, self.embed_dims)
        if surface_depth is None:
            raise RuntimeError(
                'Global sparse Surface PE requires surface_depth')
        if surface_depth.dim() != 4:
            raise ValueError(
                'surface_depth must have shape [B, Ncam, H, W], got '
                f'{tuple(surface_depth.shape)}')
        if surface_depth.shape[:2] != (batch_size, self.num_cams):
            raise ValueError(
                'surface_depth batch/camera axes do not match sparse '
                'aggregation inputs')
        if surface_confidence is not None:
            if surface_confidence.shape != surface_depth.shape:
                raise ValueError(
                    'surface_confidence must match surface_depth shape')

        # Projection and back-projection are kept in FP32 because pixel*depth
        # products and matrix inverses are not numerically safe in FP16.
        depth_map = surface_depth.float()
        confidence_map = (
            torch.ones_like(depth_map)
            if surface_confidence is None
            else surface_confidence.float())
        if self.surface_detach_depth:
            depth_map = depth_map.detach()
            confidence_map = confidence_map.detach()

        eps = self.surface_eps
        global_points = key_points[:, :num_global].float()
        lidar2img_fp32 = lidar2img.float()
        points_h = torch.cat(
            [global_points, torch.ones_like(global_points[..., :1])],
            dim=-1)
        points_cam = torch.matmul(
            lidar2img_fp32[:, :, None, None],
            points_h[:, None, :, :, :, None]).squeeze(-1)
        key_depth = points_cam[..., 2]
        pixel_xy = (
            points_cam[..., :2]
            / key_depth.unsqueeze(-1).clamp_min(eps))
        normalizer = torch.stack(
            [pad_shapes[..., 1], pad_shapes[..., 0]], dim=-1
        ).float()[:, :, None, None]
        points_2d = pixel_xy / normalizer
        finite_projection = (
            torch.isfinite(key_depth)
            & torch.isfinite(points_2d).all(dim=-1))
        projection_valid = (
            finite_projection
            & (key_depth > eps)
            & (points_2d[..., 0] >= 0.0)
            & (points_2d[..., 0] <= 1.0)
            & (points_2d[..., 1] >= 0.0)
            & (points_2d[..., 1] <= 1.0))

        sample_grid = torch.nan_to_num(
            points_2d * 2.0 - 1.0,
            nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        sample_grid = sample_grid.reshape(
            batch_size * self.num_cams, num_global, num_pts, 2)
        sampled_depth = F.grid_sample(
            depth_map.reshape(
                batch_size * self.num_cams, 1,
                depth_map.size(-2), depth_map.size(-1)),
            sample_grid,
            mode='bilinear', padding_mode='zeros',
            align_corners=False).reshape(
                batch_size, self.num_cams, num_global, num_pts)
        sampled_confidence = F.grid_sample(
            confidence_map.reshape(
                batch_size * self.num_cams, 1,
                confidence_map.size(-2), confidence_map.size(-1)),
            sample_grid,
            mode='bilinear', padding_mode='zeros',
            align_corners=False).reshape(
                batch_size, self.num_cams, num_global, num_pts)

        depth_valid = (
            torch.isfinite(sampled_depth)
            & (sampled_depth > eps)
            & (sampled_depth <= self.surface_max_depth))
        confidence_valid = torch.isfinite(sampled_confidence)
        valid = projection_valid & depth_valid & confidence_valid
        sampled_depth = torch.nan_to_num(
            sampled_depth, nan=eps,
            posinf=self.surface_max_depth, neginf=eps).clamp(
                min=eps, max=self.surface_max_depth)
        sampled_confidence = torch.nan_to_num(
            sampled_confidence, nan=0.0,
            posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # Surface depth and key-point depth lie on the same camera ray but
        # have different semantics: visible surface versus object hypothesis.
        safe_points_2d = torch.nan_to_num(
            points_2d, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        safe_pixel_xy = safe_points_2d * normalizer
        surface_image_points = torch.cat([
            safe_pixel_xy * sampled_depth.unsqueeze(-1),
            sampled_depth.unsqueeze(-1),
            torch.ones_like(sampled_depth).unsqueeze(-1),
        ], dim=-1)
        img2lidar = torch.inverse(lidar2img_fp32)
        surface_points = torch.matmul(
            img2lidar[:, :, None, None],
            surface_image_points.unsqueeze(-1)).squeeze(-1)[..., :3]

        pc_range_fp32 = pc_range.float()
        pc_extent = (
            pc_range_fp32[3:6] - pc_range_fp32[:3]).clamp_min(eps)
        surface_reference = (
            surface_points - pc_range_fp32[:3]) / pc_extent
        range_valid = (
            torch.isfinite(surface_reference).all(dim=-1)
            & (surface_reference >= 0.0).all(dim=-1)
            & (surface_reference <= 1.0).all(dim=-1))
        valid = valid & range_valid
        surface_reference = torch.nan_to_num(
            surface_reference, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(eps, 1.0 - eps)

        query_points = global_points[:, None].expand(
            -1, self.num_cams, -1, -1, -1)
        relative_xyz = (
            (query_points - surface_points) / pc_extent
        ).clamp(-self.surface_relation_clip, self.surface_relation_clip)
        depth_delta = (
            (key_depth[:, :, :num_global] - sampled_depth)
            / self.surface_max_depth
        ).clamp(-self.surface_relation_clip, self.surface_relation_clip)
        log_depth_ratio = torch.log(
            key_depth[:, :, :num_global].clamp_min(eps)
            / sampled_depth.clamp_min(eps)
        ).clamp(-self.surface_log_ratio_clip, self.surface_log_ratio_clip)
        relation = torch.cat([
            relative_xyz,
            depth_delta.unsqueeze(-1),
            log_depth_ratio.unsqueeze(-1),
            sampled_confidence.unsqueeze(-1),
        ], dim=-1)
        relation = torch.nan_to_num(
            relation, nan=0.0, posinf=0.0, neginf=0.0)

        output_dtype = weights.dtype
        surface_pe = pos2posemb3d(
            surface_reference,
            num_pos_feats=self.embed_dims // 2).to(output_dtype)
        relation = relation.to(output_dtype)
        surface_feature = (
            self.surface_position_encoder(surface_pe)
            + self.surface_relation_encoder(relation))
        surface_gate = torch.sigmoid(self.surface_gate(relation))
        surface_feature = (
            surface_feature
            * surface_gate
            * sampled_confidence.to(output_dtype).unsqueeze(-1)
            * valid.to(output_dtype).unsqueeze(-1))

        # Reuse the original camera/level/point attention weights. Surface PE
        # is level-independent, so summing over levels gives each projected
        # point the exact mass assigned by the sparse image aggregation.
        num_camera_batches = batch_size * self.num_cams
        grouped_surface = surface_feature.reshape(
            num_camera_batches, num_global, num_pts,
            self.num_groups, self.group_dims).permute(0, 1, 3, 2, 4)
        surface_weights = weights[:, :num_global].sum(dim=3)
        aggregated = (
            grouped_surface * surface_weights.unsqueeze(-1)).sum(dim=3)
        aggregated = aggregated.reshape(
            batch_size, self.num_cams, num_global,
            self.embed_dims).sum(dim=1)
        return self.surface_output_proj(aggregated)

    def forward(self,
                instance_feature,
                query_pos,
                feat_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                pc_range,
                lidar2img,
                pad_shapes,
                query_key_padding_mask,
                surface_depth=None,
                surface_confidence=None):
        if spatial_shapes.size(0) != self.num_levels:
            raise ValueError(
                f'Configured for {self.num_levels} levels, got '
                f'{spatial_shapes.size(0)}')
        if lidar2img.size(1) != self.num_cams:
            raise ValueError(
                f'Configured for {self.num_cams} cameras, got '
                f'{lidar2img.size(1)}')

        reference_points = self._global_reference_points(
            reference_points, pc_range)
        offsets = self.learnable_fc(instance_feature).view(
            instance_feature.size(0), instance_feature.size(1),
            self.num_pts, 3)
        key_points = reference_points.unsqueeze(2) + offsets
        weights = self._get_weights(
            instance_feature, query_pos, lidar2img,
            query_key_padding_mask)
        features = self._feature_sampling(
            feat_flatten, spatial_shapes, level_start_index, key_points,
            weights, lidar2img, pad_shapes)
        output = self.output_proj(features)
        if self.use_global_surface_pe:
            surface_delta = self._surface_geometry_aggregation(
                key_points, weights, surface_depth, surface_confidence,
                lidar2img, pad_shapes, pc_range)
            num_global = surface_delta.size(1)
            if num_global < instance_feature.size(1):
                surface_delta = torch.cat([
                    surface_delta,
                    surface_delta.new_zeros(
                        surface_delta.size(0),
                        instance_feature.size(1) - num_global,
                        surface_delta.size(2)),
                ], dim=1)
            output = output + surface_delta
        output = self.drop(output) + instance_feature
        return output.masked_fill(
            query_key_padding_mask.unsqueeze(-1), 0.0)
