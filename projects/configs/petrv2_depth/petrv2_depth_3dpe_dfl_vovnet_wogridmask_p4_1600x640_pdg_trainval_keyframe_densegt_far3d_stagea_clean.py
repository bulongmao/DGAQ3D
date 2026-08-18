import os

_base_ = ['./petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt.py']

# Clean 3DPPE + Far3D StageA, single-frame/keyframe model.
# This config intentionally keeps out DDN region/background depth, 2D box NMS,
# dense-depth-valid proposal filtering and temporal memory. StageA uses
# four-level FPN proposal supervision. Global and Adaptive Queries are joined
# before one P3-P6 sparse deformable decoder, matching Far3D's core query flow.

_data_root = 'data/nuscenes/'
if not _data_root.endswith('/'):
    _data_root += '/'
data_root = _data_root
_dense_depth_root = os.environ.get(
    'DENSEGT_DEPTH_ROOT',
    'data/metric3d_depth')
_2dgt_path = os.environ.get(
    'NUSCENES_2DGT_PATH',
    'data/nuscenes/2dgt/nuscenes_train_2dgt.pkl')

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)
file_client_args = dict(backend='disk')
ida_aug_conf = {
    'resize': (-0.06, 0.11),
    'rot': (0.0, 0.0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.04,
    'H': 900,
    'W': 1600,
    'final_dim': (640, 1600),
}

model = dict(
    img_backbone=dict(
        out_features=('stage2', 'stage3', 'stage4', 'stage5')),
    img_neck=dict(
        type='FPN',
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        num_outs=4),
    pts_bbox_head=dict(
        with_time=False,
        position_level=1,
        use_far3d_stagea=True,
        # The Far3D decoder consumes P3-P6 FPN features directly. The legacy
        # P4-only context branch is therefore not part of this graph.
        depthnet=dict(with_context=False),
        far3d_transformer=dict(
            type='PETRFar3DTransformer',
            num_feature_levels=4,
            num_cams=6,
            use_spatial_alignment=True,
            intrinsic_scale=1000.0,
            decoder=dict(
                type='PETRFar3DTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRFar3DDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Far3DDeformableFeatureAggregationCuda',
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=(
                        'self_attn', 'norm', 'cross_attn', 'norm',
                        'ffn', 'norm')))),
        far3d_stagea_cfg=dict(
            num_feature_levels=4,
            # Official Far3D convention: 50 metric bins + 1 terminal/background class.
            depth_num_bins=50,
            strides=[8, 16, 32, 64],
            assigner=dict(type='SimOTAAssigner', center_radius=2.5),
            depth_bin_mode='lid',
            depth_min=1.0,
            depth_max=61.2,
            depth_aggregation='window',
            depth_window=3,
            # Official Far3D inference keeps one depth hypothesis per proposal.
            depth_topk=1,
            depth_range_min=30.0,
            train_use_gt_depth=True,
            gt_depth_warmup_iters=22000,
            ddn_focal_alpha=0.25,
            ddn_focal_gamma=2.0,
            ddn_foreground_weight=13.0,
            ddn_background_weight=1.0,
            sample_max_per_cam=16,
            topk_per_cam=16,
            max_adaptive_queries=96,
            score_thr=0.1,
            use_2d_score_localmax=True,
            score_localmax_kernel=3,
            loss_score_weight=1.0,
            loss_cls_weight=1.0,
            loss_iou_weight=5.0,
            loss_bbox_weight=1.0,
            loss_center_weight=1.0,
            loss_depth_weight=0.2)))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=0,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=False,
        sweep_range=[3, 27]),
    dict(type='LoadOffline2DGT', gt2d_path=_2dgt_path, min_box_size=2.0),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImageV2', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='LoadDenseDepthFromFiles',
        depth_root=_dense_depth_root,
        src_size=(450, 800),
        input_size=ida_aug_conf['final_dim'],
        downsample=16,
        max_dist=61.2),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'depth_map',
              'depth_map_mask'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
            'transformation_3d_flow', 'img_info', 'intrinsics', 'extrinsics',
            'timestamp', 'gt2d_boxes', 'gt2d_labels', 'gt2d_depths'
        ])
]

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=train_pipeline),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_val.pkl'),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_val.pkl'))

evaluation = dict(interval=1, start=22)
