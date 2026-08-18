import os

_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_lossd005.py'
]

# Match the original 3DPPE temporal-input setup:
#   6 current views + 1 historical sweep x 6 views = 12 image slots.
# All 12 slots pass through the backbone, FPN, dense-depth branch and sparse
# decoder. OAQG supervision remains on the six current-frame views because
# the offline 2DGT describes the current keyframe.

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
    pts_bbox_head=dict(
        with_time=True,
        append_far3d_adaptive_queries=True,
        far3d_transformer=dict(
            num_cams=12,
            decoder=dict(
                transformerlayers=dict(
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
                            num_cams=12,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.0),
                    ]))),
        far3d_stagea_cfg=dict(
            supervised_num_cams=6)))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
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
    dict(
        type='ResizeCropFlipImageV2',
        data_aug_conf=ida_aug_conf,
        training=True),
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
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'depth_map',
            'depth_map_mask'
        ],
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

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=True,
        sweep_range=[3, 27]),
    dict(
        type='ResizeCropFlipImageV2',
        data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                    'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_info', 'intrinsics',
                    'extrinsics', 'timestamp'
                ])
        ])
]

data = dict(
    samples_per_gpu=1,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=train_pipeline),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_val.pkl',
        pipeline=test_pipeline),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'petr/mmdet3d_nuscenes_30f_infos_val.pkl',
        pipeline=test_pipeline))

evaluation = dict(interval=1, start=22, pipeline=test_pipeline)

work_dir = 'work_dirs/stagea_top1_ddn_sweep1_lossd005'
