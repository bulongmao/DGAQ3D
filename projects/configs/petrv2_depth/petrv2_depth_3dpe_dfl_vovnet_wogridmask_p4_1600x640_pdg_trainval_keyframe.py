import os

_base_ = ['./petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval.py']

# Clean 3DPPE 1600x640 train-only keyframe baseline for temporal ablation.
#
# This config keeps the official 1600x640 DFL/PGD 3DPPE setting unchanged
# except for removing image-level sweeps. It is intended as the high-resolution
# A0 baseline before adding StreamPETR/Far3D-style temporal memory.
#
# Differences from the official 1600x640 trainval config:
#   - sweeps_num: 1 -> 0 in train/test pipeline;
#   - pts_bbox_head.with_time: True -> False, because there is no second frame
#     timestamp for velocity normalization;
#   - train ann_file: train+val -> train only for clean val ablation.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
_data_root = 'data/nuscenes/'
if not _data_root.endswith('/'):
    _data_root += '/'
data_root = _data_root
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
        with_time=False))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=0,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=False,
        sweep_range=[3, 27]),
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
        type='LoadDepthByMapplingPoints2Images',
        src_size=(900, 1600),
        input_size=ida_aug_conf['final_dim'],
        downsample=16),
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
            'transformation_3d_flow', 'img_info', 'intrinsics',
            'extrinsics', 'timestamp'
        ])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=0,
        to_float32=True,
        pad_empty_sweeps=True,
        sweep_range=[3, 27]),
    dict(type='ResizeCropFlipImageV2', data_aug_conf=ida_aug_conf, training=False),
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
                    'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
                    'img_info', 'intrinsics', 'extrinsics', 'timestamp'
                ])
        ])
]

# For local val-set ablation, keep training strictly train-only.
# The inherited official 1600 trainval base uses train+val for test-server
# training, so we explicitly override ann_file here to avoid val leakage.
data = dict(
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
