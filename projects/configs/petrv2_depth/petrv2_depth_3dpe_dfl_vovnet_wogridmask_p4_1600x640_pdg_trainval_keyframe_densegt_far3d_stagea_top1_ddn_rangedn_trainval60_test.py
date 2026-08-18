import os

_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py'
]

# Table 2 protocol: single-frame 1600x640, train+val, DD3D initialization,
# 60 epochs, and nuScenes test-server export. Local ablations remain on the
# train/val split and must not use this config.
_data_root = os.environ.get(
    'NUSCENES_DATA_ROOT', 'data/nuscenes/').rstrip('/') + '/'
_info_prefix = _data_root + 'petr/mmdet3d_nuscenes_30f_infos_'

data = dict(
    train=dict(
        data_root=_data_root,
        ann_file=[_info_prefix + 'train.pkl', _info_prefix + 'val.pkl']),
    # Training must use --no-validate because test annotations are hidden.
    val=dict(
        data_root=_data_root,
        ann_file=_info_prefix + 'test.pkl'),
    test=dict(
        data_root=_data_root,
        ann_file=_info_prefix + 'test.pkl'))

total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=6)

# Keep the official 3DPPE test initialization explicit. The inherited
# optimizer, cosine schedule, 500-iteration warmup, and loss_scale=512 remain
# unchanged.
load_from = 'ckpts/dd3d_det_final.pth'
resume_from = None

# Disabled in practice by --no-validate; the large values also protect against
# accidental evaluation on the annotation-free test split.
evaluation = dict(interval=1000, start=1000)

work_dir = 'work_dirs/dgaq3d_keyframe_trainval60_test'
