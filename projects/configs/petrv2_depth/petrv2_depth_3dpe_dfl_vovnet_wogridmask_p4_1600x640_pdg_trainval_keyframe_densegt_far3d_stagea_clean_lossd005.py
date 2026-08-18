_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_clean.py'
]

# Joint Global/Adaptive StageA experiment with DenseGT depth loss at 0.05.
model = dict(
    pts_bbox_head=dict(
        loss_depth=dict(loss_weight=0.05)))

# Keep one sample per GPU so adaptive queries are truly dynamic per sample.
data = dict(samples_per_gpu=1)

checkpoint_config = dict(interval=1, max_keep_ckpts=10)

total_epochs = 26
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
work_dir = 'work_dirs/stagea_lossd005_joint'
