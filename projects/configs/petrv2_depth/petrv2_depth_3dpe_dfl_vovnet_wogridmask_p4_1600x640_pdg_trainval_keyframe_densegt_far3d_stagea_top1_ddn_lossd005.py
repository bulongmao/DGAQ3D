_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_clean.py'
]

# Top1 StageA with full P3 DDN supervision and 22k-iteration GT-depth warmup.
model = dict(
    pts_bbox_head=dict(
        loss_depth=dict(loss_weight=0.05),
        far3d_stagea_cfg=dict(
            depth_topk=1,
            train_use_gt_depth=True,
            gt_depth_warmup_iters=22000,
            loss_score_weight=0.2,
            loss_cls_weight=0.2,
            loss_iou_weight=1.0,
            loss_bbox_weight=0.2,
            loss_center_weight=0.0,
            loss_depth_weight=0.2)))

data = dict(samples_per_gpu=1)

checkpoint_config = dict(interval=1, max_keep_ckpts=10)

total_epochs = 26
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
work_dir = 'work_dirs/stagea_top1_ddn_warmup_lossd005'
