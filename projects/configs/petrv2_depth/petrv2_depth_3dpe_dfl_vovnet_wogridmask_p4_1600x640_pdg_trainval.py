_base_ = [
    './petrv2_depth_3dpe_vovnet_wogridmask_p4_1600x640_trainval.py'
]

# Keep the local PGD baseline and every descendant ablation on the same
# DD3D initialization. This is intentionally explicit rather than relying on
# the parent config so future baseline edits cannot change the pretraining
# protocol silently.
load_from = 'ckpts/dd3d_det_final.pth'


model = dict(
    pts_bbox_head=dict(
        depthnet=dict(
            with_context_encoder=False,
            with_pgd=True,
            depth_channels=16,
        ),
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        use_dfl=True,
        with_fpe=False,
        depth_num=16,
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=0.1),
        loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=0.25),
    )
)
