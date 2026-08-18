_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt.py'
]

# Controlled decoder ablation:
#   DenseGT supervision:       enabled
#   P3-P6 sparse decoder:      enabled
#   OAQG proposal head/loss:   disabled
#   Adaptive Query injection:  disabled
#
# The only Decoder inputs are the original 900 learnable Global Queries.
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
        use_far3d_stagea=False,
        use_sparse_multiscale_decoder=True,
        use_global_sparse_surface_pe=False,
        # Sparse decoding reads P3-P6 directly, so the legacy P4 context
        # output is not constructed. The P4 depth prediction/loss remains.
        depthnet=dict(with_context=False),
        loss_depth=dict(loss_weight=0.05),
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
                        'ffn', 'norm'))))))

data = dict(samples_per_gpu=1)

checkpoint_config = dict(interval=1, max_keep_ckpts=10)

total_epochs = 26
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
work_dir = 'work_dirs/densegt_global_sparse_decoder_lossd005'
