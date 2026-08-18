_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_lossd005.py'
]

# Global-only sparse Surface PE experiment. The validated StageA query flow
# and P3-P6 sparse decoder remain unchanged. At every decoder layer, dense P4
# surface depth is sampled only at the 13 projected sparse key points. Its
# absolute 3D PE and center-surface relation form a zero-initialized residual
# for the first 900 Global Queries; Adaptive Queries and reference points are
# never modified by this branch.
model = dict(
    pts_bbox_head=dict(
        use_global_sparse_surface_pe=True,
        far3d_transformer=dict(
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
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.0,
                            surface_pe_cfg=dict(
                                enabled=True,
                                num_global_queries=900,
                                detach_depth=True,
                                max_depth=61.2,
                                relation_clip=2.0,
                                log_ratio_clip=4.0,
                                gate_bias=-2.0,
                                eps=1e-5)),
                    ])))))

# Keep the validated training-time proposal budget unchanged. For the final
# comparison, evaluate every checkpoint with score_thr=0.05 and 144 Adaptive
# Queries, exactly as for the 0.4892 mAP / 0.5534 NDS StageA result.
work_dir = 'work_dirs/stagea_top1_ddn_global_sparse_surfacepe_lossd005'
