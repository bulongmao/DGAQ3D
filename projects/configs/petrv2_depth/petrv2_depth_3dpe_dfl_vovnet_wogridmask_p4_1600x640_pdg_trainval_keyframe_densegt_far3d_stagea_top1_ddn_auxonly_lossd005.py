_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_lossd005.py'
]

# Controlled OAQG ablation:
#   DenseGT supervision:       enabled
#   P3-P6 sparse decoder:      enabled
#   OAQG proposal head/loss:   enabled
#   Adaptive Query injection:  disabled
#
# The proposal branch still receives all configured auxiliary losses and
# updates the shared FPN. The decoder receives only the original 900 Global
# Queries, isolating auxiliary feature supervision from query injection.
model = dict(
    pts_bbox_head=dict(
        append_far3d_adaptive_queries=False))

work_dir = 'work_dirs/stagea_top1_ddn_auxonly_lossd005'
