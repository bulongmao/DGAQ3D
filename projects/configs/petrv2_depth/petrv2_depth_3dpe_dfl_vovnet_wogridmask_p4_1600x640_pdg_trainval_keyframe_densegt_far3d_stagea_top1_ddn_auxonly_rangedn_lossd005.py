_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py'
]

# Decisive Range-DN control:
#   Dense pseudo-depth:         enabled
#   P3-P6 OAQG auxiliary loss: enabled
#   Sparse multi-scale decoder: enabled
#   Range-DN:                  enabled during training
#   Adaptive Query injection:  disabled
#
# Normal detection therefore uses the original 900 Global Queries. Range-DN
# queries are prepended only during training and removed before normal losses
# and inference, isolating Range-DN from Adaptive Query injection.
model = dict(
    pts_bbox_head=dict(
        append_far3d_adaptive_queries=False))

work_dir = 'work_dirs/stagea_top1_ddn_auxonly_rangedn_lossd005'
