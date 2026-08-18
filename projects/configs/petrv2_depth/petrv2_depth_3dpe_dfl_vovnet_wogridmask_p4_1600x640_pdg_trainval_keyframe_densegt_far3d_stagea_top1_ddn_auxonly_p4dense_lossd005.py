_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_auxonly_lossd005.py'
]

# Decisive OAGA ablation:
#   DenseGT supervision:       enabled
#   P3-P6 OAGA auxiliary loss: enabled during training
#   Adaptive Query injection:  disabled
#   Decoder:                   original 3DPPE P4 dense decoder
#
# P3-P6 remain available to the proposal head, while the detector follows the
# original DenseGT path: P4 context + dense 3D image PE + 900 Global Queries.
model = dict(
    pts_bbox_head=dict(
        use_sparse_multiscale_decoder=False,
        far3d_transformer=None,
        depthnet=dict(with_context=True)))

work_dir = 'work_dirs/stagea_top1_ddn_auxonly_p4dense_lossd005'
