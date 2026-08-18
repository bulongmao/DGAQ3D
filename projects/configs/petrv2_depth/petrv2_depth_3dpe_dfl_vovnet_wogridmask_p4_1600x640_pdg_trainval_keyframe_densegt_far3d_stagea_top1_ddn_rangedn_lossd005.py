_base_ = [
    './petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_lossd005.py'
]

# Official Far3D range-modulated 3D denoising:
# one scale-modulated recoverable sample and two range-modulated negatives.
model = dict(
    pts_bbox_head=dict(
        far3d_stagea_cfg=dict(
            range_dn=dict(
                enabled=True,
                scalar=10,
                noise_scale=1.0,
                noise_trans=0.0,
                dn_weight=1.0,
                offset=0.5,
                offset_p=0.0,
                num_smp_per_gt=3,
                query_num_dn=600))))

work_dir = 'work_dirs/stagea_top1_ddn_rangedn_lossd005'
