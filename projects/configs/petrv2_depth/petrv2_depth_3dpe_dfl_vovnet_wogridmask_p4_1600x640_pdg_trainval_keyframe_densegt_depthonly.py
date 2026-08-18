import os

_base_ = ['./petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt.py']

# DenseGT depth-only pretraining/diagnosis.
# It uses the same Metric3D 0618 dense depth target as the DenseGT keyframe
# config, but skips transformer decoder, cls/bbox heads, and Hungarian matching.
# Use this only to study predicted depth map quality or to initialize a later
# full 3DPPE finetune; do not directly report mAP/NDS from this checkpoint.

model = dict(depth_only=True)

find_unused_parameters = True

total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

evaluation = dict(interval=999999, start=999999)
