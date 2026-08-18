import os

_base_ = ['./petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe.py']

# Depth-only pretraining/diagnosis for the Transformer-left part of 3DPPE Fig.3.
# It keeps image backbone + neck + depth branch, but skips position embedding,
# transformer decoder, classification, box regression, and Hungarian matching.
# This checkpoint is for depth analysis / initialization, not direct mAP/NDS eval.

model = dict(depth_only=True)

# Some FPN/detection-head parameters are intentionally unused in this mode.
find_unused_parameters = True

# Depth-only is a short diagnostic/pretrain stage by default. Override from CLI
# if a longer run is needed: --cfg-options runner.max_epochs=12 total_epochs=12
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Do not run 3D detection evaluation for depth-only checkpoints.
evaluation = dict(interval=999999, start=999999)
