# 表 2 官方测试集实验协议

本协议用于生成论文表 2 中与公开 nuScenes **test 测试集**结果进行比较的唯一主结果。
本地模块消融实验仍使用 nuScenes train 训练、val 评测，不得与本协议的结果混用。

## 一、固定实验协议

- 输入：6 个相机的当前关键帧图像（`sweeps_num=0`）
- 输入分辨率：`1600x640`
- Backbone：VoVNet-99-eSE
- 初始化权重：`ckpts/dd3d_det_final.pth`
- 训练数据：nuScenes train + val
- 训练周期：60 epochs
- 优化器：AdamW，初始学习率 `1e-4`
- 学习率策略：CosineAnnealing，前 500 iteration 线性 warmup
- 混合精度：固定 `loss_scale=512`
- 评测方式：nuScenes 官方 test server
- 最终推理参数：置信度阈值 `0.05`、每相机 Top-24、最多 144 个 Adaptive Query、Top-1 深度及 Window-3 深度聚合

3DPPE 论文将 test 主表标记为单时间戳输入。其公开的 `1600x640 trainval`
配置中虽然包含 `sweeps_num=1`，但本实验按照论文中的单帧定义以及 DGAQ-3D 的
关键帧设置，保持 `sweeps_num=0`。

## 二、为 val 数据生成并合并二维 GT

在本协议中，val 数据会参与训练。因此，OAQG 需要 val 集全部 36,114 张相机图像的
离线二维框、类别及相机坐标系目标中心深度。

首先进入训练代码目录：

```bash
cd /path/to/DGAQ3D
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

定义数据路径：

```bash
ROOT=data/nuscenes
TRAIN_INFO=$ROOT/petr/mmdet3d_nuscenes_30f_infos_train.pkl
VAL_INFO=$ROOT/petr/mmdet3d_nuscenes_30f_infos_val.pkl
TRAIN_2DGT=$ROOT/2dgt/nuscenes_train_2dgt.pkl
VAL_2DGT=$ROOT/2dgt/nuscenes_val_2dgt.pkl
TRAINVAL_2DGT=$ROOT/2dgt/nuscenes_trainval_2dgt.pkl
```

为 val 数据生成离线二维 GT：

```bash
python tools/generate_nuscenes_2dgt.py \
  "$VAL_INFO" "$VAL_2DGT" \
  --data-root "$ROOT" \
  --min-box-size 2.0 \
  --min-visible-corners 4
```

合并 train 与 val 的二维 GT，并检查两部分数据是否完整覆盖：

```bash
python tools/prepare_nuscenes_trainval_2dgt.py \
  --train-info "$TRAIN_INFO" \
  --val-info "$VAL_INFO" \
  --train-2dgt "$TRAIN_2DGT" \
  --val-2dgt "$VAL_2DGT" \
  --out-file "$TRAINVAL_2DGT"
```

脚本必须显示 train 和 val 均不存在缺失图像，即 `missing=0`。如果覆盖不完整，
脚本会直接报错并停止，避免模型在缺少 OAQG 二维监督的情况下静默训练。

训练前还需要再次运行 Metric3Dv2 builder 的 dry-run 检查。对于 val 关键帧，输出必须包含：

```text
selected_items=36114
existing_complete=36114
pending_generate=0
```

这表示 val 集的 Metric3Dv2 稠密伪深度已经全部生成完毕。

## 三、生成 nuScenes test info 文件

原始 nuScenes 数据目录必须包含：

- `v1.0-test`
- `samples`
- `sweeps`
- test 集对应的全部传感器文件

执行以下命令：

```bash
cd /path/to/DGAQ3D
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python - <<'PY'
from tools.data_converter.nuscenes_converter import create_nuscenes_infos

create_nuscenes_infos(
    root_path='data/nuscenes',
    info_prefix='petr/mmdet3d_nuscenes_30f',
    version='v1.0-test',
    max_sweeps=30)
PY

ls -lh data/nuscenes/petr/mmdet3d_nuscenes_30f_infos_test.pkl
```

生成的 test info 应包含 6,008 个样本。test 集不需要 Metric3Dv2 伪深度或离线二维 GT，
因为这些内容只用于训练监督，推理阶段只输入相机图像。

## 四、使用 train+val 训练 60 epochs

```bash
cd /path/to/DGAQ3D
set -euo pipefail

CFG=projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_trainval60_test.py
WORK=work_dirs/dgaq3d_keyframe_trainval60_test

mkdir -p "$WORK"

NUSCENES_DATA_ROOT=data/nuscenes \
DENSEGT_DEPTH_ROOT=data/metric3d_depth \
NUSCENES_2DGT_PATH=data/nuscenes/2dgt/nuscenes_trainval_2dgt.pkl \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash tools/dist_train.sh "$CFG" 8 \
  --work-dir "$WORK" \
  --no-validate \
  2>&1 | tee "$WORK/train.log"
```

注意事项：

- 必须保留 `--no-validate`。test 集没有公开 GT，训练过程中不能进行本地 test 评测。
- 不要额外覆盖 `total_epochs=26`。
- 不要修改当前学习率、warmup 或 `loss_scale`。
- 不要从已有的 26e train-only 实验恢复优化器状态。
- 训练应从 `ckpts/dd3d_det_final.pth` 初始化，并完整执行 60 epochs。
- 训练阶段仍采用每相机 16 个、总计最多 96 个 Adaptive Query；144 Query 是最终推理设置，不应写入训练配置。

## 五、导出官方 test 提交文件

训练完成后，使用 `epoch_60.pth` 和最终 144 Query 推理策略导出结果：

```bash
cd /path/to/DGAQ3D
set -euo pipefail

CFG=projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_trainval60_test.py
WORK=work_dirs/dgaq3d_keyframe_trainval60_test
CKPT=$WORK/epoch_60.pth
OUT=$WORK/test_q144

mkdir -p "$OUT"

NUSCENES_DATA_ROOT=data/nuscenes \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=29680 \
bash tools/dist_test.sh "$CFG" "$CKPT" 8 \
  --out "$OUT/test_results.pkl" \
  --format-only \
  --eval-options jsonfile_prefix="$OUT/submission" \
  --cfg-options \
  model.pts_bbox_head.far3d_stagea_cfg.score_thr=0.05 \
  model.pts_bbox_head.far3d_stagea_cfg.sample_max_per_cam=24 \
  model.pts_bbox_head.far3d_stagea_cfg.topk_per_cam=24 \
  model.pts_bbox_head.far3d_stagea_cfg.max_adaptive_queries=144 \
  model.pts_bbox_head.far3d_stagea_cfg.depth_topk=1 \
  model.pts_bbox_head.far3d_stagea_cfg.depth_aggregation=window \
  model.pts_bbox_head.far3d_stagea_cfg.depth_window=3 \
  2>&1 | tee "$OUT/test.log"

ls -lh "$OUT/submission/pts_bbox/results_nusc.json"
```

最终需要提交的文件是：

```text
work_dirs/dgaq3d_keyframe_trainval60_test/test_q144/submission/pts_bbox/results_nusc.json
```

将该文件上传到 nuScenes detection test server。论文表 2 必须填写 test server 返回的
官方指标，不能将本地 val 结果重新命名为 test 结果。

## 六、论文中的结果使用规则

表 2 中的 DGAQ-3D 主结果应注明或通过表头统一表达以下协议：

```text
nuScenes test | train+val | single frame | VoVNet-99-eSE | 1600x640 | DD3D | 60e
```

本地模块消融表继续采用：

```text
nuScenes val | train only | single frame | 26e
```

两类结果应分别放在跨论文 test 对比表和本地 val 消融表中，不直接计算跨协议模块增益。
