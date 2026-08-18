# DGAQ-3D

**Dense Geometry Regularized Global-Adaptive Query Learning for Multi-Camera 3D Object Detection**

基于稠密几何正则与全局-自适应查询协同的多相机三维目标检测。

DGAQ-3D 以原始 **3DPPE** 为基线，在保留 900 个全局查询空间覆盖能力的基础上，引入训练期稠密几何正则、对象感知 Adaptive Query，以及面向混合查询的多视角多尺度稀疏 Decoder。当前主配置面向 nuScenes 六相机、单帧 keyframe、`1600 x 640` 输入设置，并提供 one-sweep 输入扩展。

> 当前最佳 nuScenes validation 结果：关键帧 **0.5065 mAP / 0.5652 NDS**；one-sweep **0.5274 mAP / 0.6001 NDS**。
> 本地 `1600 x 640` 3DPPE keyframe BGD baseline 为 **0.4364 mAP / 0.5076 NDS**。3DPPE 论文公开的 `800 x 320` 单帧结果仅作为跨分辨率外部参考，不参与严格增益计算。

## 摘要

3DPPE 通过预测像素深度恢复三维点，并将其编码为多相机图像特征的位置表示。然而，稀疏 LiDAR 投影监督、固定全局查询和 P4 单尺度交互仍限制了对象中心深度建模、当前帧目标初始化以及跨尺度特征聚合。

DGAQ-3D 从三个方面扩展 3DPPE：

1. **双粒度深度建模**：以 Metric3Dv2 离线稠密伪深度约束 P4 场景几何，同时在 P3 上使用 50 个前景深度 bin 和 1 个背景/终点 bin 建模实例中心深度。
2. **Global-Adaptive 混合查询**：保留 900 个数据集级 Global Query，并根据当前图像中的二维 proposal、类别和深度动态生成对象感知 Adaptive Query。
3. **多视角多尺度稀疏解码**：每个查询围绕三维参考点学习 13 个采样点，在 P3-P6 与六个相机上执行 Deformable Feature Aggregation。

完整 DDN 监督、GT-depth warmup、Top1 深度解码和有界动态查询预算共同抑制低质量二维候选与深度误差向三维空间传播。

## 方法总览
![DGAQ-3D framework](figs/dgaq3d_framework.png)

## 研究动机

### 1. 稀疏深度监督难以提供连续场景几何

原始 3DPPE 使用投影到图像平面的稀疏 LiDAR 点监督深度。大量像素缺少约束，目标边界、遮挡区域和远处小目标的几何表示容易不稳定。DGAQ-3D 将 Metric3Dv2 的离线预测作为**稠密伪深度**，用于训练期共享 FPN 特征的几何正则。

DenseGT 在本仓库中表示离线稠密伪深度，不代表传感器直接采集的真实稠密深度。

### 2. 固定 Global Query 缺少当前图像的对象先验

3DPPE 的 Global Query 跨样本共享，主要学习数据集级空间先验。它能够覆盖完整三维空间，但初始参考点不一定靠近当前图像中的真实目标。DGAQ-3D 利用当前帧二维目标证据生成 Adaptive Query，在不替代 Global Query 的前提下补充样本级对象先验。

### 3. 单尺度交互难以兼顾不同目标尺度

远处小目标需要 P3 的空间细节，近处大目标和复杂区域则受益于 P5/P6 的高层语义。DGAQ-3D 使用参考点引导的稀疏采样，在四个 FPN 层和六个相机之间学习聚合权重，避免完整多尺度全局注意力的高计算开销。

## 方法

### 1. 双粒度深度协同

#### 场景级稠密几何正则

P4 Camera-Aware PGD/DFL 深度头预测像素级深度，训练时由离线 Metric3Dv2 稠密伪深度监督：

- Smooth L1 深度损失权重：`0.05`；
- DFL 用于离散深度分布学习；
- `depth_map_mae/rmse/absrel/delta1/bias/scale` 仅作为诊断指标；
- 在最终 StageA 模式中，该分支是训练期辅助任务，不直接向稀疏 Decoder 输入查询或图像位置编码。

该分支学习可见表面与场景尺度，主要用于正则共享图像特征。

#### 实例级 P3 DDN

StageA 在 P3-P6 上预测 objectness、类别、二维框和中心，并在 P3 上额外预测实例深度分布：

- 50 个 LID 非均匀前景深度 bin；
- 1 个背景/终点 bin；
- 前景区域使用对应实例中心深度；
- 二维框重叠时采用最近实例深度；
- 框外区域使用背景类别；
- DDN focal loss 的前景/背景权重为 `13:1`。

训练时，SimOTA 使用离线 nuScenes 2D GT 为多尺度 priors 建立动态匹配。前 `22000` iterations 使用 GT-depth warmup，减轻早期深度预测不稳定对 Adaptive Query 初始化的影响。

### 2. Global-Adaptive 混合查询

#### Global Query

- 数量固定为 900；
- 三维 reference 来自可学习参考点表；
- Query Content 初始化为 0；
- Query Position 由归一化三维 reference 的位置编码生成。

Global Query 提供完整场景覆盖，降低二维 proposal 漏检带来的风险。

#### Adaptive Query

StageA proposal 分数定义为：

```text
proposal_score = sigmoid(objectness) * max(sigmoid(class_logits))
```

随后依次执行 `3 x 3` 局部极大值筛选、分数阈值过滤和每相机 Top-K。P3 DDN 为保留的 proposal 解码 Top1 深度，并使用相邻三个 bin 的概率加权值降低量化误差。

对于二维中心 `(u, v)` 和预测深度 `d`，三维参考点由相机反投影得到：

```text
p_img   = [u * d, v * d, d, 1]^T
p_lidar = inverse(lidar2img) * p_img
ref_3d  = normalize(p_lidar, point_cloud_range)
```

Adaptive Query 的两个组成部分为：

```text
query_position = MLP(PE_3D(ref_3d))
query_content  = MLP([multi-scale context, relative proposal log-odds])
```

Global Query 与 Adaptive Query 沿查询维拼接后，共同参与 Decoder self-attention。无效补齐位置通过 `query_key_padding_mask` 屏蔽。

### 3. 多视角多尺度稀疏 Decoder

六层 Decoder 的每一层包含：

1. Global 与 Adaptive Query 之间的 self-attention；
2. 围绕每个三维 reference 学习 13 个三维偏移点；
3. 将采样点投影到六个相机；
4. 在 P3-P6 上采样二维语义特征；
5. 根据 Query、位置和相机标定学习视角、尺度、采样点和通道组权重；
6. 残差融合、FFN 与逐层三维框预测。

FPN 特征在进入 Decoder 前由相机内外参生成的 `gamma/beta` 进行几何调制。三维 reference 决定“去哪里采样”，P3-P6 提供“采到的二维语义特征”。

## 贡献总结

1. **场景级稠密几何与实例级离散深度协同**：将 P4 稠密伪深度正则与 P3 实例中心 DDN 解耦，分别服务于共享特征几何学习和 Adaptive Query 三维初始化。
2. **Global-Adaptive 混合查询初始化**：联合数据集级全局先验与当前帧对象先验，在保持空间覆盖的同时缩短高质量查询与真实目标之间的初始距离。
3. **面向混合查询的多视角多尺度稀疏解码**：通过 13 点 Deformable Feature Aggregation 统一处理两类查询，并从 P3-P6 和六相机中自适应聚合目标证据。

Adaptive Query、SimOTA、DDN 和 Deformable Feature Aggregation 均有明确的已有工作来源。本项目的贡献是面向 3DPPE 的职责拆分、接口适配、混合查询协同和系统实验验证，而不是声称首次提出这些基础模块。

## 实验

### 设置

| 项目 | 设置 |
|---|---|
| Dataset | nuScenes train / validation |
| Cameras | 6 |
| Input | `1600 x 640` |
| Temporal setting | single-frame, keyframe-only, `sweeps_num=0` |
| Backbone | VoVNet V-99-eSE |
| Pretraining | DD3D, `ckpts/dd3d_det_final.pth` (shared by the local PGD baseline and all ablations) |
| FPN | P3-P6, 256 channels |
| Global Queries | 900 |
| StageA depth | Top1, 50 foreground bins + 1 background bin |
| Decoder | 6 layers, 13 sampling points, 4 levels, 6 cameras |
| Training epochs | 26 |
| Evaluation | nuScenes official detection metrics |

关键帧主模型包含 Range-DN；one-sweep 仅作为同一方法的输入扩展，不使用时序 memory。

### 主结果

| 方法 | 输入与评测设置 | 角色 | mAP ↑ | NDS ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3DPPE（论文公开单帧结果） | 6 张当前 keyframe，`800 x 320` | 跨分辨率外部参考 | 0.3940 | 0.4340 | 0.7060 | - | - | 0.8480 | - |
| 3DPPE keyframe BGD baseline（本地复现） | 6 张当前 keyframe，`1600 x 640` | 同设置主 baseline | 0.4364 | 0.5076 | 0.7199 | 0.2735 | 0.3044 | 0.6217 | 0.1867 |
| 3DPPE + DenseGT (`loss_depth=0.05`), epoch 25 | 6 张当前 keyframe，`1600 x 640` | 稠密几何中间版本 | 0.4806 | 0.5384 | 0.6324 | 0.2651 | 0.3340 | 0.6006 | 0.1866 |
| DGAQ-3D keyframe (`score_thr=0.05`, Top-24, 144-query cap) | 6 张当前 keyframe，`1600 x 640` | 最终关键帧配置 | **0.5065** | **0.5652** | **0.5745** | **0.2589** | **0.2454** | **0.6235** | **0.1783** |
| DGAQ-3D one-sweep (`score_thr=0.05`, Top-24, 144-query cap) | 当前帧 + 1 历史 sweep，`1600 x 640` | 输入扩展 | **0.5274** | **0.6001** | **0.5671** | **0.2560** | **0.2324** | **0.3936** | **0.1870** |

相对同设置的本地 `3DPPE keyframe BGD baseline`，最终方法取得：

- mAP：`+0.0701`（`+7.01` 个百分点）；
- NDS：`+0.0576`（`+5.76` 个百分点）；
- mATE：`-0.1454 m`；
- mASE：`-0.0146`；
- mAOE：`-0.0590 rad`；
- mAVE：`+0.0018 m/s`；
- mAAE：`-0.0084`。

相对 `3DPPE + DenseGT` 中间版本，最终关键帧配置进一步带来 `+0.0259 mAP`、`+0.0268 NDS`、`-0.0579 mATE` 和 `-0.0886 mAOE`。

论文公开的 `800 x 320` 结果与本地 `1600 x 640` 实验输入分辨率不同，因此不能将二者差值表述为严格方法增益；本文的正式增益以同设置本地 BGD baseline 为准。

### Range-DN 前结构的跨 epoch 稳定性

以下结果使用 Range-DN 前的 Top1 DDN 模型和默认 96-query 上限：

| Epoch | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 22 | 0.4825 | 0.5491 | 0.6203 | 0.2628 | 0.2577 | 0.5998 | 0.1815 |
| 23 | 0.4802 | 0.5480 | 0.6246 | 0.2605 | 0.2538 | 0.6007 | 0.1814 |
| 24 | 0.4823 | 0.5493 | 0.6222 | 0.2606 | 0.2540 | 0.5989 | 0.1830 |
| 25 | 0.4830 | 0.5498 | 0.6193 | 0.2620 | 0.2557 | 0.5979 | 0.1818 |
| 26 | **0.4844** | **0.5504** | **0.6173** | 0.2617 | 0.2569 | 0.6001 | 0.1819 |

### 动态查询统计

推荐推理配置使用 `score_thr=0.05`、每相机最多 24 条、每帧最多 144 条 Adaptive Query：

| Epoch | Samples | Mean | P95 | Max / Cap | Cap hits | Cap hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 6019 | 54.12 | 106 | 144 / 144 | 16 | 0.2658% |
| 25 | 6019 | 53.51 | 105 | 144 / 144 | 16 | 0.2658% |

模型平均仅使用约 54 条 Adaptive Query，远低于 144 条最坏情况预算。P95 表示 95% 的样本不超过约 105-106 条查询。

## 配置

| 实验 | 配置文件 |
|---|---|
| 原始 3DPPE keyframe baseline | `projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe.py` |
| 3DPPE + DenseGT | `projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt.py` |
| StageA 公共结构配置 | `projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_clean.py` |
| 最终关键帧 Range-DN 配置 | `projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py` |

最终配置覆盖的主要参数为：

```python
model = dict(
    pts_bbox_head=dict(
        loss_depth=dict(loss_weight=0.05),
        far3d_stagea_cfg=dict(
            depth_topk=1,
            train_use_gt_depth=True,
            gt_depth_warmup_iters=22000,
            loss_score_weight=0.2,
            loss_cls_weight=0.2,
            loss_iou_weight=1.0,
            loss_bbox_weight=0.2,
            loss_center_weight=0.0,
            loss_depth_weight=0.2)))

total_epochs = 26
```

训练配置默认每相机最多 16 条、每帧最多 96 条 Adaptive Query。`24/144` 是验证集上确定的推荐推理参数，README 中所有结果均明确区分这两种设置。

## 安装

本仓库基于 OpenMMLab 1.x，建议使用 Python 3.8 和与本机 CUDA 匹配的 PyTorch 1.x 环境。核心版本约束见 [requirements.txt](requirements.txt)，完整环境说明见 [install.md](install.md)。

```bash
pip install -r requirements.txt
pip install einops
```

已使用的主要组件包括：

```text
PyTorch 1.x
MMCV-Full 1.x
MMDetection 2.24.1
MMSegmentation 0.20.2
MMDetection3D 0.17.1-compatible codebase
nuScenes devkit
```

MMCV-Full 必须按 PyTorch 与 CUDA 版本安装对应的预编译包或从源码编译，否则多尺度可变形注意力算子无法加载。

## 数据准备

需要准备三类数据：

1. nuScenes 原始数据及 3DPPE annotation pkl；
2. Metric3Dv2 离线稠密伪深度；
3. nuScenes 训练集离线 2D GT。

参考目录：

```text
/path/to/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── petr/
│   ├── mmdet3d_nuscenes_30f_infos_train.pkl
│   └── mmdet3d_nuscenes_30f_infos_val.pkl
└── 2dgt/
    └── nuscenes_train_2dgt.pkl

/path/to/metric3d_depth/
└── ... offline dense-depth files ...
```

StageA 配置支持通过环境变量覆盖两类附加数据：

```bash
export DENSEGT_DEPTH_ROOT=/path/to/metric3d_depth
export NUSCENES_2DGT_PATH=/path/to/nuscenes/2dgt/nuscenes_train_2dgt.pkl
```

nuScenes 根目录当前由配置中的 `_data_root` 指定。训练前还需将 3DPPE 使用的初始化权重放置为：

```text
ckpts/dd3d_det_final.pth
```

## 训练

```bash
cd /path/to/3dppe_clean

CFG=projects/configs/petrv2_depth/petrv2_depth_3dpe_dfl_vovnet_wogridmask_p4_1600x640_pdg_trainval_keyframe_densegt_far3d_stagea_top1_ddn_rangedn_lossd005.py
WORK=work_dirs/stagea_top1_ddn_rangedn_lossd005

mkdir -p "$WORK"
PORT=28500 bash tools/dist_train.sh "$CFG" 8 \
  --work-dir "$WORK" \
  2>&1 | tee "$WORK/train.log"
```

断点续训：

```bash
PORT=28500 bash tools/dist_train.sh "$CFG" 8 \
  --work-dir "$WORK" \
  --resume-from "$WORK/epoch_10.pth" \
  2>&1 | tee -a "$WORK/train.log"
```

配置从 epoch 22 开始每轮评测，并训练至 epoch 26。

## 评测

### 默认 96-query 配置

```bash
CKPT="$WORK/epoch_26.pth"

CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CFG" "$CKPT" \
  --gpu-id 0 \
  --eval bbox
```

### 推荐动态 144-query 配置

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CFG" "$CKPT" \
  --gpu-id 0 \
  --eval bbox \
  --cfg-options \
    model.pts_bbox_head.far3d_stagea_cfg.score_thr=0.05 \
    model.pts_bbox_head.far3d_stagea_cfg.sample_max_per_cam=24 \
    model.pts_bbox_head.far3d_stagea_cfg.topk_per_cam=24 \
    model.pts_bbox_head.far3d_stagea_cfg.max_adaptive_queries=144
```

多 GPU 评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=29500 bash tools/dist_test.sh "$CFG" "$CKPT" 4 \
  --eval bbox \
  --cfg-options \
    model.pts_bbox_head.far3d_stagea_cfg.score_thr=0.05 \
    model.pts_bbox_head.far3d_stagea_cfg.sample_max_per_cam=24 \
    model.pts_bbox_head.far3d_stagea_cfg.topk_per_cam=24 \
    model.pts_bbox_head.far3d_stagea_cfg.max_adaptive_queries=144
```

## 代码索引

| 模块 | 位置 |
|---|---|
| DenseGT 与离线 2D GT 加载 | `projects/mmdet3d_plugin/datasets/pipelines/loading.py` |
| 2D GT 与图像增强同步 | `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` |
| StageA 2D head、SimOTA target、DDN loss、Adaptive Query | `projects/mmdet3d_plugin/models/dense_heads/petrv2_depth_head.py` |
| 相机几何调制与多尺度稀疏 Decoder | `projects/mmdet3d_plugin/models/utils/petr_far3d_transformer.py` |
| 深度图诊断 | `tools/diagnose_pred_depth_map.py` |
| StageA proposal 深度诊断 | `tools/diagnose_stagea_proposal_depth.py` |

## 结果解释边界

- 关键帧模型是主方法；one-sweep 仅增加一帧历史图像输入，不包含 StreamPETR 式时序 memory。
- 仓库提供距离阈值与配对诊断脚本；关于小投影目标的结论仍需结合类别和尺寸分桶解释。
- 动态查询统计说明查询预算较稀疏，但不能替代 FLOPs、FPS、显存和参数量报告。
- 论文公开 baseline 与当前实验可能存在训练协议差异；正式投稿应补齐同设置 baseline 和多随机种子结果。
- Metric3Dv2 在训练前离线运行，最终模型推理时不运行 Metric3Dv2。

## 相关工作与致谢

- [3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shu_3DPPE_3D_Point_Positional_Encoding_for_Transformer-based_Multi-Camera_3D_Object_ICCV_2023_paper.html)
- [Far3D: Expanding the Horizon for Surround-view 3D Object Detection, AAAI 2024](https://arxiv.org/abs/2308.09616)
- [StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Exploring_Object-Centric_Temporal_Modeling_for_Efficient_Multi-View_3D_Object_Detection_ICCV_2023_paper.html)
- [Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation](https://arxiv.org/abs/2404.15506)
- [CaDDN: Categorical Depth Distribution Network for Monocular 3D Object Detection, CVPR 2021](https://arxiv.org/abs/2103.01100)
- [nuScenes: A Multimodal Dataset for Autonomous Driving, CVPR 2020](https://www.nuscenes.org/)

本项目基于 3DPPE/PETR 代码体系，并参考 Far3D 的 Adaptive Query 与稀疏透视聚合设计。感谢上述开源项目与数据集。

## License

本仓库遵循 [Apache License 2.0](LICENSE)。
