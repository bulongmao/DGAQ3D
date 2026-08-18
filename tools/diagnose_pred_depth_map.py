#!/usr/bin/env python
"""Diagnose predicted depth maps against the configured depth target.

This script is post-hoc: run it on a trained checkpoint to inspect the
Transformer-left depth prior. It does not change training or evaluation.

Outputs:
  - depth_error_by_distance.csv
  - depth_error_by_camera_distance.csv
  - summary.txt
  - vis/*.png for selected samples/cameras
"""
import argparse
import copy
import csv
import importlib
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, 'long'):
    np.long = int  # type: ignore[attr-defined]


DEFAULT_META_KEYS = [
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
    'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
    'pts_filename', 'transformation_3d_flow', 'img_info', 'intrinsics',
    'extrinsics', 'timestamp'
]


class RunningDepthStats:
    def __init__(self):
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_absrel = 0.0
        self.sum_delta1 = 0.0
        self.sum_diff = 0.0
        self.sum_pred = 0.0
        self.sum_tgt = 0.0

    def add(self, pred, tgt):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        tgt = np.asarray(tgt, dtype=np.float64).reshape(-1)
        valid = np.isfinite(pred) & np.isfinite(tgt) & (tgt > 0) & (pred > 0)
        if not np.any(valid):
            return
        pred = pred[valid]
        tgt = tgt[valid]
        diff = pred - tgt
        abs_diff = np.abs(diff)
        ratio = np.maximum(pred / np.maximum(tgt, 1e-6), tgt / np.maximum(pred, 1e-6))
        self.count += int(pred.size)
        self.sum_abs += float(abs_diff.sum())
        self.sum_sq += float((diff ** 2).sum())
        self.sum_absrel += float((abs_diff / np.maximum(tgt, 1e-6)).sum())
        self.sum_delta1 += float((ratio < 1.25).sum())
        self.sum_diff += float(diff.sum())
        self.sum_pred += float(pred.sum())
        self.sum_tgt += float(tgt.sum())

    def row(self, name):
        if self.count == 0:
            return dict(name=name, count=0, mae=np.nan, rmse=np.nan,
                        absrel=np.nan, delta1=np.nan, bias=np.nan,
                        scale=np.nan)
        return dict(
            name=name,
            count=self.count,
            mae=self.sum_abs / self.count,
            rmse=(self.sum_sq / self.count) ** 0.5,
            absrel=self.sum_absrel / self.count,
            delta1=self.sum_delta1 / self.count,
            bias=self.sum_diff / self.count,
            scale=self.sum_pred / max(self.sum_tgt, 1e-6),
        )


def parse_args():
    parser = argparse.ArgumentParser('Diagnose predicted depth map quality')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--dataset', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--out-dir', default='work_dirs/depth_map_diagnostics')
    parser.add_argument('--data-root', default=os.environ.get('NUSCENES_DATA_ROOT'))
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--print-interval', type=int, default=20)
    parser.add_argument('--distance-bins', type=float, nargs='+',
                        default=[0.0, 10.0, 20.0, 30.0, 50.0])
    parser.add_argument('--depth-max', type=float, default=61.2)
    parser.add_argument('--max-vis', type=int, default=24,
                        help='Maximum number of saved camera-view figures.')
    parser.add_argument('--vis-cams', type=int, nargs='*', default=[0],
                        help='Camera indices to visualize. Use empty value to disable.')
    parser.add_argument('--cfg-options', nargs='+', action=mmcv.DictAction)
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', None)
    if plugin_dir is None:
        module_path = '.'.join(Path(config_path).parent.parts)
    else:
        module_path = plugin_dir.rstrip('/').replace('/', '.')
    importlib.import_module(module_path)


def resolve_ann_path(data_cfg, data_root):
    if not data_root or not isinstance(data_cfg.get('ann_file', None), str):
        return
    ann = data_cfg.ann_file
    for prefix in ('./data/nuscenes/', 'data/nuscenes/'):
        if ann.startswith(prefix):
            data_cfg.ann_file = os.path.join(data_root, ann[len(prefix):])
            return


def build_depth_diag_pipeline(cfg):
    source = cfg.get('train_pipeline', None)
    if source is None:
        source = cfg.data.train.pipeline
    pipeline = []
    drop_types = {'LoadAnnotations3D', 'ObjectRangeFilter', 'ObjectNameFilter',
                  'GlobalRotScaleTransImage'}
    found_depth_loader = False
    for item in source:
        item = copy.deepcopy(item)
        t = item.get('type')
        if t in drop_types:
            continue
        if t == 'ResizeCropFlipImageV2':
            item['training'] = False
        elif t == 'LoadMultiViewImageFromMultiSweepsFiles':
            item['test_mode'] = True
        elif t in ('LoadDepthByMapplingPoints2Images', 'LoadDenseDepthFromFiles'):
            found_depth_loader = True
        elif t == 'DefaultFormatBundle3D':
            item['with_gt'] = False
            item['with_label'] = False
        elif t == 'Collect3D':
            meta_keys = item.get('meta_keys', DEFAULT_META_KEYS)
            item = dict(type='Collect3D',
                        keys=['img', 'depth_map', 'depth_map_mask'],
                        meta_keys=meta_keys)
        pipeline.append(item)
    if not found_depth_loader:
        raise RuntimeError('No depth target loader found in train_pipeline.')
    return pipeline


def build_diag_dataset(cfg, args):
    data_cfg = copy.deepcopy(getattr(cfg.data, args.dataset))
    data_cfg.test_mode = True
    data_cfg.pipeline = build_depth_diag_pipeline(cfg)
    if args.data_root:
        data_cfg.data_root = args.data_root
        resolve_ann_path(data_cfg, args.data_root)
    if args.ann_file:
        data_cfg.ann_file = args.ann_file
    return build_dataset(data_cfg)


def unwrap_data(x):
    if isinstance(x, DC):
        return unwrap_data(x.data)
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return unwrap_data(x[0])
        return [unwrap_data(v) for v in x]
    return x


def as_depth_tensor(x):
    x = unwrap_data(x)
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise TypeError(f'Unexpected depth list length: {len(x)}')
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 4 and x.size(0) == 1:
        x = x[0]
    return x.float()


def get_img_tensor(data):
    img = unwrap_data(data['img'])
    if isinstance(img, (list, tuple)):
        img = img[0]
    if img.dim() == 5 and img.size(0) == 1:
        img = img[0]
    return img.float()


def get_img_meta(data):
    meta = unwrap_data(data['img_metas'])
    if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], dict):
        meta = meta[0]
    return meta


def label_for_distance(distance, edges):
    for start, end in zip(edges[:-1], edges[1:]):
        if start <= distance < end:
            return f'{start:g}-{end:g}m'
    return f'>={edges[-1]:g}m'


def update_stats(stats, name, pred, tgt):
    if name not in stats:
        stats[name] = RunningDepthStats()
    stats[name].add(pred, tgt)


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unnormalize_image(img_chw, cfg):
    img = tensor_to_numpy(img_chw).transpose(1, 2, 0)
    norm = cfg.get('img_norm_cfg', {})
    mean = np.asarray(norm.get('mean', [0, 0, 0]), dtype=np.float32)
    std = np.asarray(norm.get('std', [1, 1, 1]), dtype=np.float32)
    to_rgb = bool(norm.get('to_rgb', False))
    img = img * std + mean
    if not to_rgb:
        img = img[..., ::-1]
    return np.clip(img, 0, 255).astype(np.uint8)


def save_depth_figure(out_path, img_chw, pred, tgt, mask, cfg, title, depth_max):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = unnormalize_image(img_chw, cfg)
    pred = np.asarray(pred, dtype=np.float32)
    tgt = np.asarray(tgt, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    valid = mask & np.isfinite(pred) & np.isfinite(tgt) & (tgt > 0) & (pred > 0)
    gt_vis = np.where(mask & (tgt > 0), tgt, np.nan)
    err_vis = np.where(valid, np.abs(pred - tgt), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('image')
    im1 = axes[0, 1].imshow(gt_vis, cmap='turbo', vmin=0, vmax=depth_max)
    axes[0, 1].set_title('GT depth target (valid pixels)')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im2 = axes[1, 0].imshow(pred, cmap='turbo', vmin=0, vmax=depth_max)
    axes[1, 0].set_title('predicted depth map')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im3 = axes[1, 1].imshow(err_vis, cmap='magma', vmin=0, vmax=min(depth_max, 30.0))
    axes[1, 1].set_title('absolute error on valid pixels')
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for ax in axes.reshape(-1):
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_stats_csv(path, stats, key_name):
    rows = []
    for name in sorted(stats.keys()):
        row = stats[name].row(name)
        row[key_name] = row.pop('name')
        rows.append(row)
    fieldnames = [key_name, 'count', 'mae', 'rmse', 'absrel', 'delta1', 'bias', 'scale']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / 'vis'
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    import_plugins(cfg, args.config)

    torch.cuda.set_device(args.gpu_id)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_diag_dataset(cfg, args)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', getattr(dataset, 'CLASSES', None))
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()

    edges = sorted(args.distance_bins)
    by_dist = {'all': RunningDepthStats()}
    by_cam_dist = {}
    vis_count = 0
    processed = 0

    for idx, data in enumerate(loader):
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        depth_tgt = as_depth_tensor(data['depth_map'])
        depth_mask = as_depth_tensor(data['depth_map_mask']).bool()
        img_tensor = get_img_tensor(data)
        img_meta = get_img_meta(data)

        model_data = dict(data)
        model_data.pop('depth_map', None)
        model_data.pop('depth_map_mask', None)
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **model_data)
            depth_pred = model.module.pts_bbox_head.depth_map.detach().float().cpu()

        if depth_pred.dim() == 4 and depth_pred.size(0) == 1:
            depth_pred = depth_pred[0]
        if depth_pred.shape[-2:] != depth_tgt.shape[-2:]:
            depth_pred = F.interpolate(
                depth_pred[:, None], size=depth_tgt.shape[-2:],
                mode='bilinear', align_corners=False)[:, 0]

        pred_np = tensor_to_numpy(depth_pred)
        tgt_np = tensor_to_numpy(depth_tgt)
        mask_np = tensor_to_numpy(depth_mask).astype(bool)
        n_cam = pred_np.shape[0]

        valid_all = mask_np & np.isfinite(pred_np) & np.isfinite(tgt_np) & (tgt_np > 0) & (pred_np > 0)
        by_dist['all'].add(pred_np[valid_all], tgt_np[valid_all])

        for cam_idx in range(n_cam):
            cam_valid = valid_all[cam_idx]
            if not np.any(cam_valid):
                continue
            cam_pred = pred_np[cam_idx]
            cam_tgt = tgt_np[cam_idx]
            for start, end in zip(edges[:-1], edges[1:]):
                m = cam_valid & (cam_tgt >= start) & (cam_tgt < end)
                label = f'{start:g}-{end:g}m'
                update_stats(by_dist, label, cam_pred[m], cam_tgt[m])
                update_stats(by_cam_dist, f'cam{cam_idx}:{label}', cam_pred[m], cam_tgt[m])
            m = cam_valid & (cam_tgt >= edges[-1])
            label = f'>={edges[-1]:g}m'
            update_stats(by_dist, label, cam_pred[m], cam_tgt[m])
            update_stats(by_cam_dist, f'cam{cam_idx}:{label}', cam_pred[m], cam_tgt[m])

        if args.max_vis > 0 and args.vis_cams and vis_count < args.max_vis:
            filenames = img_meta.get('filename', []) if isinstance(img_meta, dict) else []
            sample_idx = img_meta.get('sample_idx', idx) if isinstance(img_meta, dict) else idx
            for cam_idx in args.vis_cams:
                if cam_idx < 0 or cam_idx >= n_cam or vis_count >= args.max_vis:
                    continue
                cam_name = f'cam{cam_idx}'
                if isinstance(filenames, (list, tuple)) and cam_idx < len(filenames):
                    cam_name = Path(filenames[cam_idx]).parent.name
                title = f'sample={sample_idx} {cam_name}'
                out_path = vis_dir / f'{idx:06d}_{cam_name}.png'
                save_depth_figure(
                    out_path, img_tensor[cam_idx], pred_np[cam_idx],
                    tgt_np[cam_idx], mask_np[cam_idx], cfg,
                    title, args.depth_max)
                vis_count += 1

        processed += 1
        if args.print_interval > 0 and processed % args.print_interval == 0:
            row = by_dist['all'].row('all')
            print(f"[progress] samples={processed} mae={row['mae']:.4f} "
                  f"rmse={row['rmse']:.4f} bias={row['bias']:.4f} "
                  f"scale={row['scale']:.4f}")

    dist_rows = write_stats_csv(out_dir / 'depth_error_by_distance.csv', by_dist, 'distance_bin')
    cam_rows = write_stats_csv(out_dir / 'depth_error_by_camera_distance.csv', by_cam_dist, 'camera_distance_bin')

    summary_path = out_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'config: {args.config}\n')
        f.write(f'checkpoint: {args.checkpoint}\n')
        f.write(f'dataset: {args.dataset}\n')
        f.write(f'processed_samples: {processed}\n')
        f.write(f'visualizations: {vis_count}\n')
        f.write('\n[depth_error_by_distance]\n')
        for row in dist_rows:
            f.write(str(row) + '\n')
        f.write('\n[depth_error_by_camera_distance]\n')
        for row in cam_rows:
            f.write(str(row) + '\n')

    overall = by_dist['all'].row('all')
    print('[done]')
    print(f"  samples: {processed}")
    print(f"  visualizations: {vis_count}")
    print(f"  overall_mae: {overall['mae']:.4f}")
    print(f"  overall_rmse: {overall['rmse']:.4f}")
    print(f"  overall_absrel: {overall['absrel']:.4f}")
    print(f"  overall_delta1: {overall['delta1']:.4f}")
    print(f"  overall_bias: {overall['bias']:.4f}")
    print(f"  overall_scale: {overall['scale']:.4f}")
    print(f"  out_dir: {out_dir}")


if __name__ == '__main__':
    main()
