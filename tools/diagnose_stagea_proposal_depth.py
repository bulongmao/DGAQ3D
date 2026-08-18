#!/usr/bin/env python
"""Diagnose Far3D StageA 2D proposal depth against offline 2DGT depth.

The script is post-hoc. It loads a trained checkpoint, runs the current
StageA 2D proposal head, matches selected 2D proposals to offline 2DGT boxes,
and reports metric-depth errors in meters.
"""
import argparse
import csv
import importlib
import os
from collections import defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

# Compatibility for old mmdet3d/mmcv pipelines under NumPy >= 1.24.
if 'bool' not in np.__dict__:
    np.bool = bool

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

try:
    from mmdet.utils import compat_cfg, setup_multi_processes
except ImportError:
    from mmdet3d.utils import compat_cfg, setup_multi_processes

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose StageA 2D proposal depth error.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--dataset', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--score-thr', type=float, default=None)
    parser.add_argument('--topk-per-cam', type=int, default=None)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--class-aware', action='store_true', default=True)
    parser.add_argument('--no-class-aware', dest='class_aware',
                        action='store_false')
    parser.add_argument('--no-localmax', action='store_true',
                        help='Disable 3x3 local-max filtering in diagnosis.')
    parser.add_argument('--depth-decode', default='stagea',
                        choices=['stagea', 'top1', 'expectation'],
                        help='How to decode depth logits for diagnosis.')
    parser.add_argument('--distance-bins', type=float, nargs='+',
                        default=[0.0, 20.0, 30.0, 50.0, 80.0])
    parser.add_argument('--out-dir', default='work_dirs/stagea_depth_diag')
    parser.add_argument('--print-interval', type=int, default=20)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', None)
    if plugin_dir is None:
        module_path = '.'.join(Path(config_path).parent.parts)
    else:
        module_path = plugin_dir.rstrip('/').replace('/', '.')
    if module_path:
        importlib.import_module(module_path)


def unwrap_data_container(value):
    if hasattr(value, 'data'):
        return value.data
    return value


def get_batch_metas(data):
    metas = data['img_metas']
    if isinstance(metas, (list, tuple)) and len(metas) == 1:
        metas = metas[0]
    metas = unwrap_data_container(metas)
    while isinstance(metas, (list, tuple)) and len(metas) == 1:
        inner = unwrap_data_container(metas[0])
        if inner is metas:
            break
        metas = inner
    if isinstance(metas, dict):
        return [metas]
    return metas


def bbox_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0))
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(1e-6)


def collect_gt_from_meta(meta, cam_id, device):
    boxes_list = meta.get('gt2d_boxes', None)
    labels_list = meta.get('gt2d_labels', None)
    depths_list = meta.get('gt2d_depths', None)
    if boxes_list is None or labels_list is None or depths_list is None:
        raise RuntimeError(
            'img_metas do not contain gt2d_* fields. Use a config with '
            'LoadOffline2DGT and collect gt2d_boxes/labels/depths.')
    boxes = torch.as_tensor(np.asarray(boxes_list[cam_id], dtype=np.float32),
                            device=device).reshape(-1, 4)
    labels = torch.as_tensor(np.asarray(labels_list[cam_id], dtype=np.int64),
                             device=device).reshape(-1)
    depths = torch.as_tensor(np.asarray(depths_list[cam_id], dtype=np.float32),
                             device=device).reshape(-1)
    num = min(boxes.size(0), labels.size(0), depths.size(0))
    boxes, labels, depths = boxes[:num], labels[:num], depths[:num]
    if num == 0:
        return boxes, labels, depths
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (torch.isfinite(boxes).all(dim=1) &
            torch.isfinite(depths) & (depths > 0) &
            (wh[:, 0] > 0) & (wh[:, 1] > 0) &
            (labels >= 0))
    return boxes[keep], labels[keep], depths[keep]


def bin_label(depth, edges):
    for start, end in zip(edges[:-1], edges[1:]):
        if start <= depth < end:
            return f'{start:g}-{end:g}m'
    return f'>={edges[-1]:g}m'


def decode_depth(head, depth_logits, args):
    prob = F.softmax(depth_logits, dim=-1)
    centers = head.far3d_stagea_depth_centers.to(
        device=depth_logits.device, dtype=depth_logits.dtype)
    if args.depth_decode == 'expectation':
        return (prob * centers.view(1, 1, -1)).sum(dim=-1)
    if args.depth_decode == 'top1':
        top_idx = prob.argmax(dim=-1)
        return centers[top_idx]
    depth_values, _, _ = head._decode_far3d_stagea_depth_hypotheses(
        prob, centers, depth_topk=1)
    return depth_values[..., 0]


def flatten_stagea_predictions(head, preds_list, args):
    flat = head._flatten_far3d_stagea_preds(preds_list)
    objectness = flat['objectness'].squeeze(-1)
    cls_prob = flat['cls_logits'].sigmoid()
    cls_score, labels = cls_prob.max(dim=-1)
    official_score = objectness.sigmoid() * cls_score
    depth_prob = F.softmax(flat['depth_logits'], dim=-1)
    depth_conf = depth_prob.max(dim=-1).values
    quality = official_score
    localmax_keep = None
    if (not args.no_localmax) and getattr(head, 'far3d_stagea_use_localmax', False):
        localmax_keep = head._stagea_localmax_keep(preds_list)
    return dict(
        priors=flat['priors'],
        boxes=flat['decoded_bboxes'],
        labels=labels,
        official_score=official_score,
        quality=quality,
        depth=decode_depth(head, flat['depth_logits'], args),
        depth_conf=depth_conf,
        level_ids=flat['level_ids'],
        localmax_keep=localmax_keep)


def select_predictions(head, preds, img_idx, args):
    score_thr = args.score_thr
    if score_thr is None:
        score_thr = float(head._stagea_get('score_thr', 0.01))
    topk = args.topk_per_cam
    if topk is None:
        topk = int(head._stagea_get(
            'topk_per_cam', head._stagea_get('sample_max_per_cam', 64)))
    quality = preds['quality'][img_idx]
    official_score = preds['official_score'][img_idx]
    valid = (torch.isfinite(quality) & torch.isfinite(official_score) &
             (official_score >= score_thr))
    valid = valid & torch.isfinite(preds['boxes'][img_idx]).all(dim=-1)
    if preds['localmax_keep'] is not None:
        valid = valid & preds['localmax_keep'][img_idx]
    inds = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if inds.numel() == 0:
        empty = quality.new_empty((0,))
        return dict(
            boxes=preds['boxes'][img_idx][:0],
            labels=empty.long(),
            quality=empty,
            official_score=empty,
            depth=empty,
            depth_conf=empty,
            level_ids=empty.long())
    k = min(int(topk), int(inds.numel()))
    _, order = torch.topk(quality[inds], k=k, largest=True, sorted=True)
    inds = inds[order]
    return dict(
        boxes=preds['boxes'][img_idx][inds],
        labels=preds['labels'][img_idx][inds],
        quality=preds['quality'][img_idx][inds],
        official_score=preds['official_score'][img_idx][inds],
        depth=preds['depth'][img_idx][inds],
        depth_conf=preds['depth_conf'][img_idx][inds],
        level_ids=preds['level_ids'][inds])


def greedy_match(pred, gt_boxes, gt_labels, gt_depths, iou_thr, class_aware):
    rows = []
    if pred['boxes'].numel() == 0 or gt_boxes.numel() == 0:
        return rows
    ious = bbox_iou(pred['boxes'], gt_boxes)
    used_gt = set()
    order = torch.argsort(pred['quality'], descending=True)
    for pred_idx in order.tolist():
        cur_iou = ious[pred_idx].clone()
        if class_aware:
            cur_iou = torch.where(
                gt_labels == pred['labels'][pred_idx], cur_iou,
                cur_iou.new_full(cur_iou.shape, -1.0))
        for gt_idx in used_gt:
            cur_iou[gt_idx] = -1.0
        best_iou, best_gt = cur_iou.max(dim=0)
        if float(best_iou.item()) < iou_thr:
            continue
        gt_idx = int(best_gt.item())
        used_gt.add(gt_idx)
        diff = pred['depth'][pred_idx] - gt_depths[gt_idx]
        rows.append(dict(
            pred_idx=pred_idx,
            gt_idx=gt_idx,
            iou=float(best_iou.item()),
            pred_label=int(pred['labels'][pred_idx].item()),
            gt_label=int(gt_labels[gt_idx].item()),
            quality=float(pred['quality'][pred_idx].item()),
            official_score=float(pred['official_score'][pred_idx].item()),
            depth_conf=float(pred['depth_conf'][pred_idx].item()),
            level_id=int(pred['level_ids'][pred_idx].item()),
            pred_depth=float(pred['depth'][pred_idx].item()),
            gt_depth=float(gt_depths[gt_idx].item()),
            depth_error=float(diff.item()),
            abs_error=float(diff.abs().item())))
    return rows


class DepthStats:
    def __init__(self):
        self.pred = []
        self.gt = []
        self.err = []

    def add(self, row):
        self.pred.append(row['pred_depth'])
        self.gt.append(row['gt_depth'])
        self.err.append(row['depth_error'])

    def summary(self):
        if not self.err:
            return dict(count=0, mae=0.0, rmse=0.0, absrel=0.0,
                        bias=0.0, scale=0.0, delta1=0.0)
        pred = np.asarray(self.pred, dtype=np.float64)
        gt = np.asarray(self.gt, dtype=np.float64)
        err = np.asarray(self.err, dtype=np.float64)
        pred_safe = np.maximum(pred, 1e-3)
        gt_safe = np.maximum(gt, 1e-3)
        ratio = np.maximum(pred_safe / gt_safe, gt_safe / pred_safe)
        return dict(
            count=int(err.size),
            mae=float(np.abs(err).mean()),
            rmse=float(np.sqrt((err ** 2).mean())),
            absrel=float((np.abs(err) / gt_safe).mean()),
            bias=float(err.mean()),
            scale=float(pred_safe.mean() / max(gt_safe.mean(), 1e-3)),
            delta1=float((ratio < 1.25).mean()))


def format_stats(name, stats):
    s = stats.summary()
    return (
        f'{name:>12s}  count={s["count"]:6d}  '
        f'mae={s["mae"]:7.3f}  rmse={s["rmse"]:7.3f}  '
        f'absrel={s["absrel"]:6.3f}  bias={s["bias"]:7.3f}  '
        f'scale={s["scale"]:6.3f}  delta1={s["delta1"]:6.3f}')


def build_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    import_plugins(cfg, args.config)
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    return cfg


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'stagea_proposal_depth_matches.csv'

    cfg = build_cfg(args)
    dataset_cfg = getattr(cfg.data, args.dataset)
    if isinstance(dataset_cfg, (list, tuple)):
        raise RuntimeError('Concatenated datasets are not supported here.')
    if args.dataset != 'train':
        dataset_cfg.test_mode = False
    dataset = build_dataset(dataset_cfg)
    loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=0, dist=False,
                      shuffle=False)
    loader_cfg.update(cfg.data.get(f'{args.dataset}_dataloader', {}))
    loader_cfg['samples_per_gpu'] = 1
    loader_cfg['workers_per_gpu'] = 0
    data_loader = build_dataloader(dataset, **loader_cfg)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()
    head = model.module.pts_bbox_head

    captured = {}

    def hook(_module, _inputs, output):
        captured['stagea_preds'] = output.get('far3d_stagea_preds', None)

    handle = head.register_forward_hook(hook)
    all_stats = DepthStats()
    bin_stats = defaultdict(DepthStats)
    cam_stats = defaultdict(DepthStats)
    level_stats = defaultdict(DepthStats)
    total_gt = 0
    total_props = 0
    total_matched = 0
    rows = []
    edges = sorted(args.distance_bins)

    print(f'[depth_decode] {args.depth_decode}')
    print(f'[match] iou_thr={args.iou_thr} class_aware={args.class_aware}')
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if args.max_samples >= 0 and idx >= args.max_samples:
                break
            captured.clear()
            _ = model(return_loss=True, **data)
            preds_list = captured.get('stagea_preds', None)
            if preds_list is None:
                raise RuntimeError(
                    'Model did not return far3d_stagea_preds. Check '
                    'use_far3d_stagea=True in config.')
            img_metas = get_batch_metas(data)
            preds = flatten_stagea_predictions(head, preds_list, args)
            num_cams = preds_list[0]['num_cams']
            for b, meta in enumerate(img_metas):
                for cam_id in range(num_cams):
                    img_idx = b * num_cams + cam_id
                    pred = select_predictions(head, preds, img_idx, args)
                    gt_boxes, gt_labels, gt_depths = collect_gt_from_meta(
                        meta, cam_id, pred['boxes'].device)
                    total_gt += int(gt_boxes.size(0))
                    total_props += int(pred['boxes'].size(0))
                    matched = greedy_match(
                        pred, gt_boxes, gt_labels, gt_depths,
                        args.iou_thr, args.class_aware)
                    total_matched += len(matched)
                    filename = ''
                    if 'filename' in meta and cam_id < len(meta['filename']):
                        filename = meta['filename'][cam_id]
                    for row in matched:
                        row.update(
                            sample_idx=idx,
                            cam_id=cam_id,
                            filename=filename,
                            depth_bin=bin_label(row['gt_depth'], edges))
                        rows.append(row)
                        all_stats.add(row)
                        bin_stats[row['depth_bin']].add(row)
                        cam_stats[f'cam{cam_id}'].add(row)
                        level_stats[f'P{int(row["level_id"]) + 3}'].add(row)
            if (idx + 1) % args.print_interval == 0:
                print(f'[progress] samples={idx + 1}/{args.max_samples}')

    handle.remove()

    fieldnames = [
        'sample_idx', 'cam_id', 'filename', 'pred_idx', 'gt_idx', 'iou',
        'pred_label', 'gt_label', 'quality', 'official_score', 'depth_conf',
        'level_id', 'pred_depth', 'gt_depth', 'depth_error', 'abs_error',
        'depth_bin'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    processed_samples = len(dataset) if args.max_samples < 0 else min(
        args.max_samples, len(dataset))
    print('[summary]')
    print(f'  samples: {processed_samples}')
    print(f'  gt_boxes: {total_gt}')
    print(f'  selected_proposals: {total_props}')
    print(f'  matched_proposals: {total_matched}')
    print(f'  match_ratio_vs_gt: {total_matched / max(total_gt, 1):.4f}')
    print(f'  mean_selected_props_per_image: '
          f'{total_props / max(processed_samples * num_cams, 1):.2f}')
    print(format_stats('all', all_stats))
    print('[by_depth_bin]')
    for key in [bin_label((edges[i] + edges[i + 1]) * 0.5, edges)
                for i in range(len(edges) - 1)] + [f'>={edges[-1]:g}m']:
        if key in bin_stats:
            print(format_stats(key, bin_stats[key]))
    print('[by_camera]')
    for key in sorted(cam_stats):
        print(format_stats(key, cam_stats[key]))
    print('[by_feature_level]')
    for key in sorted(level_stats):
        print(format_stats(key, level_stats[key]))
    print(f'[csv] {csv_path}')


if __name__ == '__main__':
    main()
