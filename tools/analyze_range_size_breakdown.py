#!/usr/bin/env python
"""Paper-grade range and projected-size evaluation for nuScenes detections.

Run detector inference first with ``tools/test.py --out`` (or distributed
equivalent), then pass matched baseline and proposed-method result pickles to
this script. The analysis follows nuScenes score-ranked center matching and AP
interpolation, applies official class-specific evaluation ranges, and reports:

* absolute and class-normalized distance bins;
* max-visible-camera projected 2D size bins;
* bucket mAP and macro recall at 2 m;
* paired scene-bootstrap confidence intervals;
* Far3D-Figure-4-style AP/recall curves over 0.5/1/2/4 m thresholds.
"""

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for import_root in (REPO_ROOT, TOOLS_DIR):
    import_root = str(import_root)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
from range_size_eval_utils import (  # noqa: E402
    apply_image_transform,
    assign_bin,
    bin_labels,
    build_test_ida,
    lidar_centers_to_ego,
    match_predictions,
    max_projected_box_sizes,
    nuscenes_ap_from_tp,
    recall_from_tp,
    repeat_bootstrap_events,
    validate_edges,
)


NUSCENES_CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')

FALLBACK_CLASS_RANGES = {
    'car': 50.0,
    'truck': 50.0,
    'bus': 50.0,
    'trailer': 50.0,
    'construction_vehicle': 50.0,
    'pedestrian': 40.0,
    'motorcycle': 40.0,
    'bicycle': 40.0,
    'traffic_cone': 30.0,
    'barrier': 30.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare matched nuScenes predictions by range and 2D size.')
    parser.add_argument('config', help='Config defining the validation dataset/IDA.')
    parser.add_argument('baseline_pkl', help='Baseline result from --out.')
    parser.add_argument('ours_pkl', help='Proposed-method result from --out.')
    parser.add_argument('--baseline-name', default='3DPPE')
    parser.add_argument('--ours-name', default='Ours')
    parser.add_argument('--out-dir', default='work_dirs/range_size_breakdown')
    parser.add_argument('--data-root', default=os.environ.get('NUSCENES_DATA_ROOT'))
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--num-cams', type=int, default=6)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug only. Omit for publishable results.')
    parser.add_argument('--distance-bins', type=float, nargs='+',
                        default=[0.0, 20.0, 30.0, 40.0, 50.0])
    parser.add_argument('--relative-distance-bins', type=float, nargs='+',
                        default=[0.0, 0.4, 0.7, 1.0])
    parser.add_argument('--size-bins', type=float, nargs='+',
                        default=[0.0, 16.0, 32.0, 96.0, float('inf')],
                        help='Edges for sqrt(clipped 2D box area), in pixels.')
    parser.add_argument('--distance-thresholds', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--recall-distance-threshold', type=float, default=2.0)
    parser.add_argument('--far-relative-start', type=float, default=0.7)
    parser.add_argument('--min-recall', type=float, default=0.1)
    parser.add_argument('--min-precision', type=float, default=0.1)
    parser.add_argument('--min-projection-depth', type=float, default=0.1)
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help='Paired cluster-bootstrap draws; 0 disables CIs.')
    parser.add_argument('--bootstrap-unit', choices=['scene', 'sample'],
                        default='scene')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--paper-figure-dir', default=None,
                        help='Optional directory receiving generated PDF/PNG figures.')
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', None)
    if plugin_dir is None:
        default_plugin = REPO_ROOT / 'projects' / 'mmdet3d_plugin'
        if default_plugin.is_dir():
            plugin_dir = default_plugin.relative_to(REPO_ROOT).as_posix()
        else:
            config_parent = Path(config_path).resolve().parent
            try:
                plugin_dir = config_parent.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                plugin_dir = config_parent.as_posix()
    plugin_path = Path(plugin_dir.rstrip('/'))
    if plugin_path.is_absolute():
        try:
            plugin_path = plugin_path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(
                'plugin_dir must be importable below the repository root: '
                '{}'.format(plugin_dir)) from error
    module_path = '.'.join(plugin_path.parts)
    importlib.import_module(module_path)


def build_eval_dataset(cfg, args):
    from mmdet3d.datasets import build_dataset

    import_plugins(cfg, args.config)
    data_cfg = cfg.data.val.copy()
    data_cfg.test_mode = True
    if args.data_root:
        data_cfg.data_root = args.data_root
    if args.ann_file:
        data_cfg.ann_file = args.ann_file
    elif args.data_root and isinstance(data_cfg.get('ann_file'), str):
        ann_file = data_cfg.ann_file
        for prefix in ('./data/nuscenes/', 'data/nuscenes/'):
            if ann_file.startswith(prefix):
                data_cfg.ann_file = os.path.join(
                    args.data_root, ann_file[len(prefix):])
                break
    if hasattr(cfg, 'test_pipeline'):
        data_cfg.pipeline = cfg.test_pipeline
    return build_dataset(data_cfg)


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'tensor'):
        value = value.tensor
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def centers_from_boxes(boxes):
    if boxes is None:
        return np.zeros((0, 3), dtype=np.float32)
    if hasattr(boxes, 'gravity_center'):
        return tensor_to_numpy(boxes.gravity_center).reshape(-1, 3)
    array = tensor_to_numpy(boxes)
    return array[:, :3].reshape(-1, 3)


def corners_from_boxes(boxes):
    if boxes is None:
        return np.zeros((0, 8, 3), dtype=np.float32)
    if hasattr(boxes, 'corners'):
        corners = tensor_to_numpy(boxes.corners)
        return corners.reshape(-1, 8, 3)
    raise TypeError('Expected MMDetection3D boxes exposing a corners property.')


def unpack_prediction(result):
    if isinstance(result, dict) and 'pts_bbox' in result:
        result = result['pts_bbox']
    if not isinstance(result, dict):
        raise TypeError('Prediction entry must be a dictionary.')
    boxes = result.get('boxes_3d', result.get('boxes'))
    scores = tensor_to_numpy(result.get('scores_3d', result.get('scores')))
    labels = tensor_to_numpy(result.get('labels_3d', result.get('labels')))
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if labels is None:
        labels = np.zeros((0,), dtype=np.int64)
    return boxes, scores.reshape(-1), labels.astype(np.int64).reshape(-1)


def find_resize_transform(node):
    if isinstance(node, (list, tuple)):
        for child in node:
            found = find_resize_transform(child)
            if found is not None:
                return found
        return None
    if hasattr(node, 'items'):
        transform_type = node.get('type', None)
        if transform_type in ('ResizeCropFlipImage', 'ResizeCropFlipImageV2'):
            return transform_type, node['data_aug_conf']
        for _, child in node.items():
            found = find_resize_transform(child)
            if found is not None:
                return found
    return None


def get_test_projection_geometry(cfg):
    pipeline = getattr(cfg, 'test_pipeline', cfg.data.val.get('pipeline'))
    found = find_resize_transform(pipeline)
    if found is None:
        raise RuntimeError('No ResizeCropFlipImage transform found in test pipeline.')
    transform_type, data_aug_conf = found
    return build_test_ida(data_aug_conf, transform_type=transform_type)


def class_range_map(dataset, class_names):
    eval_configs = getattr(dataset, 'eval_detection_configs', None)
    ranges = getattr(eval_configs, 'class_range', None)
    if ranges is None:
        ranges = FALLBACK_CLASS_RANGES
    missing = [name for name in class_names if name not in ranges]
    if missing:
        raise KeyError('Missing official class ranges for: {}'.format(missing))
    return {name: float(ranges[name]) for name in class_names}


def resolve_bootstrap_units(dataset, sample_count, requested_unit, data_root):
    if requested_unit == 'sample':
        return np.arange(sample_count, dtype=np.int32), 'sample'

    scene_tokens = []
    all_embedded = True
    for info in dataset.data_infos[:sample_count]:
        scene_token = info.get('scene_token')
        if scene_token is None:
            all_embedded = False
            break
        scene_tokens.append(scene_token)

    if not all_embedded:
        try:
            from nuscenes import NuScenes
            version = getattr(dataset, 'version', 'v1.0-trainval')
            root = data_root or getattr(dataset, 'data_root', None)
            nusc = NuScenes(version=version, dataroot=root, verbose=False)
            scene_tokens = [
                nusc.get('sample', info['token'])['scene_token']
                for info in dataset.data_infos[:sample_count]
            ]
        except Exception as error:
            print('[warning] Scene bootstrap unavailable ({}). Falling back to '
                  'paired sample bootstrap.'.format(error))
            return np.arange(sample_count, dtype=np.int32), 'sample'

    mapping = {}
    unit_ids = []
    for token in scene_tokens:
        if token not in mapping:
            mapping[token] = len(mapping)
        unit_ids.append(mapping[token])
    return np.asarray(unit_ids, dtype=np.int32), 'scene'


def empty_chunk_store(model_count, thresholds, class_count):
    store = []
    for _ in range(model_count):
        by_threshold = []
        for _ in thresholds:
            by_threshold.append([
                defaultdict(list) for _ in range(class_count)])
        store.append(by_threshold)
    return store


def empty_gt_store(class_count):
    return [defaultdict(list) for _ in range(class_count)]


def append_chunk(store, key, value, dtype=None):
    value = np.asarray(value, dtype=dtype).reshape(-1)
    if value.size:
        store[key].append(value)


def concatenate_chunks(chunks, dtype):
    if not chunks:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(chunks).astype(dtype, copy=False)


def finalize_gt_store(store):
    finalized = []
    dtypes = {'unit': np.int32, 'absolute': np.int16, 'relative': np.int16,
              'size': np.int16, 'joint': np.int16}
    for class_store in store:
        finalized.append({
            key: concatenate_chunks(class_store[key], dtype)
            for key, dtype in dtypes.items()
        })
    return finalized


def finalize_event_store(store):
    finalized = []
    dtypes = {'score': np.float32, 'tp': np.int8, 'unit': np.int32,
              'absolute': np.int16, 'relative': np.int16,
              'size': np.int16, 'joint': np.int16}
    for model_store in store:
        model_final = []
        for threshold_store in model_store:
            threshold_final = []
            for class_store in threshold_store:
                arrays = {
                    key: concatenate_chunks(class_store[key], dtype)
                    for key, dtype in dtypes.items()
                }
                indices = np.arange(arrays['score'].size, dtype=np.int64)
                order = np.lexsort((-indices, -arrays['score']))
                arrays = {key: value[order] for key, value in arrays.items()}
                threshold_final.append(arrays)
            model_final.append(threshold_final)
        finalized.append(model_final)
    return finalized


def make_box_records(boxes, labels, scores, raw_info, lidar2img,
                     image_shape, class_names, class_ranges, bin_edges,
                     min_projection_depth):
    centers_lidar = centers_from_boxes(boxes)
    corners_lidar = corners_from_boxes(boxes)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores is None:
        scores = np.ones(labels.shape[0], dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(centers_lidar) == len(corners_lidar) == len(labels) == len(scores)):
        raise ValueError('Box, label, score, center, and corner counts differ.')

    rotation = raw_info.get('lidar2ego_rotation', [1.0, 0.0, 0.0, 0.0])
    translation = raw_info.get('lidar2ego_translation', [0.0, 0.0, 0.0])
    centers_ego = lidar_centers_to_ego(centers_lidar, rotation, translation)
    absolute_range = np.linalg.norm(centers_ego[:, :2], axis=1)
    class_limit = np.full(labels.shape, np.nan, dtype=np.float64)
    valid_label = (labels >= 0) & (labels < len(class_names))
    for label in np.unique(labels[valid_label]):
        class_limit[labels == label] = class_ranges[class_names[int(label)]]
    official_keep = valid_label & np.isfinite(absolute_range)
    official_keep &= absolute_range <= class_limit + 1e-6

    centers_ego = centers_ego[official_keep]
    corners_lidar = corners_lidar[official_keep]
    labels = labels[official_keep]
    scores = scores[official_keep]
    absolute_range = absolute_range[official_keep]
    class_limit = class_limit[official_keep]
    relative_range = absolute_range / class_limit

    projected_size, visible_cameras = max_projected_box_sizes(
        corners_lidar, lidar2img, image_shape,
        min_depth=min_projection_depth)

    absolute_bin = assign_bin(absolute_range, bin_edges['absolute'])
    relative_bin = assign_bin(relative_range, bin_edges['relative'])
    size_bin = assign_bin(projected_size, bin_edges['size'])
    joint_bin = relative_bin * (len(bin_edges['size']) - 1) + size_bin
    joint_bin[(relative_bin < 0) | (size_bin < 0)] = -1
    return {
        'center': centers_ego[:, :2].astype(np.float32),
        'label': labels.astype(np.int64),
        'score': scores,
        'absolute_value': absolute_range.astype(np.float32),
        'relative_value': relative_range.astype(np.float32),
        'size_value': projected_size,
        'visible_cameras': visible_cameras,
        'absolute': absolute_bin.astype(np.int16),
        'relative': relative_bin.astype(np.int16),
        'size': size_bin.astype(np.int16),
        'joint': joint_bin.astype(np.int16),
    }


def collect_events(dataset, baseline_results, ours_results, sample_units,
                   projection_geometry, class_names, class_ranges, args):
    thresholds = np.asarray(args.distance_thresholds, dtype=np.float64)
    class_count = len(class_names)
    gt_store = empty_gt_store(class_count)
    event_store = empty_chunk_store(2, thresholds, class_count)
    bin_edges = {
        'absolute': validate_edges(args.distance_bins),
        'relative': validate_edges(args.relative_distance_bins),
        'size': validate_edges(args.size_bins),
    }
    ida = projection_geometry['matrix']
    image_shape = projection_geometry['final_dim']
    sample_count = min(len(dataset), len(baseline_results), len(ours_results))
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)

    try:
        from mmcv import track_iter_progress
        iterator = track_iter_progress(range(sample_count))
    except ImportError:
        iterator = range(sample_count)

    projection_audit = {
        'gt_zero_visible': 0,
        'gt_total': 0,
        'baseline_zero_visible': 0,
        'baseline_total': 0,
        'ours_zero_visible': 0,
        'ours_total': 0,
    }
    models = (baseline_results, ours_results)
    for sample_index in iterator:
        raw_info = dataset.data_infos[sample_index]
        data_info = dataset.get_data_info(sample_index)
        lidar2img = np.asarray(data_info['lidar2img'][:args.num_cams])
        lidar2img = apply_image_transform(lidar2img, ida)
        unit = int(sample_units[sample_index])

        annotation = dataset.get_ann_info(sample_index)
        gt_boxes = annotation['gt_bboxes_3d']
        gt_labels = tensor_to_numpy(annotation['gt_labels_3d']).astype(np.int64)
        gt = make_box_records(
            gt_boxes, gt_labels, None, raw_info, lidar2img, image_shape,
            class_names, class_ranges, bin_edges, args.min_projection_depth)
        projection_audit['gt_total'] += len(gt['label'])
        projection_audit['gt_zero_visible'] += int(
            np.sum(gt['visible_cameras'] == 0))

        for class_id in range(class_count):
            indices = np.flatnonzero(gt['label'] == class_id)
            if indices.size == 0:
                continue
            class_store = gt_store[class_id]
            append_chunk(class_store, 'unit', np.full(indices.size, unit), np.int32)
            for key in ('absolute', 'relative', 'size', 'joint'):
                append_chunk(class_store, key, gt[key][indices], np.int16)

        for model_id, results in enumerate(models):
            pred_boxes, pred_scores, pred_labels = unpack_prediction(
                results[sample_index])
            pred = make_box_records(
                pred_boxes, pred_labels, pred_scores, raw_info, lidar2img,
                image_shape, class_names, class_ranges, bin_edges,
                args.min_projection_depth)
            prefix = 'baseline' if model_id == 0 else 'ours'
            projection_audit[prefix + '_total'] += len(pred['label'])
            projection_audit[prefix + '_zero_visible'] += int(
                np.sum(pred['visible_cameras'] == 0))

            for class_id in range(class_count):
                gt_indices = np.flatnonzero(gt['label'] == class_id)
                pred_indices = np.flatnonzero(pred['label'] == class_id)
                if pred_indices.size == 0:
                    continue
                gt_centers = gt['center'][gt_indices]
                pred_centers = pred['center'][pred_indices]
                class_scores = pred['score'][pred_indices]

                for threshold_id, threshold in enumerate(thresholds):
                    order, matched_local_gt = match_predictions(
                        gt_centers, pred_centers, class_scores, threshold)
                    ordered_pred = pred_indices[order]
                    matched = matched_local_gt >= 0
                    tp = matched.astype(np.int8)
                    event_attributes = {}
                    for key in ('absolute', 'relative', 'size', 'joint'):
                        values = pred[key][ordered_pred].copy()
                        if np.any(matched):
                            values[matched] = gt[key][
                                gt_indices[matched_local_gt[matched]]]
                        event_attributes[key] = values

                    class_store = event_store[model_id][threshold_id][class_id]
                    append_chunk(class_store, 'score', pred['score'][ordered_pred],
                                 np.float32)
                    append_chunk(class_store, 'tp', tp, np.int8)
                    append_chunk(class_store, 'unit',
                                 np.full(order.size, unit), np.int32)
                    for key, values in event_attributes.items():
                        append_chunk(class_store, key, values, np.int16)

    return (finalize_gt_store(gt_store), finalize_event_store(event_store),
            bin_edges, projection_audit)


def select_gt_units(gt_class, partition, bucket):
    if partition == 'full':
        return gt_class['unit']
    return gt_class['unit'][gt_class[partition] == bucket]


def select_events(event_class, partition, bucket):
    if partition == 'full':
        mask = np.ones(event_class['tp'].shape[0], dtype=bool)
    else:
        mask = event_class[partition] == bucket
    return event_class['tp'][mask], event_class['unit'][mask]


def class_metric(gt_class, event_class, partition, bucket, unit_counts,
                 min_recall, min_precision):
    gt_units = select_gt_units(gt_class, partition, bucket)
    tp, event_units = select_events(event_class, partition, bucket)
    if unit_counts is not None:
        tp, num_gt = repeat_bootstrap_events(
            tp, event_units, gt_units, unit_counts)
    else:
        num_gt = gt_units.size
    ap = nuscenes_ap_from_tp(
        tp, num_gt, min_recall=min_recall, min_precision=min_precision)
    recall = recall_from_tp(tp, num_gt)
    return ap, recall, int(num_gt)


def aggregate_curve(gt_store, event_store, model_id, partition, bucket,
                    args, unit_counts=None):
    ap_curve = []
    recall_curve = []
    class_rows = []
    for threshold_id, threshold in enumerate(args.distance_thresholds):
        class_ap = []
        class_recall = []
        threshold_rows = []
        for class_id in range(len(gt_store)):
            ap, recall, num_gt = class_metric(
                gt_store[class_id],
                event_store[model_id][threshold_id][class_id],
                partition, bucket, unit_counts,
                args.min_recall, args.min_precision)
            if num_gt <= 0:
                continue
            class_ap.append(ap)
            class_recall.append(recall)
            threshold_rows.append((class_id, threshold, num_gt, ap, recall))
        ap_curve.append(float(np.mean(class_ap)) if class_ap else 0.0)
        recall_curve.append(float(np.mean(class_recall)) if class_recall else 0.0)
        class_rows.extend(threshold_rows)
    return np.asarray(ap_curve), np.asarray(recall_curve), class_rows


def bucket_summary(gt_store, event_store, model_id, partition, bucket, args,
                   unit_counts=None):
    ap_curve, recall_curve, class_rows = aggregate_curve(
        gt_store, event_store, model_id, partition, bucket, args,
        unit_counts=unit_counts)
    thresholds = np.asarray(args.distance_thresholds, dtype=np.float64)
    recall_index = int(np.argmin(np.abs(
        thresholds - args.recall_distance_threshold)))
    if not np.isclose(thresholds[recall_index], args.recall_distance_threshold):
        raise ValueError('recall-distance-threshold must be in distance-thresholds.')
    gt_count = sum(
        select_gt_units(gt_class, partition, bucket).size
        for gt_class in gt_store)
    return {
        'bmap': float(np.mean(ap_curve)),
        'mar': float(recall_curve[recall_index]),
        'ap_curve': ap_curve,
        'recall_curve': recall_curve,
        'gt': int(gt_count),
        'class_rows': class_rows,
    }


def make_partition_specs(bin_edges):
    default_size_edges = np.array([0.0, 16.0, 32.0, 96.0, np.inf])
    if np.array_equal(bin_edges['size'], default_size_edges):
        size_labels = ['Tiny (<16 px)', 'Small (16--32 px)',
                       'Medium (32--96 px)', 'Large ($\\geq$96 px)']
    else:
        size_labels = bin_labels(bin_edges['size'], suffix=' px')
    return [
        ('absolute', bin_labels(bin_edges['absolute'], suffix=' m'),
         'Absolute distance'),
        ('relative', bin_labels(bin_edges['relative'], suffix=r' $R_c$'),
         'Class-normalized distance'),
        ('size', size_labels, 'Projected size'),
    ]


def compute_main_rows(gt_store, event_store, bin_edges, args):
    rows = []
    per_class_rows = []
    for partition, labels, title in make_partition_specs(bin_edges):
        for bucket, label in enumerate(labels):
            baseline = bucket_summary(
                gt_store, event_store, 0, partition, bucket, args)
            ours = bucket_summary(
                gt_store, event_store, 1, partition, bucket, args)
            rows.append({
                'partition': partition,
                'partition_title': title,
                'bucket': bucket,
                'bucket_label': label,
                'gt': baseline['gt'],
                'baseline_bmap': baseline['bmap'],
                'ours_bmap': ours['bmap'],
                'delta_bmap_pp': 100.0 * (ours['bmap'] - baseline['bmap']),
                'baseline_mar_2m': baseline['mar'],
                'ours_mar_2m': ours['mar'],
                'delta_mar_2m_pp': 100.0 * (ours['mar'] - baseline['mar']),
            })
            for model_name, summary in ((args.baseline_name, baseline),
                                        (args.ours_name, ours)):
                for class_id, threshold, num_gt, ap, recall in summary['class_rows']:
                    per_class_rows.append({
                        'partition': partition,
                        'bucket': bucket,
                        'bucket_label': label,
                        'model': model_name,
                        'class_id': class_id,
                        'distance_threshold': threshold,
                        'gt': num_gt,
                        'ap': ap,
                        'recall': recall,
                    })
    return rows, per_class_rows


def prepare_bootstrap_inputs(rows, gt_store, event_store, args):
    """Materialize each bucket mask once instead of once per bootstrap draw."""
    prepared = {}
    for row in rows:
        row_key = (row['partition'], row['bucket'])
        prepared[row_key] = {}
        for model_id in (0, 1):
            threshold_inputs = []
            for threshold_id, _ in enumerate(args.distance_thresholds):
                class_inputs = []
                for class_id in range(len(gt_store)):
                    gt_units = select_gt_units(
                        gt_store[class_id], row_key[0], row_key[1])
                    tp, event_units = select_events(
                        event_store[model_id][threshold_id][class_id],
                        row_key[0], row_key[1])
                    class_inputs.append((tp, event_units, gt_units))
                threshold_inputs.append(class_inputs)
            prepared[row_key][model_id] = threshold_inputs
    return prepared


def bootstrap_bucket_summary(prepared, unit_counts, args):
    ap_curve = []
    recall_curve = []
    for class_inputs in prepared:
        class_ap = []
        class_recall = []
        for tp, event_units, gt_units in class_inputs:
            sampled_tp, num_gt = repeat_bootstrap_events(
                tp, event_units, gt_units, unit_counts)
            if num_gt <= 0:
                continue
            class_ap.append(nuscenes_ap_from_tp(
                sampled_tp, num_gt, min_recall=args.min_recall,
                min_precision=args.min_precision))
            class_recall.append(recall_from_tp(sampled_tp, num_gt))
        ap_curve.append(float(np.mean(class_ap)) if class_ap else 0.0)
        recall_curve.append(
            float(np.mean(class_recall)) if class_recall else 0.0)
    thresholds = np.asarray(args.distance_thresholds, dtype=np.float64)
    recall_index = int(np.argmin(np.abs(
        thresholds - args.recall_distance_threshold)))
    return float(np.mean(ap_curve)), float(recall_curve[recall_index])


def bootstrap_confidence_intervals(rows, gt_store, event_store, num_units, args):
    if args.bootstrap <= 0:
        return {}
    rng = np.random.RandomState(args.seed)
    samples = {
        (row['partition'], row['bucket']): {'bmap': [], 'mar': []}
        for row in rows
    }
    prepared = prepare_bootstrap_inputs(rows, gt_store, event_store, args)
    report_every = max(1, args.bootstrap // 10)
    print('[bootstrap] {} paired {}-level draws'.format(
        args.bootstrap, args.bootstrap_unit))
    for draw in range(args.bootstrap):
        selected = rng.randint(0, num_units, size=num_units)
        unit_counts = np.bincount(selected, minlength=num_units)
        for row in rows:
            key = (row['partition'], row['bucket'])
            baseline_bmap, baseline_mar = bootstrap_bucket_summary(
                prepared[key][0], unit_counts, args)
            ours_bmap, ours_mar = bootstrap_bucket_summary(
                prepared[key][1], unit_counts, args)
            samples[key]['bmap'].append(100.0 * (ours_bmap - baseline_bmap))
            samples[key]['mar'].append(100.0 * (ours_mar - baseline_mar))
        if (draw + 1) % report_every == 0 or draw + 1 == args.bootstrap:
            print('  {}/{}'.format(draw + 1, args.bootstrap))

    intervals = {}
    for key, metrics in samples.items():
        intervals[key] = {}
        for metric_name, values in metrics.items():
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[key][metric_name + '_ci_low'] = float(low)
            intervals[key][metric_name + '_ci_high'] = float(high)
    return intervals


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_far3d_style(gt_store, event_store, far_bucket, args, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = ((0, args.baseline_name, '#4c566a', 'o'),
               (1, args.ours_name, '#d1495b', 's'))
    x_labels = [str(value).rstrip('0').rstrip('.')
                for value in args.distance_thresholds]
    x = np.arange(len(x_labels))
    far_title = r'Far range: ${:.1f}R_c$--$R_c$'.format(
        args.far_relative_start)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.25))
    panels = (
        ('full', 0, 'recall', 'Full official range', 'Recall (%)'),
        ('relative', far_bucket, 'recall', far_title, 'Recall (%)'),
        ('full', 0, 'ap', 'Full official range', 'mAP (%)'),
        ('relative', far_bucket, 'ap', far_title, 'mAP (%)'),
    )
    for axis, (partition, bucket, metric, title, ylabel) in zip(axes, panels):
        for model_id, name, color, marker in methods:
            summary = bucket_summary(
                gt_store, event_store, model_id, partition, bucket, args)
            curve = (summary['recall_curve'] if metric == 'recall'
                     else summary['ap_curve'])
            curve = np.asarray(curve) * 100.0
            axis.plot(x, curve, color=color, marker=marker, linewidth=1.8,
                      markersize=4.5, label=name)
        axis.set_xticks(x)
        axis.set_xticklabels(x_labels)
        axis.set_xlabel('Center-distance threshold (m)')
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(axis='y', linewidth=0.5, alpha=0.35)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False, loc='best')
    fig.tight_layout(w_pad=1.25)
    paths = []
    for extension, dpi in (('pdf', None), ('png', 240)):
        path = out_dir / ('far3d_style_range_curves.' + extension)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def plot_bucket_breakdown(rows, args, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selected_partitions = ('relative', 'size')
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    colors = ('#4c566a', '#d1495b')
    for axis, partition in zip(axes, selected_partitions):
        partition_rows = [row for row in rows if row['partition'] == partition]
        x = np.arange(len(partition_rows))
        width = 0.36
        baseline = [100.0 * row['baseline_bmap'] for row in partition_rows]
        ours = [100.0 * row['ours_bmap'] for row in partition_rows]
        axis.bar(x - width / 2, baseline, width, color=colors[0],
                 label=args.baseline_name)
        axis.bar(x + width / 2, ours, width, color=colors[1],
                 label=args.ours_name)
        axis.set_xticks(x)
        labels = [row['bucket_label'].replace(r' $R_c$', '')
                  for row in partition_rows]
        axis.set_xticklabels(labels, rotation=12, ha='right')
        axis.set_ylabel('Bucket mAP (%)')
        axis.set_title(partition_rows[0]['partition_title'])
        axis.grid(axis='y', linewidth=0.5, alpha=0.35)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.4)
    paths = []
    for extension, dpi in (('pdf', None), ('png', 240)):
        path = out_dir / ('range_size_bucket_map.' + extension)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def write_latex_rows(path, rows):
    with open(path, 'w') as file:
        for row in rows:
            file.write('{} & {} & {:.2f} & {:.2f} & {:+.2f} & {:.2f} & '
                       '{:.2f} & {:+.2f} \\\\\n'.format(
                           row['bucket_label'], row['gt'],
                           100.0 * row['baseline_bmap'],
                           100.0 * row['ours_bmap'], row['delta_bmap_pp'],
                           100.0 * row['baseline_mar_2m'],
                           100.0 * row['ours_mar_2m'],
                           row['delta_mar_2m_pp']))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isposinf(value):
            return 'inf'
        if np.isneginf(value):
            return '-inf'
        if np.isnan(value):
            return None
        return float(value)
    return value


def main():
    args = parse_args()
    from mmcv import Config, load

    args.distance_bins = validate_edges(args.distance_bins)
    args.relative_distance_bins = validate_edges(args.relative_distance_bins)
    args.size_bins = validate_edges(args.size_bins)
    args.distance_thresholds = validate_edges(
        [0.0] + list(args.distance_thresholds))[1:]
    if args.bootstrap < 0:
        raise ValueError('--bootstrap must be non-negative.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    dataset = build_eval_dataset(cfg, args)
    baseline_results = load(args.baseline_pkl)
    ours_results = load(args.ours_pkl)
    expected = len(dataset)
    if len(baseline_results) != expected or len(ours_results) != expected:
        raise ValueError(
            'Result lengths must equal dataset length {} (baseline={}, ours={}).'
            .format(expected, len(baseline_results), len(ours_results)))
    if args.max_samples is not None:
        expected = min(expected, args.max_samples)

    class_names = list(getattr(dataset, 'CLASSES', NUSCENES_CLASSES))
    class_ranges = class_range_map(dataset, class_names)
    projection_geometry = get_test_projection_geometry(cfg)
    sample_units, resolved_bootstrap_unit = resolve_bootstrap_units(
        dataset, expected, args.bootstrap_unit, args.data_root)
    args.bootstrap_unit = resolved_bootstrap_unit
    num_units = int(np.max(sample_units)) + 1 if sample_units.size else 0

    print('[protocol]')
    print('  samples: {}'.format(expected))
    print('  classes: {}'.format(len(class_names)))
    print('  bootstrap unit/count: {}/{}'.format(
        resolved_bootstrap_unit, num_units))
    print('  image transform: {} resize={:.4f} crop={}'.format(
        projection_geometry['transform_type'],
        projection_geometry['resize'], projection_geometry['crop']))
    print('  image shape: {}'.format(projection_geometry['final_dim']))

    gt_store, event_store, bin_edges, projection_audit = collect_events(
        dataset, baseline_results, ours_results, sample_units,
        projection_geometry, class_names, class_ranges, args)
    rows, per_class_rows = compute_main_rows(
        gt_store, event_store, bin_edges, args)
    intervals = bootstrap_confidence_intervals(
        rows, gt_store, event_store, num_units, args)
    for row in rows:
        interval = intervals.get((row['partition'], row['bucket']), {})
        row.update(interval)

    main_fields = [
        'partition', 'partition_title', 'bucket', 'bucket_label', 'gt',
        'baseline_bmap', 'ours_bmap', 'delta_bmap_pp',
        'baseline_mar_2m', 'ours_mar_2m', 'delta_mar_2m_pp',
        'bmap_ci_low', 'bmap_ci_high', 'mar_ci_low', 'mar_ci_high']
    for row in rows:
        for field in main_fields:
            row.setdefault(field, '')
    write_csv(out_dir / 'range_size_metrics.csv', rows, main_fields)
    per_class_fields = [
        'partition', 'bucket', 'bucket_label', 'model', 'class_id', 'class',
        'distance_threshold', 'gt', 'ap', 'recall']
    for row in per_class_rows:
        row['class'] = class_names[row['class_id']]
    write_csv(out_dir / 'range_size_per_class.csv', per_class_rows,
              per_class_fields)
    write_latex_rows(out_dir / 'range_size_table_rows.tex', rows)

    far_bucket = assign_bin(
        args.far_relative_start + 1e-8, bin_edges['relative'])
    if far_bucket < 0:
        raise ValueError('--far-relative-start lies outside relative bins.')
    full_metrics = {
        args.baseline_name: bucket_summary(
            gt_store, event_store, 0, 'full', 0, args),
        args.ours_name: bucket_summary(
            gt_store, event_store, 1, 'full', 0, args),
    }
    far_metrics = {
        args.baseline_name: bucket_summary(
            gt_store, event_store, 0, 'relative', far_bucket, args),
        args.ours_name: bucket_summary(
            gt_store, event_store, 1, 'relative', far_bucket, args),
    }
    figure_paths = []
    figure_paths.extend(plot_far3d_style(
        gt_store, event_store, far_bucket, args, out_dir))
    figure_paths.extend(plot_bucket_breakdown(rows, args, out_dir))
    if args.paper_figure_dir:
        paper_dir = Path(args.paper_figure_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        for source in figure_paths:
            shutil.copy2(str(source), str(paper_dir / source.name))

    summary = {
        'protocol': {
            'config': args.config,
            'baseline_pkl': args.baseline_pkl,
            'ours_pkl': args.ours_pkl,
            'baseline_name': args.baseline_name,
            'ours_name': args.ours_name,
            'samples': expected,
            'bootstrap_draws': args.bootstrap,
            'bootstrap_unit': resolved_bootstrap_unit,
            'bootstrap_units': num_units,
            'seed': args.seed,
            'distance_thresholds': args.distance_thresholds,
            'recall_distance_threshold': args.recall_distance_threshold,
            'absolute_edges_m': bin_edges['absolute'],
            'relative_edges': bin_edges['relative'],
            'projected_size_edges_px': bin_edges['size'],
            'projected_size_definition':
                'max over current cameras of sqrt(clipped 2D bbox area)',
            'class_ranges_m': class_ranges,
            'projection_geometry': projection_geometry,
        },
        'projection_audit': projection_audit,
        'full_official_range': {
            name: {
                'mAP': metrics['bmap'],
                'mAR_at_2m': metrics['mar'],
                'AP_by_distance_threshold': metrics['ap_curve'],
                'recall_by_distance_threshold': metrics['recall_curve'],
            } for name, metrics in full_metrics.items()
        },
        'far_normalized_range': {
            name: {
                'mAP': metrics['bmap'],
                'mAR_at_2m': metrics['mar'],
                'AP_by_distance_threshold': metrics['ap_curve'],
                'recall_by_distance_threshold': metrics['recall_curve'],
            } for name, metrics in far_metrics.items()
        },
        'rows': rows,
        'figures': [str(path) for path in figure_paths],
    }
    with open(out_dir / 'summary.json', 'w') as file:
        json.dump(json_safe(summary), file, indent=2, sort_keys=True)

    print('[projection audit]')
    for key, value in projection_audit.items():
        print('  {}: {}'.format(key, value))
    print('[full-range audit]')
    for name, metrics in full_metrics.items():
        print('  {}: mAP={:.4f}, mAR@2m={:.4f}'.format(
            name, metrics['bmap'], metrics['mar']))
    print('[far-range audit: {:.1f}Rc--Rc]'.format(
        args.far_relative_start))
    for name, metrics in far_metrics.items():
        print('  {}: mAP={:.4f}, mAR@2m={:.4f}'.format(
            name, metrics['bmap'], metrics['mar']))
    print('[summary]')
    for row in rows:
        print('  {:>8s} {:<22s} GT={:5d} b-mAP {:.2f}->{:.2f} '
              '({:+.2f} pp), mAR@2m {:.2f}->{:.2f} ({:+.2f} pp)'.format(
                  row['partition'], row['bucket_label'], row['gt'],
                  100.0 * row['baseline_bmap'], 100.0 * row['ours_bmap'],
                  row['delta_bmap_pp'], 100.0 * row['baseline_mar_2m'],
                  100.0 * row['ours_mar_2m'], row['delta_mar_2m_pp']))
    print('[done] {}'.format(out_dir))


if __name__ == '__main__':
    main()
