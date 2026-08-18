#!/usr/bin/env python
"""Export paired, paper-ready qualitative nuScenes comparisons.

The script accepts MMDetection3D ``--out`` pickles or nuScenes
``results_nusc.json`` files independently for the baseline and proposed model.
It uses the same validation samples, test-time image transform, score threshold,
and center-distance matching for both methods, then:

1. ranks examples where the proposed model recovers or localizes far/small GTs;
2. renders the same camera view for input/GT, baseline, and proposed method;
3. renders a paired BEV overlay;
4. writes case images, a contact sheet, a multi-page PDF, CSV/JSON selections,
   and a reproducibility manifest.

Example:

    python tools/export_paired_qualitative.py CONFIG BASELINE.pkl OURS.pkl \
        --baseline-name 3DPPE --ours-name DGAQ-3D \
        --data-root data/nuscenes \
        --ann-file data/nuscenes/petr/mmdet3d_nuscenes_30f_infos_val.pkl \
        --score-thr 0.25 --num-cases 6 \
        --out-dir work_dirs/paper_qualitative
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for import_root in (REPO_ROOT, TOOLS_DIR):
    import_root = str(import_root)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from analyze_range_size_breakdown import (  # noqa: E402
    build_eval_dataset,
    class_range_map,
    corners_from_boxes,
    get_test_projection_geometry,
    tensor_to_numpy,
    unpack_prediction,
)
from range_size_eval_utils import (  # noqa: E402
    BOX_EDGES,
    apply_image_transform,
    lidar_centers_to_ego,
    match_predictions,
    project_box_to_image,
    quaternion_to_matrix,
)


NUSCENES_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

COLORS = {
    'gt': (255, 196, 45, 255),
    'gt_context': (215, 215, 215, 150),
    'baseline': (30, 186, 224, 220),
    'ours': (47, 202, 105, 235),
    'grid': (82, 91, 103, 125),
    'text': (25, 28, 33, 255),
    'muted': (105, 112, 122, 255),
    'paper': (255, 255, 255, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select and render paired nuScenes qualitative examples.')
    parser.add_argument('config', help='Validation config used by both methods.')
    parser.add_argument('baseline_result', help='Baseline .pkl/.pickle or results JSON.')
    parser.add_argument('ours_result', help='Proposed .pkl/.pickle or results JSON.')
    parser.add_argument('--baseline-name', default='3DPPE')
    parser.add_argument('--ours-name', default='DGAQ-3D')
    parser.add_argument('--out-dir', default='work_dirs/paired_qualitative')
    parser.add_argument('--data-root', default=os.environ.get('NUSCENES_DATA_ROOT'))
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--score-thr', type=float, default=0.25,
                        help='Shared final 3D detection score threshold.')
    parser.add_argument('--match-distance', type=float, default=2.0,
                        help='Shared class-wise center matching threshold in meters.')
    parser.add_argument('--far-relative-start', type=float, default=0.7,
                        help='Far if distance / official class range is at least this.')
    parser.add_argument('--small-size-thr', type=float, default=32.0,
                        help='Small if max-view sqrt(projected area) is below this.')
    parser.add_argument('--min-error-gain', type=float, default=0.5,
                        help='Minimum center-error reduction for a strong gain case.')
    parser.add_argument('--num-cases', type=int, default=6)
    parser.add_argument('--num-regressions', type=int, default=1,
                        help='Reserved baseline-better cases for an honest appendix.')
    parser.add_argument('--num-cams', type=int, default=6)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug only. Omit for paper export.')
    parser.add_argument('--max-boxes-per-view', type=int, default=50)
    parser.add_argument('--min-projection-depth', type=float, default=0.1)
    parser.add_argument('--bev-range', type=float, default=55.0)
    parser.add_argument('--panel-width', type=int, default=520)
    parser.add_argument('--rows-per-page', type=int, default=2)
    parser.add_argument('--main-case-ranks', default='1,2,3',
                        help='Diagnostic ranks used by the compact paper figure.')
    parser.add_argument('--paper-figure-dir', default=None,
                        help='Optional directory receiving the final PDF and PNG.')
    return parser.parse_args()


@dataclass
class BoxSet:
    centers: np.ndarray
    corners: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    edges: tuple = BOX_EDGES

    @classmethod
    def empty(cls, edges=BOX_EDGES):
        return cls(
            centers=np.zeros((0, 3), dtype=np.float32),
            corners=np.zeros((0, 8, 3), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            scores=np.zeros((0,), dtype=np.float32),
            edges=tuple(edges),
        )

    def __len__(self):
        return int(self.labels.size)

    def take(self, indices):
        indices = np.asarray(indices)
        return BoxSet(
            centers=self.centers[indices],
            corners=self.corners[indices],
            labels=self.labels[indices],
            scores=self.scores[indices],
            edges=self.edges,
        )


def resolve_result_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError('Result does not exist: {}'.format(path))
    candidates = (
        path / 'results_nusc.json',
        path / 'pts_bbox' / 'results_nusc.json',
        path / 'results' / 'pts_bbox' / 'results_nusc.json',
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    recursive = sorted(path.glob('**/results_nusc.json'))
    if len(recursive) == 1:
        return recursive[0]
    raise FileNotFoundError(
        'Expected exactly one results_nusc.json below {}, found {}.'.format(
            path, len(recursive)))


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class PicklePredictionSource:
    kind = 'mmdet-pickle'

    def __init__(self, path):
        import mmcv
        self.path = Path(path)
        loaded = mmcv.load(str(path))
        if isinstance(loaded, dict) and isinstance(loaded.get('results'), list):
            loaded = loaded['results']
        if not isinstance(loaded, (list, tuple)):
            raise TypeError('Pickle result must contain a list of sample outputs.')
        self.results = loaded

    def __len__(self):
        return len(self.results)

    def get(self, sample_index, sample_token, raw_info, class_names):
        del sample_token, raw_info, class_names
        boxes, scores, labels = unpack_prediction(self.results[sample_index])
        if boxes is None:
            return BoxSet.empty()
        centers = tensor_to_numpy(boxes.gravity_center).reshape(-1, 3)
        corners = corners_from_boxes(boxes)
        return BoxSet(
            centers=np.asarray(centers, dtype=np.float32),
            corners=np.asarray(corners, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
            edges=BOX_EDGES,
        )


class JsonPredictionSource:
    kind = 'nuscenes-json'

    def __init__(self, path):
        self.path = Path(path)
        with open(path, 'r') as stream:
            loaded = json.load(stream)
        self.results = loaded.get('results', loaded)
        if not isinstance(self.results, dict):
            raise TypeError('nuScenes JSON must map sample tokens to detections.')

    def __len__(self):
        return len(self.results)

    def get(self, sample_index, sample_token, raw_info, class_names):
        del sample_index
        records = self.results.get(sample_token, [])
        if not records:
            return BoxSet.empty(edges=NUSCENES_BOX_EDGES)
        class_to_id = {name: index for index, name in enumerate(class_names)}
        centers = []
        corners = []
        labels = []
        scores = []
        for record in records:
            name = record.get('detection_name', record.get('name'))
            if name not in class_to_id:
                continue
            translation = np.asarray(record['translation'], dtype=np.float64)
            size = np.asarray(record['size'], dtype=np.float64)
            rotation = quaternion_to_matrix(record['rotation'])
            local = nuscenes_local_corners(size)
            global_corners = local @ rotation.T + translation.reshape(1, 3)
            centers.append(global_to_lidar(translation.reshape(1, 3), raw_info)[0])
            corners.append(global_to_lidar(global_corners, raw_info))
            labels.append(class_to_id[name])
            scores.append(float(record.get('detection_score', record.get('score', 0.0))))
        if not labels:
            return BoxSet.empty(edges=NUSCENES_BOX_EDGES)
        return BoxSet(
            centers=np.asarray(centers, dtype=np.float32),
            corners=np.asarray(corners, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
            edges=NUSCENES_BOX_EDGES,
        )


def load_prediction_source(path):
    suffix = path.suffix.lower()
    if suffix in ('.pkl', '.pickle'):
        return PicklePredictionSource(path)
    if suffix == '.json':
        return JsonPredictionSource(path)
    raise ValueError('Unsupported result extension: {}'.format(path.suffix))


def nuscenes_local_corners(size):
    """Return nuScenes Box corner ordering for size ``[width, length, height]``."""
    width, length, height = [float(value) for value in size]
    return np.array([
        [length / 2, width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [length / 2, -width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, height / 2],
        [-length / 2, -width / 2, height / 2],
        [-length / 2, -width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
    ], dtype=np.float64)


def global_to_lidar(points, raw_info):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ego_rotation = quaternion_to_matrix(raw_info['ego2global_rotation'])
    ego_translation = np.asarray(
        raw_info['ego2global_translation'], dtype=np.float64).reshape(1, 3)
    lidar_rotation = quaternion_to_matrix(raw_info['lidar2ego_rotation'])
    lidar_translation = np.asarray(
        raw_info['lidar2ego_translation'], dtype=np.float64).reshape(1, 3)
    points_ego = (points - ego_translation) @ ego_rotation
    return (points_ego - lidar_translation) @ lidar_rotation


def lidar_points_to_ego(points, raw_info):
    shape = np.asarray(points).shape
    flattened = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    converted = lidar_centers_to_ego(
        flattened,
        raw_info.get('lidar2ego_rotation', [1.0, 0.0, 0.0, 0.0]),
        raw_info.get('lidar2ego_translation', [0.0, 0.0, 0.0]),
    )
    return converted.reshape(shape)


def boxes_from_annotation(annotation):
    boxes = annotation['gt_bboxes_3d']
    labels = tensor_to_numpy(annotation['gt_labels_3d']).astype(np.int64)
    if boxes is None or labels.size == 0:
        return BoxSet.empty()
    return BoxSet(
        centers=tensor_to_numpy(boxes.gravity_center).reshape(-1, 3).astype(np.float32),
        corners=corners_from_boxes(boxes).astype(np.float32),
        labels=labels.reshape(-1),
        scores=np.ones(labels.size, dtype=np.float32),
        edges=BOX_EDGES,
    )


def filter_boxes(boxes, raw_info, class_names, class_ranges, score_thr=None):
    if len(boxes) == 0:
        return boxes
    labels = boxes.labels
    valid = (labels >= 0) & (labels < len(class_names))
    valid &= np.isfinite(boxes.centers).all(axis=1)
    valid &= np.isfinite(boxes.scores)
    if score_thr is not None:
        valid &= boxes.scores >= float(score_thr)
    centers_ego = lidar_points_to_ego(boxes.centers, raw_info)
    ranges = np.linalg.norm(centers_ego[:, :2], axis=1)
    limits = np.zeros(len(boxes), dtype=np.float64)
    for label in np.unique(labels[valid]):
        limits[labels == label] = class_ranges[class_names[int(label)]]
    valid &= ranges <= limits + 1e-6
    return boxes.take(np.flatnonzero(valid))


def resolve_scene_tokens(dataset, sample_count, data_root):
    embedded = [
        info.get('scene_token') for info in dataset.data_infos[:sample_count]
    ]
    if all(token is not None for token in embedded):
        return embedded
    try:
        from nuscenes import NuScenes
        nusc = NuScenes(
            version=getattr(dataset, 'version', 'v1.0-trainval'),
            dataroot=data_root or getattr(dataset, 'data_root', None),
            verbose=False,
        )
        return [
            nusc.get('sample', info['token'])['scene_token']
            for info in dataset.data_infos[:sample_count]
        ]
    except Exception as error:
        print('[warning] Cannot resolve scene tokens ({}); sample tokens will '
              'be used for diversity.'.format(error))
        return [info['token'] for info in dataset.data_infos[:sample_count]]


def best_camera_projection(corners, lidar2img, image_shape, min_depth):
    best = None
    best_area = -1.0
    for camera_index, matrix in enumerate(lidar2img):
        rectangle = project_box_to_image(
            corners, matrix, image_shape, min_depth=min_depth)
        if rectangle is None:
            continue
        area = max(0.0, float(rectangle[2] - rectangle[0])) * max(
            0.0, float(rectangle[3] - rectangle[1]))
        if area > best_area:
            best_area = area
            best = (camera_index, rectangle)
    if best is None:
        return None, None, 0.0
    return best[0], best[1], math.sqrt(best_area)


def matches_by_gt(gt, pred, gt_centers_ego, pred_centers_ego,
                  class_id, distance_threshold):
    gt_indices = np.flatnonzero(gt.labels == class_id)
    pred_indices = np.flatnonzero(pred.labels == class_id)
    if gt_indices.size == 0 or pred_indices.size == 0:
        return {}
    order, matched_local_gt = match_predictions(
        gt_centers_ego[gt_indices, :2],
        pred_centers_ego[pred_indices, :2],
        pred.scores[pred_indices],
        distance_threshold,
    )
    output = {}
    for rank, local_gt in enumerate(matched_local_gt):
        if local_gt < 0:
            continue
        gt_index = int(gt_indices[int(local_gt)])
        pred_index = int(pred_indices[int(order[rank])])
        error = float(np.linalg.norm(
            gt_centers_ego[gt_index, :2] - pred_centers_ego[pred_index, :2]))
        output[gt_index] = {
            'pred_index': pred_index,
            'error': error,
            'score': float(pred.scores[pred_index]),
        }
    return output


def challenge_name(relative_distance, projected_size, args):
    far = relative_distance >= args.far_relative_start
    small = projected_size < args.small_size_thr
    if far and small:
        return 'far_small'
    if far:
        return 'far'
    if small:
        return 'small'
    return 'general'


def comparison_status(baseline_match, ours_match, min_error_gain):
    if baseline_match is None and ours_match is not None:
        return 'ours_only', None
    if baseline_match is not None and ours_match is None:
        return 'baseline_only', None
    if baseline_match is None or ours_match is None:
        return None, None
    gain = baseline_match['error'] - ours_match['error']
    if gain >= min_error_gain:
        return 'localization_gain', gain
    if gain > 0.0:
        return 'modest_gain', gain
    if gain <= -min_error_gain:
        return 'localization_regression', gain
    if gain < 0.0:
        return 'modest_regression', gain
    return None, gain


def collect_candidates(dataset, baseline_source, ours_source, scene_tokens,
                       projection_geometry, class_names, class_ranges, args):
    ida = projection_geometry['matrix']
    image_shape = projection_geometry['final_dim']
    source_lengths = [len(dataset)]
    if isinstance(baseline_source, PicklePredictionSource):
        source_lengths.append(len(baseline_source))
    if isinstance(ours_source, PicklePredictionSource):
        source_lengths.append(len(ours_source))
    sample_count = min(source_lengths)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)

    rows = []
    for sample_index in range(sample_count):
        if sample_index % 250 == 0 or sample_index + 1 == sample_count:
            print('[select] {}/{}'.format(sample_index + 1, sample_count))
        raw_info = dataset.data_infos[sample_index]
        data_info = dataset.get_data_info(sample_index)
        sample_token = raw_info['token']
        lidar2img = apply_image_transform(
            np.asarray(data_info['lidar2img'][:args.num_cams]), ida)
        camera_names = list(raw_info.get('cams', {}).keys())[:args.num_cams]
        if len(camera_names) < args.num_cams:
            camera_names = [
                'camera_{}'.format(index) for index in range(args.num_cams)
            ]

        gt = filter_boxes(
            boxes_from_annotation(dataset.get_ann_info(sample_index)),
            raw_info, class_names, class_ranges)
        baseline = filter_boxes(
            baseline_source.get(
                sample_index, sample_token, raw_info, class_names),
            raw_info, class_names, class_ranges, args.score_thr)
        ours = filter_boxes(
            ours_source.get(sample_index, sample_token, raw_info, class_names),
            raw_info, class_names, class_ranges, args.score_thr)
        if len(gt) == 0:
            continue

        gt_ego = lidar_points_to_ego(gt.centers, raw_info)
        baseline_ego = lidar_points_to_ego(baseline.centers, raw_info)
        ours_ego = lidar_points_to_ego(ours.centers, raw_info)
        baseline_matches = {}
        ours_matches = {}
        for class_id in range(len(class_names)):
            baseline_matches.update(matches_by_gt(
                gt, baseline, gt_ego, baseline_ego, class_id,
                args.match_distance))
            ours_matches.update(matches_by_gt(
                gt, ours, gt_ego, ours_ego, class_id,
                args.match_distance))

        for gt_index in range(len(gt)):
            class_id = int(gt.labels[gt_index])
            camera_index, rectangle, size = best_camera_projection(
                gt.corners[gt_index], lidar2img, image_shape,
                args.min_projection_depth)
            if camera_index is None:
                continue
            distance = float(np.linalg.norm(gt_ego[gt_index, :2]))
            class_limit = class_ranges[class_names[class_id]]
            relative_distance = distance / class_limit
            baseline_match = baseline_matches.get(gt_index)
            ours_match = ours_matches.get(gt_index)
            status, gain = comparison_status(
                baseline_match, ours_match, args.min_error_gain)
            if status is None:
                continue
            rows.append({
                'sample_index': sample_index,
                'sample_token': sample_token,
                'scene_token': scene_tokens[sample_index],
                'camera_index': int(camera_index),
                'camera_name': camera_names[int(camera_index)],
                'class_id': class_id,
                'class_name': class_names[class_id],
                'gt_index': gt_index,
                'distance_m': distance,
                'relative_distance': relative_distance,
                'projected_size_px': float(size),
                'challenge': challenge_name(relative_distance, size, args),
                'status': status,
                'error_gain_m': None if gain is None else float(gain),
                'baseline_pred_index': (
                    None if baseline_match is None
                    else int(baseline_match['pred_index'])),
                'baseline_error_m': (
                    None if baseline_match is None
                    else float(baseline_match['error'])),
                'baseline_score': (
                    None if baseline_match is None
                    else float(baseline_match['score'])),
                'ours_pred_index': (
                    None if ours_match is None
                    else int(ours_match['pred_index'])),
                'ours_error_m': (
                    None if ours_match is None
                    else float(ours_match['error'])),
                'ours_score': (
                    None if ours_match is None
                    else float(ours_match['score'])),
                'target_rect': [float(value) for value in rectangle],
            })
    return rows, sample_count


def candidate_sort_key(row, regression=False):
    challenge_rank = {'far_small': 4, 'far': 3, 'small': 2, 'general': 1}
    if regression:
        status_rank = {'baseline_only': 3, 'localization_regression': 2,
                       'modest_regression': 1}
        severity = abs(row['error_gain_m'] or 0.0)
    else:
        status_rank = {'ours_only': 3, 'localization_gain': 2,
                       'modest_gain': 1}
        severity = row['error_gain_m'] or 0.0
    return (
        challenge_rank.get(row['challenge'], 0),
        status_rank.get(row['status'], 0),
        severity,
        row['relative_distance'],
        -row['projected_size_px'],
    )


def pick_diverse(rows, count, used_samples=None, used_cameras=None):
    used_samples = set() if used_samples is None else used_samples
    used_cameras = set() if used_cameras is None else used_cameras
    selected = []
    used_scenes = set()
    used_classes = set()
    passes = (
        lambda row: row["camera_name"] not in used_cameras
        and row["scene_token"] not in used_scenes
        and row["class_name"] not in used_classes,
        lambda row: row["camera_name"] not in used_cameras
        and row["scene_token"] not in used_scenes,
        lambda row: row["camera_name"] not in used_cameras,
        lambda row: row["scene_token"] not in used_scenes
        and row["class_name"] not in used_classes,
        lambda row: row["scene_token"] not in used_scenes,
        lambda row: True,
    )
    for predicate in passes:
        for row in rows:
            if len(selected) >= count:
                return selected
            if row["sample_index"] in used_samples or row in selected:
                continue
            if not predicate(row):
                continue
            selected.append(row)
            used_samples.add(row["sample_index"])
            used_cameras.add(row["camera_name"])
            used_scenes.add(row["scene_token"])
            used_classes.add(row["class_name"])
    return selected


def select_candidates(rows, args):
    regressions = {
        'baseline_only', 'localization_regression', 'modest_regression'}
    gain_rows = sorted(
        [row for row in rows if row['status'] not in regressions],
        key=lambda row: candidate_sort_key(row, regression=False), reverse=True)
    regression_rows = sorted(
        [row for row in rows if row['status'] in regressions],
        key=lambda row: candidate_sort_key(row, regression=True), reverse=True)

    regression_count = min(
        max(args.num_regressions, 0), max(args.num_cases - 1, 0))
    gain_count = max(args.num_cases - regression_count, 0)
    used_samples = set()
    used_cameras = set()
    selected = []
    recovered_rows = [row for row in gain_rows if row["status"] == "ours_only"]
    localization_rows = [
        row for row in gain_rows if row["status"] == "localization_gain"]
    if gain_count > 0:
        selected.extend(pick_diverse(
            recovered_rows, 1, used_samples, used_cameras))
    if len(selected) < gain_count:
        selected.extend(pick_diverse(
            localization_rows, 1, used_samples, used_cameras))
    if len(selected) < gain_count:
        selected.extend(pick_diverse(
            gain_rows, gain_count - len(selected),
            used_samples, used_cameras))
    selected.extend(pick_diverse(
        regression_rows, min(regression_count, args.num_cases - len(selected)),
        used_samples, used_cameras))
    if len(selected) < args.num_cases:
        remaining = [
            row for row in gain_rows + regression_rows
            if row["sample_index"] not in used_samples
        ]
        selected.extend(pick_diverse(
            remaining, args.num_cases - len(selected),
            used_samples, used_cameras))
    for rank, row in enumerate(selected, 1):
        row['selection_rank'] = rank
    return selected


def font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend([
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/tru/dejavu/DejaVuSans.ttf',
        ])
    else:
        candidates.append('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def transformed_camera_image(path, projection_geometry, data_root=None):
    path = Path(path)
    if not path.is_file() and data_root:
        value = str(path)
        for prefix in ('./data/nuscenes/', 'data/nuscenes/'):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        candidate = Path(data_root) / value
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise FileNotFoundError('Camera image not found: {}'.format(path))
    image = Image.open(str(path)).convert('RGB')
    image = image.resize(
        projection_geometry['resize_dims'], resample=Image.BILINEAR)
    return image.crop(projection_geometry['crop'])


def liang_barsky(p0, p1, width, height):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0, width - 1 - x0, y0, height - 1 - y0)
    low, high = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        ratio = qi / pi
        if pi < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None
    return ((x0 + low * dx, y0 + low * dy),
            (x0 + high * dx, y0 + high * dy))


def projected_segments(corners, matrix, edges, image_size, min_depth):
    corners = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    homogeneous = np.concatenate(
        [corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    width, height = image_size
    segments = []
    for first, second in edges:
        point0 = projected[first].copy()
        point1 = projected[second].copy()
        front0 = point0[2] >= min_depth
        front1 = point1[2] >= min_depth
        if not front0 and not front1:
            continue
        if front0 != front1:
            denominator = point1[2] - point0[2]
            if abs(denominator) < 1e-12:
                continue
            ratio = (min_depth - point0[2]) / denominator
            clipped = point0 + ratio * (point1 - point0)
            if front0:
                point1 = clipped
            else:
                point0 = clipped
        uv0 = point0[:2] / point0[2]
        uv1 = point1[:2] / point1[2]
        if not np.isfinite(np.concatenate([uv0, uv1])).all():
            continue
        clipped = liang_barsky(uv0, uv1, width, height)
        if clipped is not None:
            segments.append(clipped)
    return segments


def draw_dashed_line(draw, start, end, fill, width=2, dash=10, gap=6):
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 1e-6:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    offset = 0.0
    while offset < length:
        stop = min(offset + dash, length)
        draw.line(
            [(x0 + dx * offset, y0 + dy * offset),
             (x0 + dx * stop, y0 + dy * stop)],
            fill=fill, width=width)
        offset += dash + gap


def draw_box(draw, corners, matrix, edges, image_size, color, width,
             min_depth, dashed=False):
    segments = projected_segments(
        corners, matrix, edges, image_size, min_depth)
    for start, end in segments:
        if dashed:
            draw_dashed_line(draw, start, end, color, width=width)
        else:
            draw.line([start, end], fill=color, width=width)
    return bool(segments)


def draw_dashed_rectangle(draw, rectangle, fill, width=3):
    left, top, right, bottom = rectangle
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for start, end in zip(corners, corners[1:] + corners[:1]):
        draw_dashed_line(draw, start, end, fill, width=width)


def add_zoom_inset(image, target_rectangle):
    if target_rectangle is None:
        return image
    left, top, right, bottom = target_rectangle
    width, height = image.size
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_w = max((right - left) * 4.0, width * 0.13)
    crop_h = max((bottom - top) * 4.0, height * 0.25)
    crop_w = min(crop_w, width * 0.45)
    crop_h = min(crop_h, height * 0.55)
    crop = (
        max(0, int(center_x - crop_w / 2)),
        max(0, int(center_y - crop_h / 2)),
        min(width, int(center_x + crop_w / 2)),
        min(height, int(center_y + crop_h / 2)),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return image
    inset_w = int(width * 0.29)
    inset_h = int(height * 0.42)
    inset = image.crop(crop).resize((inset_w, inset_h), Image.BICUBIC)
    margin = max(8, int(width * 0.01))
    x = margin if center_x > width / 2 else width - inset_w - margin
    y = margin
    overlay = image.copy()
    border = Image.new('RGB', (inset_w + 8, inset_h + 8), 'white')
    border.paste(inset, (4, 4))
    overlay.paste(border, (x - 4, y - 4))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (x - 4, y - 4, x + inset_w + 3, y + inset_h + 3),
        outline=(35, 38, 43), width=2)
    draw.text((x + 7, y + 5), 'Target zoom', fill=(255, 255, 255),
              font=font(max(13, int(height * 0.025)), bold=True),
              stroke_width=2, stroke_fill=(20, 20, 20))
    return overlay


def annotate_camera(image, gt, predictions, matrix, candidate, mode, args):
    canvas = image.convert('RGBA')
    draw = ImageDraw.Draw(canvas, 'RGBA')
    image_size = canvas.size
    target_gt = int(candidate['gt_index'])

    for index in range(len(gt)):
        draw_box(
            draw, gt.corners[index], matrix, gt.edges, image_size,
            COLORS['gt_context'], 1, args.min_projection_depth)
    target_rect = project_box_to_image(
        gt.corners[target_gt], matrix, image_size,
        min_depth=args.min_projection_depth)
    draw_box(
        draw, gt.corners[target_gt], matrix, gt.edges, image_size,
        COLORS['gt'], 4, args.min_projection_depth, dashed=True)
    if target_rect is not None:
        draw_dashed_rectangle(draw, target_rect, COLORS['gt'], width=3)

    if predictions is not None and len(predictions):
        color = COLORS[mode]
        order = np.argsort(-predictions.scores)[:args.max_boxes_per_view]
        target_pred = candidate.get(mode + '_pred_index')
        for index in order:
            line_width = 4 if target_pred is not None and int(index) == target_pred else 2
            visible = draw_box(
                draw, predictions.corners[index], matrix, predictions.edges,
                image_size, color, line_width, args.min_projection_depth)
            if visible and target_pred is not None and int(index) == target_pred:
                rectangle = project_box_to_image(
                    predictions.corners[index], matrix, image_size,
                    min_depth=args.min_projection_depth)
                if rectangle is not None:
                    label = '{} {:.2f}'.format(
                        candidate['class_name'], predictions.scores[index])
                    draw.text(
                        (rectangle[0] + 3, max(2, rectangle[1] - 25)), label,
                        fill=color, font=font(20, bold=True), stroke_width=2,
                        stroke_fill=(20, 20, 20, 220))
    return add_zoom_inset(canvas.convert('RGB'), target_rect)


def make_panel(image, title, panel_width):
    image = image.convert('RGB')
    panel_height = max(1, round(image.height * panel_width / image.width))
    resized = image.resize((panel_width, panel_height), Image.LANCZOS)
    title_height = 38
    panel = Image.new('RGB', (panel_width, panel_height + title_height), 'white')
    panel.paste(resized, (0, title_height))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), title, fill=COLORS['text'][:3], font=font(20, bold=True))
    draw.line((0, title_height - 1, panel_width, title_height - 1),
              fill=(215, 218, 222), width=1)
    return panel


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(points) <= 1:
        return points

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def draw_bev_polygon(draw, corners_ego, transform, color, width=2, dashed=False):
    hull = convex_hull(np.asarray(corners_ego)[:, :2])
    if len(hull) < 2:
        return
    points = [transform(point) for point in hull]
    for start, end in zip(points, points[1:] + points[:1]):
        if dashed:
            draw_dashed_line(draw, start, end, color, width=width, dash=7, gap=5)
        else:
            draw.line([start, end], fill=color, width=width)


def render_bev(gt, baseline, ours, raw_info, candidate, args, size):
    width, height = size
    image = Image.new('RGBA', size, (248, 249, 250, 255))
    draw = ImageDraw.Draw(image, 'RGBA')
    margin = 18
    origin = (width / 2, height - margin)
    scale = (height - 2 * margin) / args.bev_range

    def transform(point):
        forward, lateral = point[0], point[1]
        return origin[0] - lateral * scale, origin[1] - forward * scale

    for distance in (10, 20, 30, 40, 50):
        if distance > args.bev_range:
            continue
        radius = distance * scale
        draw.arc((origin[0] - radius, origin[1] - radius,
                  origin[0] + radius, origin[1] + radius),
                 180, 360, fill=COLORS['grid'], width=1)
        draw.text((origin[0] + radius - 23, origin[1] - 17),
                  '{}m'.format(distance), fill=COLORS['muted'], font=font(12))
    draw.line((origin[0], margin, origin[0], origin[1]),
              fill=COLORS['grid'], width=1)
    draw.polygon([
        (origin[0], origin[1] - 10),
        (origin[0] - 7, origin[1] + 4),
        (origin[0] + 7, origin[1] + 4),
    ], fill=(30, 34, 40, 255))

    gt_corners = lidar_points_to_ego(gt.corners, raw_info)
    baseline_corners = lidar_points_to_ego(baseline.corners, raw_info)
    ours_corners = lidar_points_to_ego(ours.corners, raw_info)
    for index in range(len(gt)):
        draw_bev_polygon(
            draw, gt_corners[index], transform, COLORS['gt_context'], width=1)
    target_gt = int(candidate['gt_index'])
    draw_bev_polygon(
        draw, gt_corners[target_gt], transform, COLORS['gt'], width=4)

    baseline_order = np.argsort(-baseline.scores)[:args.max_boxes_per_view]
    ours_order = np.argsort(-ours.scores)[:args.max_boxes_per_view]
    for index in baseline_order:
        width_value = 4 if candidate.get('baseline_pred_index') == int(index) else 2
        draw_bev_polygon(
            draw, baseline_corners[index], transform, COLORS['baseline'],
            width=width_value, dashed=True)
    for index in ours_order:
        width_value = 4 if candidate.get('ours_pred_index') == int(index) else 2
        draw_bev_polygon(
            draw, ours_corners[index], transform, COLORS['ours'],
            width=width_value)

    legend_y = 9
    entries = (
        ('GT', COLORS['gt']),
        (args.baseline_name, COLORS['baseline']),
        (args.ours_name, COLORS['ours']),
    )
    x = 12
    for label, color in entries:
        draw.line((x, legend_y + 7, x + 22, legend_y + 7), fill=color, width=4)
        draw.text((x + 28, legend_y), label, fill=COLORS['text'], font=font(13))
        x += 30 + int(draw.textlength(label, font=font(13))) + 34
    return image.convert('RGB')


def error_text(value):
    return 'miss' if value is None else '{:.2f} m'.format(value)


def render_case(dataset, baseline_source, ours_source, projection_geometry,
                class_names, class_ranges, candidate, args):
    sample_index = int(candidate['sample_index'])
    raw_info = dataset.data_infos[sample_index]
    data_info = dataset.get_data_info(sample_index)
    token = raw_info['token']
    camera_index = int(candidate['camera_index'])
    matrices = apply_image_transform(
        np.asarray(data_info['lidar2img'][:args.num_cams]),
        projection_geometry['matrix'])
    matrix = matrices[camera_index]

    gt = filter_boxes(
        boxes_from_annotation(dataset.get_ann_info(sample_index)),
        raw_info, class_names, class_ranges)
    baseline = filter_boxes(
        baseline_source.get(sample_index, token, raw_info, class_names),
        raw_info, class_names, class_ranges, args.score_thr)
    ours = filter_boxes(
        ours_source.get(sample_index, token, raw_info, class_names),
        raw_info, class_names, class_ranges, args.score_thr)

    camera_paths = data_info['img_filename'][:args.num_cams]
    image = transformed_camera_image(
        camera_paths[camera_index], projection_geometry, args.data_root)
    gt_view = annotate_camera(
        image, gt, None, matrix, candidate, 'baseline', args)
    baseline_view = annotate_camera(
        image, gt, baseline, matrix, candidate, 'baseline', args)
    ours_view = annotate_camera(
        image, gt, ours, matrix, candidate, 'ours', args)

    image_panel = make_panel(gt_view, 'Input / ground truth', args.panel_width)
    baseline_title = '{}  |  target {}'.format(
        args.baseline_name, error_text(candidate['baseline_error_m']))
    ours_title = '{}  |  target {}'.format(
        args.ours_name, error_text(candidate['ours_error_m']))
    baseline_panel = make_panel(baseline_view, baseline_title, args.panel_width)
    ours_panel = make_panel(ours_view, ours_title, args.panel_width)
    bev_height = image_panel.height - 38
    bev = render_bev(
        gt, baseline, ours, raw_info, candidate, args,
        (args.panel_width, bev_height))
    bev_panel = make_panel(bev, 'Paired BEV overlay', args.panel_width)

    header_height = 48
    gap = 8
    row_width = sum(
        panel.width for panel in (image_panel, baseline_panel, ours_panel, bev_panel)
    ) + 3 * gap
    row_height = header_height + max(
        panel.height for panel in (image_panel, baseline_panel, ours_panel, bev_panel))
    row = Image.new('RGB', (row_width, row_height), 'white')
    draw = ImageDraw.Draw(row)
    header = ('Case {rank:02d} | {class_name} | {challenge} | {status} | '
              'range {distance_m:.1f} m ({relative_distance:.2f} Rc) | '
              'size {projected_size_px:.1f} px | {camera_name}').format(
                  rank=candidate['selection_rank'], **candidate)
    draw.text((12, 12), header, fill=COLORS['text'][:3], font=font(19, bold=True))
    x = 0
    for panel in (image_panel, baseline_panel, ours_panel, bev_panel):
        row.paste(panel, (x, header_height))
        x += panel.width + gap
    return row


def safe_slug(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')
    return value or 'case'


def write_candidates_csv(path, rows):
    fields = [
        'sample_index', 'sample_token', 'scene_token', 'camera_index',
        'camera_name', 'class_name', 'distance_m', 'relative_distance',
        'projected_size_px', 'challenge', 'status', 'error_gain_m',
        'baseline_pred_index', 'baseline_error_m', 'baseline_score',
        'ours_pred_index', 'ours_error_m', 'ours_score',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_pdf_pages(rows, output_dir, args):
    if not rows:
        raise RuntimeError('No rows were rendered.')
    gap = 12
    margin = 18
    page_images = []
    for start in range(0, len(rows), args.rows_per_page):
        current = rows[start:start + args.rows_per_page]
        width = max(row.width for row in current) + 2 * margin
        height = sum(row.height for row in current) + gap * (len(current) - 1) + 2 * margin
        page = Image.new('RGB', (width, height), 'white')
        y = margin
        for row in current:
            page.paste(row, (margin, y))
            y += row.height + gap
        page_index = len(page_images) + 1
        page.save(output_dir / 'paired_qualitative_page_{:02d}.png'.format(page_index))
        page_images.append(page)

    contact_height = sum(row.height for row in rows) + gap * (len(rows) - 1) + 2 * margin
    contact_width = max(row.width for row in rows) + 2 * margin
    contact = Image.new('RGB', (contact_width, contact_height), 'white')
    y = margin
    for row in rows:
        contact.paste(row, (margin, y))
        y += row.height + gap
    contact_path = output_dir / 'paired_qualitative_contact_sheet.png'
    contact.save(contact_path)

    pdf_path = output_dir / 'paired_qualitative.pdf'
    page_images[0].save(
        pdf_path, 'PDF', resolution=150.0, save_all=True,
        append_images=page_images[1:])
    return pdf_path, contact_path, page_images


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError('--num-cases must be positive.')
    if args.rows_per_page <= 0:
        raise ValueError('--rows-per-page must be positive.')
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError('--score-thr must be in [0, 1].')
    if args.data_root:
        os.environ['NUSCENES_DATA_ROOT'] = os.path.abspath(args.data_root)

    baseline_path = resolve_result_path(args.baseline_result)
    ours_path = resolve_result_path(args.ours_result)
    output_dir = Path(args.out_dir).expanduser().resolve()
    case_dir = output_dir / 'cases'
    case_dir.mkdir(parents=True, exist_ok=True)

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    dataset = build_eval_dataset(cfg, args)
    projection_geometry = get_test_projection_geometry(cfg)
    class_names = tuple(getattr(dataset, 'CLASSES'))
    class_ranges = class_range_map(dataset, class_names)
    baseline_source = load_prediction_source(baseline_path)
    ours_source = load_prediction_source(ours_path)

    source_lengths = [len(dataset)]
    if isinstance(baseline_source, PicklePredictionSource):
        source_lengths.append(len(baseline_source))
    if isinstance(ours_source, PicklePredictionSource):
        source_lengths.append(len(ours_source))
    sample_count = min(source_lengths)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
    scene_tokens = resolve_scene_tokens(dataset, sample_count, args.data_root)

    baseline_hash = file_sha256(baseline_path)
    ours_hash = file_sha256(ours_path)
    if baseline_hash == ours_hash:
        print('[warning] Baseline and proposed result files have identical SHA256.')
    print('[input] baseline={} ({})'.format(baseline_path, baseline_source.kind))
    print('[input] ours={} ({})'.format(ours_path, ours_source.kind))

    candidates, evaluated_samples = collect_candidates(
        dataset, baseline_source, ours_source, scene_tokens,
        projection_geometry, class_names, class_ranges, args)
    candidates = sorted(
        candidates,
        key=lambda row: candidate_sort_key(
            row, regression='regression' in row['status']
            or row['status'] == 'baseline_only'),
        reverse=True)
    selected = select_candidates(candidates, args)
    if not selected:
        raise RuntimeError(
            'No paired candidates found. Lower --score-thr or '
            '--min-error-gain after checking the evaluation protocol.')

    write_candidates_csv(output_dir / 'candidates.csv', candidates)
    with open(output_dir / 'selected_cases.json', 'w') as stream:
        json.dump(json_ready(selected), stream, indent=2)

    rendered_rows = []
    for candidate in selected:
        print('[render] case {}/{}: sample {} {} {}'.format(
            candidate['selection_rank'], len(selected),
            candidate['sample_index'], candidate['class_name'],
            candidate['status']))
        row = render_case(
            dataset, baseline_source, ours_source, projection_geometry,
            class_names, class_ranges, candidate, args)
        slug = safe_slug(
            '{:02d}_{}_{}_{}'.format(
                candidate['selection_rank'], candidate['class_name'],
                candidate['challenge'], candidate['sample_token'][:8]))
        row.save(case_dir / (slug + '.png'))
        row.save(case_dir / (slug + '.pdf'), 'PDF', resolution=150.0)
        rendered_rows.append(row)

    pdf_path, contact_path, _ = save_pdf_pages(
        rendered_rows, output_dir, args)
    from paper_qualitative_layout import build_main_figure
    main_png_path, main_pdf_path = build_main_figure(
        output_dir, case_ranks=args.main_case_ranks,
        baseline_name=args.baseline_name, ours_name=args.ours_name)

    manifest = {
        'config': str(Path(args.config).expanduser().resolve()),
        'baseline': {
            'name': args.baseline_name,
            'path': str(baseline_path),
            'type': baseline_source.kind,
            'sha256': baseline_hash,
        },
        'ours': {
            'name': args.ours_name,
            'path': str(ours_path),
            'type': ours_source.kind,
            'sha256': ours_hash,
        },
        'protocol': {
            'evaluated_samples': evaluated_samples,
            'score_threshold': args.score_thr,
            'match_distance_m': args.match_distance,
            'far_relative_start': args.far_relative_start,
            'small_projected_size_px': args.small_size_thr,
            'min_localization_gain_m': args.min_error_gain,
            'test_image_transform': projection_geometry,
        },
        'candidate_count': len(candidates),
        'selected_count': len(selected),
        'selected_status_counts': {
            status: sum(row['status'] == status for row in selected)
            for status in sorted(set(row['status'] for row in selected))
        },
        'outputs': {
            'pdf': str(pdf_path),
            'contact_sheet': str(contact_path),
            'main_figure_pdf': str(main_pdf_path),
            'main_figure_png': str(main_png_path),
            'candidate_csv': str(output_dir / 'candidates.csv'),
            'selected_json': str(output_dir / 'selected_cases.json'),
        },
        'arguments': vars(args),
    }
    with open(output_dir / 'manifest.json', 'w') as stream:
        json.dump(json_ready(manifest), stream, indent=2)

    if args.paper_figure_dir:
        figure_dir = Path(args.paper_figure_dir).expanduser().resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, figure_dir / pdf_path.name)
        shutil.copy2(contact_path, figure_dir / contact_path.name)
        shutil.copy2(main_pdf_path, figure_dir / main_pdf_path.name)
        shutil.copy2(main_png_path, figure_dir / main_png_path.name)
        print('[paper] copied final assets to {}'.format(figure_dir))

    print('[done] candidates: {}'.format(output_dir / 'candidates.csv'))
    print('[done] selection:  {}'.format(output_dir / 'selected_cases.json'))
    print('[done] PDF:        {}'.format(pdf_path))
    print('[done] PNG:        {}'.format(contact_path))
    print('[done] Main PDF:   {}'.format(main_pdf_path))
    print('[done] Main PNG:   {}'.format(main_png_path))


if __name__ == '__main__':
    main()
