#!/usr/bin/env python
"""Pure NumPy helpers for range- and projected-size detection analysis.

The integration script lives in ``tools/analyze_range_size_breakdown.py``.
Keeping the metric and projection primitives independent from MMDetection3D
makes their behavior testable without loading the detector stack.
"""

import math

import numpy as np


# Corner ordering used by MMDetection3D's BaseInstance3DBoxes.
BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def validate_edges(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('Bin edges must be a one-dimensional sequence of length >= 2.')
    if np.any(np.isnan(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError('Bin edges must be strictly increasing and cannot contain NaN.')
    return edges


def assign_bin(values, edges):
    """Assign values to half-open bins, with the final right edge included.

    Values outside the supplied range and NaNs receive ``-1``. The function
    accepts either a scalar or an array and returns the matching shape.
    """
    edges = validate_edges(edges)
    values_arr = np.asarray(values, dtype=np.float64)
    flat = values_arr.reshape(-1)
    out = np.full(flat.shape, -1, dtype=np.int16)
    valid = np.isfinite(flat) & (flat >= edges[0]) & (flat <= edges[-1])
    if np.isinf(edges[-1]):
        valid = (~np.isnan(flat)) & (flat >= edges[0]) & (flat <= edges[-1])
    if np.any(valid):
        ids = np.searchsorted(edges, flat[valid], side='right') - 1
        ids[ids == edges.size - 1] = edges.size - 2
        out[valid] = ids.astype(np.int16)
    out = out.reshape(values_arr.shape)
    return int(out) if values_arr.ndim == 0 else out


def bin_labels(edges, suffix='', precision=2):
    edges = validate_edges(edges)

    def fmt(value):
        if np.isposinf(value):
            return 'inf'
        if float(value).is_integer():
            return str(int(value))
        return ('{:.' + str(precision) + 'f}').format(value).rstrip('0').rstrip('.')

    labels = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        closing = ']' if index == edges.size - 2 else ')'
        labels.append('[{}, {}{}{}'.format(fmt(left), fmt(right), closing, suffix))
    return labels


def quaternion_to_matrix(quaternion):
    """Convert a nuScenes ``[w, x, y, z]`` quaternion to a rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError('Quaternion has zero norm.')
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lidar_centers_to_ego(centers, lidar2ego_rotation, lidar2ego_translation):
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    rotation = quaternion_to_matrix(lidar2ego_rotation)
    translation = np.asarray(lidar2ego_translation, dtype=np.float64).reshape(1, 3)
    return centers @ rotation.T + translation


def build_test_ida(data_aug_conf, transform_type='ResizeCropFlipImageV2'):
    """Reproduce the deterministic validation image transform.

    Returns a 4x4 homogeneous image-domain transform plus useful geometry
    metadata. Test-time flip and rotation are zero in both repository pipeline
    implementations.
    """
    conf = dict(data_aug_conf)
    height = int(conf['H'])
    width = int(conf['W'])
    final_h, final_w = [int(v) for v in conf['final_dim']]

    if transform_type == 'ResizeCropFlipImageV2':
        resize = float(final_w) / float(width)
        resize += float(conf.get('resize_test', 0.0))
        crop_key = 'crop_h'
    elif transform_type == 'ResizeCropFlipImage':
        resize = max(float(final_h) / float(height),
                     float(final_w) / float(width))
        crop_key = 'bot_pct_lim'
    else:
        raise ValueError('Unsupported image transform: {}'.format(transform_type))

    resized_w = int(width * resize)
    resized_h = int(height * resize)
    crop_values = conf.get(crop_key, (0.0, 0.0))
    crop_h = int((1.0 - float(np.mean(crop_values))) * resized_h) - final_h
    crop_w = int(max(0, resized_w - final_w) / 2)

    ida = np.eye(4, dtype=np.float64)
    ida[0, 0] = resize
    ida[1, 1] = resize
    ida[0, 2] = -crop_w
    ida[1, 2] = -crop_h
    return {
        'matrix': ida,
        'resize': resize,
        'resize_dims': (resized_w, resized_h),
        'crop': (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
        'final_dim': (final_h, final_w),
        'transform_type': transform_type,
    }


def apply_image_transform(lidar2img, ida_matrix):
    lidar2img = np.asarray(lidar2img, dtype=np.float64)
    ida_matrix = np.asarray(ida_matrix, dtype=np.float64).reshape(4, 4)
    if lidar2img.ndim == 2:
        return ida_matrix @ lidar2img
    return np.stack([ida_matrix @ matrix for matrix in lidar2img], axis=0)


def project_box_to_image(corners, lidar2img, image_shape, min_depth=0.1):
    """Project a 3D box and return its clipped 2D rectangle or ``None``.

    Box edges crossing the camera plane are clipped at ``min_depth``. This is
    more robust than dropping all behind-camera corners for nearby objects.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    matrix = np.asarray(lidar2img, dtype=np.float64).reshape(4, 4)
    image_h, image_w = [float(v) for v in image_shape]

    homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ matrix.T
    depth = projected[:, 2]
    points = [projected[i] for i in range(8) if depth[i] >= min_depth]

    for first, second in BOX_EDGES:
        z_first, z_second = depth[first], depth[second]
        if (z_first >= min_depth) == (z_second >= min_depth):
            continue
        denominator = z_second - z_first
        if abs(denominator) < 1e-12:
            continue
        ratio = (min_depth - z_first) / denominator
        points.append(projected[first] + ratio * (projected[second] - projected[first]))

    if not points:
        return None
    points = np.asarray(points, dtype=np.float64)
    uv = points[:, :2] / points[:, 2:3]
    uv = uv[np.isfinite(uv).all(axis=1)]
    if uv.shape[0] == 0:
        return None

    left = max(0.0, float(np.min(uv[:, 0])))
    top = max(0.0, float(np.min(uv[:, 1])))
    right = min(image_w, float(np.max(uv[:, 0])))
    bottom = min(image_h, float(np.max(uv[:, 1])))
    if right <= left or bottom <= top:
        return None
    return np.array([left, top, right, bottom], dtype=np.float64)


def max_projected_box_size(corners, lidar2img_matrices, image_shape,
                           min_depth=0.1):
    """Return max-camera ``sqrt(clipped width * clipped height)`` in pixels."""
    best = 0.0
    visible_cameras = 0
    for matrix in np.asarray(lidar2img_matrices):
        rectangle = project_box_to_image(
            corners, matrix, image_shape, min_depth=min_depth)
        if rectangle is None:
            continue
        visible_cameras += 1
        width = rectangle[2] - rectangle[0]
        height = rectangle[3] - rectangle[1]
        best = max(best, math.sqrt(max(width * height, 0.0)))
    return best, visible_cameras


def max_projected_box_sizes(corners, lidar2img_matrices, image_shape,
                            min_depth=0.1, chunk_size=4096):
    """Project many boxes efficiently, with exact near-plane fallback."""
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 8, 3)
    matrices = np.asarray(lidar2img_matrices, dtype=np.float64).reshape(-1, 4, 4)
    image_h, image_w = [float(value) for value in image_shape]
    sizes = np.zeros(corners.shape[0], dtype=np.float32)
    visible_counts = np.zeros(corners.shape[0], dtype=np.int16)

    for start in range(0, corners.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), corners.shape[0])
        current = corners[start:end]
        homogeneous = np.concatenate([
            current,
            np.ones((current.shape[0], 8, 1), dtype=np.float64),
        ], axis=2)
        projected = np.einsum('cij,nkj->ncki', matrices, homogeneous)
        depth = projected[..., 2]
        front = depth >= min_depth
        safe_depth = np.where(front, depth, 1.0)
        uv = projected[..., :2] / safe_depth[..., None]
        valid = front & np.isfinite(uv).all(axis=-1)

        minimum = np.min(np.where(valid[..., None], uv, np.inf), axis=2)
        maximum = np.max(np.where(valid[..., None], uv, -np.inf), axis=2)
        left = np.maximum(minimum[..., 0], 0.0)
        top = np.maximum(minimum[..., 1], 0.0)
        right = np.minimum(maximum[..., 0], image_w)
        bottom = np.minimum(maximum[..., 1], image_h)
        camera_area = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
        camera_valid = np.any(valid, axis=2) & np.isfinite(camera_area)
        camera_valid &= camera_area > 0
        camera_area = np.where(camera_valid, camera_area, 0.0)
        sizes[start:end] = np.sqrt(np.max(camera_area, axis=1)).astype(np.float32)
        visible_counts[start:end] = np.sum(camera_valid, axis=1).astype(np.int16)

        # Corner-only projection truncates boxes crossing a camera plane. Such
        # boxes are uncommon, so apply exact edge clipping only to this subset.
        crossing = np.any(np.any(front, axis=2) & np.any(~front, axis=2), axis=1)
        for local_index in np.flatnonzero(crossing):
            exact_size, exact_count = max_projected_box_size(
                current[local_index], matrices, image_shape,
                min_depth=min_depth)
            sizes[start + local_index] = exact_size
            visible_counts[start + local_index] = exact_count
    return sizes, visible_counts


def match_predictions(gt_centers, pred_centers, pred_scores, distance_threshold):
    """Perform nuScenes-style score-ordered nearest-center matching.

    Inputs contain one class and one sample. Returned indices are in descending
    prediction-score order; ``matched_gt`` contains ``-1`` for false positives.
    """
    gt_centers = np.asarray(gt_centers, dtype=np.float64).reshape(-1, 2)
    pred_centers = np.asarray(pred_centers, dtype=np.float64).reshape(-1, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    if pred_centers.shape[0] != pred_scores.shape[0]:
        raise ValueError('Prediction centers and scores have different lengths.')

    indices = np.arange(pred_scores.size, dtype=np.int64)
    # Official devkit sorts (score, index) ascending and reverses the result.
    order = np.lexsort((-indices, -pred_scores))
    matched_gt = np.full(order.shape, -1, dtype=np.int64)
    taken = np.zeros(gt_centers.shape[0], dtype=bool)
    for rank, pred_index in enumerate(order):
        available = np.flatnonzero(~taken)
        if available.size == 0:
            break
        distances = np.linalg.norm(
            gt_centers[available] - pred_centers[pred_index], axis=1)
        local_index = int(np.argmin(distances))
        if float(distances[local_index]) < float(distance_threshold):
            gt_index = int(available[local_index])
            matched_gt[rank] = gt_index
            taken[gt_index] = True
    return order, matched_gt


def nuscenes_ap_from_tp(tp_sorted, num_gt, min_recall=0.1,
                        min_precision=0.1):
    """Compute AP using the official nuScenes 101-point interpolation."""
    tp_sorted = np.asarray(tp_sorted, dtype=np.float64).reshape(-1)
    num_gt = float(num_gt)
    if num_gt <= 0 or tp_sorted.size == 0 or np.sum(tp_sorted) <= 0:
        return 0.0
    fp_sorted = 1.0 - tp_sorted
    tp_cumulative = np.cumsum(tp_sorted)
    fp_cumulative = np.cumsum(fp_sorted)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    recall = tp_cumulative / num_gt
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_grid = np.interp(recall_grid, recall, precision, right=0.0)
    first_index = round(100 * float(min_recall)) + 1
    clipped = precision_grid[first_index:] - float(min_precision)
    clipped[clipped < 0] = 0
    return float(np.mean(clipped)) / (1.0 - float(min_precision))


def recall_from_tp(tp_sorted, num_gt):
    num_gt = float(num_gt)
    if num_gt <= 0:
        return 0.0
    return float(np.sum(np.asarray(tp_sorted, dtype=np.float64))) / num_gt


def repeat_bootstrap_events(tp_sorted, event_units_sorted, gt_units,
                            unit_counts):
    """Expand pre-matched events for an exact paired cluster bootstrap draw."""
    tp_sorted = np.asarray(tp_sorted, dtype=np.int8).reshape(-1)
    event_units_sorted = np.asarray(event_units_sorted, dtype=np.int64).reshape(-1)
    gt_units = np.asarray(gt_units, dtype=np.int64).reshape(-1)
    unit_counts = np.asarray(unit_counts, dtype=np.int64).reshape(-1)
    if tp_sorted.shape[0] != event_units_sorted.shape[0]:
        raise ValueError('Event TP flags and unit ids have different lengths.')
    event_repeats = unit_counts[event_units_sorted]
    sampled_tp = np.repeat(tp_sorted, event_repeats)
    sampled_num_gt = int(np.sum(unit_counts[gt_units])) if gt_units.size else 0
    return sampled_tp, sampled_num_gt
