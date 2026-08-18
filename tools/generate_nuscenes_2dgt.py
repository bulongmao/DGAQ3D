#!/usr/bin/env python
"""Generate offline image-space 2D GT from nuScenes 3D GT boxes.

The output is consumed by LoadOffline2DGT in the 3DPPE training pipeline.
Boxes are saved in original nuScenes image coordinates, before resize/crop/flip.
"""

import argparse
import os
from collections import defaultdict

import mmcv
import numpy as np

try:
    from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
except Exception as exc:  # pragma: no cover
    LiDARInstance3DBoxes = None
    _IMPORT_ERROR = exc


DEFAULT_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project nuScenes 3D GT boxes to per-camera 2D GT boxes.')
    parser.add_argument('ann_file', help='nuScenes mmdet3d info pkl file')
    parser.add_argument('out_file', help='output pkl path')
    parser.add_argument(
        '--data-root', default='',
        help='Optional prefix to strip from image paths in output keys')
    parser.add_argument(
        '--classes', nargs='+', default=DEFAULT_CLASSES,
        help='Class order used by the 3D detector')
    parser.add_argument('--image-width', type=int, default=1600)
    parser.add_argument('--image-height', type=int, default=900)
    parser.add_argument('--min-depth', type=float, default=1e-5)
    parser.add_argument('--min-box-size', type=float, default=2.0)
    parser.add_argument('--min-visible-corners', type=int, default=4)
    parser.add_argument(
        '--use-valid-flag', action='store_true', default=True,
        help='Filter boxes by valid_flag when the info file contains it')
    parser.add_argument(
        '--no-valid-flag', dest='use_valid_flag', action='store_false')
    parser.add_argument(
        '--min-lidar-pts', type=int, default=0,
        help='Fallback lidar-point filter when valid_flag is unavailable')
    parser.add_argument(
        '--max-samples', type=int, default=-1,
        help='Debug option: only process the first N samples')
    parser.add_argument(
        '--dump-coco', default='',
        help='Optional COCO-format json output for visualization/training 2D detector')
    return parser.parse_args()


def normalize_key(path, data_root=''):
    path = str(path).replace('\\', '/')
    data_root = str(data_root).replace('\\', '/').rstrip('/')
    if data_root and path.startswith(data_root + '/'):
        path = path[len(data_root) + 1:]
    for marker in ('samples/', 'sweeps/'):
        if marker in path:
            return path[path.index(marker):]
    return path


def lidar2img_from_cam(cam_info):
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def build_box_mask(info, use_valid_flag=True, min_lidar_pts=0):
    num = len(info.get('gt_names', []))
    mask = np.ones(num, dtype=bool)
    if use_valid_flag and 'valid_flag' in info:
        mask &= np.asarray(info['valid_flag'], dtype=bool)
    elif min_lidar_pts > 0 and 'num_lidar_pts' in info:
        mask &= np.asarray(info['num_lidar_pts']) >= min_lidar_pts
    return mask


def project_sample(info, class_to_idx, args):
    if LiDARInstance3DBoxes is None:
        raise ImportError(
            'mmdet3d LiDARInstance3DBoxes is required. Original error: '
            f'{_IMPORT_ERROR}')

    gt_boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
    gt_names = np.asarray(info.get('gt_names', []))
    if gt_boxes.size == 0 or len(gt_names) == 0:
        return {}

    mask = build_box_mask(info, args.use_valid_flag, args.min_lidar_pts)
    class_mask = np.asarray([name in class_to_idx for name in gt_names], dtype=bool)
    mask &= class_mask
    if mask.sum() == 0:
        return {}

    gt_boxes = gt_boxes[mask]
    gt_names = gt_names[mask]
    gt_labels = np.asarray([class_to_idx[name] for name in gt_names], dtype=np.int64)

    boxes3d = LiDARInstance3DBoxes(
        gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
    corners = boxes3d.corners.numpy().astype(np.float32)
    centers = boxes3d.gravity_center.numpy().astype(np.float32)
    corners_h = np.concatenate(
        [corners, np.ones((*corners.shape[:2], 1), dtype=np.float32)], axis=-1)
    centers_h = np.concatenate(
        [centers, np.ones((centers.shape[0], 1), dtype=np.float32)], axis=-1)

    sample_results = {}
    for _, cam_info in info['cams'].items():
        lidar2img = lidar2img_from_cam(cam_info)
        proj = corners_h @ lidar2img.T
        center_proj = centers_h @ lidar2img.T
        center_depth = center_proj[:, 2]
        boxes2d, labels2d, depths2d = [], [], []
        for obj_id in range(corners.shape[0]):
            depth = proj[obj_id, :, 2]
            valid = depth > args.min_depth
            if int(valid.sum()) < args.min_visible_corners:
                continue
            pts = proj[obj_id, valid]
            u = pts[:, 0] / np.maximum(pts[:, 2], args.min_depth)
            v = pts[:, 1] / np.maximum(pts[:, 2], args.min_depth)
            x1 = np.clip(u.min(), 0, args.image_width - 1)
            y1 = np.clip(v.min(), 0, args.image_height - 1)
            x2 = np.clip(u.max(), 0, args.image_width - 1)
            y2 = np.clip(v.max(), 0, args.image_height - 1)
            if x2 - x1 < args.min_box_size or y2 - y1 < args.min_box_size:
                continue
            if not np.isfinite(center_depth[obj_id]) or center_depth[obj_id] <= args.min_depth:
                continue
            boxes2d.append([x1, y1, x2, y2])
            labels2d.append(gt_labels[obj_id])
            depths2d.append(center_depth[obj_id])

        key = normalize_key(cam_info['data_path'], args.data_root)
        sample_results[key] = dict(
            boxes=np.asarray(boxes2d, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels2d, dtype=np.int64),
            depths=np.asarray(depths2d, dtype=np.float32))
    return sample_results


def dump_coco(priors, out_file, classes, image_width, image_height):
    images, annotations = [], []
    ann_id = 1
    for image_id, key in enumerate(sorted(priors.keys()), start=1):
        entry = priors[key]
        images.append(dict(
            id=image_id, file_name=key, width=image_width, height=image_height))
        boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(entry['labels'], dtype=np.int64).reshape(-1)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            annotations.append(dict(
                id=ann_id,
                image_id=image_id,
                category_id=int(label) + 1,
                bbox=[x1, y1, w, h],
                area=float(max(w, 0.0) * max(h, 0.0)),
                iscrowd=0))
            ann_id += 1
    categories = [dict(id=i + 1, name=name) for i, name in enumerate(classes)]
    mmcv.dump(dict(images=images, annotations=annotations, categories=categories), out_file)


def main():
    args = parse_args()
    class_to_idx = {name: idx for idx, name in enumerate(args.classes)}
    data = mmcv.load(args.ann_file)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    if args.max_samples > 0:
        infos = infos[:args.max_samples]

    priors = {}
    stats = defaultdict(int)
    prog_bar = mmcv.ProgressBar(len(infos))
    for info in infos:
        sample_priors = project_sample(info, class_to_idx, args)
        for key, entry in sample_priors.items():
            priors[key] = entry
            stats['images'] += 1
            stats['boxes'] += int(len(entry['boxes']))
            if len(entry['boxes']) == 0:
                stats['empty_images'] += 1
        stats['samples'] += 1
        prog_bar.update()

    out = dict(
        meta=dict(
            type='nuscenes_projected_2dgt',
            ann_file=args.ann_file,
            data_root=args.data_root,
            classes=list(args.classes),
            image_width=args.image_width,
            image_height=args.image_height,
            min_depth=args.min_depth,
            min_box_size=args.min_box_size,
            min_visible_corners=args.min_visible_corners,
            use_valid_flag=args.use_valid_flag,
            min_lidar_pts=args.min_lidar_pts),
        annotations=priors)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    mmcv.dump(out, args.out_file)
    if args.dump_coco:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_coco)), exist_ok=True)
        dump_coco(priors, args.dump_coco, args.classes,
                  args.image_width, args.image_height)
    print('\n[summary]')
    print(f"  samples: {stats['samples']}")
    print(f"  images: {stats['images']}")
    print(f"  boxes: {stats['boxes']}")
    print(f"  empty_images: {stats['empty_images']}")
    print(f"  out_file: {args.out_file}")
    if args.dump_coco:
        print(f"  coco_file: {args.dump_coco}")


if __name__ == '__main__':
    main()
