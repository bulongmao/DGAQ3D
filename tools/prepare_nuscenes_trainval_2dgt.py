#!/usr/bin/env python
"""Audit and merge projected nuScenes 2D GT for train+val training."""

import argparse
import os

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-info', required=True)
    parser.add_argument('--val-info', required=True)
    parser.add_argument('--train-2dgt', required=True)
    parser.add_argument('--val-2dgt', required=True)
    parser.add_argument('--out-file', required=True)
    parser.add_argument(
        '--audit-only', action='store_true',
        help='Audit the input files without writing the merged asset.')
    return parser.parse_args()


def annotations_from(data, path):
    annotations = data.get('annotations', data) if isinstance(data, dict) else data
    if not isinstance(annotations, dict):
        raise TypeError('{} does not contain a 2D-GT annotation dict'.format(path))
    return annotations


def info_list(path):
    data = mmcv.load(path)
    infos = data.get('infos', data) if isinstance(data, dict) else data
    if not isinstance(infos, list):
        raise TypeError('{} does not contain an info list'.format(path))
    return infos


def candidate_keys(filename):
    filename = str(filename).replace('\\', '/')
    keys = [filename]
    if filename.startswith('./'):
        keys.append(filename[2:])
    for marker in ('samples/', 'sweeps/'):
        if marker in filename:
            rel = filename[filename.index(marker):]
            keys.extend((rel, './data/nuscenes/' + rel, 'data/nuscenes/' + rel))
    keys.append(os.path.basename(filename))
    return keys


def audit_split(name, infos, annotations):
    image_count = 0
    matched = 0
    empty = 0
    boxes = 0
    missing = []
    for info in infos:
        for cam in info.get('cams', {}).values():
            image_count += 1
            ann = None
            for key in candidate_keys(cam['data_path']):
                if key in annotations:
                    ann = annotations[key]
                    break
            if ann is None:
                if len(missing) < 10:
                    missing.append(cam['data_path'])
                continue
            matched += 1
            num_boxes = len(np.asarray(ann.get('boxes', [])).reshape(-1, 4))
            boxes += num_boxes
            empty += int(num_boxes == 0)
    print('[{}] samples={} images={} matched={} missing={} boxes={} empty={}'.format(
        name, len(infos), image_count, matched, image_count - matched, boxes, empty))
    for path in missing:
        print('  missing: {}'.format(path))
    if matched != image_count:
        raise RuntimeError('{} 2D GT coverage is incomplete: {}/{}'.format(
            name, matched, image_count))


def merge_annotations(train_annotations, val_annotations):
    merged = dict(train_annotations)
    duplicate_count = 0
    for key, value in val_annotations.items():
        if key in merged:
            duplicate_count += 1
            lhs = merged[key]
            for field in ('boxes', 'labels', 'depths'):
                if not np.array_equal(
                        np.asarray(lhs.get(field, [])),
                        np.asarray(value.get(field, []))):
                    raise RuntimeError(
                        'Conflicting duplicate key in train/val 2D GT: {}'.format(key))
        else:
            merged[key] = value
    return merged, duplicate_count


def main():
    args = parse_args()
    train_infos = info_list(args.train_info)
    val_infos = info_list(args.val_info)
    train_data = mmcv.load(args.train_2dgt)
    val_data = mmcv.load(args.val_2dgt)
    train_annotations = annotations_from(train_data, args.train_2dgt)
    val_annotations = annotations_from(val_data, args.val_2dgt)

    audit_split('train', train_infos, train_annotations)
    audit_split('val', val_infos, val_annotations)
    merged, duplicate_count = merge_annotations(
        train_annotations, val_annotations)
    print('[merge] train_keys={} val_keys={} duplicates={} merged_keys={}'.format(
        len(train_annotations), len(val_annotations), duplicate_count, len(merged)))

    if args.audit_only:
        print('[done] audit only')
        return

    out = dict(
        meta=dict(
            type='nuscenes_projected_2dgt_trainval',
            train_info=os.path.abspath(args.train_info),
            val_info=os.path.abspath(args.val_info),
            train_2dgt=os.path.abspath(args.train_2dgt),
            val_2dgt=os.path.abspath(args.val_2dgt)),
        annotations=merged)
    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    os.makedirs(out_dir, exist_ok=True)
    mmcv.dump(out, args.out_file)
    print('[done] {}'.format(args.out_file))


if __name__ == '__main__':
    main()
