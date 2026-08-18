# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import cv2
import os
import numpy as np
from mmdet.datasets.builder import PIPELINES
from einops import rearrange

# Compatibility for older OpenMMLab code on newer NumPy.
if 'bool' not in np.__dict__:
    np.bool = np.bool_
if 'long' not in np.__dict__:
    np.long = np.int_

@PIPELINES.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results

@PIPELINES.register_module()
class LoadOffline2DGT(object):
    """Load offline 2D boxes/depths keyed by nuScenes image path.

    The asset is expected to contain per-image ``boxes``, ``labels`` and
    ``depths``. The depth value is the projected 3D box center depth in the
    camera frame, not dense/surface depth.
    """

    def __init__(self, gt2d_path, min_box_size=1.0):
        self.gt2d_path = gt2d_path
        self.min_box_size = min_box_size
        data = mmcv.load(gt2d_path)
        if isinstance(data, dict) and 'annotations' in data:
            data = data['annotations']
        self.annotations = data

    def _candidate_keys(self, filename):
        keys = [filename]
        if filename.startswith('./'):
            keys.append(filename[2:])
        marker = 'samples/'
        if marker in filename:
            rel = filename[filename.index(marker):]
            keys.extend([rel, './data/nuscenes/' + rel, 'data/nuscenes/' + rel])
        marker = 'sweeps/'
        if marker in filename:
            rel = filename[filename.index(marker):]
            keys.extend([rel, './data/nuscenes/' + rel, 'data/nuscenes/' + rel])
        keys.append(os.path.basename(filename))
        return keys

    def _load_one(self, filename):
        ann = None
        for key in self._candidate_keys(filename):
            if key in self.annotations:
                ann = self.annotations[key]
                break
        if ann is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )

        boxes = np.asarray(ann.get('boxes', []), dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(ann.get('labels', []), dtype=np.int64).reshape(-1)
        depths = np.asarray(ann.get('depths', []), dtype=np.float32).reshape(-1)
        num = min(len(boxes), len(labels), len(depths))
        boxes, labels, depths = boxes[:num], labels[:num], depths[:num]

        if num > 0:
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            keep = (
                np.isfinite(boxes).all(axis=1)
                & np.isfinite(depths)
                & (depths > 0)
                & (wh[:, 0] >= self.min_box_size)
                & (wh[:, 1] >= self.min_box_size)
            )
            boxes, labels, depths = boxes[keep], labels[keep], depths[keep]
        return boxes, labels, depths

    def __call__(self, results):
        boxes_list, labels_list, depths_list = [], [], []
        for filename in results['filename']:
            boxes, labels, depths = self._load_one(filename)
            boxes_list.append(boxes)
            labels_list.append(labels)
            depths_list.append(depths)
        results['gt2d_boxes'] = boxes_list
        results['gt2d_labels'] = labels_list
        results['gt2d_depths'] = depths_list
        return results




@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']   # List[(H, W, 3), (H, W, 3), ...]
        img_timestamp = results['img_timestamp']    # List[float, float, ...]
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]    # List[float, float, ...]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.sweeps_num <= 0:
            results['img'] = sweep_imgs_list
            results['timestamp'] = timestamp_imgs_list
            return results
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
                    results['ori_intrinsics'].append(np.copy(results['ori_intrinsics'][j]))
                    results['ori_lidar2img'].append(np.copy(results['ori_lidar2img'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    # 在sweep_range中选择对应的sweep
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                # (H, W, 3, Num_views)
                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]   # List[(H, W, 3), (H, W, 3), ...]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6 for sensor in self.sensors]      # List[float, float, ...]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
                    results['ori_intrinsics'].append(sweep[sensor]['intrinsics'].copy())
                    results['ori_lidar2img'].append(sweep[sensor]['lidar2img'].copy())
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadDepthByMapplingPoints2Images(object):
    def __init__(self, src_size, input_size, downsample=1, min_dist=1e-5, max_dist=None):
        self.src_size = src_size
        self.input_size = input_size
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def mask_points_by_range(self, points_2d, depths, img_size):
        """
        Args:
            points2d: (N, 2)
            depths:   (N, )
            img_size: (H, W)
        Returns:
            points2d: (N', 2)
            depths:   (N', )
        """
        H, W = img_size
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, points_2d[:, 0] >= 0)
        mask = np.logical_and(mask, points_2d[:, 0] < W)
        mask = np.logical_and(mask, points_2d[:, 1] >= 0)
        mask = np.logical_and(mask, points_2d[:, 1] < H)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def mask_points_by_dist(self, points_2d, depths, min_dist, max_dist):
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, depths >= min_dist)
        if max_dist is not None:
            mask = np.logical_and(mask, depths <= max_dist)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def transform_points2d(self, points_2d, depths, resize, crop, flip, rotate):
        points_2d = points_2d * resize
        points_2d = points_2d - crop[:2]  # (N_points, 2)
        points_2d, depths = self.mask_points_by_range(points_2d, depths, (crop[3] - crop[1], crop[2] - crop[0]))

        if flip:
            # A = np.array([[-1, 0], [0, 1]])
            # b = np.array([crop[2] - crop[0] - 1, 0])
            # points_2d = points_2d.dot(A.T) + b
            points_2d[:, 0] = (crop[2] - crop[0]) - 1 - points_2d[:, 0]

        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.dot(-b) + b

        points_2d = points_2d.dot(A.T) + b

        points_2d, depths = self.mask_points_by_range(points_2d, depths, self.input_size)

        return points_2d, depths

    def __call__(self, results):
        imgs = results["img"]  # List[(H, W, 3), (H, W, 3), ...]
        # extrinsics = results["extrinsics"]      # List[(4, 4), (4, 4), ...]
        # ori_intrinsics = results["ori_intrinsics"]      # List[(4, 4), (4, 4), ...]
        ori_lidar2imgs = results["ori_lidar2img"]       # List[(4, 4), (4, 4), ...]

        assert len(imgs) == len(ori_lidar2imgs), \
            f'imgs length {len(imgs)} != ori_lidar2imgs length {len(ori_lidar2imgs)}'

        resize = results['multi_view_resize']      # float
        resize_dims = results['multi_view_resize_dims']     # (2, )   (resize_W, resize_H)
        crop = results['multi_view_crop']    # (4, )     (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = results['multi_view_flip']          # bool
        rotate = results['multi_view_rotate']      # float

        # augmentation (resize, crop, horizontal flip, rotate)
        # resize: float, resize的比例
        # resize_dims: Tuple(W, H), resize后的图像尺寸
        # crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # flip: bool
        # rotate: float 旋转角度

        N_views = len(imgs)
        H, W = self.input_size
        dH, dW = H // self.downsample, W // self.downsample
        depth_map_list = []
        depth_map_mask_list = []

        points_lidar = results['points'].tensor[:, :3].numpy()     # (N_points, 3)
        points_lidar = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1))], axis=-1)   # (N_points, 4)

        for idx in range(N_views):
            # # lidar --> camera
            # lidar2camera = extrinsics[idx]  # (4, 4)
            # points_camera = points_lidar.dot(lidar2camera.T)   # (N_points, 4)     4: (x, y, z, 1)
            #
            # # camera --> img
            # ori_intrins = ori_intrinsics[idx]   # (4, 4)
            # points_image = points_camera.dot(ori_intrins.T)    # (N_points, 4)     4: (du, dv, d, 1)

            # lidar --> img
            points_image = points_lidar.dot(ori_lidar2imgs[idx].T)
            points_image = points_image[:, :3]  # (N_points, 3)     3: (du, dv, d)
            points_2d = points_image[:, :2] / points_image[:, 2:3]  # (N_points, 2)     2: (u, v)
            depths = points_image[:, 2]  # (N_points, )
            points_2d, depths = self.mask_points_by_range(points_2d, depths, self.src_size)

            # aug
            points_2d, depths = self.transform_points2d(points_2d, depths, resize, crop, flip, rotate)
            points_2d, depths = self.mask_points_by_dist(points_2d, depths, self.min_dist, self.max_dist)

            # downsample
            points_2d = np.round(points_2d / self.downsample)
            points_2d, depths = self.mask_points_by_range(points_2d, depths, (dH, dW))

            depth_map = np.zeros(shape=(dH, dW), dtype=np.float32)  # (dH, dW)
            depth_map_mask = np.zeros(shape=(dH, dW), dtype=np.bool)   # (dH, dW)

            ranks = points_2d[:, 0] + points_2d[:, 1] * dW
            sort = (ranks + depths / 1000.).argsort()
            points_2d, depths, ranks = points_2d[sort], depths[sort], ranks[sort]

            kept = np.ones(points_2d.shape[0], dtype=np.bool)
            kept[1:] = (ranks[1:] != ranks[:-1])
            points_2d, depths = points_2d[kept], depths[kept]
            points_2d = points_2d.astype(np.long)

            depth_map[points_2d[:, 1], points_2d[:, 0]] = depths
            depth_map_mask[points_2d[:, 1], points_2d[:, 0]] = 1
            depth_map_list.append(depth_map)
            depth_map_mask_list.append(depth_map_mask)

        depth_map = np.stack(depth_map_list, axis=0)      # (N_view, dH, dW)
        depth_map_mask = np.stack(depth_map_mask_list, axis=0)    # (N_view, dH, dW)


        # for vis
        # import cv2
        # for idx in range(len(imgs)):
        #     ori_img = imgs[idx]  # (H, W, 3)
        #     ori_img = ori_img.astype(np.uint8)
        #     cv2.imshow("ori_img", ori_img)
        #
        #     curr_img = cv2.resize(src=ori_img,
        #                           dsize=(ori_img.shape[1]//self.downsample, ori_img.shape[0]//self.downsample))
        #     cv2.imshow("curr_img", curr_img)
        #
        #     cv2.imshow("mask", depth_map_mask[idx].astype(np.uint8) * 255)
        #     cur_depth_map = depth_map[idx]
        #     cur_depth_map_mask = depth_map_mask[idx]
        #
        #     cur_depth_map = cur_depth_map / 60 * 255
        #     cur_depth_map = cur_depth_map.astype(np.uint8)
        #     cur_depth_map = cv2.applyColorMap(cur_depth_map, cv2.COLORMAP_RAINBOW)
        #
        #     # cur_depth_map = cv2.resize(src=cur_depth_map, dsize=(img.shape[1], img.shape[0]))
        #
        #     curr_img[cur_depth_map_mask] = cur_depth_map[cur_depth_map_mask]
        #     cv2.imshow("depth map", curr_img)
        #
        #     while(True):
        #         k = cv2.waitKey(0)
        #         if k == 27:
        #             cv2.destroyAllWindows()
        #             break

        results['depth_map'] = depth_map
        results['depth_map_mask'] = depth_map_mask

        results.pop('multi_view_resize')
        results.pop('multi_view_resize_dims')
        results.pop('multi_view_crop')
        results.pop('multi_view_flip')
        results.pop('multi_view_rotate')
        return results

@PIPELINES.register_module()
class LoadDenseDepthFromFiles(object):
    """Load precomputed dense depth maps as depth supervision targets.

    This transform is intentionally parallel to
    ``LoadDepthByMapplingPoints2Images``: it keeps the same output keys
    (``depth_map`` and ``depth_map_mask``), applies the same image
    augmentation metadata, and downsamples to the detector depth-head
    feature stride.

    Expected dense depth asset path per image:
        ``<depth_root>/samples/<CAM>/<image_stem>.npz``

    The npz file is expected to contain a ``depth`` array. Invalid pixels can
    be encoded as ``-1``, ``0``, NaN, or Inf.
    """

    def __init__(self, depth_root, src_size, input_size, downsample=1,
                 min_dist=1e-5, max_dist=None):
        self.depth_root = depth_root
        self.src_size = src_size
        self.input_size = input_size
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def _load_depth_and_mask(self, img_filename):
        cam = img_filename.split('/')[-2]
        base = img_filename.split('/')[-1].replace('.jpg', '.npz')
        depth_path = f'{self.depth_root}/samples/{cam}/{base}'

        try:
            depth = np.load(depth_path)['depth'].astype(np.float32)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f'LoadDenseDepthFromFiles: missing dense depth file '
                f'{depth_path}') from exc
        except KeyError as exc:
            raise KeyError(
                f'LoadDenseDepthFromFiles: {depth_path} does not contain '
                f'a "depth" array') from exc

        mask = np.isfinite(depth) & (depth > 0)
        depth = np.where(mask, depth, 0).astype(np.float32)
        return depth, mask

    def _apply_aug(self, depth, mask, resize, resize_dims, crop, flip, rotate):
        rW, rH = resize_dims
        depth = mmcv.imresize(
            depth, (rW, rH), interpolation='bilinear').astype(np.float32)
        mask = mmcv.imresize(
            mask.astype(np.uint8), (rW, rH),
            interpolation='nearest').astype(np.bool_)

        x1, y1, x2, y2 = crop
        depth = depth[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        if flip:
            depth = np.ascontiguousarray(np.fliplr(depth))
            mask = np.ascontiguousarray(np.fliplr(mask))

        if abs(rotate) > 1e-6:
            H, W = depth.shape[:2]
            center = (W / 2.0, H / 2.0)
            M = cv2.getRotationMatrix2D(center, rotate, 1.0)
            depth = cv2.warpAffine(
                depth, M, (W, H), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(
                mask.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0).astype(np.bool_)

        return depth, mask

    def __call__(self, results):
        imgs = results['img']
        filenames = results['filename']
        assert len(imgs) == len(filenames), \
            f'imgs length {len(imgs)} != filenames length {len(filenames)}'

        resize = results['multi_view_resize']
        resize_dims = results['multi_view_resize_dims']
        crop = results['multi_view_crop']
        flip = results['multi_view_flip']
        rotate = results['multi_view_rotate']

        H, W = self.input_size
        dH, dW = H // self.downsample, W // self.downsample
        depth_map_list = []
        depth_map_mask_list = []

        for img_filename in filenames:
            depth, mask = self._load_depth_and_mask(img_filename)
            depth, mask = self._apply_aug(
                depth, mask, resize, resize_dims, crop, flip, rotate)

            if self.min_dist is not None:
                mask = mask & (depth >= self.min_dist)
            if self.max_dist is not None:
                mask = mask & (depth <= self.max_dist)

            if self.downsample > 1:
                depth = cv2.resize(depth, (dW, dH),
                                   interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask.astype(np.uint8), (dW, dH),
                                  interpolation=cv2.INTER_NEAREST).astype(
                                      np.bool_)

            depth[~mask] = 0
            depth_map_list.append(depth.astype(np.float32))
            depth_map_mask_list.append(mask)

        results['depth_map'] = np.stack(depth_map_list, axis=0)
        results['depth_map_mask'] = np.stack(depth_map_mask_list, axis=0)

        results.pop('multi_view_resize')
        results.pop('multi_view_resize_dims')
        results.pop('multi_view_crop')
        results.pop('multi_view_flip')
        results.pop('multi_view_rotate')
        return results
