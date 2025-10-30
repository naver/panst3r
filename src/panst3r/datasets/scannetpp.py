# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import json
import os.path as osp
import numpy as np
import cv2

from dust3r.utils.image import imread_cv2
from dust3r.datasets import ScanNetpp as DUSt3R_ScanNetpp
from must3r.datasets.base.tuple_maker import select_tuple_from_pairs
from panst3r.datasets.base import PanSt3RBaseDataset
from panst3r.datasets.utils import rgb2id

CLS_SEP = 256

class ScanNetppPanoptic(DUSt3R_ScanNetpp, PanSt3RBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, split='train', **kwargs)

        self.is_metric_scale = True
        self.panoptic = True

        self.pairs_per_image = [set() for _ in range(len(self.images))]
        for idx1, idx2 in self.pairs:
            self.pairs_per_image[idx1].add(idx2)
            self.pairs_per_image[idx2].add(idx1)

    def _load_data(self):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)

            if 'cls_sep' in data:
                self.cls_sep = data['cls_sep'].item()
            else:
                print(f"WARN: cls_sep not in metadata, using default (={CLS_SEP})")
                self.cls_sep = CLS_SEP

        with open(osp.join(self.ROOT, 'categories.json'), 'r') as f:
            self.categories = json.load(f)
            self.classes = [cat['name'] for cat in self.categories]


    def _load_view(self, idx, view_idx, resolution, rng):
        scene_id = self.sceneids[view_idx]
        scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

        intrinsics = self.intrinsics[view_idx]
        camera_pose = self.trajectories[view_idx]
        basename = self.images[view_idx]

        # Load RGB image
        rgb_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
        # Load depthmap
        depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000
        depthmap[~np.isfinite(depthmap)] = 0  # invalid

        # Panoptic
        panoptic_seg = imread_cv2(osp.join(scene_dir, 'panoptic', basename + '.png'))
        panoptic_id = rgb2id(panoptic_seg)
        inst_id = panoptic_id // self.cls_sep
        cls_id = panoptic_id % self.cls_sep

        rgb_image, (depthmap, inst_id, cls_id), intrinsics = self._crop_resize_if_necessary(
            rgb_image, (depthmap, inst_id, cls_id), intrinsics, resolution, rng=rng, info=view_idx)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsics.astype(np.float32),
            dataset='ScanNet++',
            label=self.scenes[scene_id] + '_' + basename,
            instance=f'{str(idx)}_{str(view_idx)}',
            pan_inst_id=inst_id,
            pan_cls_id=cls_id,
            class_set=';'.join(self.classes)
        )



    def _get_views(self, idx, resolution, memory_num_views, rng):
        idx1, idx2 = self.pairs[idx]
        def get_pairs(view_idx): return self.pairs_per_image[view_idx]
        def get_view(view_idx, rng): return self._load_view(idx, view_idx, resolution, rng)
        views = select_tuple_from_pairs(get_pairs, get_view, self.num_views, memory_num_views, rng, idx1, idx2)
        return views

