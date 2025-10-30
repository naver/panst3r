# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import json
import os.path as osp
import numpy as np
import cv2
from PIL import Image

from dust3r.utils.image import imread_cv2
from dust3r.datasets import ScanNetpp as DUSt3R_ScanNetpp
from must3r.datasets.base.must3r_base_dataset import *
from must3r.datasets.base.tuple_maker import select_tuple_from_pairs

from panst3r.datasets.utils import rgb2id
import panst3r.datasets.cropping as cropping

class EasyDataset_PanSt3R(EasyDataset_MUSt3R):
    def __add__(self, other):
        return CatDataset_PanSt3R([self, other])

    def __rmul__(self, factor):
        return MulDataset_PanSt3R(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset_PanSt3R(factor, self)


class CatDataset_PanSt3R(CatDataset_MUSt3R, EasyDataset_PanSt3R):

    @property
    def classes(self):
        class_set = set()
        for ds in self.datasets:
            class_set.update(ds.classes)
        return list(class_set)


class MulDataset_PanSt3R(MulDataset_MUSt3R, EasyDataset_PanSt3R):

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def categories(self):
        return self.dataset.categories


class ResizedDataset_PanSt3R(ResizedDataset_MUSt3R, EasyDataset_PanSt3R):

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def categories(self):
        return self.dataset.categories


class PanSt3RBaseDataset(MUSt3RBaseDataset, EasyDataset_PanSt3R):

    def _crop_resize_if_necessary(self, image, masks, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, masks, intrinsics = cropping.crop_image_and_masks(image, masks, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, masks, intrinsics = cropping.rescale_image_and_masks(image, masks, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, masks, intrinsics2 = cropping.crop_image_and_masks(image, masks, intrinsics, crop_bbox)

        return image, masks, intrinsics2
