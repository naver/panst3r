# Copyright (C) 2025-present Naver Corporation. All rights reserved.
from dust3r.datasets.utils.cropping import *

def crop_image_and_masks(image, masks, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    masks_out = [mask[t:b, l:r] for mask in masks]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), masks_out, camera_intrinsics


def rescale_image_and_masks(image, masks, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale an image and corresponding masks
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)

    for mask in masks:
        assert tuple(mask.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), masks, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)

    masks_out = []
    for mask in masks:
        mask_out = cv2.resize(mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
        masks_out.append(mask_out)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), masks_out, camera_intrinsics
