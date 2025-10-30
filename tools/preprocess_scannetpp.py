# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python3 datasets_preprocess/preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --precomputed_pairs /path/to/scannetpp_pairs --pyopengl-platform egl
# --------------------------------------------------------
import os
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import numpy as np
import cv2
import PIL.Image as Image
import json
import traceback
import pandas as pd
from scipy.spatial import distance

from dust3r.datasets.utils.cropping import rescale_image_depthmap
import dust3r.utils.geometry as geometry

from panst3r.datasets.utils import rgb2id, id2rgb

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r'^.*DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'.*frame_(?P<frameid>\d+).jpg$')

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])

CLS_SEP = 256
MIN_INST_AREA = 50


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, help="Path to the ScanNet++ dataset root directory")
    parser.add_argument('--pairs_dir', required=True, help="Directory with precomputed image pairs")
    parser.add_argument('--output_dir', default='data/scannetpp_processed')
    parser.add_argument('--target_resolution', default=920, type=int, help="images resolution")
    parser.add_argument('--pyopengl-platform', type=str, default='', help='PyOpenGL env variable')

    parser.add_argument('--class_list', default='metadata/semantic_benchmark/top100.txt')
    parser.add_argument('--instance_list', default='metadata/semantic_benchmark/top100_instance.txt')
    parser.add_argument('--mapping_file', default='metadata/semantic_benchmark/map_benchmark.csv')
    parser.add_argument('--export_crowd', action='store_true', help="Export crowd instances")
    parser.add_argument('--cls_sep', default=CLS_SEP, type=int, help="Class separation value (when encoded into RGB). Total number of classes should be < cls_sep")
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world

# Visualization colors
def generate_points_with_max_distance(n_points, initial_points=None, space_dim=3, num_candidates=100):
    """Generate n_points points in hypercube trying to maximize the distance between points."""
    if initial_points is not None:
        points = initial_points
    else:
        # Start with an initial random point
        points = [np.random.rand(space_dim)]

    for _ in range(len(points), n_points):
        max_min_dist = 0
        best_candidate = None

        # Try a number of random candidates to find the one maximizing the minimum distance
        for _ in range(num_candidates):  # You can increase this for better results
            candidate = np.random.rand(space_dim)
            min_dist = min(distance.euclidean(candidate, p) for p in points)

            # Update if this candidate has a larger minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = candidate

        # Add the best candidate to the list of points
        points.append(best_candidate)

    return np.array(points)

def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    try:
        matches = re.match(regex_expr, name)
        return matches['frameid']
    except Exception as e:
        print(f'Error when parsing {name}')
        raise ValueError(f'Invalid name {name}')


def load_sfm(sfm_dir, cam_type='dslr'):
    # load cameras
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        prefixes = ['iphone/', 'video/']
        for prefix in prefixes:
            if img_name.startswith(prefix):
                img_name = img_name[len(prefix):]
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    # load 3D points
    with open(os.path.join(sfm_dir, 'points3D.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    points3D = {}
    observations = {idx: [] for idx in img_infos.keys()}
    for point in tqdm(raw, position=1, leave=False):
        point = point.split()
        point_3d_idx = int(point[0])
        points3D[point_3d_idx] = tuple(map(float, point[1:4]))
        if len(point) > 8:
            for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                if idx not in observations:
                    continue
                observations[idx].append((point_3d_idx, int(point_2d_idx)))

    return img_idx, img_infos, points3D, observations


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val['path'] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]['frame_id'])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = geometry.opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask

def read_semantics(segments_path, annotations_path, lbl2id, crowd_classes, num_points, cls_sep, export_crowd=False):
    with open(segments_path, 'r') as f:
        segments = json.load(f)

    assert segments['segIndices'] == list(range(num_points)), "Segment indices do not match"

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    pts_pan_id = np.full(num_points, 0, dtype=int)

    segments = []
    inst_id = 1
    for seg_info in annotations['segGroups']:
        vert_idx = np.array(seg_info['segments'])

        if seg_info['label'] not in lbl2id:
            # print(f"WARN: Unknown label - {seg_info['label']}")
            continue

        cls_id = lbl2id[seg_info['label']]
        iscrowd = seg_info['label'] in crowd_classes

        if iscrowd and not export_crowd:
            continue

        segments.append(dict(
            id = inst_id * cls_sep + cls_id,
            instance_id=inst_id,
            class_id=cls_id,
            orig_class_name=seg_info['label'],
            iscrowd=iscrowd,
        ))

        pts_pan_id[vert_idx] = inst_id * cls_sep + cls_id
        inst_id += 1

    return segments, pts_pan_id

def process_scenes(root, class_list, instance_list, mapping_file, pairsdir, output_dir, target_resolution, cls_sep=CLS_SEP, export_crowd=False):
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    scenes = os.listdir(pairsdir)

    # Read class list
    with open(osp.join(root, class_list), 'r') as f:
        semantic_cls = {l.strip(): i for i, l in enumerate(f.readlines())}

    with open(osp.join(root, instance_list), 'r') as f:
        thing_cls = {l.strip() for l in f.readlines()}

    # Save categories info
    categories = []
    for sem_cls in semantic_cls:
        categories.append(dict(
            id=semantic_cls[sem_cls],
            name=sem_cls,
            isthing=int(sem_cls in thing_cls)
        ))

    # Read mapping file
    crowd_cls = set()
    if mapping_file is not None:
        map_df = pd.read_csv(osp.join(root, mapping_file))
        for i,row in map_df.iterrows():
            if pd.isna(row['semantic_map_to']) and pd.isna(row['instance_map_to']):
                continue
            cls_name = row['class']
            sem_cls = row['semantic_map_to']
            inst_cls = row['instance_map_to']

            if sem_cls not in semantic_cls and inst_cls not in semantic_cls:
                print(f"WARN: Remaped class {cls_name} not in output class set")
                continue

            if not pd.isna(inst_cls):
                # Mapping to thing class exists
                semantic_cls[cls_name] = semantic_cls[inst_cls]
            elif sem_cls not in thing_cls:
                # Mapping to stuff class exists
                semantic_cls[cls_name] = semantic_cls[sem_cls]
            else:
                # Mapping to thing class from stuff class (iscrowd=1)
                semantic_cls[cls_name] = semantic_cls[sem_cls]
                crowd_cls.add(cls_name)
                pass


    with open(osp.join(output_dir, 'categories.json'), 'w') as f:
        json.dump(categories, f)

    # for each of these, we will select some dslr images and some iphone images
    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)
    for scene in tqdm(scenes, position=0, leave=True):
        data_dir = os.path.join(root, 'data', scene)
        dir_dslr = os.path.join(data_dir, 'dslr')
        dir_iphone = os.path.join(data_dir, 'iphone')
        dir_scans = os.path.join(data_dir, 'scans')

        assert os.path.isdir(data_dir) and os.path.isdir(dir_dslr) \
            and os.path.isdir(dir_iphone) and os.path.isdir(dir_scans)

        output_dir_scene = os.path.join(output_dir, scene)
        scene_metadata_path = osp.join(output_dir_scene, 'scene_metadata.npz')
        if osp.isfile(scene_metadata_path):
            continue

        pairs_dir_scene = os.path.join(pairsdir, scene)
        pairs_dir_scene_selected_pairs = os.path.join(pairs_dir_scene, 'selected_pairs.npz')
        assert osp.isfile(pairs_dir_scene_selected_pairs)
        selected_npz = np.load(pairs_dir_scene_selected_pairs)
        selection, pairs = selected_npz['selection'], selected_npz['pairs']

        # set up the output paths
        output_dir_scene_rgb = os.path.join(output_dir_scene, 'images')
        output_dir_scene_depth = os.path.join(output_dir_scene, 'depth')
        output_dir_scene_panoptic = os.path.join(output_dir_scene, 'panoptic')
        output_dir_scene_panoptic_vis = os.path.join(output_dir_scene, 'panoptic_vis')
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)
        os.makedirs(output_dir_scene_panoptic, exist_ok=True)
        os.makedirs(output_dir_scene_panoptic_vis, exist_ok=True)

        ply_path = os.path.join(dir_scans, 'mesh_aligned_0.05.ply')
        segments_path = os.path.join(dir_scans, 'segments.json')
        annotations_path = os.path.join(dir_scans, 'segments_anno.json')

        sfm_dir_dslr = os.path.join(dir_dslr, 'colmap')
        rgb_dir_dslr = os.path.join(dir_dslr, 'resized_images')
        mask_dir_dslr = os.path.join(dir_dslr, 'resized_anon_masks')

        sfm_dir_iphone = os.path.join(dir_iphone, 'colmap')
        rgb_dir_iphone = os.path.join(dir_iphone, 'rgb')
        mask_dir_iphone = os.path.join(dir_iphone, 'rgb_masks')

        # load the mesh
        with open(ply_path, 'rb') as f:
            mesh_kwargs = trimesh.exchange.ply.load_ply(f)
        mesh_scene = trimesh.Trimesh(process=False, **mesh_kwargs)

        # read colmap reconstruction, we will only use the intrinsics and pose here
        img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr = load_sfm(sfm_dir_dslr, cam_type='dslr')
        dslr_paths = {
            "in_colmap": sfm_dir_dslr,
            "in_rgb": rgb_dir_dslr,
            "in_mask": mask_dir_dslr,
        }

        img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = load_sfm(
            sfm_dir_iphone, cam_type='iphone')
        iphone_paths = {
            "in_colmap": sfm_dir_iphone,
            "in_rgb": rgb_dir_iphone,
            "in_mask": mask_dir_iphone,
        }

        # Load semantics
        try:
            segments, pts_pan_id = read_semantics(segments_path, annotations_path, semantic_cls, crowd_cls, len(mesh_scene.vertices), cls_sep, export_crowd)
        except Exception as e:
            print(f"Error reading semantics for {scene}")
            traceback.print_exc()
            continue

        with open(os.path.join(output_dir_scene, 'segments.json'), 'w') as f:
            json.dump(segments, f)

        pts_pan_rgb = id2rgb(pts_pan_id)

        # Colors for panoptic mask visualization
        valid_pan_ids = set(np.unique(pts_pan_id)) - {0}
        colors = generate_points_with_max_distance(len(valid_pan_ids)+1, initial_points=[np.array([0,0,0])], space_dim=3)
        colors = (colors * 255).astype(np.uint8)
        vis_colors = {pan_id: rgb2id(color) for pan_id, color in zip(valid_pan_ids, colors[1:])}

        # Add alpha channel
        alpha = np.full((pts_pan_rgb.shape[0], 1), 255, dtype='uint8')
        pts_pan_rgb = np.concatenate([pts_pan_rgb, alpha], axis=1)

        mesh_scene.visual.vertex_colors = pts_pan_rgb
        mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)

        pyrender_scene = pyrender.Scene()
        pyrender_scene.add(mesh)

        selection_dslr = [imgname + '.JPG' for imgname in selection if imgname.startswith('DSC')]
        selection_iphone = [imgname + '.jpg' for imgname in selection if imgname.startswith('frame_')]

        # resize the image to a more manageable size and render depth
        for selection_cam, img_idx, img_infos, paths_data in [(selection_dslr, img_idx_dslr, img_infos_dslr, dslr_paths),
                                                              (selection_iphone, img_idx_iphone, img_infos_iphone, iphone_paths)]:
            rgb_dir = paths_data['in_rgb']
            mask_dir = paths_data['in_mask']
            for imgname in tqdm(selection_cam, position=1, leave=False):
                imgidx = img_idx[imgname]
                img_infos_idx = img_infos[imgidx]
                rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx['path'])))
                mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx['path'][:-3] + 'png')))
                rgb[mask==0] = 0 # Color masked regions black (instead of purple)

                _, _, K, rgb, mask = undistort_images(img_infos_idx['intrinsics'], rgb, mask)

                # rescale_image_depthmap assumes opencv intrinsics
                intrinsics = geometry.colmap_to_opencv_intrinsics(K)
                image, mask, intrinsics = rescale_image_depthmap(
                    rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4))

                W, H = image.size
                intrinsics = geometry.opencv_to_colmap_intrinsics(intrinsics)

                # update inpace img_infos_idx
                img_infos_idx['intrinsics'] = intrinsics
                rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx['path'][:-3] + 'jpg')
                image.save(rgb_outpath)

                depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx['path'][:-3] + 'png')

                # render depth image
                renderer.viewport_width, renderer.viewport_height = W, H
                fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
                camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(camera, pose=img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)

                flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SEG_VERT
                # flags = pyrender.RenderFlags.FLAT
                pan_mask_rgb, depth = renderer.render(pyrender_scene, flags)

                pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

                depth = (depth * 1000).astype('uint16')
                # invalidate depth from mask before saving
                depth_mask = (mask < 255)
                depth[depth_mask] = 0
                Image.fromarray(depth).save(depth_outpath)

                # Export panoptic masks (and visualization with random colors for each instance)
                mask_f = (mask / 255.)[..., None]
                pan_mask_rgb = (pan_mask_rgb * mask_f).astype(np.uint8)
                pan_mask_ids = rgb2id(pan_mask_rgb)
                pan_mask_vis = np.zeros_like(pan_mask_ids)

                # Filter noisy segments
                for inst_id in np.unique(pan_mask_ids):
                    if inst_id == 0:
                        continue
                    mask_inst = pan_mask_ids == inst_id
                    if mask_inst.sum() < MIN_INST_AREA or inst_id not in valid_pan_ids:
                        pan_mask_ids[mask_inst] = 0
                        continue

                    pan_mask_vis[mask_inst] = vis_colors[inst_id]

                pan_mask_rgb = id2rgb(pan_mask_ids)
                pan_mask_vis = id2rgb(pan_mask_vis)
                panoptic_outpath = os.path.join(output_dir_scene_panoptic, img_infos_idx['path'][:-3] + 'png')
                panoptic_vis_outpath = os.path.join(output_dir_scene_panoptic_vis, img_infos_idx['path'][:-3] + 'png')
                Image.fromarray(pan_mask_rgb).save(panoptic_outpath)
                Image.fromarray(pan_mask_vis).save(panoptic_vis_outpath)

        trajectories = []
        intrinsics = []
        for imgname in selection:
            if imgname.startswith('DSC'):
                imgidx = img_idx_dslr[imgname + '.JPG']
                img_infos_idx = img_infos_dslr[imgidx]
            elif imgname.startswith('frame_'):
                imgidx = img_idx_iphone[imgname + '.jpg']
                img_infos_idx = img_infos_iphone[imgidx]
            else:
                raise ValueError('invalid image name')

            intrinsics.append(img_infos_idx['intrinsics'])
            trajectories.append(img_infos_idx['cam_to_world'])

        intrinsics = np.stack(intrinsics, axis=0)
        trajectories = np.stack(trajectories, axis=0)
        # save metadata for this scene
        np.savez(scene_metadata_path,
                 trajectories = trajectories,
                 intrinsics = intrinsics,
                 images = selection,
                 pairs = pairs,
                 cls_sep = cls_sep
                 )

        del img_infos
        del pyrender_scene

    # concat all scene_metadata.npz into a single file
    scene_data = {}
    for scene_subdir in scenes:
        scene_metadata_path = osp.join(output_dir, scene_subdir, 'scene_metadata.npz')
        with np.load(scene_metadata_path) as data:
            trajectories = data['trajectories']
            intrinsics = data['intrinsics']
            images = data['images']
            pairs = data['pairs']
        scene_data[scene_subdir] = {'trajectories': trajectories,
                                    'intrinsics': intrinsics,
                                    'images': images,
                                    'pairs': pairs}

    offset = 0
    counts = []
    scenes = []
    sceneids = []
    images = []
    intrinsics = []
    trajectories = []
    pairs = []
    for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
        num_imgs = data['images'].shape[0]
        img_pairs = data['pairs']

        scenes.append(scene_subdir)
        sceneids.extend([scene_idx] * num_imgs)

        images.append(data['images'])

        intrinsics.append(data['intrinsics'])
        trajectories.append(data['trajectories'])

        # offset pairs
        img_pairs[:, 0:2] += offset
        pairs.append(img_pairs)
        counts.append(offset)

        offset += num_imgs

    images = np.concatenate(images, axis=0)
    intrinsics = np.concatenate(intrinsics, axis=0)
    trajectories = np.concatenate(trajectories, axis=0)
    pairs = np.concatenate(pairs, axis=0)
    np.savez(osp.join(output_dir, 'all_metadata.npz'),
        counts = counts,
        scenes = scenes,
        sceneids = sceneids,
        images = images,
        intrinsics = intrinsics,
        trajectories = trajectories,
        pairs = pairs,
        cls_sep = cls_sep)
    print('all done')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ['PYOPENGL_PLATFORM'] = args.pyopengl_platform
    process_scenes(args.root_dir, args.class_list, args.instance_list, args.mapping_file, args.pairs_dir, args.output_dir, args.target_resolution, args.cls_sep, args.export_crowd)
