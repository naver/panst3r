# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
from __future__ import annotations
import asyncio
import os
import argparse
import gradio
import torch
import numpy as np
import functools
import datetime
import time
import roma
import PIL.Image
import json
import copy
from contextlib import nullcontext
import viser
import viser.transforms as tf
import socket
from matplotlib.colors import hsv_to_rgb

from must3r.model import get_pointmaps_activation
from must3r.tools.image import get_resize_function
from must3r.engine.inference import postprocess
from must3r.demo.inference import SceneState
from must3r.demo.gradio import get_3D_model_from_scene as get_3D_model_from_scene_munst3r
from must3r.tools.image import is_valid_pil_image_file

from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import geotrf
from dust3r.viz import rgb
from must3r.datasets import ImgNorm

from panst3r import PanSt3R
from panst3r.datasets import id2rgb, rgb2id
from panst3r.engine import panoptic_inference_v1, panoptic_inference_v2, panoptic_inference_qubo
from panst3r.utils import get_colors_grid
from panst3r.tqdm import tqdm

import matplotlib.pyplot as pl

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

CLASS_NAMES = {
    'scannet': ["backpack","printer","paper bag","shoe rack","heater","bowl","power strip","sofa","table","bottle","telephone","crate","toilet brush","towel","trash can","ceiling lamp","toilet paper","stapler","plant pot","picture","pillow","sink","wall","pan","bag","storage cabinet","paper towel","jacket","exhaust fan","cup","tv","paper","blind rail","binder","basket","container","kitchen cabinet","curtain","socket","refrigerator","table lamp","coat hanger","bookshelf","soap dispenser","doorframe","clock","speaker","blinds","office chair","slippers","jar","book","cutting board","laptop","tissue box","air vent","kitchen counter","box","bucket","spray bottle","computer tower","kettle","marker","cloth","mouse","smoke detector","clothes hanger","chair","plant","cabinet","shoes","poster","door","ceiling","keyboard","shelf","cushion","floor","painting","microwave","toilet","window","tap","board","whiteboard","blanket","whiteboard eraser","monitor","headphones","light switch","windowsill","pot","window frame","suitcase","electrical duct","pipe","rack","file folder","bed","shower wall"],
    'coco': ["fire hydrant","backpack","road","banner","toothbrush","bicycle","door-stuff","bowl","frisbee","tent","donut","wall-brick","mountain-merged","building-other-merged","hair drier","teddy bear","bird","umbrella","surfboard","orange","bottle","towel","sky-other-merged","potted plant","wine glass","sand","grass-merged","fork","scissors","flower","pillow","sink","railroad","dog","playingfield","sea","cup","floor-other-merged","truck","tv","light","cabinet-merged","oven","cow","spoon","tie","handbag","giraffe","snowboard","elephant","floor-wood","curtain","motorcycle","gravel","refrigerator","boat","bench","skis","carrot","counter","tree-merged","knife","clock","zebra","fruit","couch","rock-merged","book","water-other","wall-stone","remote","laptop","food-other-merged","hot dog","airplane","bear","car","snow","stairs","stop sign","river","cardboard","platform","toaster","cell phone","wall-tile","train","parking meter","mouse","window-other","ceiling-merged","kite","dining table","cat","person","chair","paper-merged","pavement-merged","banana","traffic light","skateboard","baseball glove","baseball bat","roof","table-merged","keyboard","shelf","apple","microwave","dirt-merged","tennis racket","toilet","vase","pizza","cake","mirror-stuff","horse","broccoli","blanket","net","wall-wood","wall-other-merged","sports ball","bridge","house","rug-merged","window-blind","suitcase","bus","fence-merged","bed","sheep","sandwich"],
    'ade20k': ["clothes","basket, handbasket","armchair","painting, picture","cradle","booth","bicycle","barrel, cask","rug","desk","street lamp","food, solid food","stove","toilet, can, commode, crapper, pot, potty, stool, throne","tent","escalator, moving staircase, moving stairway","sofa","table","swivel chair","minibike, motorbike","ship","hovel, hut, hutch, shack, shanty","stage","bottle","van","wardrobe, closet, press","mountain, mount","towel","hill","blanket, cover","trash can","field","computer","chest of drawers, chest, bureau, dresser","sand","lake","fence","flower","fan","road, route","pillow","sink","coffee table","wall","lamp","bridge, span","pool table, billiard table, snooker table","grandstand, covered stand","rock, stone","bag","bookcase","sea","ottoman, pouf, pouffe, puff, hassock","truck","building","tv","light","glass, drinking glass","ball","falls","palm, palm tree","oven","blind, screen","mirror","bar","stairway, staircase","poster, posting, placard, notice, bill, card","awning, sunshade, sunblind","curtain","fireplace","animal","bannister, banister, balustrade, balusters, handrail","earth, ground","boat","bench","canopy","counter","hood, exhaust hood","bulletin board","clock","case, display case, showcase, vitrine","tree","base, pedestal, stand","book","flag","tower","runway","trade name","plate","skyscraper","column, pillar","sconce","car","box","stairs","crt screen","river","shower","buffet, counter, sideboard","grass","pier","person","chair","pool","water","kitchen island","plant","plane","conveyer belt, conveyor belt, conveyer, conveyor, transporter","cabinet","land, ground, soil","traffic light","fountain","tank, storage tank","door","ceiling","shelf","cushion","floor","screen door, screen","plaything, toy","refrigerator, icebox","step, stair","microwave","countertop","signboard, sign","path","arcade machine","vase","washer, automatic washer, washing machine","rail","radiator","sky","dishwasher","tray","tub","window ","screen","sidewalk, pavement","monitor","stool","house","seat","dirt track","pot","sculpture","bus","pole","chandelier","bed"]
}


class PanSt3R_SceneState(SceneState):
    def __init__(self, x_out, imgs, cls_vis, true_shape, focals, cams2world, image_list):
        super().__init__(x_out, imgs, true_shape, focals, cams2world, image_list)
        self.cls_vis = cls_vis


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 384, 224, 336, 448, 768], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."), default=None)
    parser.add_argument("--viser_port", type=int, help=("Port for the viser visualizer. "
                                                        "If None, will search for an available port starting at 5000."), default=None)
    parser.add_argument("--encoder", type=str, help="MUSt3R encoder configuration", default=None)
    parser.add_argument("--decoder", type=str, help="MUSt3R decoder configuration", default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser.add_argument("--retrieval", type=str, help="path to the retrieval weights", default=None)
    parser.add_argument("--camera_animation", action='store_true', help="Enable camera animation controls in the visualizer")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument('-q', '--silent', '--quiet', action='store_false', dest='verbose')

    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False,
                        help="Use Automatic Mixed Precision, fp16 might be unstable")
    parser.add_argument("--allow_local_files", action='store_true', default=False)
    return parser


def load_images(folder_content, size, patch_size=16, normalization='dust3r', verbose=True):
    imgs = []
    if normalization.lower() == 'dust3r':
        transform = ImgNorm
    else:
        raise ValueError(f'did not recognize image {normalization=}')

    for path in folder_content:
        rgb_image = PIL.Image.open(path).convert('RGB')
        rgb_image.load()
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(size, patch_size, H, W)
        rgb_tensor = resize_func(transform(rgb_image))
        imgs.append(dict(img=rgb_tensor, true_shape=np.int32([rgb_tensor.shape[-2], rgb_tensor.shape[-1]])))
        if verbose:
            print(f' - adding {path} with resolution {W}x{H} --> {rgb_tensor.shape[-1]}x{rgb_tensor.shape[-2]}')

    if len(imgs) == 1:
        imgs = imgs * 2 # create pair

    return imgs


def prepare_preds(pan_preds):
    confs = [conf.cpu().detach().numpy() for conf in pan_preds['conf']]
    pan_masks = [pan.cpu().detach().numpy() for pan in pan_preds['pan']]

    # Vis colors
    colors = get_colors_grid(len(pan_preds['segments_info']))
    id2color = {seg['id']: rgb2id(color) for seg, color in zip(pan_preds['segments_info'], colors)}

    pan_vis = [np.zeros_like(pan) for pan in pan_masks]
    inst_id = 1
    for seg in pan_preds['segments_info']:
        color = id2color[seg['id']]
        for pan_vis_i, pan_mask_i in zip(pan_vis, pan_masks):
            pan_vis_i[pan_mask_i == seg['id']] = color
        inst_id += 1
    pan_vis = [id2rgb(pan)/255. for pan in pan_vis]

    return pan_vis, confs


def pastel_colors(n=8, s_range=(0.25, 0.60), v_range=(0.92, 1.00),
                  distinct=True, seed=None):
    rng = np.random.default_rng(seed)

    # Hues
    if distinct:
        # Golden ratio conjugate spacing for well-separated hues
        phi = 0.6180339887498949
        h0 = rng.random()
        H = (h0 + phi * np.arange(n)) % 1.0
        i = np.arange(n)
        rng.shuffle(i)
        H = H[i]
    else:
        H = rng.random(n)

    # Low saturation, high value = pastel
    S = rng.uniform(*s_range, size=n)
    V = rng.uniform(*v_range, size=n)

    hsv = np.stack([H, S, V], axis=1)            # shape (n, 3)
    rgb = hsv_to_rgb(hsv)                        # floats in [0,1], shape (n, 3)

    return rgb

def _generate_colors(pan):
    """Generate distinct colors for each unique label in pan."""
    uniq_labels = np.unique(pan)
    # np.random.seed(0)  # For reproducibility
    colors = pastel_colors(n=len(uniq_labels), distinct=True)
    label_to_color = {label: colors[i] for i, label in enumerate(uniq_labels)}
    label_to_color[0] = np.array([0, 0, 0])  # Background color

    # Convert pan to pan_vis
    out = np.zeros((pan.shape[0], 3), dtype=np.float32)
    for label, color in label_to_color.items():
        out[pan == label] = color
    return out

def get_3D_model_from_scene(outdir, verbose, scene, min_conf_thr=3,
                            transparent_cams=False, local_pointmaps=False, cam_size=0.05, camera_conf_thr=0.0,
                            alpha=0.50,
                            filename='scene.glb'):
    scene_tmp = copy.copy(scene)
    scene_tmp.imgs = [rgbi * (1 - alpha) + pan_visi * alpha
                      for rgbi, pan_visi in zip(scene_tmp.imgs, scene_tmp.cls_vis)]
    return get_3D_model_from_scene_munst3r(outdir, verbose, scene_tmp, min_conf_thr, True, transparent_cams,
                                           local_pointmaps, cam_size, camera_conf_thr=camera_conf_thr,
                                           filename=filename)

@torch.no_grad()
def get_reconstructed_scene(outdir, model: PanSt3R, device, verbose, image_size, amp, visualizer,
                            filelist, loaded_files, num_mem_images, use_retrieval=False,
                            class_set=['scannet'], postprocess_fn='qubo',
                            progress=gradio.Progress()):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """

    def my_tqdm(iterable, desc=None, total=None, unit='it', *args, **kwargs):
        return progress.tqdm(iterable, desc, total, unit)

    with tqdm.wrap_tqdm(my_tqdm):
        max_bs = 1
        label_mode = model.panoptic_decoder.label_mode
        filelist = filelist or loaded_files.split("\n")

        print('loading images')
        time_start = datetime.datetime.now()
        views = load_images(filelist, size=image_size, patch_size=model.must3r_encoder.patch_size)
        nimgs = len(views)

        imgs = [b['img'].to(device) for b in views]
        true_shape = [torch.from_numpy(b['true_shape']).to(device) for b in views]
        true_shape = torch.stack(true_shape, dim=0)

        ellapsed = (datetime.datetime.now() - time_start)
        print(f'loaded in {ellapsed}')

        print('running inference')
        time_start = datetime.datetime.now()

        pointmaps_activation = get_pointmaps_activation(model.must3r_decoder)
        post_process=lambda x: postprocess(x, pointmaps_activation=pointmaps_activation)

        all_classes = []
        for s in class_set:
            assert s in CLASS_NAMES, f'did not recognize class set {s}'
            all_classes += CLASS_NAMES[s]
        all_classes = list(set(all_classes))

        assert len(imgs) >= 2
        num_keyframes = max(num_mem_images, 2)

        out_3D, pan_out = model.forward_inference_multi_ar(imgs, true_shape, all_classes, num_keyframes=num_keyframes,
                                                           use_retrieval=use_retrieval, max_bs=max_bs, outdevice='cpu', amp=amp)

        # Panoptic postprocessing
        size = true_shape.cpu().numpy()
        if postprocess_fn == 'qubo':
            pan_preds = panoptic_inference_qubo(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)
        elif postprocess_fn == 'standard_v1':
            pan_preds = panoptic_inference_v1(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)
        elif postprocess_fn == 'standard_v2':
            pan_preds = panoptic_inference_v2(pan_out['pred_logits'], pan_out['pred_masks'], size, label_mode=label_mode, device='cpu', multi_ar=True)
        else:
            raise ValueError(f'did not recognize {postprocess_fn=}')

        x_out = [post_process(pmi[0]) for pmi in out_3D]

        pan_vis, confs = prepare_preds(pan_preds[0])

        ellapsed = (datetime.datetime.now() - time_start)
        print(f'inference in {ellapsed}')
        try:
            print(str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB")
        except Exception as e:
            pass
        print('preparing pointcloud')
        time_start = datetime.datetime.now()
        focals = []
        cams2world = []
        true_shape = true_shape.cpu()
        for i in range(nimgs):
            H, W = true_shape[i]
            pp = torch.tensor((W/2, H/2), device=device)
            focal = float(estimate_focal_knowing_depth(x_out[i]['pts3d_local'].unsqueeze(0).to(device),
                                                    pp, focal_mode='weiszfeld'))
            focals.append(focal)

            R, T = roma.rigid_points_registration(
                x_out[i]['pts3d_local'].reshape(-1, 3).to(device),
                x_out[i]['pts3d'].reshape(-1, 3).to(device),
                weights=x_out[i]['conf'].ravel().to(device) - 1.0, compute_scaling=False)

            c2w = torch.eye(4, device=device)
            c2w[:3, :3] = R
            c2w[:3, 3] = T.ravel()

            cams2world.append(c2w.cpu())

        # x_out to cpu
        for i in range(len(x_out)):
            for k in x_out[i].keys():
                x_out[i][k] = x_out[i][k].cpu()

        rgbimg = [rgb(imgs[i].cpu(), true_shape[i]) for i in range(nimgs)]
        scene = PanSt3R_SceneState(x_out, rgbimg, pan_vis, true_shape, focals, cams2world, filelist)

        # get optimized values from scene
        x_out, imgs = scene.x_out, scene.imgs
        focals, cams2world = scene.focals, scene.cams2world
        nimgs = len(imgs)
        pts3d = [x_out[i]['pts3d'].cpu() for i in range(nimgs)]
        conf = [x_out[i]['conf'].cpu() for i in range(nimgs)]
        pts3d_local = [geotrf(cams2world[i], x_out[i]['pts3d_local'].cpu()) for i in range(nimgs)]

        for seg in pan_preds[0]['segments_info']:
            if 'category_name' in seg:
                continue
            seg['category_name'] = all_classes[seg['category_id']]

        visualizer.show_pointcloud(pts3d, pts3d_local, imgs, conf, scene.cams2world, scene.focals, pan_preds[0]['pan'], pan_preds[0]['segments_info'])

        ellapsed = (datetime.datetime.now() - time_start)
        print(f'pointcloud prepared in {ellapsed}')


def load_local_files(textinput):
    if textinput is not None and textinput:
        files = os.listdir(textinput)
        files = [os.path.join(textinput, f) for f in files]
        files = [f for f in files if is_valid_pil_image_file(f)]
        files = sorted(files)
    loaded_files = gradio.TextArea(value="\n".join(files), visible=True)

    return loaded_files, set_execution_params(files)


def upload_files(inputfiles, loaded_files):
    if inputfiles is not None:
        loaded_files = gradio.TextArea(value="", interactive=False, visible=False)
        valid_files = [f for f in inputfiles if is_valid_pil_image_file(f)]
        inputfiles_component = gradio.File(value=valid_files, file_count="multiple", file_types=['image'])
    elif loaded_files:
        inputfiles = loaded_files.split("\n")
        loaded_files = gradio.TextArea(interactive=False, value=loaded_files, visible=True)
        inputfiles_component = gradio.File(value=None, file_count="multiple", file_types=['image'])
    else:
        loaded_files = gradio.TextArea(value="", interactive=False, visible=False)
        inputfiles_component = gradio.File(value=None, file_count="multiple", file_types=['image'])

    return inputfiles_component, loaded_files, set_execution_params(inputfiles)

def remove_files(inputfiles, num_mem_images):
    return set_execution_params(inputfiles, num_mem_images)


def set_execution_params(inputfiles, num_mem_images=None):
    num_mem_images_out = gradio.Slider(label="Number of memory images", value=1,
                                   minimum=1, maximum=1, step=1, visible=True)
    if inputfiles is None or len(inputfiles) == 0:
        return num_mem_images_out

    num_files = len(inputfiles)
    prev = 50 if num_mem_images is None else num_mem_images
    current_num_mem_images = min(num_files, prev)

    num_mem_images_out = gradio.Slider(label="Number of keyframes", value=current_num_mem_images,
                                   minimum=1, maximum=num_files, step=1, visible=True)
    return num_mem_images_out


def _blend_colors(c1, c2, alpha):
    return (1.0 - alpha) * c1 + alpha * c2


class ClientState():
    def __init__(self, client: viser.ClientHandle, parent: ViserVisualizer):
        self.client = client
        self.parent = parent

        self.camera = client.camera

        self.animating = False
        self.start_camera = None
        self.max_el = np.deg2rad(85)  # max elevation angle

        self.client.gui.add_button("Reset camera", order=0).on_click(self.reset_camera)

    def reset(self):
        self.animating = False

        self._world_up = np.array([0, 1, 0], dtype=np.float64)
        self._center = self.camera.look_at
        self._up = self.camera.up_direction / np.linalg.norm(self.camera.up_direction)
        self._offset0 = self.camera.position - self._center
        radius = np.linalg.norm(self._offset0)
        self._forward0 = -self._offset0 / radius
        cos_el0 = np.clip(np.dot(self._forward0, self._world_up), -1.0, 1.0)
        self._el0 = np.arcsin(cos_el0)  # [-90°, +90°] where +90° means looking straight up

        self._prev_time = time.time()
        self._t = 0

    def reset_camera(self, _ev=None):
        if self.parent.data is None:
            return

        # Set initial position of the camera to the first frame (zoomed out)
        c2w = self.parent.data['cams2world'][0].cpu().numpy()
        direction = c2w[:3, 3]
        zoom_factor = 1.5
        new_position = direction * zoom_factor
        new_rot = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
        with self.client.atomic():
            self.client.camera.wxyz = new_rot
            self.client.camera.position = new_position

    def animate_step(self, speed, radius):
        if not self.animating:
            return

        cur_time = time.time()
        delta = cur_time - self._prev_time
        self._prev_time = cur_time

        self._t += (delta * speed) % (2 * np.pi)
        # Smooth ease-in-out for a nice feel:
        s1 = 0.5 * np.cos(np.pi * self._t - np.pi/2)
        s2 = 0.5 * np.sin(np.pi * self._t - np.pi/2)

        az = np.deg2rad(radius) * s1
        el = np.deg2rad(radius) * s2

        # --- 1) horizontal rotation around world_up ---
        R_az = tf.SO3.exp(self._world_up * az)
        off1 = R_az.apply(self._offset0)

        # --- 2) vertical rotation around current "right" axis ---
        # Recompute forward and right from the intermediate position
        forward = -off1 / (np.linalg.norm(off1) + 1e-9)   # camera -> center
        right = np.cross(self._world_up, forward)
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            # Degenerate (looking straight up/down); skip elevation for this frame
            off2 = off1
        else:
            right /= nr

            # Clamp target elevation to avoid flipping over poles
            # New elevation would be el0 + (el - 0) = el0 + el*s; we clamp stepwise.
            target_el = np.clip(self._el0 + el, -self.max_el, self.max_el)
            el_step = target_el - self._el0

            R_el = tf.SO3.exp(right * el_step)
            off2 = R_el.apply(off1)

        new_pos = self._center + off2

        # Atomically set both so `look_at` stays fixed (no jitter/auto-offset)
        with self.client.atomic():
            self.client.camera.position = new_pos
            self.client.camera.look_at = self._center

        self.client.flush()


class ViserVisualizer():

    def __init__(self, host='localhost', port=None, camera_animation=False):
        port = port or self._find_port(5000)
        self.server = viser.ViserServer(host=host, port=port)

        self.server.on_client_connect(self.add_client)
        self.server.on_client_disconnect(self.remove_client)

        self.server.scene.set_up_direction('-y')

         # GUI
        with self.server.gui.add_folder("Points", expand_by_default=True):
            self.confidence_thr = self.server.gui.add_slider(
                "Min. confidence threshold", min=0.0, max=10.0, step=0.1, initial_value=3.0
            )
            self.point_size = self.server.gui.add_slider(
                "Point size (cm)", min=0.1, max=2.0, step=0.1, initial_value=0.5
            )
            self.local_pointmaps = self.server.gui.add_checkbox("Local pointmaps", initial_value=True)

        with self.server.gui.add_folder("Segmentation", expand_by_default=True):
            self.opacity = self.server.gui.add_slider(
                "Segmentation opacity", min=0.0, max=1.0, step=0.01, initial_value=0.5
            )
            self.show_labels = self.server.gui.add_checkbox("Show labels", initial_value=True)
            self.btn_regenerate_colors = self.server.gui.add_button("Regenerate colors")

        with self.server.gui.add_folder("Poses", expand_by_default=True):
            self.show_poses = self.server.gui.add_checkbox("Show poses", initial_value=False)
            self.camera_size = self.server.gui.add_slider(
                "Camera size", min=0.1, max=2.0, step=0.1, initial_value=0.5
            )


        if camera_animation:
            with self.server.gui.add_folder("Camera", expand_by_default=True):
                self.animate_camera_btn = self.server.gui.add_button("Start camera animation")
                self.fps_slider = self.server.gui.add_slider(
                    "Animation FPS", min=1, max=60, step=1, initial_value=60
                )
                self.animation_speed = self.server.gui.add_slider(
                    "Animation speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
                )
                self.camera_radius = self.server.gui.add_slider(
                    "Camera radius", min=0.5, max=30, step=0.5, initial_value=10
                )

            self.animate_camera_btn.on_click(self.set_animate)

        self.confidence_thr.on_update(self.update_conf_thresh)
        self.point_size.on_update(self.update_point_size)
        self.local_pointmaps.on_update(self.update_points)

        self.opacity.on_update(self.update_opacity)
        self.show_labels.on_update(self.update_show_labels)
        self.btn_regenerate_colors.on_click(self.regenerate_colors)

        self.show_poses.on_update(self.update_poses)
        self.camera_size.on_update(self.update_poses)


        self.pc = None
        self.labels = None
        self.poses = None
        self.data = None
        self.data_f = None

        self.clients: dict[int, ClientState] = {}
        self.animating = False


        if camera_animation:
            self.server.get_event_loop().create_task(self.animate_loop())

    @property
    def address(self):
        return f"{self.server.get_host()}:{self.server.get_port()}"

    def _find_port(self, start_port, max_port=65535):
        for port in range(start_port, max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No free ports found in the given range")

    def update_opacity(self, _ev):
        if self.pc is None:
            return

        self.pc.colors = _blend_colors(self.data_f['rgb'], self.data_f['pan_vis'], float(self.opacity.value))

    def update_point_size(self, _ev):
        if self.pc is None:
            return

        self.pc.point_size = self.point_size.value * 0.01

    def update_show_labels(self, _ev):
        if self.labels is None:
            return

        for lbl in self.labels:
            lbl.visible = self.show_labels.value

    def update_poses(self, _ev):
        if self.poses is None:
            return

        for pose in self.poses:
            pose.visible = self.show_poses.value
            pose.scale = self.camera_size.value * 0.1

    def update_points(self, _ev):
        if self.data is None or self.pc is None:
            return

        self.pc.points = self.data_f['pts_local'] if self.local_pointmaps.value else self.data_f['pts']

    def regenerate_colors(self, _ev):
        if self.data is None:
            return

        self.data['pan_vis'] = _generate_colors(self.data['pan'])
        self.data_f['pan_vis'] = self.data['pan_vis'][self.data_f['mask']]

        if self.pc is not None:
            self.pc.colors = _blend_colors(self.data_f['rgb'], self.data_f['pan_vis'], float(self.opacity.value))

    def update_conf_thresh(self, _ev):
        if self.data is None:
            return

        m = self.data['conf'] >= float(self.confidence_thr.value)
        self.data_f = dict(pts=self.data['pts'][m],
                           rgb=self.data['rgb'][m],
                           pan=self.data['pan'][m],
                           pan_vis=self.data['pan_vis'][m],
                           mask=m)

        self.pc.points = self.data_f['pts']
        self.pc.colors = _blend_colors(self.data_f['rgb'], self.data_f['pan_vis'], float(self.opacity.value))

    def add_client(self, client: viser.ClientHandle) -> None:
        self.clients[client.client_id] = ClientState(client, self)

    def remove_client(self, client: viser.ClientHandle) -> None:
        self.clients.pop(client.client_id, None)

    def set_animate(self, _ev) -> None:
        if self.animating:
            self.animating = False
            for state in self.clients.values():
                state.animating = False
                state.start_camera = None
            self.animate_camera_btn.label = "Start camera animation"
        else:
            self.animating = True
            for state in self.clients.values():
                state.reset()
                state.animating = True
                state.start_camera = state.client.camera.position
            self.animate_camera_btn.label = "Stop camera animation"


    async def animate_loop(self):
        while True:
            for client in self.clients.values():
                client.animate_step(self.animation_speed.value, self.camera_radius.value)

            await asyncio.sleep(1/self.fps_slider.value)


    def show_pointcloud(self, pts3d, pts3d_local, rgb, conf, cams2world, focals, pan, segments_info):
        self.server.scene.reset()
        self.pc = None
        self.labels = None

        conf = np.concatenate([c.flatten() for c in conf])          # (N)
        pts_f = np.concatenate([p.reshape(-1, 3) for p in pts3d])  # (N, 3)
        pts_local_f = np.concatenate([p.reshape(-1, 3) for p in pts3d_local])  # (N, 3)
        rgb_f = np.concatenate([r.reshape(-1, 3) for r in rgb])    # (N, 3)
        pan_f = np.concatenate([p.reshape(-1) for p in pan])       # (N)
        self.data = dict(pts=pts_f, pts_local=pts_local_f, rgb=rgb_f, pan=pan_f, pan_vis=_generate_colors(pan_f), conf=conf, cams2world=cams2world)
        m = conf >= float(self.confidence_thr.value)
        self.data_f = dict(pts=self.data['pts'][m],
                           pts_local=self.data['pts_local'][m],
                           rgb=self.data['rgb'][m],
                           pan=self.data['pan'][m],
                           pan_vis=self.data['pan_vis'][m],
                           mask=m)

        # Create point cloud with initial colors.
        self.pc = self.server.scene.add_point_cloud(
            name="pointcloud",
            points=self.data_f['pts_local'] if self.local_pointmaps.value else self.data_f['pts'],
            colors=_blend_colors(self.data_f['rgb'], self.data_f['pan_vis'], float(self.opacity.value)),
            point_size=self.point_size.value * 0.01,
            point_shape='circle',
        )

        self.labels = []
        for i, seg in enumerate(segments_info):
            u = seg['id']
            lbl = seg['category_name'] if 'category_name' in seg else f"#{int(u)}"
            m = (self.data_f['pan'] == u)
            if not np.any(m):
                continue
            c = np.median(self.data_f['pts_local'][m], axis=0)

            # 3D label text at the medoid of each segment
            lbl = self.server.scene.add_label(
                name=f"labels/lbl_{i}",
                text=lbl,
                position=(float(c[0]), float(c[1]), float(c[2])),
                visible=self.show_labels.value,
            )
            self.labels.append(lbl)

        self.poses = []
        for i, (c2w, focal) in enumerate(zip(cams2world, focals)):
            c2w = c2w.cpu().numpy()
            rgb_i = rgb[i]
            fov = 2 * np.arctan2(rgb_i.shape[0] / 2, focal)
            aspect = rgb_i.shape[1] / rgb_i.shape[0]
            pose = self.server.scene.add_camera_frustum(
                f"poses/pose_{i}",
                fov=fov,
                aspect=aspect,
                scale=self.camera_size.value * 0.1,
                image=rgb_i,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                position=c2w[:3, 3],
                visible=self.show_poses.value,
            )
            self.poses.append(pose)

        for client in self.clients.values():
            client.reset_camera()


def main_demo(tmpdirname, model: PanSt3R, device, image_size, server_name, server_port, viser_port,
              verbose=True, allow_local_files=False, camera_animation=False, amp=False):
    visualizer = ViserVisualizer(host=server_name, port=viser_port, camera_animation=camera_animation)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, verbose, image_size, amp, visualizer)
    # model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, verbose)




    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="PanSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">PanSt3R Demo</h2>')
        with gradio.Column():
            with gradio.Row():
                with gradio.Column():
                    optional_tab = lambda x: gradio.Tab(x) if allow_local_files else nullcontext()
                    with optional_tab("Upload images"):
                        inputfiles = gradio.File(file_count="multiple", file_types=['image'], height=250)
                    with optional_tab("Local path"):
                        textinput = gradio.Textbox(label="Path to a local image directory", visible=allow_local_files)
                        load_files = gradio.Button("Load", visible=allow_local_files)
                        loaded_files = gradio.Textbox(label='Found images', value="", interactive=False, visible=False, max_lines=5)


                with gradio.Column():
                    num_mem_images = gradio.Slider(label="Number of keyframes", value=1,
                                                    minimum=1, maximum=1, step=1, visible=True)
                    use_retrieval = gradio.Checkbox(value=False, label="Use retrieval for keyframe selection",
                                                    visible=model.retrieval is not None)
                    class_set = gradio.CheckboxGroup(choices=[('ScanNet++ (100)', 'scannet'), ('COCO', 'coco'), ('ADE20k', 'ade20k')], value=['scannet'], label="Class set",
                                                     visible=True, interactive=True)
                    postprocess_fn = gradio.Radio(choices=[('QUBO', 'qubo'),('Standard (v1)', 'standard_v1'), ('Standard (v2)', 'standard_v2')], label="Panoptic postprocessing",
                                                     value=model.postprocess_default, visible=True, interactive=True)
                    run_btn = gradio.Button("Run", variant="primary")

                    status = gradio.Markdown(value='Upload images and click **Run** to start', height=50)

            # local_pointmaps = gradio.Checkbox(value=False, label="viz local pointmaps pointcloud")

            # outmodel = gradio.Model3D()
            outmodel = gradio.HTML(f"""<div style="width:100%; height:600px; border:1px solid #e4e4e7; border-radius: 4px; resize:vertical; overflow:auto;">
                <div style="padding: 5px 12px"><span style="color: #71717a">Visualization</span><span style="float: right"><a href="http://{visualizer.address}" target="_blank">Full screen</a><span></span></span></div>
                <iframe
                    src="http://{visualizer.address}"
                    style="width:100%; height: calc(100% - 36px); border:none;">
                </iframe>
                </div>""")


            # events
            inputfiles.upload(upload_files,
                              inputs=[inputfiles, loaded_files],
                              outputs=[inputfiles, loaded_files, num_mem_images])

            inputfiles.delete(remove_files,
                              inputs=[inputfiles, num_mem_images],
                              outputs=[num_mem_images])

            inputfiles.clear(remove_files,
                             inputs=[inputfiles, num_mem_images],
                             outputs=[num_mem_images])

            if allow_local_files:
                load_files.click(fn=load_local_files,
                                 inputs=[textinput],
                                 outputs=[loaded_files, num_mem_images])

            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, loaded_files, num_mem_images, use_retrieval, class_set, postprocess_fn],
                          show_progress_on=status)

    demo.launch(share=False, server_name=server_name, server_port=server_port)
