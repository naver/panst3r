
import torch
import torch.nn.functional as F
import numpy as np
from panst3r.tqdm import tqdm

# Adapted for PanSt3R from Mask2Former (https://github.com/facebookresearch/Mask2Former) by Meta Platforms, Inc.
# Original code licensed under the MIT License.
def panoptic_inference_v1(*args, mask_threshold=0.5, overlap_threshold=0.8, **kwargs):
    return panoptic_inference_v2(*args, mask_threshold=mask_threshold, overlap_threshold=overlap_threshold,
                                 niters=1, **kwargs)

@torch.no_grad()
def panoptic_inference_v2(mask_cls, mask_pred, true_shape, label_mode='sigmoid',
                       cls_threshold=0.1, temperature=None, mask_threshold=0.25,
                       overlap_threshold=0.5, niters=2, void_confidence=0.1, device=None, multi_ar=False):

    if multi_ar:
        for i in range(len(mask_pred)):
            mask_pred[i] = mask_pred[i].sigmoid().to(device)
            mask_pred[i] = F.interpolate(mask_pred[i], size=true_shape[i].tolist(), mode="bilinear", align_corners=False)
        mask_pred = torch.nested.nested_tensor(mask_pred)
        mask_pred = mask_pred.to_padded_tensor(0.).transpose(0, 1)
    else:
        mask_pred = mask_pred.sigmoid().to(device)
        mask_pred = F.interpolate(mask_pred, size=true_shape, mode="bilinear", align_corners=False)
        mask_pred = torch.stack(mask_pred)

    if device is not None:
        mask_cls = mask_cls.to(device)

    results = []
    B,V = mask_pred.shape[:2]

    for bi in range(mask_cls.shape[0]):
        mask_cls_i = mask_cls[bi]
        mask_pred_i = mask_pred[bi].transpose(0,1)
        indices_i = torch.arange(mask_cls_i.shape[0], device=mask_cls_i.device)

        if label_mode == 'sigmoid':
            scores, labels = mask_cls_i.sigmoid().max(-1)
            keep = scores >  cls_threshold

            # temperature softmax scaling, make the class prediction sharper
            if temperature is not None:
                T = temperature
                scores, labels = F.softmax(mask_cls_i.sigmoid() / T, dim=-1).max(-1)
        else:
            scores, labels = F.softmax(mask_cls_i, dim=-1).max(-1)
            num_classes = mask_cls_i.shape[-1] - 1
            keep = labels.ne(num_classes) & (scores > cls_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred_i[keep]
        cur_mask_cls = mask_cls_i[keep]
        cur_indices = indices_i[keep]

        if label_mode == 'softmax':
            cur_mask_cls = cur_mask_cls[:, :-1]

        # TODO: update
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks


        for it in range(niters):
            panoptic_seg = torch.zeros(cur_masks.shape[-3:], dtype=torch.int32, device=cur_masks.device)
            conf = torch.zeros(cur_masks.shape[-3:], dtype=torch.float, device=cur_masks.device) + void_confidence
            segments_info = []
            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                break

            # take argmax
            segments_info = []
            current_segment_id = 0
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            selected = []
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                query_id = cur_indices[k].item()
                # isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                isthing = True # TODO: temporary fix, treat everything as things
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= mask_threshold)
                mask_area = mask.sum().item()

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    selected.append(k)

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    conf[mask] = cur_masks[k][mask]

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "query_id": query_id,
                            "category_id": int(pred_class),
                        }
                    )

            selected = torch.tensor(selected, device=cur_masks.device, dtype=torch.int64)
            cur_prob_masks = cur_prob_masks[selected]
            cur_classes = cur_classes[selected]
            cur_indices = cur_indices[selected]
            cur_masks = cur_masks[selected]

        if multi_ar:
            panoptic_seg = [panoptic_seg[i, :H, :W].contiguous() for i, (H, W) in enumerate(true_shape)]
            conf = [conf[i, :H, :W].contiguous() for i, (H, W) in enumerate(true_shape)]

        results.append({
            'pan': panoptic_seg,
            'segments_info': segments_info,
            'conf': conf
        })

    return results

# Copyright (C) 2025-present Naver Corporation. All rights reserved.
@torch.no_grad()
def panoptic_inference_qubo(mask_cls, mask_pred, true_shape, label_mode='sigmoid', temperature=None, device='cuda', num_redo=20, prob_threshold=0.01, silent=False, multi_ar=False):

    if multi_ar:
        for i in range(len(mask_pred)):
            mask_pred[i] = mask_pred[i].sigmoid().to(device)
            mask_pred[i] = F.interpolate(mask_pred[i], size=true_shape[i].tolist(), mode="bilinear", align_corners=False)
        mask_pred = torch.nested.nested_tensor(mask_pred)
        mask_pred = mask_pred.to_padded_tensor(0.).transpose(0, 1)
    else:
        mask_pred = mask_pred.sigmoid().to(device)
        mask_pred = F.interpolate(mask_pred, size=true_shape, mode="bilinear", align_corners=False)
        mask_pred = torch.stack(mask_pred)

    if device is not None:
        mask_cls = mask_cls.to(device)

    results = []
    for bi in range(mask_cls.shape[0]):
        mask_cls_i = mask_cls[bi]
        mask_pred_i = mask_pred[bi].transpose(0,1)

        if label_mode == 'sigmoid':
            mask_cls_i = mask_cls_i.sigmoid()

            # temperature softmax scaling, make the class prediction sharper
            if temperature is not None:
                T = temperature
                mask_cls_i = F.softmax(mask_cls_i.sigmoid() / T, dim=-1)
        else:
            mask_cls_i = F.softmax(mask_cls_i, dim=-1)

        if label_mode == 'softmax':
            cur_mask_cls = cur_mask_cls[:, :-1]

        # Optimization done on CPU
        mask_pred_i = mask_pred_i.cpu()
        mask_cls_i = mask_cls_i.cpu()

        # Use optimization to maximize output coverage
        masks, W = weight_from_masks(mask_pred_i, mask_cls_i, silent=silent)

        solution, obj_val = solve_qubo_simulated_annealing(W, redo=num_redo, silent=silent)

        # Parse results to get the panoptic segmentation
        solution = torch.from_numpy(solution).bool()
        n_instances = solution.sum()
        cls_probs, cls_ids = mask_cls_i[solution].max(dim=1)

        conf, instance_ids = masks[solution].max(dim=0)
        true_instances = torch.unique(instance_ids)

        panoptic_seg = torch.zeros_like(instance_ids)
        new_inst_id = 1
        segments_info = []
        for inst_id in true_instances:
            cls_id = cls_ids[inst_id]
            cls_prob = cls_probs[inst_id]
            mask_conf = conf[instance_ids == inst_id].mean()

            if cls_prob * mask_conf < prob_threshold:
                continue
            panoptic_seg[instance_ids == inst_id] = new_inst_id

            segments_info.append({
                'id': new_inst_id,
                'query_id': inst_id.item(),
                'class_prob': cls_prob.item(),
                'mask_conf': mask_conf.item(),
                'category_id': cls_id,
                'area': (instance_ids == inst_id).sum().item(),
            })
            new_inst_id += 1


        if multi_ar:
            panoptic_seg = [panoptic_seg[i, :H, :W].contiguous() for i, (H, W) in enumerate(true_shape)]
            conf = [conf[i, :H, :W].contiguous() for i, (H, W) in enumerate(true_shape)]

        results.append({
            'pan': panoptic_seg,
            'segments_info': segments_info,
            'conf': conf
        })

    return results

def random_weights(N=200):
    # Create a random symmetric weight matrix W:
    # Off-diagonals are nonnegative, diagonal entries are strictly negative.
    W = np.random.rand(N, N)
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, -np.abs(np.diag(W)))  # Set diagonal entries to be strictly negative
    return W

def weight_from_masks(masks, cls_probs, min_cls_prob=0.0, penalty=1, cutoff=0, prob_weighted=False, silent=False):
    # min_cls_prob .. keep at 0. Thresholds by class probability
    # penalty .. weight for overlaps (range 0 - 2)
    # cutoff .. minimum overlap to consider (leave at 0)
    n_masks, n_imgs, im_height, im_width = masks.shape

    if prob_weighted:
        cls_max_probs, _ = cls_probs.max(dim=1)
        # N, V, H, W ---- N
        masks = masks * cls_max_probs.view(-1, 1, 1, 1)

    # remove bad masks
    bad_masks = (cls_probs < min_cls_prob).all(dim=1)
    # print(f'Removing {bad_masks.sum()} bad masks')
    masks[bad_masks] = 0 # set to 0

    # diagonal weight = area of the mask
    W = torch.diag(masks.reshape(n_masks,-1).sum(dim=1))

    # off-diagonal elements = overlaps
    for i in tqdm(range(1, n_masks), desc='[QUBO] Computing weights', disable=silent):
        mask_i = masks[i].flatten()[None]
        mask_rest = masks[:i].flatten(1)
        overlap = torch.minimum(mask_i, mask_rest).sum(dim=1)
        overlap = F.threshold(overlap, cutoff, 0, inplace=True)
        W[i, :i] = W[:i, i] = -(1+penalty) * overlap / 2

    # normalize weights:
    W /= (im_height * im_width) # independent of image size
    W /= n_imgs # independent of the number of images

    return masks, -W.numpy()

def energy(alpha, W, lambda_reg):
    """Compute the energy E = alpha^T W alpha."""
    return alpha.dot(W).dot(alpha) + lambda_reg * alpha.mean()

def solve_qubo_simulated_annealing(W, num_iters=10000, T0=0.5, T_end=1e-4, lambda_reg=1e-3, redo=20, random_init=True, silent=False):
    """
    Solve the QUBO problem using simulated annealing.

    Parameters:
      - W: numpy array of shape (N, N), the weight matrix.
      - num_iters: number of iterations.
      - T0: initial temperature. Range (0.1, 10, log)
      - Tend: final temperature. Range (1e-4, 1e-2)

    Returns:
      - best_x: best found binary solution.
      - best_energy: energy of the best solution.
    """
    # cooling_rate: multiplicative factor to cool the temperature.
    cooling_rate = (T_end / T0) ** (1/num_iters)

    N = W.shape[0]
    best_x2, best_energy2 = None, float('inf')
    for _ in tqdm(range(redo), desc='[QUBO] Optimizing', disable=silent):

        # Initialize a random binary vector (0 or 1 for each entry)
        if random_init:
            x = np.random.randint(0, 2, size=N) # TODO: try starting from 0
        else:
            x = np.zeros(N)

        best_x = np.copy(x)
        best_energy = current_energy = energy(x, W, lambda_reg)
        T = T0
        history_energy = []

        for i in range(num_iters):
            # Select a random index to flip
            j = np.random.randint(N)
            new_x = np.copy(x)
            new_x[j] = 1 - new_x[j]  # Flip the bit

            if True: #(i+1) % 100 == 0:
                # compute true exact solution from times to times
                new_energy = energy(new_x, W, lambda_reg)
            # elif new_x[j]:
            #     # went from 0 --> 1
            #     new_energy = current_energy + 2*(W[j] @ new_x) - W[j,j]
            # else:
            #     # went from 1 --> 0
            #     new_energy = current_energy - 2*(W[j] @ x) + W[j,j]

            delta = new_energy - current_energy

            # Accept new solution if energy decreases,
            # or with a probability that depends on the current temperature.
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                x = new_x
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = np.copy(x)
            # history_energy.append(current_energy)

            # Cool the temperature
            T *= cooling_rate

        # pl.plot(history_energy)
        # breakpoint()

        if best_energy < best_energy2:
            best_energy2 = best_energy
            best_x2 = best_x

    return best_x2, best_energy2
