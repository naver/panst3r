# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from dust3r.losses import MultiLoss
import numpy as np

from .matcher import HungarianMatcher
from .panoptic import SetCriterion


class PanopticLoss(MultiLoss):
    """Loss for panoptic segmentation."""
    def __init__(
        self,
        dec_layers=6,
        deep_supervision=True,
        class_weight=1.0,
        mask_weight=20.0,
        dice_weight=1.0,
        no_obj_weight=0.1,
        num_points=2048, # Default: 4096
        oversample_ratio=1.0,
        importance_sample_ratio=1.0,
        label_mode="softmax",
    ):
        super().__init__()

        self.label_mode = label_mode

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_obj_weight,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            label_mode=self.label_mode,
        )

    def get_name(self):
        return f'PanopticLoss()'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def _prepare_targets(self, gts, classes, device):
        # prepare targets
        targets = []
        # assert gt1['pan_inst_id'].shape[0] == gt2['pan_inst_id'].shape[0]
        class2id = {c: i for i, c in enumerate(classes)}
        inst_ids = torch.stack([gt['pan_inst_id'] for gt in gts], dim=1)
        cls_ids = torch.stack([gt['pan_cls_id'] for gt in gts], dim=1)
        for b in range(inst_ids.shape[0]):
            labels = []
            masks = []
            inst_ids_b = inst_ids[b]
            cls_ids_b = cls_ids[b]
            class_set = gts[0]['class_set'][b].split(';')

            # Support for multi-dataset: only compute loss on classes that are in the class_set
            output_mask = torch.tensor(np.isin(classes, class_set), device=device)

            n,h,w = inst_ids_b.shape
            for iid in torch.unique(inst_ids_b):
                if iid == 0:
                    continue
                mask = inst_ids_b == iid
                label_all = cls_ids_b[mask]
                assert (label_all == label_all[0]).all(), f"Error, different classes in the same instance ID={iid}"
                # Get global class ID (dataset-independent)
                class_name = class_set[label_all[0]]
                labels.append(class2id[class_name])
                masks.append(mask)

            if len(labels) == 0:
                # No instances in the image pair
                targets.append({
                        "labels": torch.zeros(0, device=device, dtype=torch.long),
                        "output_mask": output_mask,
                        "masks": torch.zeros(0, n, h, w, device=device),
                    })
            else:
                targets.append({
                        "labels": torch.tensor(labels, device=device).long(),
                        "output_mask": output_mask,
                        "masks": torch.stack(masks).to(device),
                    })

        return targets

    def compute_loss(self, gts, preds, classes, **kw):
        targets = self._prepare_targets(gts, classes, preds['pred_logits'].device)

        losses = self.criterion(preds, targets)
        total_loss = 0.
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                total_loss += losses[k] * self.criterion.weight_dict[k]
                losses[k] = losses[k].item()
            else:
                # remove this loss if not specified in `weight_dict`
                print("WARNING: no weight specified, ignoring loss", k)
                losses.pop(k)

        return total_loss, {'panoptic_loss': total_loss, **losses}
