# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Baseline RT-DETR criterion with clean ACRHM class reliability."""


import torch

import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from .rtdetrv2_criterion import RTDETRCriterionv2

from ...core import register

@register()
class ACRHMCriterion(RTDETRCriterionv2):
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, *args, use_x3_classification_weight=True,
                 use_x3_localization_weight=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.matcher, 'x3'):
            raise TypeError('ACRHMCriterion requires ACRHMMatcher')
        self.x3 = self.matcher.x3
        self.use_x3_classification_weight = bool(use_x3_classification_weight)
        self.use_x3_localization_weight = bool(use_x3_localization_weight)
        print(
            f'[ACRHMCriterion] use_x3_classification_weight='
            f'{self.use_x3_classification_weight} '
            f'use_x3_localization_weight={self.use_x3_localization_weight}'
        )

    def set_total_updates(self, total_updates: int) -> None:
        self.x3.set_total_updates(total_updates)

    def forward(self, outputs, targets, **kwargs):
        self.matcher.begin_step()
        return super().forward(outputs, targets, **kwargs)

    def _matched_role_weights(self, targets, indices, device, localization=False):
        labels = torch.cat([
            target['labels'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]).to(device)
        if labels.numel() == 0:
            return labels, None
        classes = self.x3.map_labels(labels)
        source = (
            self.x3.localization_weights(device)
            if localization else self.x3.classification_weights(device)
        )
        return labels, source[classes].detach()

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([
                target['boxes'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values
        src_logits = outputs['pred_logits']
        labels, weights = self._matched_role_weights(targets, indices, src_logits.device)
        if not self.use_x3_classification_weight:
            weights = None
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = labels
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = src_logits.sigmoid().detach()
        focal_weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        raw = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=focal_weight, reduction='none')
        if weights is not None:
            batch_indices, query_indices = idx
            raw = raw.clone()
            raw[batch_indices, query_indices, labels] *= weights.to(raw.dtype)
        return {'loss_vfl': raw.mean(1).sum() * src_logits.shape[1] / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            target['boxes'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0)
        _, weights = self._matched_role_weights(
            targets, indices, src_boxes.device, localization=True)
        if not self.use_x3_localization_weight:
            weights = None
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(-1)
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        if boxes_weight is not None:
            loss_giou *= boxes_weight
        if weights is not None:
            loss_bbox *= weights.to(loss_bbox.dtype)
            loss_giou *= weights.to(loss_giou.dtype)
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }


