# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Clean Hungarian matcher adapter for ACRHM."""


from typing import Dict

import torch

import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from .matcher import HungarianMatcher

from .acrhm import ACRHM

from ...core import register

@register()
class ACRHMMatcher(HungarianMatcher):
    """Baseline Hungarian matching with ACRHM on class cost only."""

    def __init__(self, *args, annotation_file, category_ids, total_updates=None,
                 ema_momentum=0.93, observer_only=False, reliability_matcher_alpha=0.0,
                 role_strength_scale=1.0, delayed_start_ratio=None, delayed_ramp_ratio=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.x3 = ACRHM(
            annotation_file=annotation_file,
            category_ids=category_ids,
            total_updates=total_updates,
            ema_momentum=ema_momentum,
            role_strength_scale=role_strength_scale,
            delayed_start_ratio=delayed_start_ratio,
            delayed_ramp_ratio=delayed_ramp_ratio,
        )
        self._update_armed = False
        self.observer_only = bool(observer_only)
        self.reliability_matcher_alpha = float(reliability_matcher_alpha)
        self.external_reliability_weights = None
        self.pro2_external_class_cost_scale = None
        self.pro2_last_external_scale_mean = 1.0
        self.pro2_last_external_scale_min = 1.0
        self.pro2_last_external_scale_max = 1.0
        self.pro2_last_external_mean_one_error = 0.0
        self.pro2_disable_baseline_matching = False
        self.pro2_baseline_matching_called = False
        # These are training-only observations consumed by the autonomous
        # PFHM/ACRHM balancer.  They describe what ACRHM changed relative to the
        # baseline assignment on the same batch.
        self.last_assignment_agreement = 1.0
        self.last_assignment_switch = 0.0
        self.last_weighted_iou = 0.0
        self.last_baseline_iou = 0.0
        self.last_iou_delta = 0.0
        self.last_matched_count = 0.0
        self.last_weighted_iou_lt_050 = 0.0
        self.last_weighted_iou_lt_075 = 0.0
        self.last_weighted_iou_mean = 0.0
        self.last_weighted_iou_p25 = 0.0
        self.last_weighted_iou_p50 = 0.0
        self.last_weighted_iou_p75 = 0.0
        self.last_baseline_iou_mean = 0.0

    def begin_step(self):
        self._update_armed = True

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs['pred_logits'].flatten(0, 1))
        else:
            out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([target['labels'] for target in targets])
        tgt_bbox = torch.cat([target['boxes'] for target in targets])

        if self.use_focal_loss:
            selected = out_prob[:, tgt_ids]
            negative = (1 - self.alpha) * selected.pow(self.gamma) * (-(1 - selected + 1e-8).log())
            positive = self.alpha * (1 - selected).pow(self.gamma) * (-(selected + 1e-8).log())
            cost_class = positive - negative
        else:
            cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        class_indices = self.x3.map_labels(tgt_ids)
        if self.external_reliability_weights is not None:
            column_weights = self.external_reliability_weights.to(tgt_ids.device)[class_indices]
        elif self.reliability_matcher_alpha > 0:
            diag = self.x3.diagnostics()
            need = diag["difficulty"].to(tgt_ids.device).float()
            need = (need / need.mean().clamp(min=1e-6)).clamp(.25, 4.0)
            iou = diag["iou"].to(tgt_ids.device).float().clamp(.05, 1.0)
            reliability = (iou / iou.mean().clamp(min=1e-6)).clamp(.25, 1.5)
            residual = (need * reliability).clamp(0.0, 1.0)
            column_weights = (1.0 + self.reliability_matcher_alpha * residual).clamp(.95, 1.05)
            column_weights = column_weights / column_weights.mean().clamp(min=1e-6)
            column_weights = column_weights[class_indices]
        else:
            column_weights = self.x3.matcher_weights(tgt_ids.device)[class_indices]
        if self.pro2_external_class_cost_scale is not None:
            external = self.pro2_external_class_cost_scale.detach().to(
                device=tgt_ids.device, dtype=column_weights.dtype
            )
            external_columns = external[class_indices]
            if torch.isfinite(external_columns).all():
                mean = external_columns.mean().clamp(min=1.0e-6)
                external_columns = external_columns / mean
                column_weights = column_weights * external_columns
                self.pro2_last_external_scale_mean = float(external_columns.mean().item())
                self.pro2_last_external_scale_min = float(external_columns.min().item())
                self.pro2_last_external_scale_max = float(external_columns.max().item())
                self.pro2_last_external_mean_one_error = float((external_columns.mean() - 1.0).abs().item())
            else:
                self.pro2_last_external_scale_mean = 1.0
                self.pro2_last_external_scale_min = 1.0
                self.pro2_last_external_scale_max = 1.0
                self.pro2_last_external_mean_one_error = 0.0
        baseline_cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        weighted_cost = (
            self.cost_class * cost_class * column_weights.unsqueeze(0)
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        sizes = [len(target['boxes']) for target in targets]
        weighted_chunks = weighted_cost.view(bs, num_queries, -1).cpu().split(sizes, -1)
        weighted_indices = [linear_sum_assignment(item[i]) for i, item in enumerate(weighted_chunks)]
        if self.pro2_disable_baseline_matching:
            baseline_indices = weighted_indices
            self.pro2_baseline_matching_called = False
        else:
            baseline_chunks = baseline_cost.view(bs, num_queries, -1).cpu().split(sizes, -1)
            baseline_indices = [linear_sum_assignment(item[i]) for i, item in enumerate(baseline_chunks)]
            self.pro2_baseline_matching_called = True

        if self._update_armed and tgt_ids.numel() > 0:
            prob_by_image = out_prob.view(bs, num_queries, -1)
            class_cost_by_image = cost_class.view(bs, num_queries, -1)
            giou_by_image = giou.view(bs, num_queries, -1)
            iou_by_image = iou.view(bs, num_queries, -1)
            baseline_by_image = baseline_cost.view(bs, num_queries, -1)
            observations = {key: [] for key in (
                'labels', 'wrong', 'difficulty', 'iou', 'margin',
                'class_gap', 'full_gap', 'switch', 'weighted_iou',
                'baseline_iou')}
            offset = 0
            for image_index, target in enumerate(targets):
                labels = target['labels'].to(tgt_ids.device)
                count = labels.numel()
                if count == 0:
                    continue
                local_columns = torch.arange(count, device=tgt_ids.device)
                probability = prob_by_image[image_index]
                true_probability = probability[:, labels]
                best_query = true_probability.argmax(dim=0)
                best_true = true_probability[best_query, local_columns].clamp(1e-6, 1.0)
                chosen = probability[best_query].clone()
                chosen[local_columns, labels] = -1.0
                wrong_probability, wrong_label = chosen.max(dim=1)
                margin = best_true - wrong_probability

                true_cost = class_cost_by_image[image_index, best_query, offset + local_columns]
                if self.use_focal_loss:
                    p = wrong_probability.clamp(1e-6, 1 - 1e-6)
                    negative = (1 - self.alpha) * p.pow(self.gamma) * (-(1 - p).log())
                    positive = self.alpha * (1 - p).pow(self.gamma) * (-p.log())
                    wrong_cost = positive - negative
                else:
                    wrong_cost = -wrong_probability
                class_gap = wrong_cost - true_cost
                class_gap = class_gap / class_gap.abs().mean().clamp(min=1e-6)

                local_full = baseline_by_image[image_index, :, offset:offset + count]
                two_best = torch.topk(local_full, k=min(2, num_queries), dim=0, largest=False).values
                full_gap = (
                    two_best[1] - two_best[0]
                    if two_best.shape[0] > 1 else torch.ones_like(two_best[0])
                )
                full_gap = full_gap / full_gap.mean().clamp(min=1e-6)
                localization_quality = giou_by_image[
                    image_index, :, offset:offset + count].max(dim=0).values.clamp(0, 1)

                weighted_map = torch.full((count,), -1, device=tgt_ids.device, dtype=torch.long)
                baseline_map = torch.full_like(weighted_map, -1)
                wp, wg = weighted_indices[image_index]
                weighted_map[torch.as_tensor(wg, device=tgt_ids.device)] = torch.as_tensor(wp, device=tgt_ids.device)
                if self.pro2_disable_baseline_matching:
                    baseline_map.copy_(weighted_map)
                else:
                    bp, bg = baseline_indices[image_index]
                    baseline_map[torch.as_tensor(bg, device=tgt_ids.device)] = torch.as_tensor(bp, device=tgt_ids.device)
                switched = (weighted_map != baseline_map).float()
                valid_weighted = weighted_map >= 0
                valid_baseline = baseline_map >= 0
                weighted_iou = torch.zeros_like(switched)
                baseline_iou = torch.zeros_like(switched)
                if bool(valid_weighted.any()):
                    target_positions = torch.arange(count, device=tgt_ids.device)[valid_weighted]
                    weighted_iou[valid_weighted] = iou_by_image[image_index][
                        weighted_map[valid_weighted], target_positions
                    ]
                if bool(valid_baseline.any()):
                    target_positions = torch.arange(count, device=tgt_ids.device)[valid_baseline]
                    baseline_iou[valid_baseline] = iou_by_image[image_index][
                        baseline_map[valid_baseline], target_positions
                    ]
                wrong_internal = torch.full_like(wrong_label, -1)
                for category_id, internal_index in self.x3.category_id_to_index.items():
                    wrong_internal = torch.where(
                        wrong_label == category_id,
                        torch.as_tensor(internal_index, device=wrong_label.device),
                        wrong_internal,
                    )
                observations['labels'].append(labels)
                observations['wrong'].append(wrong_internal)
                observations['difficulty'].append(-best_true.log())
                observations['iou'].append(localization_quality)
                observations['margin'].append(margin)
                observations['class_gap'].append(class_gap.clamp(-2, 2))
                observations['full_gap'].append(full_gap.clamp(0, 2))
                observations['switch'].append(switched)
                observations['weighted_iou'].append(weighted_iou)
                observations['baseline_iou'].append(baseline_iou)
                offset += count
            if observations['labels']:
                cat = lambda key: torch.cat(observations[key])
                switch = cat('switch')
                weighted_iou = cat('weighted_iou')
                baseline_iou = cat('baseline_iou')
                self.x3.update(
                    cat('labels'), cat('difficulty'), cat('iou'),
                    wrong_classes=cat('wrong'),
                    probability_margin=cat('margin'),
                    class_cost_gap=cat('class_gap'),
                    full_cost_gap=cat('full_gap'),
                    assignment_switch=cat('switch'),
                )
                self.last_assignment_switch = float(switch.mean().item())
                self.last_assignment_agreement = 1.0 - self.last_assignment_switch
                self.last_weighted_iou = float(weighted_iou.mean().item())
                self.last_baseline_iou = float(baseline_iou.mean().item())
                self.last_iou_delta = self.last_weighted_iou - self.last_baseline_iou
                self.last_matched_count = float(weighted_iou.numel())
                self.last_weighted_iou_lt_050 = float((weighted_iou < 0.50).float().mean().item())
                self.last_weighted_iou_lt_075 = float((weighted_iou < 0.75).float().mean().item())
                self.last_weighted_iou_mean = self.last_weighted_iou
                self.last_baseline_iou_mean = self.last_baseline_iou
                if weighted_iou.numel() > 0:
                    q = torch.quantile(
                        weighted_iou.detach().float(),
                        weighted_iou.new_tensor([0.25, 0.50, 0.75], dtype=torch.float32),
                    )
                    self.last_weighted_iou_p25 = float(q[0].item())
                    self.last_weighted_iou_p50 = float(q[1].item())
                    self.last_weighted_iou_p75 = float(q[2].item())
            self._update_armed = False

        chosen_indices = baseline_indices if self.observer_only else weighted_indices
        return {'indices': [
            (torch.as_tensor(pred, dtype=torch.int64), torch.as_tensor(gt, dtype=torch.int64))
            for pred, gt in chosen_indices
        ]}


