# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Modified to support both single-stream (Tensor `samples`) and dual-stream
(dict `samples = {'rgb': Tensor, 'ir': Tensor}`) inputs.
"""

import sys
import math
import json
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


# ---------------------------------------------------------------------------
# Helper: move samples (Tensor OR dict-of-Tensors) to device
# ---------------------------------------------------------------------------

def _samples_to_device(samples, device):
    """Move single-stream Tensor or dual-stream dict to device."""
    if isinstance(samples, dict):
        return {k: v.to(device) for k, v in samples.items()}
    return samples.to(device)


# ---------------------------------------------------------------------------

def _inject_teacher_outputs(criterion, outputs, samples, targets):
    teacher = getattr(criterion, 'teacher_model', None)
    if teacher is None:
        return outputs
    was_training = teacher.training
    teacher.eval()
    with torch.no_grad():
        teacher_outputs = teacher(samples, targets=targets)
    if was_training:
        teacher.train()
    outputs['teacher_pred_logits'] = teacher_outputs['pred_logits'].detach()
    outputs['teacher_pred_boxes'] = teacher_outputs['pred_boxes'].detach()
    return outputs


# ---------------------------------------------------------------------------

def _param_bucket(name: str) -> str:
    lowered = name.lower()
    if ".x1." in lowered or lowered.startswith("x1.") or "x1_fine" in lowered or "fine_head" in lowered:
        return "x1"
    if "fusion" in lowered:
        return "fusion"
    if "decoder" in lowered or "encoder" in lowered or "backbone" in lowered:
        return "shared"
    return "other"


def _grad_norms_by_bucket(model: torch.nn.Module) -> dict:
    sq = {"x1": 0.0, "fusion": 0.0, "shared": 0.0, "other": 0.0, "total": 0.0}
    nonfinite = {key: 0.0 for key in sq}
    elems = {key: 0 for key in sq}
    counts = {key: 0 for key in sq}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        safe_grad = torch.where(finite, grad, torch.zeros_like(grad))
        value = float(safe_grad.pow(2).sum().item())
        bucket = _param_bucket(name)
        sq[bucket] += value
        sq["total"] += value
        nonfinite[bucket] += float((~finite).float().sum().item())
        nonfinite["total"] += float((~finite).float().sum().item())
        elems[bucket] += int(grad.numel())
        elems["total"] += int(grad.numel())
        counts[bucket] += 1
        counts["total"] += 1
    stats = {f"runtime_grad_{key}_norm": math.sqrt(value) for key, value in sq.items()}
    stats.update({
        f"runtime_grad_{key}_nonfinite_ratio": nonfinite[key] / max(1, elems[key])
        for key in sq
    })
    stats.update({f"runtime_grad_{key}_params": float(value) for key, value in counts.items()})
    return stats


def _runtime_loss_grad_cosine(model: torch.nn.Module, loss_dict: dict) -> dict:
    x1_loss = loss_dict.get("loss_x1_fine")
    if x1_loss is None or not hasattr(x1_loss, "requires_grad") or not x1_loss.requires_grad:
        return {}
    main_terms = [
        value for key, value in loss_dict.items()
        if key != "loss_x1_fine" and hasattr(value, "requires_grad") and value.requires_grad
    ]
    if not main_terms:
        return {}
    main_loss = sum(main_terms)
    params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and _param_bucket(name) in {"shared", "fusion"}
    ]
    if not params:
        return {}
    gx = torch.autograd.grad(x1_loss, params, retain_graph=True, allow_unused=True)
    gm = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    device = x1_loss.device
    dot = torch.zeros((), device=device)
    nx = torch.zeros((), device=device)
    nm = torch.zeros((), device=device)
    used = 0
    for a, b in zip(gx, gm):
        if a is None or b is None:
            continue
        a = a.detach().float()
        b = b.detach().float()
        dot = dot + (a * b).sum()
        nx = nx + a.pow(2).sum()
        nm = nm + b.pow(2).sum()
        used += 1
    if used == 0:
        return {"runtime_grad_cos_x1_main_shared": 0.0, "runtime_grad_cos_shared_params": 0.0}
    cosine = dot / (nx.sqrt() * nm.sqrt()).clamp(min=1.0e-12)
    return {
        "runtime_grad_cos_x1_main_shared": float(cosine.item()),
        "runtime_grad_x1_shared_norm": float(nx.sqrt().item()),
        "runtime_grad_main_shared_norm": float(nm.sqrt().item()),
        "runtime_grad_cos_shared_params": float(used),
    }


@torch.no_grad()
def _ema_drift(model: torch.nn.Module, ema: ModelEMA) -> dict:
    if ema is None or getattr(ema, "module", None) is None:
        return {}
    ema_params = dict(ema.module.named_parameters())
    sq = {"x1": 0.0, "fusion": 0.0, "shared": 0.0, "other": 0.0, "total": 0.0}
    denom = {"x1": 0.0, "fusion": 0.0, "shared": 0.0, "other": 0.0, "total": 0.0}
    for name, param in model.named_parameters():
        other = ema_params.get(name)
        if other is None:
            continue
        bucket = _param_bucket(name)
        diff = (param.detach().float() - other.detach().float()).pow(2).sum().item()
        base = param.detach().float().pow(2).sum().item()
        sq[bucket] += float(diff)
        sq["total"] += float(diff)
        denom[bucket] += float(base)
        denom["total"] += float(base)
    return {
        f"runtime_ema_{key}_rms_drift": math.sqrt(value / max(denom[key], 1.0e-12))
        for key, value in sq.items()
    }


def _matcher_runtime_stats(criterion: torch.nn.Module) -> dict:
    stats = {}
    for module in criterion.modules():
        for name in (
            "last_assignment_agreement",
            "last_assignment_switch",
            "last_weighted_iou",
            "last_baseline_iou",
            "last_iou_delta",
            "last_matched_count",
            "last_weighted_iou_lt_050",
            "last_weighted_iou_lt_075",
            "last_weighted_iou_mean",
            "last_weighted_iou_p25",
            "last_weighted_iou_p50",
            "last_weighted_iou_p75",
            "last_baseline_iou_mean",
        ):
            if hasattr(module, name):
                stats[f"runtime_matcher_{name}"] = float(getattr(module, name))
    return stats


def _write_runtime_diagnostics(path: str, record: dict) -> None:
    if not path or not dist_utils.is_main_process():
        return
    import pathlib
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _sanitize_runtime_stats(stats: dict) -> dict:
    clean = {}
    flags = {}
    for key, value in stats.items():
        if hasattr(value, "detach"):
            value = value.detach().float().mean().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean[key] = value
        else:
            clean[key] = 0.0
            flags[f"{key}_nonfinite_flag"] = 1.0
    clean.update(flags)
    return clean


def _fine_param_bucket(name: str) -> str:
    lowered = name.lower()
    if "fusion" in lowered and (lowered.endswith(".alpha") or "coord_logit" in lowered):
        return "spsf_alpha"
    if "fusion" in lowered and "common" in lowered:
        return "spsf_common"
    if "fusion" in lowered and "private" in lowered:
        return "spsf_private"
    if "fusion" in lowered and ("out" in lowered or "projection" in lowered):
        return "spsf_output"
    if "fusion" in lowered:
        return "spsf_other"
    if "backbone" in lowered and ("ir" in lowered or "infra" in lowered):
        return "ir_backbone_last"
    if "backbone" in lowered:
        return "rgb_backbone_last"
    if "input_proj" in lowered or "input_projection" in lowered:
        return "encoder_input_projection"
    if "encoder" in lowered:
        return "encoder"
    if "decoder" in lowered:
        return "decoder"
    if "x1" in lowered or "fine_head" in lowered or "x1_fine" in lowered:
        return "x1_fine_head"
    if "class" in lowered or "score" in lowered:
        return "main_class_head"
    if "bbox" in lowered:
        return "bbox_head"
    return "other"


def _grad_norms_by_fine_bucket(model: torch.nn.Module) -> dict:
    buckets = [
        "spsf_alpha", "spsf_common", "spsf_private", "spsf_output", "spsf_other",
        "rgb_backbone_last", "ir_backbone_last", "encoder_input_projection",
        "encoder", "decoder", "x1_fine_head", "main_class_head", "bbox_head", "other",
    ]
    sq = {key: 0.0 for key in buckets}
    psq = {key: 0.0 for key in buckets}
    max_abs = {key: 0.0 for key in buckets}
    nonfinite = {key: 0.0 for key in buckets}
    counts = {key: 0 for key in buckets}
    elems = {key: 0 for key in buckets}
    for name, param in model.named_parameters():
        bucket = _fine_param_bucket(name)
        if bucket not in sq:
            bucket = "other"
        pdata = param.detach().float()
        psq[bucket] += float(pdata.pow(2).sum().item())
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = torch.isfinite(grad)
        sq[bucket] += float(torch.where(finite, grad, torch.zeros_like(grad)).pow(2).sum().item())
        max_abs[bucket] = max(max_abs[bucket], float(torch.where(finite, grad.abs(), torch.zeros_like(grad)).max().item()))
        nonfinite[bucket] += float((~finite).float().sum().item())
        counts[bucket] += 1
        elems[bucket] += int(grad.numel())
    stats = {}
    for key in buckets:
        gnorm = math.sqrt(sq[key])
        pnorm = math.sqrt(psq[key])
        prefix = f"probe_grad_{key}"
        stats[f"{prefix}_l2"] = gnorm
        stats[f"{prefix}_param_l2"] = pnorm
        stats[f"{prefix}_max_abs"] = max_abs[key]
        stats[f"{prefix}_nonfinite_ratio"] = nonfinite[key] / max(1, elems[key])
        stats[f"{prefix}_param_groups"] = float(counts[key])
        stats[f"{prefix}_update_weight_ratio"] = gnorm / max(pnorm, 1.0e-12)
    return stats


def _fusion_probe_stats(model: torch.nn.Module) -> dict:
    module = dist_utils.de_parallel(model)
    fusion = getattr(module, "fusion", None)
    layers = getattr(fusion, "fusions", []) if fusion is not None else []
    stats = {}
    for idx, layer in enumerate(layers):
        prefix = f"probe_fusion_s{idx + 3}"
        for name in (
            "last_common_norm", "last_private_norm", "last_weighted_private_norm",
            "last_alpha", "last_pr_rms", "last_nr_rms", "last_residual_norm",
            "last_private_sparsity", "last_private_spatial_cv",
            "last_common_pair_alpha", "last_pair_common_norm", "last_private_limiter",
            "last_effective_alpha", "last_base_norm", "last_candidate_norm",
            "last_delta_norm", "last_output_delta_norm",
            "last_delta_base_ratio", "last_output_delta_base_ratio",
            "last_concat_base_norm", "last_spsf_delta_norm", "last_coord",
            "last_delta_limiter",
            "last_gate_mean", "last_gate_std", "last_risk", "last_semantic_risk",
            "last_energy_risk", "last_private_risk", "last_strength",
            "last_progress", "last_semantic_cos", "last_private_common_ratio",
        ):
            value = getattr(layer, name, None)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().float().mean().item()
            else:
                value = float(value)
            stats[f"{prefix}_{name}"] = value
        common = stats.get(f"{prefix}_last_common_norm")
        private = stats.get(f"{prefix}_last_private_norm")
        weighted = stats.get(f"{prefix}_last_weighted_private_norm")
        if common is not None and private is not None:
            stats[f"{prefix}_private_common_l2_ratio"] = private / max(common, 1.0e-12)
        if common is not None and weighted is not None:
            stats[f"{prefix}_weighted_private_common_l2_ratio"] = weighted / max(common, 1.0e-12)
    return stats


def _write_online_probe(path: str, record: dict) -> None:
    if not path or not dist_utils.is_main_process():
        return
    import pathlib
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    max_train_batches = kwargs.get('max_train_batches', None)
    runtime_diagnostics = bool(kwargs.get('runtime_diagnostics', False))
    runtime_diagnostics_interval = max(1, int(kwargs.get('runtime_diagnostics_interval', 200)))
    runtime_diagnostics_path = kwargs.get('runtime_diagnostics_path', '')
    online_probe_path = kwargs.get('online_probe_path', '')
    online_probe_interval = max(1, int(kwargs.get('online_probe_interval', 1)))
    accumulation_steps = max(1, int(kwargs.get('gradient_accumulation_steps', 1)))
    planned_batches = len(data_loader)
    if max_train_batches is not None:
        planned_batches = min(planned_batches, int(max_train_batches))
    optimizer.zero_grad()

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if max_train_batches is not None and i >= int(max_train_batches):
            print(f"Stopping train epoch early after {i} batches due to max_train_batches={max_train_batches}")
            break
        samples = _samples_to_device(samples, device)        # ← MODIFIED
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)
        runtime_sample = runtime_diagnostics and global_step % runtime_diagnostics_interval == 0
        online_probe_sample = bool(online_probe_path) and global_step % online_probe_interval == 0
        runtime_stats = {}

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            with torch.autocast(device_type=str(device), enabled=False):
                outputs = _inject_teacher_outputs(criterion, outputs, samples, targets)
                loss_dict = criterion(outputs, targets, **metas)

            if runtime_sample:
                runtime_stats.update(_runtime_loss_grad_cosine(model, loss_dict))
            loss = sum(loss_dict.values()) / accumulation_steps
            scaler.scale(loss).backward()
            do_optimizer_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == planned_batches)
            grads_unscaled = False
            if do_optimizer_step and (runtime_sample or online_probe_sample or max_norm > 0):
                scaler.unscale_(optimizer)
                grads_unscaled = True

            for _mod in model.modules():
                _theta = getattr(_mod, 'theta', None)
                if _theta is not None and hasattr(_mod, 'channel_alpha'):
                    _grad = getattr(_theta, 'grad', None)
                    if _grad is not None:
                        _mod._csspsf_last_theta_grad_norm = _grad.detach().float().norm().item()

            if runtime_sample:
                runtime_stats.update(_grad_norms_by_bucket(model))
                runtime_stats.update(_matcher_runtime_stats(criterion))
            if online_probe_sample:
                probe_stats = {
                    "epoch": float(epoch),
                    "step": float(i),
                    "global_step": float(global_step),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{f"loss_{k}": float(v.detach().float().item()) for k, v in loss_dict.items() if hasattr(v, "detach") and v.numel() == 1},
                    **_matcher_runtime_stats(criterion),
                    **_fusion_probe_stats(model),
                    **_grad_norms_by_fine_bucket(model),
                }
                _write_online_probe(online_probe_path, probe_stats)
            if do_optimizer_step:
                if max_norm > 0:
                    if not grads_unscaled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            outputs = _inject_teacher_outputs(criterion, outputs, samples, targets)
            loss_dict = criterion(outputs, targets, **metas)

            if runtime_sample:
                runtime_stats.update(_runtime_loss_grad_cosine(model, loss_dict))
            loss : torch.Tensor = sum(loss_dict.values()) / accumulation_steps
            loss.backward()

            for _mod in model.modules():
                _theta = getattr(_mod, 'theta', None)
                if _theta is not None and hasattr(_mod, 'channel_alpha'):
                    _grad = getattr(_theta, 'grad', None)
                    if _grad is not None:
                        _mod._csspsf_last_theta_grad_norm = _grad.detach().float().norm().item()

            if runtime_sample:
                runtime_stats.update(_grad_norms_by_bucket(model))
                runtime_stats.update(_matcher_runtime_stats(criterion))
            if online_probe_sample:
                probe_stats = {
                    "epoch": float(epoch),
                    "step": float(i),
                    "global_step": float(global_step),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{f"loss_{k}": float(v.detach().float().item()) for k, v in loss_dict.items() if hasattr(v, "detach") and v.numel() == 1},
                    **_matcher_runtime_stats(criterion),
                    **_fusion_probe_stats(model),
                    **_grad_norms_by_fine_bucket(model),
                }
                _write_online_probe(online_probe_path, probe_stats)
            do_optimizer_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == planned_batches)
            if do_optimizer_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()

        # ema
        if do_optimizer_step and ema is not None:
            ema.update(model)
        if runtime_sample:
            runtime_stats.update(_ema_drift(model, ema))
            runtime_stats.update({
                "epoch": float(epoch),
                "step": float(i),
                "global_step": float(global_step),
            })
            runtime_stats = _sanitize_runtime_stats(runtime_stats)
            metric_logger.update(**runtime_stats)
            _write_runtime_diagnostics(runtime_diagnostics_path, runtime_stats)

        if do_optimizer_step and lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # v10 gate stats logging
        _stats_acc, _stats_cnt = {}, 0
        for _mod in model.modules():
            _s = getattr(_mod, '_gate_stats', None)
            if isinstance(_s, dict):
                _stats_cnt += 1
                for k, v in _s.items():
                    _stats_acc[k] = _stats_acc.get(k, 0.0) + float(v)
        if _stats_cnt > 0:
            metric_logger.update(**{f'g_{k}': _stats_acc[k]/_stats_cnt for k in _stats_acc})

        for _idx, _mod in enumerate(model.modules()):
            if hasattr(_mod, 'theta') and hasattr(_mod, 'channel_alpha'):
                _alpha = _mod.channel_alpha().detach().float().flatten()
                if _alpha.numel() > 0:
                    metric_logger.update(**{
                        f'csspsf_alpha_mean': _alpha.mean().item(),
                        f'csspsf_alpha_std': _alpha.std(unbiased=False).item(),
                        f'csspsf_alpha_min': _alpha.min().item(),
                        f'csspsf_alpha_max': _alpha.max().item(),
                        f'csspsf_theta_grad_norm': float(getattr(_mod, '_csspsf_last_theta_grad_norm', 0.0)),
                    })
            _alpha_param = getattr(_mod, 'alpha', None)
            if _alpha_param is not None:
                _grad = getattr(_alpha_param, 'grad', None)
                metric_logger.update(**{
                    'spsf_residual_alpha': _alpha_param.detach().float().mean().item(),
                    'spsf_residual_alpha_grad_norm': 0.0 if _grad is None else _grad.detach().float().norm().item(),
                })
            _diag_prefix = getattr(_mod, 'spsf_diagnostic_prefix', None)
            if _diag_prefix and hasattr(_mod, 'diagnostics'):
                _diag = _mod.diagnostics()
                _grad_stats = {
                    f'{_diag_prefix}_{k}': float(v)
                    for k, v in _diag.items()
                    if k.endswith('_grad_norm') and isinstance(v, (int, float))
                }
                if _grad_stats:
                    metric_logger.update(**_grad_stats)

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, max_eval_batches=None):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if max_eval_batches is not None and i >= int(max_eval_batches):
            print(f"Stopping evaluation early after {i} batches due to max_eval_batches={max_eval_batches}")
            break
        samples = _samples_to_device(samples, device)        # ← MODIFIED
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        pad_target_sizes = None
        if targets and 'pad_size' in targets[0]:
            pad_target_sizes = torch.stack([t["pad_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes, pad_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
