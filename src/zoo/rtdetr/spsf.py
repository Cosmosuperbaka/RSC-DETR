# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Parameter-efficient symmetric RGB/IR fusion probes.

These modules are registered for explicit configs only.  They do not alter the
default concat fusion path.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from ...core import register
from .concat_fusion import _ConcatFusionLayer

def _conv_bn_act(in_ch: int, out_ch: int, kernel_size: int=1, groups: int=1) -> nn.Sequential:
    pad = kernel_size // 2
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, groups=groups, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))

class _SPSFLayer(nn.Module):

    def __init__(self, channels: int, mid_channels: int, enable_diagnostics: bool=False):
        super().__init__()
        self.enable_diagnostics = bool(enable_diagnostics)
        self.shared = _conv_bn_act(channels, mid_channels, 1)
        self.common_dw = _conv_bn_act(mid_channels, mid_channels, 3, groups=mid_channels)
        self.common_out = _conv_bn_act(mid_channels, channels, 1)
        self.private_act = nn.GELU()
        self.private_dw = _conv_bn_act(mid_channels, mid_channels, 3, groups=mid_channels)
        self.private_out = _conv_bn_act(mid_channels, channels, 1)
        self.last_common_norm: Optional[torch.Tensor] = None
        self.last_private_norm: Optional[torch.Tensor] = None

    def _private(self, x: torch.Tensor) -> torch.Tensor:
        return self.private_out(self.private_dw(self.private_act(x)))

    def diagnostics(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for name in ['last_common_norm', 'last_private_norm', 'last_weighted_private_norm', 'last_alpha_mean', 'last_alpha_std', 'last_pr_rms', 'last_nr_rms', 'last_residual_norm', 'last_gain_mean', 'last_gain_std']:
            value = getattr(self, name, None)
            if value is not None:
                stats[name.removeprefix('last_')] = float(value.detach().cpu())
        if self.last_alpha is not None:
            alpha = self.last_alpha.detach().float()
            stats['alpha_min'] = float(alpha.min().cpu())
            stats['alpha_max'] = float(alpha.max().cpu())
        for prefix in ['shared', 'common_dw', 'common_out', 'private_dw', 'private_out', 'private_alpha']:
            total = 0.0
            for (name, param) in self.named_parameters():
                if name == prefix or name.startswith(prefix + '.'):
                    if param.grad is not None:
                        total += float(param.grad.detach().float().norm().cpu())
            stats[f'{prefix}_grad_norm'] = total
        return stats

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        pr = self.shared(rgb)
        pi = self.shared(ir)
        u = 0.5 * (pr + pi)
        er = pr - u
        ei = pi - u
        common = self.common_out(self.common_dw(u))
        private = 0.5 * (self._private(er) + self._private(ei))
        if self.enable_diagnostics:
            self.last_common_norm = common.detach().float().norm()
            self.last_private_norm = private.detach().float().norm()
        return common + private

class _SPSFFixedAlphaLayer(_SPSFLayer):
    """Fixed static private residual strength: F = B + alpha_fixed * C."""

    def __init__(self, channels: int, mid_channels: int, alpha_fixed: float=0.5, enable_diagnostics: bool=False):
        super().__init__(channels, mid_channels, enable_diagnostics=enable_diagnostics)
        self.alpha_fixed = float(alpha_fixed)
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_weighted_private_norm: Optional[torch.Tensor] = None

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        pr = self.shared(rgb)
        pi = self.shared(ir)
        u = 0.5 * (pr + pi)
        er = pr - u
        ei = pi - u
        common = self.common_out(self.common_dw(u))
        private = 0.5 * (self._private(er) + self._private(ei))
        weighted = self.alpha_fixed * private
        if self.enable_diagnostics:
            self.last_common_norm = common.detach().float().norm()
            self.last_private_norm = private.detach().float().norm()
            self.last_weighted_private_norm = weighted.detach().float().norm()
            self.last_alpha = private.detach().new_tensor(self.alpha_fixed).float()
        return common + weighted

def fusion_param_count_formula(kind: str, channels: int, mid: int) -> int:
    """Parameter count for one scale, including Conv and BatchNorm affine."""
    c = int(channels)
    m = int(mid)
    if kind == 'peconcat':
        return 3 * c * m + 13 * m + 2 * c
    if kind == 'pesf_s':
        return 2 * c * m + 13 * m + 2 * c
    if kind == 'pesf_ad':
        return 3 * c * m + 24 * m + 4 * c
    if kind == 'spsf':
        return 3 * c * m + 24 * m + 4 * c
    if kind == 'spsf_common':
        return 2 * c * m + 13 * m + 2 * c
    if kind == 'spsf_private_1x1':
        return 3 * c * m + 16 * m + 4 * c
    raise ValueError(kind)

def _nearest_width_for_budget(kind: str, channels: int, target: int) -> int:
    best_m = 1
    best_err = None
    for m in range(1, channels * 2 + 1):
        params = fusion_param_count_formula(kind, channels, m)
        err = abs(params - target)
        if best_err is None or err < best_err:
            best_m = m
            best_err = err
    return best_m

def solve_pesf_widths(in_channels: List[int], base_ratio: float=0.5) -> Dict[str, Dict[str, List[int] | int | float]]:
    """Solve per-scale widths against PE-Concat C/2 budget.

    The budget B0 is defined by PE-Concat with mid ~= C/2 at each scale.  PESF-S
    and PESF-AD widths are chosen per scale to match the same per-scale budget
    as closely as integer channel counts allow.
    """
    base_mid = [max(1, int(round(c * base_ratio))) for c in in_channels]
    base_budget = [fusion_param_count_formula('peconcat', c, m) for (c, m) in zip(in_channels, base_mid)]
    out: Dict[str, Dict[str, List[int] | int | float]] = {}
    for kind in ['peconcat', 'pesf_s', 'pesf_ad', 'spsf']:
        mids = base_mid if kind == 'peconcat' else [_nearest_width_for_budget(kind, c, b) for (c, b) in zip(in_channels, base_budget)]
        params = [fusion_param_count_formula(kind, c, m) for (c, m) in zip(in_channels, mids)]
        total = int(sum(params))
        target = int(sum(base_budget))
        out[kind] = {'mid_channels': mids, 'per_scale_params': params, 'total_params': total, 'target_params': target, 'relative_error': (total - target) / max(1, target)}
    return out

def _normalize_scales(scales: List[int | str]) -> List[int]:
    mapping = {'s3': 0, 'c3': 0, '0': 0, 's4': 1, 'c4': 1, '1': 1, 's5': 2, 'c5': 2, '2': 2}
    out: List[int] = []
    for scale in scales:
        if isinstance(scale, int):
            idx = scale
        else:
            key = str(scale).strip().lower()
            if key not in mapping:
                raise ValueError(f'Unknown SPSF scale {scale!r}; expected S3/S4/S5 or 0/1/2')
            idx = mapping[key]
        if idx < 0 or idx > 2:
            raise ValueError(f'SPSF scale index out of range: {idx}')
        out.append(idx)
    return sorted(set(out))

class _StaticSPSFSingleScaleBase(nn.Module):
    layer_cls = _SPSFLayer

    def __init__(self, in_channels: List[int]=[512, 1024, 2048], base_ratio: float=0.75, enabled_scales: Optional[List[int | str]]=None, enable_diagnostics: bool=False, **layer_kwargs):
        super().__init__()
        if enabled_scales is None:
            enabled_scales = ['S5']
        solved = solve_pesf_widths(in_channels, base_ratio)
        mids = list(solved['spsf']['mid_channels'])
        self.in_channels = list(in_channels)
        self.out_channels = list(in_channels)
        self.mid_channels = mids
        self.enabled_scales = _normalize_scales(enabled_scales)
        self.enable_diagnostics = bool(enable_diagnostics)
        self.fusions = nn.ModuleList()
        for (idx, (channels, mid)) in enumerate(zip(in_channels, mids)):
            if idx in self.enabled_scales:
                self.fusions.append(self.layer_cls(channels, mid, enable_diagnostics=enable_diagnostics, **layer_kwargs))
            else:
                self.fusions.append(_ConcatFusionLayer(channels))

    def forward(self, rgb_feats: List[torch.Tensor], ir_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(rgb_feats) == len(ir_feats) == len(self.fusions)
        return [f(r, i) for (f, r, i) in zip(self.fusions, rgb_feats, ir_feats)]

    def set_epoch(self, epoch: int) -> None:
        for layer in self.fusions:
            setter = getattr(layer, 'set_epoch', None)
            if setter is not None:
                setter(epoch)

@register()
class SPSF(_StaticSPSFSingleScaleBase):
    layer_cls = _SPSFFixedAlphaLayer
