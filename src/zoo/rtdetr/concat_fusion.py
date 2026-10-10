# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Simple RGB-IR feature concatenation fusion for RT-DETR.

Each backbone level keeps the downstream channel count unchanged:
RGB/IR concat -> 1x1 compression -> 3x3 refinement.
"""


import torch

import torch.nn as nn

class _ConcatFusionLayer(nn.Module):
    def __init__(self, channels: int, mid_channels: int = None):
        super().__init__()
        mid = mid_channels if mid_channels is not None else channels
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([rgb, ir], dim=1))


