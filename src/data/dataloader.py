# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFunction',
    'BatchImagePadCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets


@register()
class BatchImagePadCollateFunction(BaseCollateFunction):
    """Native-resolution single-stream collate with right/bottom padding."""

    def __init__(self, size_divisibility=32, fill=0.0, adjust_normalized_boxes=True, scales=None, stop_epoch=None) -> None:
        super().__init__()
        self.size_divisibility = int(size_divisibility or 1)
        self.fill = float(fill)
        self.adjust_normalized_boxes = bool(adjust_normalized_boxes)

    @staticmethod
    def _round_up(value, divisor):
        if divisor <= 1:
            return int(value)
        return int((int(value) + divisor - 1) // divisor * divisor)

    def __call__(self, items):
        images = [x[0] for x in items]
        targets = [x[1] for x in items]
        shapes = [tuple(image.shape[-2:]) for image in images]
        max_h = self._round_up(max(h for h, _ in shapes), self.size_divisibility)
        max_w = self._round_up(max(w for _, w in shapes), self.size_divisibility)

        first = images[0]
        batch = first.new_full((len(images), first.shape[0], max_h, max_w), self.fill)
        for idx, image in enumerate(images):
            h, w = image.shape[-2:]
            batch[idx, :, :h, :w] = image

        for target, (h, w) in zip(targets, shapes):
            target['pad_size'] = torch.as_tensor([max_w, max_h], dtype=torch.int64)
            target['valid_size'] = torch.as_tensor([w, h], dtype=torch.int64)
            if self.adjust_normalized_boxes and 'boxes' in target and target['boxes'].numel() > 0:
                scale = target['boxes'].new_tensor([w / max_w, h / max_h, w / max_w, h / max_h])
                target['boxes'] = target['boxes'] * scale

        return batch, targets
