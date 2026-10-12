# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""VEDAI dual-stream (RGB + IR) detection dataset.

VEDAI (Vehicules1024) stores both modalities in one flat directory:
    XXXXXXXX_co.png   — RGB (color)
    XXXXXXXX_ir.png   — IR

After x2coco_vedai.py, file_name in the JSON is 'XXXXXXXX_co.png'.
IR path = same directory, '_co.' → '_ir.'.

Categories (8, 1-indexed matching VEDAI class IDs):
    1=car  2=truck  3=pickup  4=tractor  5=camping_car  6=boat  7=plane  8=van
"""
import os
from PIL import Image

from .coco_dataset import CocoDetection
from ...core import register


__all__ = ['VEDAIDetection']


@register()
class VEDAIDetection(CocoDetection):
    """Dual-stream COCO-format dataset for VEDAI.

    img_folder — directory containing both _co.png and _ir.png files.
    ann_file   — COCO JSON produced by x2coco_vedai.py.
    transforms — must be a DualCompose instance.

    __getitem__ returns ((rgb_pil, ir_pil), target).
    """

    __inject__ = ['transforms']
    __share__ = []

    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=False,
        )

    def __getitem__(self, idx):
        rgb_pil, target = self.load_item(idx)

        image_id  = self.ids[idx]
        img_info  = self.coco.loadImgs(image_id)[0]
        rgb_fname = os.path.basename(img_info['file_name'])
        ir_fname  = rgb_fname.replace('_co.', '_ir.')

        ir_path = os.path.join(self.img_folder, ir_fname)
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f'IR image not found: {ir_path}')
        ir_pil = Image.open(ir_path).convert('RGB')

        if self._transforms is not None:
            (rgb_pil, ir_pil), target, _ = self._transforms(
                (rgb_pil, ir_pil), target, self)

        return (rgb_pil, ir_pil), target
