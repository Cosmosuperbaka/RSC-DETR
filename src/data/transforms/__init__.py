# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from ._transforms import (
    EmptyTransform,
    RandomPhotometricDistort,
    RandomZoomOut,
    RandomIoUCrop,
    RandomHorizontalFlip,
    Resize,
    LetterboxResize,
    PadToSize,
    SanitizeBoundingBoxes,
    RandomCrop,
    Normalize,
    ConvertBoxes,
    ConvertPILImage,
)
from .container import Compose
from .mosaic import Mosaic

# Dual-stream transforms container
from ._transforms_dual import DualCompose

# Domain-targeted augmentations for RGB+IR drone vehicle detection
from ._transforms_dual_aug import (
    RandomVerticalFlip,
    IRChannelNoise,
    IRGlobalBrightness,
    RGBNightSimulation,
    ModalityDropout,
    CrossModalCutOut,
    SmallVehicleCopyPaste,
    RandomRotate90,
)
