# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from .coco_dataset import CocoDetection, mscoco_category2name, mscoco_category2label, mscoco_label2category
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .vedai_dataset import VEDAIDetection
from .m3fd_dataset import M3FDDetection
