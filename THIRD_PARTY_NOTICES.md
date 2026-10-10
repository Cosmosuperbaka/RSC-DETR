# Third-party source notices

The following upstream projects are identified by copyright notices or source links in the retained code. Original notices are preserved. Modified Python files carry a packaging modification comment, and the transformation record is in [the final version audit](docs/FINAL_VERSION_AUDIT.json).

| Upstream | Source provenance | License copy |
| --- | --- | --- |
| [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) | Backbone, detector, training infrastructure and related utilities | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [DETR](https://github.com/facebookresearch/detr) | Matching, box operations, logging, distributed and training utilities identified by source-file notices | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [torchvision](https://github.com/pytorch/vision) | COCO dataset/evaluation helpers and utilities identified by source-file links | [BSD-3-Clause](LICENSES/torchvision-BSD-3-Clause.txt) |
| [pytorch-image-models / timm](https://github.com/huggingface/pytorch-image-models) | Model EMA implementation attributed in `src/optim/ema.py` | [Apache-2.0](LICENSES/Apache-2.0.txt) |

Upstream license references: [RT-DETR](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE), [DETR](https://github.com/facebookresearch/detr/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE), [timm](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE).

The Apache-2.0 text is the standard license text. The torchvision notice is copied from the installed torchvision 0.16.2 distribution. Python dependencies listed in `requirements.txt` are installed separately; their full source distributions are not included in this package.

The original local source did not identify an upstream commit for every copied file. These notices record available source provenance; they do not claim a complete historical source audit. Licensing status for RSC-DETR-specific contributions is described in [LICENSE](LICENSE).
