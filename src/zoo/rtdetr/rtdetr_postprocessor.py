# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from ...core import register

def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        quality_score_beta=0.5,
        quality_score_mode="pow",
        quality_residual_tau=0.2,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.quality_score_beta = float(quality_score_beta)
        self.quality_score_mode = str(quality_score_mode)
        self.quality_residual_tau = float(quality_residual_tau)
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor, pad_target_sizes: torch.Tensor = None):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        scale_sizes = pad_target_sizes if pad_target_sizes is not None else orig_target_sizes
        bbox_pred *= scale_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            quality_logits = outputs.get("pred_quality_logits")
            if quality_logits is not None:
                if self.quality_score_mode == "residual":
                    quality_factor = torch.exp(self.quality_residual_tau * torch.tanh(quality_logits))
                else:
                    quality_factor = torch.sigmoid(quality_logits).pow(self.quality_score_beta)
                scores = scores * quality_factor.unsqueeze(-1)
            score_calibration_logits = outputs.get("pred_score_calibration_logits")
            if score_calibration_logits is not None:
                score_factor = torch.exp(self.quality_residual_tau * torch.tanh(score_calibration_logits))
                scores = scores * score_factor
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            if pad_target_sizes is not None:
                orig_w, orig_h = orig_target_sizes[idx].unbind(0)
                center = (box[:, :2] + box[:, 2:]) * 0.5
                keep = (center[:, 0] <= orig_w) & (center[:, 1] <= orig_h)
                lab, box, sco = lab[keep], box[keep], sco[keep]
                box[:, 0::2].clamp_(min=0, max=orig_w)
                box[:, 1::2].clamp_(min=0, max=orig_h)
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 


