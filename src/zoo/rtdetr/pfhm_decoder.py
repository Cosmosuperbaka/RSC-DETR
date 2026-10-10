# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""RT-DETR decoder with final PFHM prototype supervision.

The decoder returns matched-query features to PFHM during training.
The underlying RT-DETR detection parameters and outputs are preserved.
"""


import torch

import torch.nn as nn

import torch.nn.functional as F

from .rtdetrv2_decoder import RTDETRTransformerv2, TransformerDecoder, inverse_sigmoid

from .denoising import get_contrastive_denoising_training_group

from .pfhm import PFHM

from src.core import register

class PFHMTransformerDecoder(TransformerDecoder):
    """Same as TransformerDecoder but additionally returns final query features.
    
    No __init__ needed — fully inherits parent. Only forward is overridden.
    """
    
    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        """Identical to parent forward, but returns (bboxes, logits, final_output)."""
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        output = target
        
        final_output = None  # capture last-layer query features
        
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, memory, memory_spatial_shapes,
                           attn_mask, memory_mask, query_pos_embed)
            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            
            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                final_output = output  # training: last iter's output = last layer
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                final_output = output  # eval: take output at eval_idx
                break
            
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
        
        return (torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                final_output)


@register()
class PFHMDecoder(RTDETRTransformerv2):
    """Thin RT-DETR adapter for the clean, dataset-agnostic PFHM module."""

    def __init__(self, annotation_file, category_ids, x1_num_classes=None, fine_temperature=8.0,
                 fine_learnable_temperature=True, x1_query_residual_adapter=False,
                 x1_qra_bottleneck_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.decoder.__class__ = PFHMTransformerDecoder
        x1_num_classes = len(category_ids) if x1_num_classes is None else int(x1_num_classes)
        self.x1 = PFHM(
            annotation_file=annotation_file,
            num_classes=x1_num_classes,
            hidden_dim=self.hidden_dim,
            category_ids=category_ids,
            temperature=fine_temperature,
            learnable_temperature=fine_learnable_temperature,
        )
        self.x1_query_residual_adapter = bool(x1_query_residual_adapter)
        if self.x1_query_residual_adapter:
            bottleneck = max(1, self.hidden_dim // int(x1_qra_bottleneck_ratio))
            self.x1_qra = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, self.hidden_dim),
            )
            nn.init.zeros_(self.x1_qra[-1].weight)
            nn.init.zeros_(self.x1_qra[-1].bias)
        else:
            self.x1_qra = None

    def _x1_query_features(self, query_features):
        if self.x1_qra is None:
            return query_features
        # QRA: PFHM sees a detached query base plus an adapter path.  This blocks
        # PFHM's direct gradient to the main decoder query while still allowing PFHM
        # to train the adapter from the original, non-detached query features.
        return query_features.detach() + self.x1_qra(query_features)

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                None, None, None, None

        content, ref_points, enc_boxes, enc_logits = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        out_boxes, out_logits, query_features = self.decoder(
            content, ref_points, memory, spatial_shapes,
            self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_boxes, out_boxes = torch.split(out_boxes, dn_meta['dn_num_split'], dim=2)
            dn_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            query_features = query_features[:, dn_meta['dn_num_split'][0]:, :]

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_boxes[-1]}
        out.update(self.x1(self._x1_query_features(query_features)))
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_boxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_logits, enc_boxes)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_logits, dn_boxes)
                out['dn_meta'] = dn_meta
        return out


