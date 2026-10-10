# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dual-Stream RT-DETR with optional query refiner modules."""


import torch

import torch.nn as nn

import torch.nn.functional as F

from ...core import register

from .s4_s5_relation_preservation import S4S5RelationPreservationLoss

@register()
class RSCDETR(nn.Module):
    __inject__ = ['backbone_rgb', 'backbone_ir', 'fusion', 'encoder', 'decoder', 'maqr', 'qmrm', 'rgb_enhancer', 'damr']

    def __init__(self, backbone_rgb, backbone_ir, fusion, encoder, decoder,
                 maqr=None, qmrm=None, rgb_enhancer=None, damr=None,
                 modal_dropout_p: float = 0.0, cmcl_weight: float = 0.0,
                 s4_s5_relation_preservation: bool = False,
                 s4_s5_relation_roi_size: int = 3,
                 s4_s5_relation_max_objects: int = 128,
                 s4_s5_relation_sample_seed: int = 3407):
        super().__init__()
        self.backbone_rgb    = backbone_rgb
        self.backbone_ir     = backbone_ir
        self.fusion          = fusion
        self.encoder         = encoder
        self.decoder         = decoder
        self.maqr            = maqr
        self.qmrm            = qmrm
        self.damr            = damr
        self.rgb_enhancer    = rgb_enhancer
        self.modal_dropout_p = modal_dropout_p  # prob to zero IR stream each iter (Modal Dropout probe)
        self.cmcl_weight     = cmcl_weight      # weight for cross-modal consistency loss (CMCL probe)
        self.s4_s5_relation = (
            S4S5RelationPreservationLoss(
                roi_size=s4_s5_relation_roi_size,
                max_objects=s4_s5_relation_max_objects,
                sample_seed=s4_s5_relation_sample_seed,
            )
            if s4_s5_relation_preservation else None
        )

    @staticmethod
    def _feats_to_memory(feats, input_proj):
        """backbone feats [512,1024,2048] -> (B, SUM_HW, 256) via encoder input_proj"""
        parts = []
        for i, feat in enumerate(feats):
            proj = input_proj[i](feat)   # ConvNorm: C -> hidden_dim
            parts.append(proj.flatten(2).permute(0, 2, 1))
        return torch.cat(parts, dim=1)

    def forward(self, samples, targets=None):
        if isinstance(samples, dict):
            rgb, ir = samples['rgb'], samples['ir']
        else:
            rgb = ir = samples

        rgb_feats = self.backbone_rgb(rgb)
        if self.rgb_enhancer is not None:
            rgb_feats = self.rgb_enhancer(rgb_feats, rgb)
        ir_feats = self.backbone_ir(ir)
        # Modal Dropout: zero out IR stream with probability p during training
        if self.training and self.modal_dropout_p > 0 and torch.rand(1).item() < self.modal_dropout_p:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]
        fused = self.fusion(rgb_feats, ir_feats)
        relation_loss = None
        relation_stats = None
        if self.training and self.s4_s5_relation is not None and targets is not None:
            relation_loss = self.s4_s5_relation(
                fused[1],
                fused[2],
                targets,
                image_hw=(rgb.shape[-2], rgb.shape[-1]),
            )
            relation_stats = dict(self.s4_s5_relation.last_stats)
        enc_feats = self.encoder(fused)

        if self.maqr is not None or self.qmrm is not None or self.damr is not None:
            dec = self.decoder
            rgb_memory     = self._feats_to_memory(rgb_feats, self.encoder.input_proj)
            ir_memory      = self._feats_to_memory(ir_feats,  self.encoder.input_proj)
            spatial_shapes = [(f.shape[2], f.shape[3]) for f in rgb_feats]

            memory, enc_spatial_shapes = dec._get_encoder_input(enc_feats)

            if dec.training and dec.num_denoising > 0:
                from .denoising import get_contrastive_denoising_training_group
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(
                        targets, dec.num_classes, dec.num_queries,
                        dec.denoising_class_embed,
                        num_denoising=dec.num_denoising,
                        label_noise_ratio=dec.label_noise_ratio,
                        box_noise_scale=dec.box_noise_scale)
            else:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    None, None, None, None

            content, ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
                dec._get_decoder_input(memory, enc_spatial_shapes,
                                       denoising_logits, denoising_bbox_unact)

            if dn_meta is not None:
                dn_num      = dn_meta['dn_num_split'][0]
                dn_content  = content[:, :dn_num]
                det_content = content[:, dn_num:]
                det_ref     = ref_points_unact[:, dn_num:]
            else:
                dn_content  = None
                det_content = content
                det_ref     = ref_points_unact

            # MAQR: modifies content queries before the decoder runs
            if self.maqr is not None:
                det_content = self.maqr(
                    content        = det_content,
	                    ref_points     = F.sigmoid(det_ref),
	                    rgb_memory     = rgb_memory,
	                    ir_memory      = ir_memory,
	                    spatial_shapes = spatial_shapes,
                )

            # QMRM: light query-level RGB/IR reliability gate.
            if self.qmrm is not None:
                det_content = self.qmrm(
                    content        = det_content,
                    ref_points     = F.sigmoid(det_ref),
                    rgb_memory     = rgb_memory,
                    ir_memory      = ir_memory,
                    spatial_shapes = spatial_shapes,
                )

            if dn_content is not None:
                content = torch.cat([dn_content, det_content], dim=1)
            else:
                content = det_content

            # DAMR: passed into the decoder to apply inside the last layer
            damr_memories = (rgb_memory, ir_memory, spatial_shapes) if self.damr is not None else None

            out_bboxes, out_logits = dec.decoder(
                content, ref_points_unact, memory, enc_spatial_shapes,
                dec.dec_bbox_head, dec.dec_score_head, dec.query_pos_head,
                attn_mask=attn_mask,
                damr=self.damr, damr_memories=damr_memories)

            if dec.training and dn_meta is not None:
                dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
            if dec.training and dec.aux_loss:
                out['aux_outputs']     = dec._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
                out['enc_aux_outputs'] = dec._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
                out['enc_meta']        = {'class_agnostic': dec.query_select_method == 'agnostic'}
                if dn_meta is not None:
                    out['dn_aux_outputs'] = dec._set_aux_loss(dn_out_logits, dn_out_bboxes)
                    out['dn_meta']        = dn_meta
            if self.training and hasattr(self.fusion, 'get_consistency_pair'):
                pair = self.fusion.get_consistency_pair()
                if pair is not None:
                    out['consistency_pair'] = pair
            if relation_loss is not None:
                out['loss_s4_s5_rel'] = relation_loss
                out['s4_s5_relation_stats'] = relation_stats
            return out

        out = self.decoder(enc_feats, targets)
        if self.training and hasattr(self.fusion, 'align_loss') and self.fusion.align_loss is not None:
            out['align_loss'] = self.fusion.align_loss
        # Cross-Modal Consistency Loss: encourage RGB and IR features to share a common representation
        if self.training and self.cmcl_weight > 0:
            cmcl = sum(
                (1.0 - (F.normalize(rf, dim=1) * F.normalize(irf, dim=1)).sum(dim=1)).mean()
                for rf, irf in zip(rgb_feats, ir_feats)
            )
            out['cmcl_loss'] = self.cmcl_weight * cmcl / len(rgb_feats)
        if self.training and hasattr(self.fusion, 'get_consistency_pair'):
            pair = self.fusion.get_consistency_pair()
            if pair is not None:
                out['consistency_pair'] = pair
        if relation_loss is not None:
            out['loss_s4_s5_rel'] = relation_loss
            out['s4_s5_relation_stats'] = relation_stats
        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy') and m is not self:
                m.convert_to_deploy()
        return self


