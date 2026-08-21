# filip/model/filip_ecg_model.py

import torch
import torch.nn as nn

from filip.model.vision_encoder import FILIPVisionEncoder
from filip.model.feature_alignment import FeatureAlignmentHead
from filip.model.report_alignment import ReportAlignmentHead
from filip.data.vocab import get_feature_vocab, get_diagnosis_vocab

class FILIPECGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        _, self.feature_list = get_feature_vocab()
        _, self.diagnosis_list = get_diagnosis_vocab()
        
        self.num_features = len(self.feature_list)
        self.num_diagnosis = config.get('model', {}).get('num_classes', config.get('model', {}).get('num_diagnosis', len(self.diagnosis_list)))
        
        # Vision Encoder
        vision_model_name = config.get('model', {}).get('vision_encoder', 'openai/clip-vit-base-patch32')
        image_size = config.get('model', {}).get('image_size', 224)
        patch_size = config.get('model', {}).get('patch_size', 32)
        self.vision_encoder = FILIPVisionEncoder(
            model_name=vision_model_name,
            image_size=image_size,
            patch_size=patch_size
        )
        self.hidden_size = self.vision_encoder.hidden_size

        # Stage-1 raw-report alignment. The text encoder is used here rather
        # than converting reports into the legacy 20-class target vocabulary.
        self.use_report_alignment = config.get('model', {}).get('use_report_alignment', False)
        if self.use_report_alignment:
            from transformers import CLIPTextModel

            text_model_name = config.get('model', {}).get('text_encoder', vision_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
            text_hidden_size = self.text_encoder.config.hidden_size
            self.report_alignment_head = ReportAlignmentHead(
                image_hidden_size=self.hidden_size,
                text_hidden_size=text_hidden_size,
                align_dim=config.get('model', {}).get('report_align_dim', 256),
                topk=config.get('model', {}).get('report_topk', 8),
            )
        
        # Feature Alignment Head
        self.use_feature_alignment = config.get('model', {}).get('use_feature_alignment', True)
        if self.use_feature_alignment:
            align_dim = config.get('model', {}).get('feature_align_dim', 256)
            topk = config.get('model', {}).get('feature_topk', 8)
            self.feature_alignment_head = FeatureAlignmentHead(
                hidden_size=self.hidden_size,
                num_features=self.num_features,
                align_dim=align_dim,
                topk=topk
            )
        
        # Diagnosis Head
        self.diagnosis_from_features = config.get('model', {}).get('diagnosis_from_features', False)
        if self.diagnosis_from_features:
            # Stage 2: Predict from feature logits
            self.diagnosis_head = nn.Linear(self.num_features, self.num_diagnosis)
        else:
            # Standard prediction from pooled image features if needed
            self.diagnosis_head = nn.Linear(self.hidden_size, self.num_diagnosis)
            
        # Regular Prediction Dropout
        self.dropout = nn.Dropout(p=0.1)

        # JEPA configuration
        self.use_jepa = config.get('model', {}).get('use_jepa', False)
        if self.use_jepa:
            self.target_encoder = FILIPVisionEncoder(
                model_name=vision_model_name,
                image_size=image_size,
                patch_size=patch_size
            )
            self.target_encoder.load_state_dict(self.vision_encoder.state_dict())
            for param in self.target_encoder.parameters():
                param.requires_grad = False
                
            predictor_dim = config.get('model', {}).get('jepa_predictor_dim', self.hidden_size)
            self.num_patches = (image_size // patch_size) ** 2
            
            self.predictor_mask_token = nn.Parameter(torch.zeros(predictor_dim))
            nn.init.normal_(self.predictor_mask_token, std=0.02)
            
            self.predictor_pos_embedding = nn.Embedding(self.num_patches, predictor_dim)
            nn.init.normal_(self.predictor_pos_embedding.weight, std=0.02)
            
            self.predictor_proj = nn.Linear(self.hidden_size, predictor_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=16,
                dim_feedforward=2048,
                batch_first=True
            )
            self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            if predictor_dim != self.hidden_size:
                self.predictor_out_proj = nn.Linear(predictor_dim, self.hidden_size)

    def forward(self, images, input_ids=None, attention_mask=None, content_mask=None):
        outputs = {}
        
        # 1. Vision Encoder
        patch_features = self.vision_encoder(images) # [B, P, H]
        outputs["patch_features"] = patch_features

        if self.use_report_alignment and input_ids is not None:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_features = text_outputs.last_hidden_state
            if content_mask is None:
                content_mask = attention_mask
            report_logits, patch_report_similarity = self.report_alignment_head(
                patch_features, token_features, content_mask
            )
            outputs["report_logits"] = report_logits
            outputs["patch_report_similarity"] = patch_report_similarity
        
        # 2. Feature Alignment
        if self.use_feature_alignment:
            feature_logits, patch_feature_similarity = self.feature_alignment_head(patch_features)
            outputs["feature_logits"] = feature_logits
            outputs["patch_feature_similarity"] = patch_feature_similarity
            
        # 3. Diagnosis Prediction
        # If we are in Stage 2, diagnosis_head will exist and we use it
        if hasattr(self, 'diagnosis_head'):
            if self.diagnosis_from_features:
                diagnosis_logits = self.diagnosis_head(outputs["feature_logits"])
            else:
                pooled_image = patch_features.mean(dim=1)
                pooled_image = self.dropout(pooled_image)
                diagnosis_logits = self.diagnosis_head(pooled_image)
            outputs["diagnosis_logits"] = diagnosis_logits
            
        return outputs

    def forward_text_prompts(self, images, input_ids, attention_mask, content_mask=None):
        """Score each ECG against each diagnosis/report prompt.

        Unlike :meth:`forward`, image and text batch sizes may differ. This is
        the downstream text-classifier path: ``B`` images by ``C`` prompts.
        """
        if not self.use_report_alignment:
            raise RuntimeError("Text-prompt scoring requires use_report_alignment=true")
        patch_features = self.vision_encoder(images)
        token_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if content_mask is None:
            content_mask = attention_mask
        logits, similarities = self.report_alignment_head.score_prompts(
            patch_features, token_features, content_mask
        )
        return {
            "patch_features": patch_features,
            "diagnosis_logits": logits,
            "patch_prompt_similarity": similarities,
        }

    @torch.no_grad()
    def update_target_encoder(self):
        for param_c, param_t in zip(self.vision_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.mul_(0.996).add_(param_c.data, alpha=0.004)

    def forward_jepa(self, images, mask):
        B = images.shape[0]
        # Target representations from unmasked EMA target encoder
        with torch.no_grad():
            target_masked_latents = self.target_encoder(images) # [B, P, H]
            
        # Context representations from masked context encoder
        context_features = self.vision_encoder(images, mask=mask) # [B, P, H]
        
        # Predictor forward pass
        proj_features = self.predictor_proj(context_features) # [B, P, D]
        
        mask_token_expanded = self.predictor_mask_token.expand(B, self.num_patches, -1)
        predictor_input = torch.where(mask.unsqueeze(-1), mask_token_expanded, proj_features)
        
        pos_ids = torch.arange(self.num_patches, device=images.device).unsqueeze(0) # [1, P]
        pos_embed = self.predictor_pos_embedding(pos_ids) # [1, P, D]
        predictor_input = predictor_input + pos_embed
        
        predicted_latents = self.predictor(predictor_input) # [B, P, D]
        
        if hasattr(self, 'predictor_out_proj'):
            predicted_latents = self.predictor_out_proj(predicted_latents)
            
        return predicted_latents, target_masked_latents
