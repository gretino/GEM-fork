# filip/model/filip_ecg_model.py

import torch
import torch.nn as nn

from filip.model.vision_encoder import FILIPVisionEncoder
from filip.model.feature_alignment import FeatureAlignmentHead
from filip.data.vocab import get_feature_vocab, get_diagnosis_vocab

class FILIPECGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        _, self.feature_list = get_feature_vocab()
        _, self.diagnosis_list = get_diagnosis_vocab()
        
        self.num_features = len(self.feature_list)
        self.num_diagnosis = config.get('model', {}).get('num_diagnosis', len(self.diagnosis_list))
        
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

    def forward(self, images):
        outputs = {}
        
        # 1. Vision Encoder
        patch_features = self.vision_encoder(images) # [B, P, H]
        outputs["patch_features"] = patch_features
        
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
                # Fallback: global mean pooling over patches
                pooled_image = patch_features.mean(dim=1)
                diagnosis_logits = self.diagnosis_head(pooled_image)
            outputs["diagnosis_logits"] = diagnosis_logits
            
        return outputs
