# filip/model/vision_encoder.py

import math
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel

class FILIPVisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", image_size=224, patch_size=32):
        super().__init__()
        
        config = CLIPVisionConfig.from_pretrained(model_name)
        default_image_size = config.image_size
        default_patch_size = config.patch_size
        
        # If sizes differ from pretrained defaults, construct custom config and load weights with interpolation
        if image_size != default_image_size or patch_size != default_patch_size:
            config.image_size = image_size
            config.patch_size = patch_size
            self.encoder = CLIPVisionModel(config)
            
            # Load pretrained model to extract original weights
            pretrained_model = CLIPVisionModel.from_pretrained(model_name)
            state_dict = pretrained_model.state_dict()
            
            # 1. Interpolate patch embedding weights if patch size has changed
            if patch_size != default_patch_size:
                patch_embed = state_dict['vision_model.embeddings.patch_embedding.weight']
                new_patch_embed = torch.nn.functional.interpolate(
                    patch_embed,
                    size=(patch_size, patch_size),
                    mode='bicubic',
                    align_corners=False
                )
                state_dict['vision_model.embeddings.patch_embedding.weight'] = new_patch_embed
            
            # 2. Interpolate positional embeddings if grid geometry has changed
            pos_embed = state_dict['vision_model.embeddings.position_embedding.weight']
            cls_pos_embed = pos_embed[0:1, :]
            grid_pos_embed = pos_embed[1:, :]
            old_grid_size = int(round(math.sqrt(grid_pos_embed.shape[0])))
            new_grid_size = image_size // patch_size
            
            grid_pos_embed = grid_pos_embed.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
            grid_pos_embed = torch.nn.functional.interpolate(
                grid_pos_embed,
                size=(new_grid_size, new_grid_size),
                mode='bicubic',
                align_corners=False
            )
            grid_pos_embed = grid_pos_embed.permute(0, 2, 3, 1).reshape(new_grid_size * new_grid_size, -1)
            
            new_pos_embed = torch.cat([cls_pos_embed, grid_pos_embed], dim=0)
            state_dict['vision_model.embeddings.position_embedding.weight'] = new_pos_embed
            
            # Load modified state dict
            self.encoder.load_state_dict(state_dict)
            print(f"Initialized dynamic CLIPVisionModel from {model_name} with interpolated image_size={image_size}, patch_size={patch_size}")
        else:
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            print(f"Initialized standard CLIPVisionModel from {model_name} with image_size={image_size}, patch_size={patch_size}")
        
    def forward(self, images):
        # outputs.hidden_states is a tuple of all layer outputs if output_hidden_states=True
        outputs = self.encoder(images, output_hidden_states=True)
        # Select the last hidden state
        last_hidden = outputs.hidden_states[-1]
        # Drop the CLS token (index 0) to return only patch tokens
        patch_features = last_hidden[:, 1:, :] # [B, P, H]
        return patch_features
        
    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size
