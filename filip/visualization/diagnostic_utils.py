# filip/visualization/diagnostic_utils.py

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPVisionModel, CLIPTextModel

# Ensure parent path is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from filip.model.filip_ecg_model import FILIPECGModel
from filip.data.dataset import ECGImageDataset

SUBCLASS_DESCRIPTIONS = {
    'AMI': 'anterior myocardial infarction',
    'ALMI': 'anterolateral myocardial infarction',
    'ASMI': 'anteroseptal myocardial infarction',
    'ILMI': 'inferolateral myocardial infarction',
    'IMI': 'inferior myocardial infarction',
    'SEHYP': 'septal hypertrophy',
    'CRBBB': 'complete right bundle branch block',
    'WPW': 'Wolff-Parkinson-White syndrome',
    'LAO/LAE': 'left atrial overload or enlargement',
    'NORM': 'normal sinus rhythm',
    'ISC': 'ischemic ST-T changes',
    'AVB': 'atrioventricular block',
    'RAO/RAE': 'right atrial overload or enlargement',
    'LMI': 'lateral myocardial infarction',
    'ISCI': 'ischemic infarction',
    'ISCA': 'ischemic anterior changes',
    'NST': 'non-specific ST-T changes',
    'CLBBB': 'complete left bundle branch block',
    'ILBBB': 'incomplete left bundle branch block',
    'IRBBB': 'incomplete right bundle branch block',
    'PMI': 'posterior myocardial infarction',
    'RVH': 'right ventricular hypertrophy',
    'IVCD': 'intraventricular conduction delay'
}


class ExpandToSquare(object):
    """Pad rectangle image to square with solid background color."""
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, img):
        width, height = img.size
        if width == height:
            return img
        max_dim = max(width, height)
        result = Image.new(img.mode, (max_dim, max_dim), self.background_color)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        result.paste(img, (left, top))
        return result


def get_exact_training_transform(image_size=224):
    """Return the exact image preprocessing pipeline used during FILIP model training."""
    return transforms.Compose([
        ExpandToSquare(background_color=(255, 255, 255)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def load_model_from_checkpoint(checkpoint_path, config_path=None, device="cuda"):
    """Load a FILIP model from checkpoint path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("config")
    if config is None and config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    if config is None:
        # Default fallback config for ViT-Large report aligned models
        config = {
            "model": {
                "vision_encoder": "openai/clip-vit-large-patch14",
                "text_encoder": "openai/clip-vit-large-patch14",
                "image_size": 224,
                "patch_size": 14,
                "use_report_alignment": True,
                "use_feature_alignment": False,
                "report_align_dim": 256,
                "num_classes": 23
            }
        }

    model = FILIPECGModel(config).to(device)
    sd = ckpt.get("model_state_dict", ckpt)

    # Filter shape mismatch
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    return model, config


def load_base_clip_model(model_name="openai/clip-vit-large-patch14", device="cuda"):
    """Load original un-finetuned HuggingFace CLIP model components."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    vision_encoder = CLIPVisionModel.from_pretrained(model_name).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    vision_encoder.eval()
    text_encoder.eval()
    return vision_encoder, text_encoder, tokenizer


def compute_patch_norms(model, image_tensor):
    """
    Compute spatial patch norms for a given image tensor [1, 3, H, W]:
    - raw_patch_norm: ||z_i||_2 (output of vision encoder)
    - proj_patch_norm: ||W_i z_i||_2 (projection before L2 normalization in alignment head)
    - reg_norms: ||r_k||_2 (register token norms if num_registers > 0)
    """
    device = image_tensor.device
    reg_norms = None
    with torch.no_grad():
        if hasattr(model, "vision_encoder"):
            if getattr(model.vision_encoder, "num_registers", 0) > 0:
                patch_features, reg_features = model.vision_encoder(image_tensor, return_register_tokens=True)
                reg_norms = torch.norm(reg_features[0], dim=-1).cpu().numpy()
            else:
                patch_features = model.vision_encoder(image_tensor)
        else:
            # HuggingFace VisionModel directly
            patch_features = model(image_tensor).last_hidden_state[:, 1:]

        # Raw ViT patch norm
        raw_patch_norm = torch.norm(patch_features[0], dim=-1).cpu().numpy()

        # Projected patch norm
        proj_patch_norm = None
        if hasattr(model, "report_alignment_head") and model.report_alignment_head is not None:
            proj_features = model.report_alignment_head.image_projection(patch_features)
            proj_patch_norm = torch.norm(proj_features[0], dim=-1).cpu().numpy()
        elif hasattr(model, "feature_alignment_head") and model.feature_alignment_head is not None:
            proj_features = model.feature_alignment_head.image_projection(patch_features)
            proj_patch_norm = torch.norm(proj_features[0], dim=-1).cpu().numpy()

    return raw_patch_norm, proj_patch_norm, reg_norms


def compute_token_patch_similarity(model, tokenizer, image_tensor, text_phrase, device="cuda"):
    """
    Compute cosine similarity map between each token in text_phrase and all image patches.
    No token-index prompt averaging. Returns per-token similarity grids and statistics.
    """
    model.eval()
    tokens = tokenizer(text_phrase, return_tensors="pt", padding=True).to(device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        if hasattr(model, "vision_encoder"):
            patch_features = model.vision_encoder(image_tensor) # [1, P, H_i]
        else:
            patch_features = model(image_tensor).last_hidden_state[:, 1:]

        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_features = text_outputs.last_hidden_state # [1, T, H_t]
            head = model.report_alignment_head
            img_proj = F.normalize(head.image_projection(patch_features), dim=-1) # [1, P, A]
            txt_proj = F.normalize(head.text_projection(token_features), dim=-1)   # [1, T, A]
            similarities = torch.einsum("bpa,bta->bpt", img_proj, txt_proj)[0] # [P, T]
        else:
            raise RuntimeError("Model does not have text_encoder/report_alignment_head.")

    token_ids = input_ids[0].cpu().tolist()
    token_str_list = [tokenizer.decode([t_id]).strip() for t_id in token_ids]

    # Filter special tokens like <|startoftext|>, <|endoftext|>
    valid_indices = [
        i for i, t_id in enumerate(token_ids)
        if t_id not in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    ]

    per_token_sims = {}
    for idx in valid_indices:
        token_name = token_str_list[idx]
        sim_map = similarities[:, idx].cpu().numpy() # [P]
        per_token_sims[token_name] = {
            "token_id": token_ids[idx],
            "similarity_map": sim_map,
            "min": float(sim_map.min()),
            "max": float(sim_map.max()),
            "mean": float(sim_map.mean()),
            "topk": sorted(sim_map.tolist(), reverse=True)[:5]
        }

    return per_token_sims, token_str_list


def compute_word_level_patch_similarity(model, tokenizer, image_tensor, text_phrase, pooling="mean", device="cuda"):
    """
    Pool subword tokens belonging to each word to produce whole-word heatmaps.
    For example: 'myocardial infarction' -> heatmaps for 'myocardial' and 'infarction'.
    """
    per_token_sims, _ = compute_token_patch_similarity(model, tokenizer, image_tensor, text_phrase, device=device)
    words = text_phrase.strip().split()

    tokens = tokenizer(text_phrase, return_tensors="pt")["input_ids"][0]
    subword_tokens = [tokenizer.decode([t_id]).strip() for t_id in tokens if t_id not in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)]

    word_sims = {}
    curr_token_idx = 0

    for word in words:
        # Match subwords to word
        matched_subword_maps = []
        assembled = ""
        while curr_token_idx < len(subword_tokens):
            sub_tok = subword_tokens[curr_token_idx]
            sub_key = sub_tok
            if sub_key in per_token_sims:
                matched_subword_maps.append(per_token_sims[sub_key]["similarity_map"])
            clean_sub = sub_tok.replace("</w>", "").strip()
            assembled += clean_sub
            curr_token_idx += 1
            if assembled.lower() >= word.lower().replace("-", ""):
                break

        if matched_subword_maps:
            stacked = np.stack(matched_subword_maps, axis=0) # [num_subwords, P]
            if pooling == "max":
                pooled_map = np.max(stacked, axis=0)
            else:
                pooled_map = np.mean(stacked, axis=0)

            word_sims[word] = {
                "similarity_map": pooled_map,
                "min": float(pooled_map.min()),
                "max": float(pooled_map.max()),
                "mean": float(pooled_map.mean()),
                "max_patch_idx": int(np.argmax(pooled_map)),
                "topk": sorted(pooled_map.tolist(), reverse=True)[:5]
            }

    return word_sims


def compute_phrase_level_patch_similarity(model, tokenizer, image_tensor, text_phrase, pooling="mean", device="cuda"):
    """
    Pool all content subwords across the phrase to produce 1 unified heatmap for the phrase.
    """
    per_token_sims, _ = compute_token_patch_similarity(model, tokenizer, image_tensor, text_phrase, device=device)
    all_maps = [v["similarity_map"] for v in per_token_sims.values()]
    stacked = np.stack(all_maps, axis=0)
    if pooling == "max":
        pooled_map = np.max(stacked, axis=0)
    else:
        pooled_map = np.mean(stacked, axis=0)

    return {
        "phrase": text_phrase,
        "similarity_map": pooled_map,
        "min": float(pooled_map.min()),
        "max": float(pooled_map.max()),
        "mean": float(pooled_map.mean()),
        "max_patch_idx": int(np.argmax(pooled_map)),
        "topk": sorted(pooled_map.tolist(), reverse=True)[:5]
    }


def compute_attention_rollout(model, image_tensor, zero_border=False, border_size=1, zero_left_right=False, return_register_attention=False):
    """
    Compute ViT self-attention rollout matrix from vision encoder attentions.
    Correctly supports models with num_registers > 0 by properly routing register tokens
    through the vision encoder and slicing the spatial patch tokens.
    If zero_border=False (default), the complete, uncut attention rollout is returned.
    """
    num_registers = 0
    with torch.no_grad():
        if hasattr(model, "vision_encoder"):
            ve = model.vision_encoder
            num_registers = getattr(ve, "num_registers", 0)
            if num_registers > 0:
                B = image_tensor.shape[0]
                hidden_states = ve.encoder.vision_model.embeddings(image_tensor)
                cls_embed = hidden_states[:, :1, :]
                patch_embeds = hidden_states[:, 1:, :]
                reg_tokens = ve.register_tokens.expand(B, -1, -1).to(hidden_states.dtype)
                hidden_states = torch.cat([cls_embed, reg_tokens, patch_embeds], dim=1)
                hidden_states = ve.encoder.vision_model.pre_layrnorm(hidden_states)
                encoder_outputs = ve.encoder.vision_model.encoder(
                    inputs_embeds=hidden_states,
                    output_attentions=True,
                )
                attentions = encoder_outputs.attentions
            else:
                outputs = ve.encoder(image_tensor, output_attentions=True)
                attentions = outputs.attentions
            image_size = getattr(ve, "image_size", 224)
            patch_size = getattr(ve, "patch_size", 14)
        else:
            outputs = model(image_tensor, output_attentions=True)
            attentions = outputs.attentions
            image_size = getattr(model, "image_size", getattr(model.config, "image_size", 224))
            patch_size = getattr(model, "patch_size", getattr(model.config, "patch_size", 14))

    num_tokens = attentions[0].size(-1)
    result = torch.eye(num_tokens)

    with torch.no_grad():
        for attn in attentions:
            attn_heads_avg = torch.mean(attn[0], dim=0).cpu() # [Tokens, Tokens]
            attn_with_residual = 0.5 * attn_heads_avg + 0.5 * torch.eye(num_tokens)
            attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_normalized, result)

    grid_size = image_size // patch_size
    patch_start_idx = 1 + num_registers
    cls_rollout = result[0, patch_start_idx:].numpy().reshape(grid_size, grid_size)
    if zero_border:
        cls_rollout = zero_border_squares(cls_rollout, border_size=border_size, fill_value=0.0, zero_top_bottom=True, zero_left_right=zero_left_right)

    if return_register_attention:
        reg_attn = result[0, 1:1 + num_registers].numpy() if num_registers > 0 else np.array([])
        return cls_rollout, reg_attn

    return cls_rollout


def run_text_independence_test(model, tokenizer, image_tensor, test_terms=None, device="cuda"):
    """
    Test whether background hotspots win across completely unrelated text terms.
    Returns dictionary mapping terms to their highest scoring patch index and score.
    """
    if test_terms is None:
        test_terms = [
            "normal", "sinus", "atrial", "fibrillation",
            "infarction", "hypertrophy", "block", "tachycardia"
        ]

    results = {}
    for term in test_terms:
        per_token, _ = compute_token_patch_similarity(model, tokenizer, image_tensor, term, device=device)
        first_key = list(per_token.keys())[0]
        sim_map = per_token[first_key]["similarity_map"]
        max_idx = int(np.argmax(sim_map))
        max_val = float(np.max(sim_map))
        results[term] = {
            "max_patch_idx": max_idx,
            "max_patch_score": max_val,
            "mean_score": float(np.mean(sim_map)),
            "min_score": float(np.min(sim_map))
        }

    max_indices = [v["max_patch_idx"] for v in results.values()]
    most_common_patch = max(set(max_indices), key=max_indices.count)
    same_patch_freq = max_indices.count(most_common_patch) / len(test_terms)

    return results, most_common_patch, same_patch_freq


def zero_border_squares(sim_map, border_size=1, fill_value=0.0, zero_top_bottom=True, zero_left_right=False):
    """
    Set the bordering patch squares on top and bottom to fill_value (default 0.0),
    while retaining left and right borders (unless zero_left_right=True).
    Works for both 2D grids [H, W] and 1D flattened patch arrays [P].
    """
    if isinstance(sim_map, np.ndarray):
        is_1d = (sim_map.ndim == 1)
        if is_1d:
            grid_size = int(np.sqrt(sim_map.shape[0]))
            grid = np.copy(sim_map).reshape(grid_size, grid_size)
        else:
            grid = np.copy(sim_map)

        if zero_top_bottom:
            grid[:border_size, :] = fill_value
            grid[-border_size:, :] = fill_value
        if zero_left_right:
            grid[:, :border_size] = fill_value
            grid[:, -border_size:] = fill_value

        return grid.flatten() if is_1d else grid
    elif torch.is_tensor(sim_map):
        is_1d = (sim_map.dim() == 1)
        if is_1d:
            grid_size = int(np.sqrt(sim_map.size(0)))
            grid = sim_map.clone().view(grid_size, grid_size)
        else:
            grid = sim_map.clone()

        if zero_top_bottom:
            grid[:border_size, :] = fill_value
            grid[-border_size:, :] = fill_value
        if zero_left_right:
            grid[:, :border_size] = fill_value
            grid[:, -border_size:] = fill_value

        return grid.flatten() if is_1d else grid
    else:
        return sim_map


def get_warm_red_transparent_cmap():
    """
    Return a custom Warm Red vs. Transparent Colormap:
    Low values (min similarity / <= 0) are completely transparent (alpha=0.0).
    High values (peak similarity) glow in warm red (alpha=0.85).
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (1.0, 0.0, 0.0, 0.0),   # min: fully transparent red
        (1.0, 0.25, 0.0, 0.35), # mid: semi-transparent warm orange-red
        (1.0, 0.0, 0.0, 0.85)   # peak: glowing warm red
    ]
    cmap = LinearSegmentedColormap.from_list('WarmRedTransparent', colors)
    cmap.set_bad(alpha=0.0)
    cmap.set_under(alpha=0.0)
    return cmap


def get_padding_info(raw_shape, image_size=224, patch_size=14):
    """
    Given raw_shape=(width, height) or PIL image size (width, height),
    compute exact padding offsets, canvas dimensions, and patch grid bounds.
    """
    if isinstance(raw_shape, (tuple, list)):
        raw_width, raw_height = raw_shape[0], raw_shape[1]
    elif hasattr(raw_shape, 'width') and hasattr(raw_shape, 'height'):
        raw_width, raw_height = raw_shape.width, raw_shape.height
    else:
        raw_width, raw_height = image_size, image_size

    max_dim = max(raw_width, raw_height)
    left_pad_px = (max_dim - raw_width) // 2
    top_pad_px = (max_dim - raw_height) // 2
    right_pad_px = max_dim - raw_width - left_pad_px
    bottom_pad_px = max_dim - raw_height - top_pad_px

    grid_size = image_size // patch_size
    top_pad_frac = top_pad_px / max_dim
    bottom_pad_frac = (top_pad_px + raw_height) / max_dim
    left_pad_frac = left_pad_px / max_dim
    right_pad_frac = (left_pad_px + raw_width) / max_dim

    start_row = int(np.floor(top_pad_frac * grid_size))
    end_row = int(np.ceil(bottom_pad_frac * grid_size))
    start_col = int(np.floor(left_pad_frac * grid_size))
    end_col = int(np.ceil(right_pad_frac * grid_size))

    scale = image_size / max_dim
    return {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "max_dim": max_dim,
        "left_pad_px": left_pad_px,
        "top_pad_px": top_pad_px,
        "right_pad_px": right_pad_px,
        "bottom_pad_px": bottom_pad_px,
        "grid_size": grid_size,
        "start_row": start_row,
        "end_row": end_row,
        "start_col": start_col,
        "end_col": end_col,
        "raw_extent_px": [
            left_pad_px * scale,
            (left_pad_px + raw_width) * scale,
            (top_pad_px + raw_height) * scale,
            top_pad_px * scale
        ]
    }


def map_patch_to_raw_pixels(row, col, raw_shape, image_size=224, patch_size=14):
    """
    Map a patch grid coordinate (row, col) back to pixel coordinates (x_min, y_min, x_max, y_max)
    on the original unpadded raw image.
    """
    info = get_padding_info(raw_shape, image_size=image_size, patch_size=patch_size)
    max_dim = info["max_dim"]
    patch_dim_px = max_dim / info["grid_size"]

    x_min_canvas = col * patch_dim_px
    y_min_canvas = row * patch_dim_px
    x_max_canvas = (col + 1) * patch_dim_px
    y_max_canvas = (row + 1) * patch_dim_px

    x_min_raw = max(0.0, min(float(info["raw_width"]), x_min_canvas - info["left_pad_px"]))
    x_max_raw = max(0.0, min(float(info["raw_width"]), x_max_canvas - info["left_pad_px"]))
    y_min_raw = max(0.0, min(float(info["raw_height"]), y_min_canvas - info["top_pad_px"]))
    y_max_raw = max(0.0, min(float(info["raw_height"]), y_max_canvas - info["top_pad_px"]))

    is_padding = (x_min_raw >= x_max_raw) or (y_min_raw >= y_max_raw)

    return {
        "x_min": x_min_raw,
        "x_max": x_max_raw,
        "y_min": y_min_raw,
        "y_max": y_max_raw,
        "is_padding": is_padding
    }


def plot_warm_red_overlay(ax, base_image, sim_grid, vmin=None, vmax=None, title=None, raw_shape=None, zero_border=False, border_size=1, zero_left_right=False):
    """
    Plot base ECG image and overlay the similarity grid with Warm Red vs. Transparent colormap.
    Precisely constrains the heatmap and view limits to match the original unpadded image bounds.
    If zero_border=False (default), displays the complete, uncut heatmap without artificially cutting borders.
    If zero_border=True, bordering squares on top and bottom are set to zero/transparent (rendering as white space),
    while retaining left and right borders (unless zero_left_right=True).
    """
    sim_grid = np.copy(sim_grid).astype(float)
    if zero_border:
        sim_grid[:border_size, :] = np.nan
        sim_grid[-border_size:, :] = np.nan
        if zero_left_right:
            sim_grid[:, :border_size] = np.nan
            sim_grid[:, -border_size:] = np.nan

    valid_vals = sim_grid[~np.isnan(sim_grid)]
    if vmin is None:
        vmin = 0.0 if (len(valid_vals) > 0 and valid_vals.min() >= 0.0) else (valid_vals.min() if len(valid_vals) > 0 else 0.0)
    if vmax is None:
        vmax = valid_vals.max() if (len(valid_vals) > 0 and valid_vals.max() > vmin) else (vmin + 1.0)

    cmap = get_warm_red_transparent_cmap()
    cmap.set_bad(alpha=0.0)
    cmap.set_under(alpha=0.0)

    W, H = base_image.width, base_image.height
    if raw_shape is not None:
        if isinstance(raw_shape, (tuple, list)):
            raw_w, raw_h = raw_shape[0], raw_shape[1]
        elif hasattr(raw_shape, 'width') and hasattr(raw_shape, 'height'):
            raw_w, raw_h = raw_shape.width, raw_shape.height
        else:
            raw_w, raw_h = W, H
    else:
        raw_w, raw_h = W, H

    # 1. Plot base ECG image and overlay heatmap
    if W == H and raw_w != raw_h:
        # base_image is padded square canvas (e.g. 224x224), but raw image is non-square (e.g. 2200x1700)
        info = get_padding_info((raw_w, raw_h), image_size=W)
        ext = info["raw_extent_px"] # [x_min, x_max, y_max, y_min]
        ax.imshow(base_image, extent=[0, W, H, 0])
        im = ax.imshow(sim_grid, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, W, H, 0], interpolation='bicubic')
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    elif W != H:
        # base_image is original raw_image directly (e.g. 2200x1700, full crisp resolution)
        max_dim = max(W, H)
        left_pad = (max_dim - W) // 2
        top_pad = (max_dim - H) // 2
        right_pad = max_dim - W - left_pad
        bottom_pad = max_dim - H - top_pad
        ax.imshow(base_image, extent=[0, W, H, 0])
        im = ax.imshow(sim_grid, cmap=cmap, vmin=vmin, vmax=vmax, extent=[-left_pad, W + right_pad, H + bottom_pad, -top_pad], interpolation='bicubic')
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    else:
        ax.imshow(base_image, extent=[0, W, H, 0])
        im = ax.imshow(sim_grid, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, W, H, 0], interpolation='bicubic')

    if title:
        ax.set_title(title)
    ax.axis('off')
    return im


def create_ecg_lead_roi_mask(image_size=224, patch_size=14, raw_shape=None, margin_top_rows=2, margin_bottom_rows=2, margin_side_cols=0, inner_margin_rows=1, inner_margin_cols=0, exclude_border=True, exclude_side_borders=False):
    """
    Generate a 2D spatial boolean mask for the active lead region of an ECG.
    If raw_shape=(width, height) is provided, dynamically calculates the exact patch bounds
    based on the raw image's aspect ratio and padding geometry.
    Removes top and bottom margins/borders while retaining the full left and right active lead signals.
    """
    grid_size = image_size // patch_size
    mask = np.zeros((grid_size, grid_size), dtype=bool)

    if raw_shape is not None:
        info = get_padding_info(raw_shape, image_size=image_size, patch_size=patch_size)
        start_row = min(grid_size - 1, max(0, info["start_row"] + inner_margin_rows))
        end_row = min(grid_size, max(start_row + 1, info["end_row"] - inner_margin_rows))
        start_col = min(grid_size - 1, max(0, info["start_col"] + inner_margin_cols))
        end_col = min(grid_size, max(start_col + 1, info["end_col"] - inner_margin_cols))
    else:
        start_row = margin_top_rows
        end_row = grid_size - margin_bottom_rows
        start_col = margin_side_cols
        end_col = grid_size - margin_side_cols

    mask[start_row:end_row, start_col:end_col] = True

    if exclude_border:
        mask[0, :] = False
        mask[-1, :] = False
        if exclude_side_borders:
            mask[:, 0] = False
            mask[:, -1] = False

    return mask


def apply_roi_mask_to_similarity(sim_map, roi_mask, fill_value=0.0):
    """
    Apply spatial ROI mask to 1D or 2D similarity grid.
    Non-lead background patches are replaced with fill_value.
    """
    original_shape = sim_map.shape
    grid_size = roi_mask.shape[0]

    sim_flat = sim_map.flatten()
    mask_flat = roi_mask.flatten()

    masked_sim = np.copy(sim_flat)
    masked_sim[~mask_flat] = fill_value

    return masked_sim.reshape(original_shape)


def compute_bias_subtracted_patch_similarity(model, tokenizer, image_tensor, query_phrase, baseline_phrase="normal", lambda_sub=1.0, device="cuda"):
    """
    Compute register-bias subtracted similarity map:
    Sim_corrected(i) = Sim(i, query) - lambda_sub * Sim(i, baseline)
    """
    query_phrase_sim = compute_phrase_level_patch_similarity(model, tokenizer, image_tensor, query_phrase, pooling="mean", device=device)
    baseline_phrase_sim = compute_phrase_level_patch_similarity(model, tokenizer, image_tensor, baseline_phrase, pooling="mean", device=device)

    q_map = query_phrase_sim["similarity_map"]
    b_map = baseline_phrase_sim["similarity_map"]

    corrected_map = q_map - lambda_sub * b_map

    return {
        "query_phrase": query_phrase,
        "baseline_phrase": baseline_phrase,
        "raw_similarity_map": q_map,
        "baseline_similarity_map": b_map,
        "corrected_similarity_map": corrected_map,
        "min": float(corrected_map.min()),
        "max": float(corrected_map.max()),
        "mean": float(corrected_map.mean()),
        "topk": sorted(corrected_map.tolist(), reverse=True)[:5]
    }

