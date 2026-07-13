import os
import argparse
import json
import torch
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

from filip.model.filip_ecg_model import FILIPECGModel
from filip.train.train_mimic_feature import ExpandToSquare

def main():
    parser = argparse.ArgumentParser(description="Visualize FILIP ECG model patch-to-feature attention heatmaps.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (e.g. outputs/filip/mimic_feature_pretrain/checkpoints/best.pt)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input ECG image PNG file")
    parser.add_argument("--feature", type=str, default=None,
                        help="Name of the target feature/diagnosis class to visualize. If not specified, will print available classes.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output visualization PNG. Defaults to outputs/filip/.../alignment_visualizations/")
    
    args = parser.parse_args()
    
    # 1. Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    print(f"Loading checkpoint from: {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint does not contain config dictionary.")
        
    # 2. Reconstruct Model and load weights
    model = FILIPECGModel(config).to(device)
    state_dict = checkpoint["model_state_dict"]
    
    # Apply robust shape matching filter just in case
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # 3. Resolve Vocab list
    # Check if Stage 1 or Stage 2 checkpoint
    is_stage2 = "diagnosis_vocab" in checkpoint
    if is_stage2:
        vocab = checkpoint["diagnosis_vocab"]
        vocab_type = "diagnosis"
    else:
        vocab = checkpoint.get("feature_vocab", {})
        vocab_type = "feature"
        
    if isinstance(vocab, dict):
        vocab_list = list(vocab.keys())
    else:
        vocab_list = list(vocab)
        
    if not vocab_list:
        # Fallbacks
        from filip.data.vocab import MIMIC_FEATURES, PTBXL_SUPERCLASSES
        vocab_list = PTBXL_SUPERCLASSES if is_stage2 else MIMIC_FEATURES
        
    print(f"Detected {vocab_type} vocabulary: {vocab_list}")
    
    if args.feature is None:
        print("\n[INFO] Please specify a class using --feature. Available classes are listed above.")
        return
        
    if args.feature not in vocab_list:
        raise ValueError(f"Requested class '{args.feature}' is not in the checkpoint vocabulary: {vocab_list}")
        
    feat_idx = vocab_list.index(args.feature)
    
    # 4. Load and preprocess image
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"ECG image not found at: {args.image_path}")
        
    original_image = Image.open(args.image_path).convert("RGB")
    
    # Apply preprocessing transform (pad to square, resize to model's image_size, normalize)
    image_size = config.get("model", {}).get("image_size", 224)
    patch_size = config.get("model", {}).get("patch_size", 32)
    transform = transforms.Compose([
        ExpandToSquare(background_color=(255, 255, 255)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 5. Model Inference
    print("Running model inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        
    # Get similarity map from alignment head
    # similarity shape: [B, P, F] -> [1, 49, F]
    similarity = outputs.get("patch_feature_similarity")
    if similarity is None:
        raise ValueError("Model forward pass did not return 'patch_feature_similarity'. Make sure use_feature_alignment is active.")
        
    # Extract similarity scores for the target feature
    if is_stage2:
        diag_weight = model.diagnosis_head.weight
        patch_diagnosis_similarity = torch.matmul(similarity, diag_weight.t()).detach()
        sim_scores = patch_diagnosis_similarity[0, :, feat_idx].cpu().numpy()
    else:
        sim_scores = similarity[0, :, feat_idx].cpu().numpy() # Shape [49]
    
    # Reshape to grid
    grid_size = image_size // patch_size
    sim_grid = sim_scores.reshape(grid_size, grid_size)
    
    # Normalize grid to [0, 1] for visualization scaling
    sim_min = sim_grid.min()
    sim_max = sim_grid.max()
    sim_range = (sim_max - sim_min) if (sim_max - sim_min) > 1e-5 else 1.0
    sim_grid_norm = (sim_grid - sim_min) / sim_range
    
    # 6. Generate upsampled heatmap
    # Convert normalized grid to PIL image, upsample with Bicubic interpolation to match original image dimensions
    heatmap_pil = Image.fromarray((sim_grid_norm * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize(original_image.size, Image.Resampling.BICUBIC)
    
    # Convert heatmap to color map (Jet colormap)
    heatmap_np = np.array(heatmap_pil) / 255.0
    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap_np) # shape [H, W, 4]
    heatmap_color_rgb = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    heatmap_color_pil = Image.fromarray(heatmap_color_rgb)
    
    # Blend the heatmap overlay with the original ECG image
    blended_image = Image.blend(original_image, heatmap_color_pil, alpha=0.35)
    
    # 7. Create Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Original ECG
    axes[0].imshow(original_image)
    axes[0].set_title("Original ECG Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Right: Overlay Heatmap
    im = axes[1].imshow(blended_image)
    axes[1].set_title(f"Alignment Heatmap: {args.feature} ({vocab_type.capitalize()})", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    # Add Colorbar scaled to original similarity range
    # Create a dummy scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=sim_min, vmax=sim_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation="vertical", shrink=0.7)
    cbar.set_label("FILIP Patch Similarity Score", fontsize=12)
    
    plt.tight_layout()
    
    # 8. Save visualization
    if args.output_path is None:
        # Save in the default output directory
        exp_name = config.get("experiment_name", config.get("task", "pretrain"))
        out_dir = f"outputs/filip/{exp_name}/alignment_visualizations"
        os.makedirs(out_dir, exist_ok=True)
        img_id = os.path.splitext(os.path.basename(args.image_path))[0]
        args.output_path = os.path.join(out_dir, f"{img_id}_{args.feature}.png")
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n[SUCCESS] Attention heatmap visualization saved to: {args.output_path}")
    
    # Save a JSON metadata sidecar as required by Section 14
    top_indices = np.argsort(sim_scores)[::-1][:8].tolist()
    top_scores = sorted(sim_scores.tolist(), reverse=True)[:8]
    
    json_path = os.path.splitext(args.output_path)[0] + ".json"
    sidecar_data = {
        "sample_id": os.path.splitext(os.path.basename(args.image_path))[0],
        "feature_id": args.feature,
        "top_patch_indices": top_indices,
        "top_patch_scores": top_scores
    }
    with open(json_path, 'w') as f:
        json.dump(sidecar_data, f, indent=4)
    print(f"Saved metadata sidecar to: {json_path}")

if __name__ == "__main__":
    main()
