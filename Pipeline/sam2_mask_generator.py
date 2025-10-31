# sam2_mask_generator.py
# Python script to automatically generate object masks from surgical instrument images
# using Meta's Segment Anything Model (SAM2) and zero-shot CLIP filtering.

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import clip
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save segmentation masks for surgical instruments using SAM2 + CLIP filter"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input surgical instrument image"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SAM model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="Backbone model type for SAM"
    )
    parser.add_argument(
        "--output_dir", type=str, default="masks_output",
        help="Directory to save generated masks"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="CLIP similarity threshold for filtering masks"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show overlayed masks in a matplotlib window"
    )
    return parser.parse_args()


def load_sam(checkpoint_path: str, model_type: str):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    return sam


def setup_clip(device: str):
    model, preprocess = clip.load("ViT-B/32", device=device)
    # prepare text embedding
    text_tokens = clip.tokenize(["surgical instrument"]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return model, preprocess, text_feat


def filter_masks_with_clip(masks, image: np.ndarray, clip_model, preprocess, text_feat, threshold: float, device: str):
    filtered = []
    for m in masks:
        x, y, w, h = map(int, m['bbox'])
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        pil_crop = Image.fromarray(crop)
        img_input = preprocess(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(img_input)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()
        if sim >= threshold:
            filtered.append(m)
    return filtered


def save_masks(masks, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        seg = mask['segmentation'].astype(np.uint8) * 255
        out_path = os.path.join(output_dir, f"mask_{i:03d}.png")
        cv2.imwrite(out_path, seg)
    print(f"Saved {len(masks)} masks to {output_dir}")


def visualize_masks(image: np.ndarray, masks: list):
    def show_anns(masks_list):
        if not masks_list:
            return
        sorted_anns = sorted(masks_list, key=lambda x: x['area'], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random((1, 3)).tolist()[0]
            masked = np.zeros_like(image)
            for c in range(3):
                masked[:, :, c] = m * color_mask[c]
            ax.imshow(np.dstack((masked, m * 0.35)))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()


def main():
    args = parse_args()

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read and prepare image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Load SAM2 model
    print("Loading SAM model...")
    sam = load_sam(args.checkpoint, args.model_type)

    # Generate masks
    print("Generating masks with SAM...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks")

    # Setup CLIP and filter masks
    print("Loading CLIP model and filtering masks...")
    clip_model, preprocess, text_feat = setup_clip(device)
    masks = filter_masks_with_clip(masks, image, clip_model, preprocess, text_feat, args.threshold, device)
    print(f"Filtered down to {len(masks)} surgical instrument masks")

    # Save masks
    save_masks(masks, args.output_dir)

    # Optional visualization
    if args.visualize:
        visualize_masks(image, masks)

if __name__ == "__main__":
    main()
