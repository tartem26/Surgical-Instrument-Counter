# pipeline.py
# Full pipeline:
# 1) Preprocess raw images (resize, color, normalize, augment)
# 2) SAM2 detect bounding boxes and fine masks
# 3) Crop per-instrument images
# 4) Auto-label masks into classes via CLIP
# 5) (Optional) Train Mask R-CNN on labeled data

import os
import cv2
import numpy as np
import argparse
import torch
import clip
import shutil
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

# --- 1) Preprocessing Setup ---
# determine project root (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# define directories
DATA_DIR      = os.path.join(PROJECT_ROOT, 'Data')          # raw images
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Processed')     # preprocessed output
TEST_DIR      = os.path.join(PROJECT_ROOT, 'Test')          # augmented test images
# create output dirs if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
# preprocessing parameters
TARGET_SIZE     = (500, 500)                    # CNN input size
ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]     # augmentation
MEAN = np.array([0.485, 0.456, 0.406])          # ImageNet means
STD  = np.array([0.229, 0.224, 0.225])          # ImageNet stds

def adjust_contrast_saturation(img, alpha=1.1, beta=0, sat_scale=1.1):
    """Adjust contrast (alpha), brightness (beta) and saturation (sat_scale)."""
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# preprocess single image
def preprocess_image(src_path, dst_dir):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Failed to read {src_path}")
        return

    # a) resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # b) contrast & saturation
    img = adjust_contrast_saturation(img)

    # c) normalize & standardize
    tmp = img.astype(np.float32) / 255.0
    tmp = (tmp - MEAN) / STD

    # d) convert back to uint8 and re-scale for saving
    img_out = np.clip((tmp * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)

    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(dst_dir, base + '.jpg')
    cv2.imwrite(out_path, img_out)

# rotate for test set
def augment_test(src_dir, dst_dir):
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(src_dir, fname))
        if img is None:
            continue

        h, w = img.shape[:2]
        base = os.path.splitext(fname)[0]
        for angle in ROTATION_ANGLES:
            M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(dst_dir, f"{base}_rot{angle}.jpg"), rot)

# --- 2) SAM2 Setup ---
def load_sam(checkpoint, model_type):
    return sam_model_registry[model_type](checkpoint=checkpoint)

# detect bounding boxes with SAM2 and return masks list
# we assume mask['bbox'] and mask['segmentation'] available

def detect_masks(sam, image):
    gen = SamAutomaticMaskGenerator(sam)
    return gen.generate(image)

# --- 3) Cropping & Saving Masks ---
def save_crops_and_masks(image, masks, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(masks):
        x, y, w, h = map(int, m['bbox'])
        crop = image[y:y+h, x:x+w]
        # extract mask region matching the crop size
        mask_full = m['segmentation']
        mask_bin = mask_full[y:y+h, x:x+w].astype(bool)
        # apply mask to crop: keep original color where mask=True
        masked_crop = np.zeros_like(crop)
        for c in range(3):
            masked_crop[:,:,c] = crop[:,:,c] * mask_bin
        # save both crop and masked_crop
        crop_name = f"{prefix}_crop_{i:03d}.png"
        mask_name = f"{prefix}_mask_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, crop_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, mask_name), cv2.cvtColor(masked_crop, cv2.COLOR_RGB2BGR))

# --- Main CLI ---
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='End-to-end pipeline')
    parser.add_argument('--checkpoint', required=True, help='SAM2 .pth file')
    parser.add_argument('--model_type', default='vit_h', choices=['vit_h','vit_l','vit_b'])
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Instrument classes, e.g. right_angle_clamp curved_mosquito DeBakey_forceps angled_bulldog_clamp')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Step 1: Preprocess Data
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            preprocess_image(os.path.join(DATA_DIR, f), PROCESSED_DIR)
    augment_test(PROCESSED_DIR, TEST_DIR)
    print('Preprocessing done')

    # Step 2: Detect, filter with CLIP & Save Crops+Masks
    sam = load_sam(args.checkpoint, args.model_type)
    # Zero-shot CLIP setup for "surgical instrument" filtering
    def setup_clip_filter(device):
        import clip
        clip_model, clip_pre = clip.load('ViT-B/32', device=device)
        txt = clip.tokenize(["surgical instrument"]).to(device)
        with torch.no_grad():
            txt_feat = clip_model.encode_text(txt)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        return clip_model, clip_pre, txt_feat

    clip_model, clip_pre, text_feat = setup_clip_filter(args.device)

    output_vis = os.path.join(PROJECT_ROOT, 'VisualizedMasks')
    shutil.rmtree(output_vis, ignore_errors=True)
    os.makedirs(output_vis, exist_ok=True)

    for f in os.listdir(PROCESSED_DIR):
        if not f.lower().endswith('.jpg'): continue
        img_path = os.path.join(PROCESSED_DIR, f)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = detect_masks(sam, img_rgb)
        # filter masks via CLIP cosine similarity
        filtered = []
        for m in masks:
            x,y,w,h = map(int, m['bbox'])
            crop = img_rgb[y:y+h, x:x+w]
            if crop.size == 0: continue
            pil = Image.fromarray(crop)
            inp = clip_pre(pil).unsqueeze(0).to(args.device)
            with torch.no_grad():
                img_f = clip_model.encode_image(inp)
                img_f /= img_f.norm(dim=-1, keepdim=True)
                sim = (img_f @ text_feat.T).item()
            if sim >= args.threshold:
                filtered.append(m)
        print(f"{f}: {len(filtered)}/{len(masks)} kept as instruments")
        # save only filtered masks
        prefix = os.path.splitext(f)[0]
        save_crops_and_masks(img_rgb, filtered, output_vis, prefix)
    print('Filtered crops and precise masks saved to VisualizedMasks/')