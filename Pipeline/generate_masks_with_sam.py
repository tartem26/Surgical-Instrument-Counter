# generate_masks_with_sam.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2

# --- Configuration ---
input_dir = "Training Data Set"
output_dir = "GeneratedMasks"
sam_checkpoint = "sam_vit_h_4b8939.pth"  # download this from Meta's SAM repo
model_type = "vit_h"
image_exts = ['.jpg', '.jpeg', '.png']

os.makedirs(output_dir, exist_ok=True)

# --- Load SAM ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# --- Process each image ---
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_dir, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    for fname in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
        if not any(fname.lower().endswith(ext) for ext in image_exts):
            continue

        image_path = os.path.join(class_path, fname)
        image = np.array(Image.open(image_path).convert("RGB"))

        masks = mask_generator.generate(image)

        if masks:
            # Combine all masks into one binary mask
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for m in masks:
                combined_mask = np.logical_or(combined_mask, m["segmentation"]).astype(np.uint8)

            # Save mask
            out_mask_path = os.path.join(output_class_path, os.path.splitext(fname)[0] + ".png")
            Image.fromarray(combined_mask * 255).save(out_mask_path)
        else:
            print(f"No masks found for {image_path}")
