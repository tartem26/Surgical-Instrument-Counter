import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import traceback
import math
import re # For sanitizing filenames

# --- Required Libraries ---
# !pip install torch torchvision torchaudio opencv-python matplotlib Pillow transformers
# !pip install git+https://github.com/facebookresearch/segment-anything.git

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("ERROR: segment_anything library not found.")
    print("Please install it: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit()

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# --- Configuration ---

# --- SAM Configuration ---
SAM_IMAGE_INPUT = "compressed.jpeg" # Using image name from standalone script example
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# --- == SAM Generation Parameters ALIGNED TO STANDALONE SCRIPT == ---
SAM_POINTS_PER_SIDE = 32
SAM_PRED_IOU_THRESH = 0.9         # Matched to standalone script
SAM_STABILITY_SCORE_THRESH = 0.92 # Matched to standalone script
SAM_MIN_MASK_REGION_AREA = 50000  # Matched to standalone script's likely value
SAM_BOX_NMS_THRESH = 0.7          # Matched to standalone script
# --- ========================================================= ---

# --- Post-SAM BBox Area Filter Configuration (Matched to standalone) ---
APPLY_FRACTIONAL_AREA_FILTER = True # Standalone script applies this filter
MIN_BBOX_AREA_FRACTION = 0.03   # Matched to standalone script
MAX_BBOX_AREA_FRACTION = 0.15   # Matched to standalone script

# --- CLIP Configuration ---
CLIP_MODEL_CHECKPOINT = "openai/clip-vit-large-patch14"
OUTPUT_FOLDER = "classified_crops_matched_params" # New folder name to reflect changes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CLIP Candidate Labels ---
CANDIDATE_LABELS = [
    "Scalpel", "Forceps", "Surgical Scissors", "Needle Holder", "Retractor",
    "Surgical Clamp", "Suction Tube", "Bone Saw", "Drill", "Hemostat",
    "Towel Clamp", "Adson Forceps", "DeBakey Forceps", "Mayo Scissors",
    "Metzenbaum Scissors", "photo of a Scalpel", "photo of Forceps",
    "photo of Surgical Scissors", "photo of a Needle Holder", "photo of a Retractor",
    "image of a Surgical Clamp", "picture of a Hemostat", "Gauze Pad",
    "Suture Needle", "Surgical Drape", "Finger", "Hand", "Blue Drape Background",
    "Wrinkled Fabric", "Not an instrument", "Object on surgical drape"
]
# --- End Configuration ---


# --- Helper Function for SAM Visualization (Optional) ---
# (Code unchanged)
def show_boxes_on_image(image_rgb, boxes_with_area, title_prefix="SAM"):
    img_copy = image_rgb.copy();
    if len(img_copy.shape) == 2: img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    elif img_copy.shape[2] == 1: img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    for box_data in boxes_with_area:
        if len(box_data) != 5: continue
        xmin, ymin, xmax, ymax, area = map(int, box_data)
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        area_text = f"{area}"; font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; font_thickness = 1
        text_size, _ = cv2.getTextSize(area_text, font, font_scale, font_thickness)
        text_x = xmin + 5; text_y = ymin + text_size[1] + 5
        bg_pt1 = (text_x - 2, ymin + 2); bg_pt2 = (text_x + text_size[0] + 2, text_y + 2)
        cv2.rectangle(img_copy, bg_pt1, bg_pt2, (0,0,0), cv2.FILLED)
        cv2.putText(img_copy, area_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    plt.figure(figsize=(12, 12)); plt.imshow(img_copy); plt.axis('off'); plt.title(f'{title_prefix} Proposals ({len(boxes_with_area)})'); plt.show()


# --- SAM Proposal Generation Function ---
# (Code unchanged - uses parameters passed to it)
def generate_sam_bounding_box_proposals(image_path, model_type, checkpoint_path, device, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area, box_nms_thresh):
    print(f"--- Starting SAM Proposal Generation ---"); print(f"Processing image: {image_path}"); print(f"Using SAM model: {model_type}...")
    if not os.path.exists(image_path): raise FileNotFoundError(f"SAM Input Image not found: {image_path}")
    image_bgr = cv2.imread(image_path);
    if image_bgr is None: raise IOError(f"Could not read SAM image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image loaded (shape: {image_rgb.shape})"); print(f"Approx image area: {image_rgb.shape[0] * image_rgb.shape[1]} pixels")
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"; print("WARNING: CUDA not available, using CPU for SAM.")
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"SAM Checkpoint not found: {checkpoint_path}")
    print(f"Loading SAM model to {device}..."); sam = sam_model_registry[model_type](checkpoint=checkpoint_path); sam.to(device=device); print("SAM model loaded.")
    print("Initializing SamAutomaticMaskGenerator...");
    # Log the parameters actually being used
    print(f"  Using: points_per_side={points_per_side}, pred_iou_thresh={pred_iou_thresh}, stability_score_thresh={stability_score_thresh}, min_mask_region_area={min_mask_region_area}, box_nms_thresh={box_nms_thresh}")
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, min_mask_region_area=min_mask_region_area, box_nms_thresh=box_nms_thresh, crop_n_layers=0, crop_n_points_downscale_factor=1)
    print("Generating SAM masks..."); masks = mask_generator.generate(image_rgb); print(f"Generated {len(masks)} mask annotations after SAM filtering/NMS.")
    if not masks: return [], image_rgb, []
    areas = [ann['area'] for ann in masks if 'area' in ann];
    if areas: print(f"Mask area range post-SAM: [{min(areas)} - {max(areas)}]")
    proposals_with_area = []
    for ann in masks:
        if 'bbox' not in ann or not ann['bbox'] or 'area' not in ann: continue
        try:
            x_min, y_min, w, h = map(int, ann['bbox']); area = int(ann['area'])
            if w > 0 and h > 0 and area > 0: proposals_with_area.append([x_min, y_min, x_min + w, y_min + h, area])
        except (ValueError, TypeError, KeyError) as e: print(f"Warning: Error processing annotation: {e}"); continue
    print(f"Extracted {len(proposals_with_area)} proposals with area."); print(f"--- Finished SAM Proposal Generation ---")
    return proposals_with_area, image_rgb, masks


# --- Filename Sanitizer ---
# (Code unchanged)
def sanitize_filename(name):
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-_\.]', '', name)
    return name

# --- Main Execution ---
if __name__ == "__main__":

    print(f"Using device: {DEVICE}")

    # --- Load CLIP Model and Processor ONCE ---
    print(f"\n--- Loading CLIP Model ({CLIP_MODEL_CHECKPOINT}) ---")
    try:
        clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_CHECKPOINT)
        clip_model = AutoModelForZeroShotImageClassification.from_pretrained(CLIP_MODEL_CHECKPOINT).to(DEVICE)
        clip_model.eval()
        print("CLIP model and processor loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load CLIP model/processor: {e}"); exit()

    # --- Generate SAM Proposals ---
    try:
        # Pass the specific parameters from the Configuration section
        proposals_data, loaded_image_rgb, raw_masks = generate_sam_bounding_box_proposals(
            image_path=SAM_IMAGE_INPUT,
            model_type=SAM_MODEL_TYPE,
            checkpoint_path=SAM_CHECKPOINT,
            device=DEVICE,
            points_per_side=SAM_POINTS_PER_SIDE,
            pred_iou_thresh=SAM_PRED_IOU_THRESH,
            stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
            min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
            box_nms_thresh=SAM_BOX_NMS_THRESH
        )
    except Exception as e: print(f"FATAL ERROR during SAM proposal generation: {e}"); traceback.print_exc(); exit()

    if not proposals_data: print("\nNo proposals generated by SAM. Exiting."); exit()

    print(f"\nGenerated {len(proposals_data)} proposals from SAM.")

    # --- Apply Fractional BBox Area Filter ---
    # (Logic unchanged, uses MIN/MAX_BBOX_AREA_FRACTION from config)
    filtered_proposals = proposals_data
    if APPLY_FRACTIONAL_AREA_FILTER:
        print(f"\n--- Applying Fractional BBox Area Filter ---")
        print(f"Filtering proposals to keep bbox area between {MIN_BBOX_AREA_FRACTION*100:.1f}% and {MAX_BBOX_AREA_FRACTION*100:.1f}% of total image area.")
        img_area_total = loaded_image_rgb.shape[0] * loaded_image_rgb.shape[1]
        proposals_before_filter = len(proposals_data)
        temp_filtered = []
        for prop in proposals_data:
            xmin, ymin, xmax, ymax, mask_area = prop
            bbox_area = (xmax - xmin) * (ymax - ymin)
            frac = bbox_area / img_area_total if img_area_total > 0 else 0
            if MIN_BBOX_AREA_FRACTION <= frac <= MAX_BBOX_AREA_FRACTION:
                temp_filtered.append(prop)
        filtered_proposals = temp_filtered
        removed_count = proposals_before_filter - len(filtered_proposals)
        if removed_count > 0: print(f"Filtered out {removed_count} proposals based on fractional bbox area.")
        else: print("No proposals removed by fractional area filter.")
        print(f"--- Finished Fractional Area Filter ---")
    else:
        print("\nFractional BBox Area Filter is disabled.")

    if not filtered_proposals: print("\nNo proposals remaining after filtering. Exiting."); exit()

    # At this point, len(filtered_proposals) should match the number of boxes
    # shown by the standalone script (e.g., 8 in your case).
    print(f"\nProceeding to classify {len(filtered_proposals)} proposals (This should match the standalone script's output count).")

    # --- Optional: Visualize proposals *after* filtering ---
    # If you uncomment this, the plot should look identical to the standalone script's plot
    # show_boxes_on_image(loaded_image_rgb, filtered_proposals, "Filtered SAM Proposals (Should Match Standalone)")

    # --- Prepare Output Directory ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n--- Starting Classification and Saving Crops to: {OUTPUT_FOLDER} ---")
    label_counts = {}

    # --- Loop Through FILTERED Proposals, Classify, and Save ---
    # (Code unchanged)
    for i, proposal in enumerate(filtered_proposals):
        xmin, ymin, xmax, ymax, area_val = proposal
        print(f"\nProcessing filtered proposal {i+1}/{len(filtered_proposals)}: Box=[{xmin},{ymin},{xmax},{ymax}], MaskArea={area_val}")
        crop_xmin = max(0, xmin); crop_ymin = max(0, ymin); crop_xmax = min(loaded_image_rgb.shape[1], xmax); crop_ymax = min(loaded_image_rgb.shape[0], ymax)
        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin: print(f"  Skipping invalid crop dimensions."); continue
        cropped_image_rgb = loaded_image_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        try: pil_image = Image.fromarray(cropped_image_rgb)
        except Exception as e: print(f"  Error converting crop {i+1} to PIL Image: {e}"); continue
        top_label = "unknown"; top_score = 0.0
        with torch.no_grad():
            try:
                inputs = clip_processor(images=pil_image, text=CANDIDATE_LABELS, return_tensors="pt", padding=True); inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = clip_model(**inputs); logits_per_image = outputs.logits_per_image.squeeze(); probs = logits_per_image.softmax(dim=0); scores = probs.tolist()
                label_scores = sorted(zip(CANDIDATE_LABELS, scores), key=lambda x: x[1], reverse=True)
                if label_scores: top_label = label_scores[0][0]; top_score = label_scores[0][1]; print(f"  CLIP Top Prediction: '{top_label}' (Conf: {top_score:.4f})")
                else: print("  Warning: CLIP returned no scores.")
            except Exception as e: print(f"  Error during CLIP inference: {e}")
        sanitized_label = sanitize_filename(top_label); count = label_counts.get(sanitized_label, 0); filename = f"{sanitized_label}_{count}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename); label_counts[sanitized_label] = count + 1
        try: pil_image.save(filepath); print(f"  Saved crop to: {filepath}")
        except Exception as e: print(f"  Error saving image {filepath}: {e}")

    print(f"\n--- Finished Processing All Filtered Proposals ---")
    print(f"Saved classified crops to folder: {OUTPUT_FOLDER}")
