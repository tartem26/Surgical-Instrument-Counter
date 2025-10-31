import os
import cv2
import numpy as np

# Determine project root (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define directories
DATA_DIR      = os.path.join(PROJECT_ROOT, 'Data')       # raw images
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Processed')  # preprocessed output
TEST_DIR      = os.path.join(PROJECT_ROOT, 'Test')       # augmented test images

# Create output dirs if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Preprocessing parameters
TARGET_SIZE     = (224, 224)                         # CNN input size
ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]           # augmentation
MEAN = np.array([0.485, 0.456, 0.406])                # ImageNet means
STD  = np.array([0.229, 0.224, 0.225])                # ImageNet stds

def adjust_contrast_saturation(img, alpha=1.1, beta=0, sat_scale=1.1):
    """Adjust contrast (alpha), brightness (beta) and saturation (sat_scale)."""
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * sat_scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess_image(src_path, dst_dir, base_name):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Failed to read {src_path}")
        return

    # 1) Resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # 2) Contrast & saturation
    img = adjust_contrast_saturation(img)
    # 3) Normalize & standardize
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # 4) Convert back to uint8 for saving
    img_out = np.clip((img * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)

    dst_path = os.path.join(dst_dir, f"{base_name}.jpg")
    cv2.imwrite(dst_path, img_out)

# --- STEP 1: Preprocess originals from Data → Processed ---
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    src = os.path.join(DATA_DIR, fname)
    name, _ = os.path.splitext(fname)
    preprocess_image(src, PROCESSED_DIR, name)

# --- STEP 2: Create rotated test set from Processed → Test ---
for fname in os.listdir(PROCESSED_DIR):
    if not fname.lower().endswith('.jpg'):
        continue
    img = cv2.imread(os.path.join(PROCESSED_DIR, fname))
    if img is None:
        continue

    h, w = img.shape[:2]
    base, _ = os.path.splitext(fname)
    for angle in ROTATION_ANGLES:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out_name = f"{base}_rot{angle}.jpg"
        cv2.imwrite(os.path.join(TEST_DIR, out_name), rotated)

print("Preprocessing and test-augmentation complete.")
