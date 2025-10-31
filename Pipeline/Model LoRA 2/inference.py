# inference.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from peft import PeftModel
import os
import matplotlib.pyplot as plt
from collections import Counter

# --- Config ---
image_path = "Test Images"  # <== CHANGE TO YOUR TEST IMAGE PATH
model_path = "./Trained Model"  # <== path where you saved your trained model
checkpoint = "nvidia/segformer-b0-finetuned-ade-512-512"
image_size = (128, 128)

# --- Label mappings from training ---
id2label = {
    0: "background",
    1: "4-4mm Dilators",
    2: "Angled Bulldog Clamp (Large)",
    3: "Angled Bulldog Clamp (Medium)",
    4: "Crile Wood Needleholder (Straight)",
    5: "Curved Crile Clamp (Kelly Clamp)",
    6: "Curved Mayo Scissors",
    7: "Curved Metzenbaum Scissors",
    8: "Curved Mosquito (Jacobson)",
    9: "DeBakey Forceps",
    10: "Jacobson Locking Straight Needleholder (Straight Castroviejo)",
    11: "Right Angle Clamp",
    12: "Ryder Needle Holder",
    13: "Short Gerald Forceps Without Teeth",
    14: "Straight Mayo Scissors (Suture Scissors)"
}

# --- Load model ---
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
base_model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, ignore_mismatched_sizes=True)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

# --- Load and preprocess image ---
img = Image.open(image_path).convert("RGB")
original_size = img.size
img = TF.resize(img, image_size)
img_tensor = TF.to_tensor(img).unsqueeze(0)

# --- Inference ---
with torch.no_grad():
    outputs = model(img_tensor)
    logits = outputs.logits
    logits = torch.nn.functional.interpolate(logits, size=image_size, mode='bilinear', align_corners=False)
    pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

# --- Count instruments ---
counts = Counter(pred_mask.flatten())
counts.pop(0, None)  # remove background

# --- Report ---
print(f"Results for: {os.path.basename(image_path)}")
for class_id, count in counts.items():
    instrument = id2label.get(class_id, f"Class {class_id}")
    print(f"→ {instrument}: {count} pixels")

# --- Optional: show segmented mask ---
plt.imshow(pred_mask, cmap="nipy_spectral")
plt.title("Predicted Segmentation")
plt.axis("off")
plt.show()
