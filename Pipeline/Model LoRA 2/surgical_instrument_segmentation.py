# surgical_instrument_segmentation.py
# Main training script

import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from scipy.ndimage import label
import random

# --- Configuration ---
save_path = os.path.join("Trained Model")  # save to 'Trained Model' folder
image_root = './Training Data Set'
mask_root = None  # set to path with mask images if available
image_size = (128, 128)  # match SegFormer output size to avoid loss mismatch
batch_size = 4
epochs = 20
learning_rate = 5e-5

# --- Dataset Preparation ---
class_names = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
label2id = {name: idx+1 for idx, name in enumerate(class_names)}
id2label = {idx+1: name for idx, name in enumerate(class_names)}
id2label[0] = "background"
num_classes = len(class_names) + 1

class InstrumentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.transform = transform
        for class_name in os.listdir(image_dir):
            folder = os.path.join(image_dir, class_name)
            if not os.path.isdir(folder):
                continue
            label_id = label2id[class_name]
            for fname in os.listdir(folder):
                img_path = os.path.join(folder, fname)
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                mask_path = None
                if mask_dir:
                    mask_path = os.path.join(mask_dir, class_name, fname)
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.labels.append(label_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.mask_paths[idx]:
            mask_img = Image.open(self.mask_paths[idx])
            mask = np.array(mask_img, dtype=np.int64)
        else:
            mask = np.zeros((img.height, img.width), dtype=np.int64)
            mask[:, :] = self.labels[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, torch.tensor(mask, dtype=torch.long), self.labels[idx]

def transform_train(img, mask):
    img = TF.resize(img, image_size)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
    mask = TF.resize(mask, image_size, interpolation=Image.NEAREST)
    mask = np.array(mask, dtype=np.uint8)
    if random.random() < 0.5:
        img = TF.hflip(img)
        mask = np.ascontiguousarray(np.fliplr(mask))
    angle = random.uniform(-15, 15)
    img = TF.rotate(img, angle)
    mask = np.ascontiguousarray(np.array(TF.rotate(Image.fromarray(mask), angle, fill=0)))
    img = TF.adjust_brightness(img, 1 + (0.2 * (random.random() - 0.5)))
    img = TF.adjust_contrast(img, 1 + (0.2 * (random.random() - 0.5)))
    img = TF.adjust_saturation(img, 1 + (0.2 * (random.random() - 0.5)))
    img = TF.adjust_hue(img, 0.1 * (random.random() - 0.5))
    img_tensor = TF.to_tensor(img)
    return img_tensor, mask

def transform_val(img, mask):
    img = TF.resize(img, image_size)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
    mask = TF.resize(mask, image_size, interpolation=Image.NEAREST)
    mask = np.array(mask, dtype=np.uint8)
    img_tensor = TF.to_tensor(img)
    return img_tensor, mask

# --- Load and split dataset ---
dataset = InstrumentDataset(image_root, mask_root, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- Model Setup ---
checkpoint = "nvidia/segformer-b0-finetuned-ade-512-512"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1, bias="none", modules_to_save=["decode_head"])
model = get_peft_model(model, lora_config)

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")
print_trainable_parameters(model)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, epochs * len(train_loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        logits = outputs.logits
        logits = nn.functional.interpolate(logits, size=image_size, mode='bilinear', align_corners=False)
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, masks, class_id in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            logits = outputs.logits
            logits = nn.functional.interpolate(logits, size=image_size, mode='bilinear', align_corners=False)
            val_loss += criterion(logits, masks).item()
            pred_mask = logits.argmax(dim=1).cpu().numpy()[0]
            probs = logits.softmax(dim=1).cpu().numpy()[0]
            unique_classes = np.unique(pred_mask)
            unique_classes = [c for c in unique_classes if c != 0]
            pred_class = max({c: np.sum(pred_mask == c) for c in unique_classes}.items(), key=lambda x: x[1])[0] if unique_classes else 0
            y_true.append(int(class_id.item()))
            y_pred.append(int(pred_class))
            y_prob.append([np.max(probs[c]) for c in range(1, num_classes)])
    avg_val_loss = val_loss / len(val_loader)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {pixel_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=list(range(1, num_classes))))
print("\nClassification Report:")

# --- Save trained LoRA model ---
model.save_pretrained(save_path)
print(f"LoRA model saved to: {save_path}")

print(classification_report(y_true, y_pred, labels=list(label2id.values()), target_names=list(label2id.keys())))
try:
    roc_auc = roc_auc_score(np.eye(num_classes-1)[np.array(y_true)-1], np.array(y_prob), average='macro', multi_class='ovr')
    print(f"ROC AUC (macro): {roc_auc:.3f}")
except ValueError as e:
    print(f"ROC AUC could not be computed: {e}")
