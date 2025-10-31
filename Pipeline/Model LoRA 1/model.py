#!/usr/bin/env python3
"""
instrument_classifier.py

Pre-requisites:
  - Python 3.8+
  - pip install torch torchvision transformers[torch] peft scikit-learn numpy tf-keras tqdm
  - export TF_ENABLE_ONEDNN_OPTS=0  # optional

Usage:
  python instrument_classifier.py `
      --data_dir "/path/to/Training Data Set" `
      --test_dir "/path/to/Test Data Set" `
      --model_name "google/vit-base-patch16-224-in21k" `
      --output_dir "./vit-lora-finetuned" `
      --batch_size 16 `
      --epochs 5 `
      --lr 5e-5
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, log_loss, roc_auc_score
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Disable TF backend
os.environ["TRANSFORMERS_NO_TF"] = "1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--test_dir', type=str, default=None)
    p.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k')
    p.add_argument('--output_dir', type=str, default='./vit-lora-finetuned')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-5)
    return p.parse_args()


def compute_metrics(labels, preds, probs, num_labels):
    acc = (preds == labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    ll = log_loss(np.eye(num_labels)[labels], probs)
    try:
        auc = roc_auc_score(np.eye(num_labels)[labels], probs, average='macro', multi_class='ovo')
    except ValueError:
        auc = float('nan')
    return acc, prec, rec, f1, ll, auc


def eval_loop(model, loader, device, num_labels, loss_fn):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    losses = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=imgs).logits
            loss = loss_fn(logits, labels).item()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss)
    probs_concat = np.vstack(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    acc, prec, rec, f1, ll, auc = compute_metrics(labels_arr, preds_arr, probs_concat, num_labels)
    return np.mean(losses), acc, prec, rec, f1, ll, auc, confusion_matrix(labels_arr, preds_arr)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    dataset = ImageFolder(args.data_dir, transform=transform)
    num_labels = len(dataset.classes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    test_loader = None
    if args.test_dir and os.path.isdir(args.test_dir):
        test_ds = ImageFolder(args.test_dir, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model & LoRA
    model = ViTForImageClassification.from_pretrained(args.model_name, num_labels=num_labels)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query","value"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    # Monkey-patch: drop unwanted kwargs
    orig_forward = model.forward
    orig_base = model.base_model.forward
    def forward_no_input_ids(*args, **kwargs):
        for k in ["input_ids", "attention_mask", "inputs_embeds"]:
            kwargs.pop(k, None)
        return orig_forward(*args, **kwargs)
    def base_no_input_ids(*args, **kwargs):
        for k in ["input_ids", "attention_mask", "inputs_embeds"]:
            kwargs.pop(k, None)
        return orig_base(*args, **kwargs)
    model.forward = forward_no_input_ids
    model.base_model.forward = base_no_input_ids

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    # Training loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(pixel_values=imgs).logits
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_ll, val_auc, val_cm = eval_loop(
            model, val_loader, device, num_labels, loss_fn
        )
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        print("Val Confusion Matrix:\n", val_cm)
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

    # Testing
    if test_loader:
        test_loss, test_acc, test_prec, test_rec, test_f1, test_ll, test_auc, test_cm = eval_loop(
            model, test_loader, device, num_labels, loss_fn
        )
        print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print("Test Confusion Matrix:\n", test_cm)

if __name__ == '__main__':
    main()
