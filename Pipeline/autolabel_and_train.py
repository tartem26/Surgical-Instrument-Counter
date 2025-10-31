# autolabel_and_train.py
# Two-stage pipeline:
# 1) Auto-label SAM-generated masks into instrument classes using CLIP
# 2) Train a Mask R-CNN segmentation model on the labeled dataset

import os
import shutil
import argparse
import json
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ---------------------
# 1) Auto-labeling
# ---------------------
def autolabel_masks(masks_dir, images_dir, output_dir, classes, threshold=0.3, device="cpu"):
    # Setup CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(classes).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    # Prepare output folder structure
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # Iterate over mask files
    for root, _, files in os.walk(masks_dir):
        for fname in files:
            if not fname.lower().endswith('.png'):
                continue
            mask_path = os.path.join(root, fname)
            # derive image name: assume filenames like image001_000.png
            base = fname.split('_')[0]
            # load mask and original image crop
            mask = np.array(Image.open(mask_path))
            if mask.sum() == 0:
                continue
            # load full image
            img_path = os.path.join(images_dir, f"{base}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, f"{base}.png")
            image = np.array(Image.open(img_path).convert("RGB"))
            # crop to bounding box
            ys, xs = np.where(mask > 0)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            crop = Image.fromarray(image[ymin:ymax+1, xmin:xmax+1])

            # CLIP inference
            img_input = preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_feats.T).squeeze(0)
                top_idx = sims.argmax().item()
                score = sims[top_idx].item()

            if score >= threshold:
                cls = classes[top_idx]
                dst = os.path.join(output_dir, cls, fname)
                shutil.copy(mask_path, dst)
    print("Auto-labeling complete.")

# ---------------------
# 2) Training Dataset
# ---------------------
class InstrumentDataset(Dataset):
    def __init__(self, root, classes, transforms=None):
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.classes = ['__background__'] + classes
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.images = sorted(os.listdir(self.img_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = img_name.rsplit('.',1)[0]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        masks = []
        boxes = []
        labels = []
        # gather masks from each class folder
        for cls in self.classes[1:]:
            cls_dir = os.path.join(self.mask_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for mname in os.listdir(cls_dir):
                if not mname.startswith(base + '_'):
                    continue
                mask = np.array(Image.open(os.path.join(cls_dir, mname)))
                mask_bin = (mask > 0).astype(np.uint8)
                ys, xs = np.where(mask_bin)
                if ys.size == 0:
                    continue
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                boxes.append([xmin, ymin, xmax, ymax])
                masks.append(mask_bin)
                labels.append(self.class_to_idx[cls])

        if not boxes:
            # no objects; return empty target
            target = {"boxes": torch.zeros((0,4), dtype=torch.float32),
                      "labels": torch.zeros((0,), dtype=torch.int64),
                      "masks": torch.zeros((0, h, w), dtype=torch.uint8)}
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms:
            img, target = self.transforms(img, target)
        return T.ToTensor()(img), target

# ---------------------
# 3) Training Loop
# ---------------------
def train_model(dataset_root, classes, epochs=10, batch_size=2, lr=0.005, device="cpu"):
    device = torch.device(device)
    dataset = InstrumentDataset(dataset_root, classes)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))

    model = maskrcnn_resnet50_fpn(num_classes=len(classes)+1, pretrained=True)
    model.to(device).train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = list(img.to(device) for img in images)
            tgt = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, tgt)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f"Epoch {epoch+1} total loss: {epoch_loss:.4f}")
    # Save checkpoint
    ckpt_path = os.path.join(dataset_root, 'maskrcnn_checkpoint.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

# ---------------------
# Argparse entry point
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Auto-label masks & train segmentation")
    sub = parser.add_subparsers(dest='command')

    # autolabel command
    p1 = sub.add_parser('autolabel')
    p1.add_argument('--masks_dir', required=True)
    p1.add_argument('--images_dir', required=True)
    p1.add_argument('--output_dir', required=True)
    p1.add_argument('--classes', nargs='+', required=True,
                    help='List of instrument class names')
    p1.add_argument('--threshold', type=float, default=0.3)
    p1.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # train command
    p2 = sub.add_parser('train')
    p2.add_argument('--dataset_root', required=True,
                    help='Root folder with subfolders images/ and masks/<class>/')
    p2.add_argument('--classes', nargs='+', required=True)
    p2.add_argument('--epochs', type=int, default=10)
    p2.add_argument('--batch_size', type=int, default=2)
    p2.add_argument('--lr', type=float, default=0.005)
    p2.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    if args.command == 'autolabel':
        autolabel_masks(args.masks_dir, args.images_dir, args.output_dir,
                        args.classes, args.threshold, args.device)
    elif args.command == 'train':
        train_model(args.dataset_root, args.classes,
                    args.epochs, args.batch_size, args.lr, args.device)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

