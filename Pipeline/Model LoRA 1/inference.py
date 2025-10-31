# inference.py
# Load fine-tuned ViT+LoRA model and run multi-object detection via SAM2 + classifier on a single image

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel

# Optional: SAM2 imports (assuming you have segment-anything installed)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM2 not installed: multi-object detection will not run.")


def load_classifier(model_dir, base_model_name=None, device='cuda'):
    """
    Load the fine-tuned ViT LoRA model from `model_dir`. If base_model_name is provided,
    reload base checkpoint by that name.
    Also loads label names from 'label_names.txt' or model config.
    """
    processor = ViTImageProcessor.from_pretrained(model_dir)

    if base_model_name:
        base_model = ViTForImageClassification.from_pretrained(base_model_name)
    else:
        base_model = ViTForImageClassification.from_pretrained(model_dir)

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)

    orig_forward = model.forward
    orig_base = model.base_model.forward
    def forward_no_kw(*args, **kwargs):
        for k in ["input_ids","attention_mask","inputs_embeds"]:
            kwargs.pop(k, None)
        return orig_forward(*args, **kwargs)
    def base_no_kw(*args, **kwargs):
        for k in ["input_ids","attention_mask","inputs_embeds"]:
            kwargs.pop(k, None)
        return orig_base(*args, **kwargs)
    model.forward = forward_no_kw
    model.base_model.forward = base_no_kw
    model.eval()

    # Load label names
    label_names_path = os.path.join(model_dir, 'label_names.txt')
    if os.path.exists(label_names_path):
        with open(label_names_path) as f:
            label_names = [line.strip() for line in f.readlines()]
    else:
        label_names = [v for k, v in sorted(model.config.id2label.items())]

    return model, processor, label_names


def classify_patch(model, processor, image_patch, device='cuda'):
    """
    Classify a single image patch and return (label_idx, confidence).
    """
    inputs = processor(images=image_patch, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()[0].numpy()
        probs = np.exp(logits) / np.exp(logits).sum()
        idx = int(probs.argmax())
        confidence = float(probs[idx])
    return idx, confidence


def segment_and_count(image_path, model, processor, label_names, device='cuda', sam_checkpoint=None, sam_model_type="vit_h"):
    """
    Uses SAM2 to segment each instrument, then classify each mask crop.
    Returns dict of {instrument_name: count} and list of detections.
    """
    if not SAM_AVAILABLE:
        raise RuntimeError("SAM2 not available; install segment-anything to use segmentation.")

    img = np.array(Image.open(image_path).convert("RGB"))

    if sam_checkpoint is None:
        raise ValueError("sam_checkpoint must be provided for SAM mode.")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        multimask_output=True
    )

    raw_counts = {}
    detections = []
    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            continue
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        patch = Image.fromarray(img[y1:y2+1, x1:x2+1])
        label_idx, conf = classify_patch(model, processor, patch, device)
        raw_counts[label_idx] = raw_counts.get(label_idx, 0) + 1
        detections.append({
            'mask_index': i,
            'label': label_names[label_idx],
            'confidence': conf,
            'bbox': (x1, y1, x2, y2)
        })

    # Convert to instrument name: count
    named_counts = {label_names[k]: v for k, v in raw_counts.items()}
    return named_counts, detections



def simple_classify(image_path, model, processor, device='cuda'):
    """
    Single-instrument classification on whole image.
    """
    img = Image.open(image_path).convert("RGB")
    idx, conf = classify_patch(model, processor, img, device)
    return idx, conf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run instrument inference on an image.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to fine-tuned model dir")
    parser.add_argument('--base_model', type=str, default=None, help="Name or path of base model if separate")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--use_sam', action='store_true', help="Enable SAM-based segmentation and counting")
    parser.add_argument('--sam_checkpoint', type=str, default=None, help="Path to SAM checkpoint")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor, label_names = load_classifier(args.model_dir, args.base_model, device)

    if args.use_sam:
        counts, detections = segment_and_count(
            args.image, model, processor, label_names, device,
            sam_checkpoint=args.sam_checkpoint
        )
        print("Instrument counts:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
    else:
        idx, conf = simple_classify(args.image, model, processor, device)
        print(f"Predicted instrument: {label_names[idx]} (index {idx}), confidence: {conf:.4f}")
