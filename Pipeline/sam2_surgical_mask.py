import argparse
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. Load image
def load_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)

# 2. Initialize SAMv2 model and mask generator
def init_sam(model_type: str, checkpoint: str, device: str):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

# 3. Generate masks for instruments
def generate_masks(mask_generator, image_np: np.ndarray):
    return mask_generator.generate(image_np)

# 4. Choose the surgical instrument mask(s)
def select_instrument_mask(masks, method: str = "largest"):
    # Currently supports 'largest' by area
    if method == "largest":
        areas = [(mask['segmentation'].sum(), mask['segmentation']) for mask in masks]
        largest = max(areas, key=lambda x: x[0])[1]
        return largest.astype(bool)
    else:
        raise ValueError(f"Unknown selection method: {method}")

# 5. Crop image and mask
def crop_bottom_fraction(image_np: np.ndarray, mask: np.ndarray, top_fraction: float = 0.33):
    h, w = mask.shape
    # Calculate start row cutting top_fraction
    start_row = int(h * top_fraction)
    # Keep the bottom portion from start_row to bottom
    cropped_image = image_np[start_row:, :, :]
    cropped_mask = mask[start_row:, :]
    return cropped_image, cropped_mask

# 6. Apply the mask to original image, preserving color on transparent background
def export_masked_image(image_np: np.ndarray, mask: np.ndarray, out_path: str):
    h, w, _ = image_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3][mask] = image_np[mask]
    rgba[..., 3][mask] = 255
    out = Image.fromarray(rgba)
    out.save(out_path)
    print(f"Saved masked image to {out_path}")

# 7. Full pipeline
def segment_instrument(input_path: str, output_path: str,
                        model_type: str, checkpoint: str,
                        device: str, method: str):
    image_np = load_image(input_path)
    mask_gen = init_sam(model_type, checkpoint, device)
    masks = generate_masks(mask_gen, image_np)
    instrument_mask = select_instrument_mask(masks, method)
    cropped_image, cropped_mask = crop_bottom_fraction(image_np, instrument_mask, top_fraction=0.33)
    export_masked_image(cropped_image, cropped_mask, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment surgical instrument with SAMv2 and crop bottom 66%")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path for output PNG mask")
    parser.add_argument("--model-type", default="vit_h", choices=list(sam_model_registry.keys()), help="SAM model type")
    parser.add_argument("--checkpoint", default="sam_v2.pth", help="Path to SAMv2 checkpoint file")
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="Computation device")
    parser.add_argument("--method", default="largest", choices=["largest"], help="Mask selection method")
    args = parser.parse_args()

    segment_instrument(
        input_path=args.input,
        output_path=args.output,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        method=args.method
    )
