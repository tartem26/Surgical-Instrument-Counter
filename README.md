# Surgical Instrument Counter
The app automates surgical instrument counts during operations, flags mismatches, and is powered by a modular end-to-end computer vision pipeline. It performs preprocessing (contrast/saturation, ImageNet normalization, rotation), generates masks with SAM/SAM2, and uses CLIP for pseudo-labeling/classification; then trains Mask R-CNN, ViT+LoRA (zero-shot→fine-tune), and SegFormer+LoRA, with a YOLOv8 training path for cross-validation. The system handles complex/overlapping shapes, auto-labels instruments, and supports reliable, real-time counting.

## Diagrams
<img width="992" height="896" alt="High-Level Block Diagram" src="https://github.com/user-attachments/assets/a78e16aa-bd77-4c52-8881-e91efadb0212" />
<img width="1341" height="752" alt="Pipeline: Preprocessing and Segmentation" src="https://github.com/user-attachments/assets/960e0f92-bc2c-4302-83dd-351523d16122" />
<img width="1343" height="752" alt="Pipeline: Classification and Fine-Tuning" src="https://github.com/user-attachments/assets/0fde9cd7-eba0-4a51-9e96-17f0afe745bb" />

## Components
1. `preprocessing.py` — resize to `224×224`, adjust contrast/brightness/saturation, ImageNet mean/std normalization, and rotation-based test augmentation. Outputs cleaned training images and rotated test images.
2. `preprocessing_expanded.py` — as above, plus CLIP cosine similarity and SAM2 crop to improve labeling/cropping quality.
3. `sam2_mask_generator.py` — generate segmentation masks with SAM2; (in-progress) CLIP filtering to keep masks semantically aligned (e.g., "surgical instrument").
4. `autolabel_and_train.py`
    - Auto-label SAM masks with CLIP into class folders.
  - Train Mask R-CNN (torchvision) on the labeled dataset for instance segmentation.
5. `classified_crops.py` — SAM proposals → filter bboxes → CLIP zero-shot classification of crops; saves crops into label-named folders; optional visualization.
6. `sam2_surgical_mask.py` — select best SAMv2 mask (e.g., largest area), optionally crop bottom region, export instrument with transparent background (RGBA PNG).
7. `generate_masks_with_sam.py` — create binary segmentation masks for an entire, class-foldered dataset (SAM).
8. Model LoRA 1
  - `model.py` — fine-tune ViT with LoRA (classification).
  - `inference.py` — classify whole image or segment-and-count via SAM2 + ViT+LoRA.
9. Model LoRA 2
  - `surgical_instrument_segmentation.py` — train SegFormer + LoRA (semantic segmentation).
  - `inference.py` — run semantic segmentation and count pixels per class.
10. Model YOLO LoRA — dataset conversion to YOLO format, YOLOv8 training, and inference; logs/plots under `runs/detect/train/`.

## Key Parameters (examples)
- Preprocessing: `TARGET_SIZE=(224,224)`, ImageNet `MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`, rotation angles e.g., `[-15,-10,-5,5,10,15]`.
- SAM proposals (typical): `points_per_side=32`, `pred_iou_thresh=0.9`, `stability_score_thresh=0.92`, `min_mask_region_area=50000`, `box_nms_thresh=0.7`.
- BBox filtering: fractional area filter, e.g., `[0.03, 0.15]` relative to image.
- CLIP: model `openai/clip-vit-large-patch14`, threshold e.g., `0.3`.

## Setup
Use `Python 3.8+` in a fresh virtual environment.

``` python
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy matplotlib seaborn
pip install transformers peft
pip install imbalanced-learn scikit-learn
# SAM / SAM2 (see Meta's repo for checkpoints)
# YOLOv8
pip install ultralytics
```

> Download the SAM/SAM2 checkpoints file and place it in the `Pipeline` folder (e.g., `sam_vit_h_4b8939.pth`).

## Typical Workflows
1. Preprocess raw images
  - Resize, normalize with ImageNet stats, adjust contrast/saturation, and create a rotated test set.
2. Generate masks
  - Per-image masks: `sam2_mask_generator.py` (SAM2).
  - Dataset-wide binary masks: `generate_masks_with_sam.py` (SAM).
3. Auto-label with CLIP → train Mask R-CNN
  ```sh
  # Auto-label masks into class folders
  python autolabel_and_train.py autolabel \
    --masks_dir sam_output/ \
    --images_dir dataset/images/ \
    --output_dir dataset/masks/ \
    --classes right_angle_clamp curved_mosquito DeBakey_forceps angled_bulldog_clamp \
    --threshold 0.3 \
    --device cuda
  
  # Train Mask R-CNN
  python autolabel_and_train.py train \
    --dataset_root dataset/ \
    --classes right_angle_clamp curved_mosquito DeBakey_forceps angled_bulldog_clamp \
    --epochs 15 \
    --batch_size 4 \
    --lr 0.005 \
    --device cuda
  ```
4. Crop proposals and classify with CLIP (zero-shot)
  - Run `classified_crops.py` to produce class-organized crops; tune SAM params and area filters as needed.
5. ViT+LoRA classification
  ```sh
  # Train
  python model.py \
    --data_dir ./train \
    --test_dir ./test \
    --model_name google/vit-base-patch16-224-in21k \
    --output_dir ./vit-lora-finetuned \
    --batch_size 16 \
    --epochs 5 \
    --lr 5e-5
  
  # Inference (single image)
  python inference.py --model_dir ./vit-lora-finetuned --image ./sample.jpg
  
  # Inference with SAM2 segment-and-count
  python inference.py --model_dir ./vit-lora-finetuned \
    --image ./grouped_instruments.jpg --use_sam \
    --sam_checkpoint ./sam_vit_h.pth
  ```
6. SegFormer+LoRA semantic segmentation
  - Train with `surgical_instrument_segmentation.py`, then run `inference.py` to get classwise pixel counts and a visual mask overlay.
7. YOLOv8 training path
  - Convert foldered dataset to YOLO format → train YOLOv8 (`yolov8n.pt`).
  - Metrics (confusion matrix, PR/F1 curves, etc.) are saved to `runs/detect/train/`.

## Notes
- Ensure binary masks are `0/255` and that image/mask naming is consistent for training.
- For smaller tools, reduce `SAM_MIN_MASK_REGION_AREA`; for fewer false positives, raise `pred_iou_thresh`/`stability_score_thresh`.
