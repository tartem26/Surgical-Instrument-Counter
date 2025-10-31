import os
from PIL import Image
from ultralytics import YOLO
import yaml

# === Step 1: Configuration ===
RAW_DATASET_DIR = "Training Data Set"   # input folder with class folders
YOLO_DATASET_DIR = "yolo_dataset"       # output folder for YOLO-format data
DATA_YAML_PATH = "custom_data.yaml"     # YOLO data config
EPOCHS = 20
IMG_SIZE = 640

# === Step 2: Prepare YOLO Dataset ===
def convert_to_yolo_format():
    os.makedirs(f"{YOLO_DATASET_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_DATASET_DIR}/labels/train", exist_ok=True)

    class_names = sorted(os.listdir(RAW_DATASET_DIR))
    class_map = {i: name for i, name in enumerate(class_names)}

    for class_id, class_name in class_map.items():
        class_path = os.path.join(RAW_DATASET_DIR, class_name)
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.j.peg', '.png')):
                continue
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            w, h = img.size

            # Save image
            base = f"{class_name}_{os.path.splitext(img_file)[0]}"
            new_img_path = os.path.join(YOLO_DATASET_DIR, "images/train", base + ".jpg")
            img.save(new_img_path)

            # Simulate bounding box (entire image)
            label_path = os.path.join(YOLO_DATASET_DIR, "labels/train", base + ".txt")
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    # Save data.yaml
    data_yaml = {
        "path": os.path.abspath(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/train",  # using train for validation (tiny dataset)
        "names": [name for _, name in sorted(class_map.items())]
    }
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f)

    print("YOLO format dataset prepared.")
    return class_map

# === Step 3: Train YOLOv8 ===
def train_model():
    model = YOLO("yolov8n.pt")  # Use small model (can try yolov8s.pt for better accuracy)
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device='cpu'  # or 'cuda' if you have GPU
    )
    print("Training complete.")

# === Step 4: Inference Example ===
def run_inference(model_path, test_image, class_map):
    model = YOLO(model_path)
    results = model(test_image)[0]

    counts = {}
    for cls in results.boxes.cls.tolist():
        cls = int(cls)
        label = class_map.get(cls, f"class_{cls}")
        counts[label] = counts.get(label, 0) + 1
    print("Detected instruments:", counts)

# === MAIN ===
if __name__ == "__main__":
    class_map = convert_to_yolo_format()
    train_model()

    # OPTIONAL: run test
    run_inference("runs/detect/train/weights/best.pt", "test1.jpeg", {v: k for k, v in class_map.items()})
