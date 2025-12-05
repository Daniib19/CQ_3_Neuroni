from ultralytics import YOLO

# Încarcă model pre-antrenat
model = YOLO("yolov8n.pt")  # sau yolov8s.pt / yolov8m.pt

# Fine-tuning complet pe dataset
model.train(
    data="dataset/data.yaml",  # fișierul YAML existent
    epochs=5,
    imgsz=320,
    batch=16,
    lr0=0.001,        # learning rate inițial
    lrf=0.01,         # learning rate final (factor)
    name="face_finetune",
    project="runs/train"
)
