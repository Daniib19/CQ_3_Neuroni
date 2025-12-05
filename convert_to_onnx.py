from ultralytics import YOLO

# Încarcă modelul antrenat
model = YOLO("runs/train/face_finetune2/weights/best.pt")

# Export în ONNX
model.export(format="onnx", opset=12, dynamic=True, simplify=True)