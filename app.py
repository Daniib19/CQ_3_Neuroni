import config as cfg
from ultralytics import YOLO

yolo_model = YOLO(cfg.YOLO_PATH)

if __name__ == "__main__":
  print("Model loaded from:", cfg.YOLO_PATH)