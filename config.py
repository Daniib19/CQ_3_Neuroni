import os

BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH  = os.path.join(BASE_PATH, "models")
DATA_PATH    = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

YOLO_PATH    = os.path.join(MODELS_PATH, "yolov8n.pt")
YOLO_FACE_PATH    = os.path.join(MODELS_PATH, "yolov8n-face.pt")
DATA_YML     = os.path.join(DATA_PATH,   "dataset.yml")