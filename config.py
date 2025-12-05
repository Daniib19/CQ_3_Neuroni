import os

# Config paths
BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH  = os.path.join(BASE_PATH, "models")
DATA_PATH    = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

# Models
YOLO_PATH         = os.path.join(MODELS_PATH, "yolov8n.pt")
YOLO_FACE_PATH    = os.path.join(MODELS_PATH, "yolov8n-face.pt")
DATA_YML          = os.path.join(DATA_PATH,   "dataset.yml")

FACE_EMBEDD_MODEL = os.path.join(MODELS_PATH, "w600k_mbf.onnx")
# FACE_LANMARKS = os.path.join(MODELS_PATH, "2d106det.onnx")
# FACE_EMBEDD_MODEL = os.path.join(MODELS_PATH, "r100_casia.onnx")