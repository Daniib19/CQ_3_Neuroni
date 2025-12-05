import config as cfg
import cv2
import os
import torch
import sys
import time
import numpy as np
import warnings
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- 0. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

# --- CHECK DEVICE ---
# FACTOR 1: Hardware Acceleration (providers)
# We prioritize 'CUDAExecutionProvider' to run computations on the NVIDIA GPU.
# This offers a massive speed boost (approx 10x-20x faster) compared to CPU.
if torch.cuda.is_available():
    import onnxruntime as ort

    print(f"ONNX Providers: {ort.get_available_providers()}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE.upper()}")

# --- 1. SETUP MODELS ---
print(f"Loading Local YOLO model from: {cfg.YOLO_PATH}")
if not os.path.exists(cfg.YOLO_PATH):
    print(f"CRITICAL ERROR: Model not found at {cfg.YOLO_PATH}")
    exit()

yolo_model = YOLO(cfg.YOLO_PATH)
yolo_model.to(DEVICE)

print("Loading SCRFD Face Detector...")

# FACTOR 2: Model Selection (name='buffalo_s')
# We switched from 'buffalo_l' (Large) to 'buffalo_s' (Small).
# - 'buffalo_l': Higher accuracy, but much slower (~100ms+ inference).
# - 'buffalo_s': Extremely fast (~10ms), lighter memory usage.
# STRATEGY: We compensate for the small model's lower accuracy by "zooming in"
# on the face (cropping) before detection. This makes the face appear large
# enough for even the small model to detect easily.
face_detector = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'],
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# FACTOR 3: Detection Resolution (det_size=(160, 160))
# This parameter controls the size of the input image tensor for the face detector.
# - Standard: Usually (640, 640).
# - Optimized: (160, 160).
# WHY IT WORKS: Since we are running this ONLY on a tight crop of the head (not the full image),
# the input image is already small. Scaling it up to 640x640 is wasteful.
# Keeping it at 160x160 maintains sufficient detail for the "buffalo_s" model
# while maximizing inference speed (Sub-10ms).
face_detector.prepare(ctx_id=0, det_size=(160, 160), det_thresh=0.1)


def warmup_models():
    """Runs dummy inference to initialize CUDA kernels."""
    print("WARMING UP GPU...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo_model.predict(dummy_frame, verbose=False, half=False)
    face_detector.get(dummy_frame)
    print("Warmup Complete.\n")


def get_best_person(frame, conf_thresh=0.15, zoom_size=1280):
    # YOLO Detects the full body first
    results = yolo_model.predict(frame, conf=conf_thresh, imgsz=zoom_size, verbose=False, classes=[0])[0]
    valid_persons = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < 300:
            continue
        valid_persons.append((x1, y1, x2, y2, float(box.conf[0])))

    if not valid_persons:
        return None, 0.0

    # Sort largest person first
    valid_persons.sort(key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), reverse=True)
    return valid_persons[0][:4], valid_persons[0][4]


def get_head_crop(frame, person_box):
    img_h, img_w = frame.shape[:2]
    px1, py1, px2, py2 = person_box
    pw, ph = px2 - px1, py2 - py1

    # FACTOR 4: Optimized Crop Factor (0.55)
    # This value was determined by running an analysis loop (np.arange(0.1, 1.005, 0.005))
    # to test detection confidence at every possible crop size.
    # - 0.25 (Head only): Too tight, often misses chin/forehead if person is moving.
    # - 1.00 (Full body): Too zoomed out, face becomes pixelated/blurry.
    # - 0.55 (Head + Torso): THE WINNER. It provides the perfect balance of "Context"
    #   (so the AI knows it's a person) and "Resolution" (enough pixels for the face).
    crop_h = int(ph * 0.55)
    crop_y2 = min(img_h, py1 + crop_h)

    # Add 20% padding to sides to account for head tilt/movement
    pad_x = int(pw * 0.20)
    crop_x1 = max(0, px1 - pad_x)
    crop_x2 = min(img_w, px2 + pad_x)

    crop_img = frame[py1:crop_y2, crop_x1:crop_x2]

    return crop_img, (crop_x1, py1, crop_x2, crop_y2)


def process_image(image_path):
    if not os.path.exists(image_path):
        image_path = image_path.replace(".png", ".jpg")
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    frame = cv2.imread(image_path)
    print(f"Processing: {image_path}")

    t_start = time.time()

    # 1. Detect Person
    person_box, person_conf = get_best_person(frame, conf_thresh=0.15, zoom_size=1280)

    if person_box is None:
        print("No persons detected.")
        return

    # 2. Get Optimized Head Crop
    head_img, crop_coords = get_head_crop(frame, person_box)

    if head_img.size == 0:
        print("Error: Head crop failed.")
        return

    # 3. Detect Face
    faces = face_detector.get(head_img)

    if len(faces) > 0:
        face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]

        t_end = time.time()
        print(f"  [Result] Face Detected!")
        print(f"  [Scores] Person: {person_conf:.2%} | Face: {face.det_score:.2%}")
        print(f"  [Time] Execution: {t_end - t_start:.4f}s")

        # Save results...
        os.makedirs(cfg.RESULTS_PATH, exist_ok=True)
        cv2.imwrite(os.path.join(cfg.RESULTS_PATH, "final_output.jpg"), frame)
    else:
        print("No face found on the person.")


if __name__ == "__main__":
    warmup_models()
    test_image = "data/dataset/test/Outdoor/Non-masked/Cristina - Outdoor - 30C.png"
    process_image(test_image)