import os
import cv2
import json
import torch
import numpy as np
import config as cfg
import onnxruntime as ort
import warnings

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# --- 0. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

# --- 1. SETUP HARDWARE ---
if torch.cuda.is_available():
    print(f"[INFO] ONNX Providers: {ort.get_available_providers()}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEVICE.upper()}")

# --- 2. SETUP MODELS ---

print(f"[INFO] Loading YOLO model: {cfg.YOLO_PATH}")
if not os.path.exists(cfg.YOLO_PATH):
    print(f"CRITICAL ERROR: Model not found at {cfg.YOLO_PATH}")
    exit()
yolo_model = YOLO(cfg.YOLO_PATH)
yolo_model.to(DEVICE)

print("[INFO] Loading SCRFD Face Detector (buffalo_s)...")
face_detector = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'],
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_detector.prepare(ctx_id=0, det_size=(160, 160), det_thresh=0.25)

print(f"[INFO] Loading Embedding Model: {cfg.FACE_EMBEDD_MODEL}")
embed_session = ort.InferenceSession(cfg.FACE_EMBEDD_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
embed_input_name = embed_session.get_inputs()[0].name
embed_output_name = embed_session.get_outputs()[0].name


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_all_persons(frame, conf_thresh=0.25, zoom_size=1280):
    """
    Detects ALL people in the frame.
    Returns: list of (x1, y1, x2, y2, confidence), Sorted by Area.
    """
    results = yolo_model.predict(frame, conf=conf_thresh, imgsz=zoom_size, verbose=False, classes=[0])[0]

    persons = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) * (y2 - y1) < 1000: continue
        persons.append((x1, y1, x2, y2, float(box.conf[0])))

    if not persons: return []
    persons.sort(key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), reverse=True)
    return persons


def get_head_crop(frame, person_box):
    """
    [RESTORED] Original 55% Height Logic
    """
    img_h, img_w = frame.shape[:2]
    px1, py1, px2, py2 = person_box
    pw, ph = px2 - px1, py2 - py1

    # 1. Height: Take only top 55%
    crop_h = int(ph * 0.55)
    crop_y2 = min(img_h, py1 + crop_h)

    # 2. Width: Add 20% padding
    pad_x = int(pw * 0.20)
    crop_x1 = max(0, px1 - pad_x)
    crop_x2 = min(img_w, px2 + pad_x)

    # Returns image + offset (x1, y1)
    crop_img = frame[py1:crop_y2, crop_x1:crop_x2]
    return crop_img, (crop_x1, py1)


def detect_face_in_crop(head_crop):
    faces = face_detector.get(head_crop)
    if not faces: return None, None
    best_face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
    return best_face, head_crop


def preprocess_embedding(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    return np.transpose(img, (2, 0, 1))[None]


def generate_embedding(full_head_img, face_obj):
    aligned_face = face_align.norm_crop(full_head_img, landmark=face_obj.kps)
    blob = preprocess_embedding(aligned_face)
    emb = embed_session.run([embed_output_name], {embed_input_name: blob})[0][0]
    return emb / np.linalg.norm(emb)


# ============================================================
# MAIN DATABASE BUILDER (SMART CLUSTERING)
# ============================================================

def build_database_optimized(db_root="data/dataset/db", output_root="db_1"):
    print(f"\n[START] Building database from {db_root}...")

    persons_dict = {}
    for root, _, files in os.walk(db_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                person_name = f.split("-")[0].strip()
                img_path = os.path.join(root, f)
                persons_dict.setdefault(person_name, []).append(img_path)

    # Limit to first 10 people for test
    selected_persons = list(persons_dict.keys())[:10]
    print(f"[INFO] Processing persons: {selected_persons}")

    os.makedirs(output_root, exist_ok=True)

    db_json = []

    # --- REGISTRY: Stores {label, ref_embedding, data_entries} ---
    unknown_registry = []

    DUPLICATE_THRESH = 0.60
    SAME_PERSON_THRESH = 0.50

    for person_label in selected_persons:
        print(f"\n--- Processing: {person_label} ---")
        person_out_dir = os.path.join(output_root, f"person_{person_label}")
        os.makedirs(person_out_dir, exist_ok=True)

        target_embeddings_list = []
        image_paths = persons_dict[person_label][:2]

        for img_idx, img_path in enumerate(image_paths, 1):
            print(f"  Reading: {os.path.basename(img_path)}")
            frame = cv2.imread(img_path)
            if frame is None: continue

            all_persons = get_all_persons(frame, conf_thresh=0.25, zoom_size=1280)
            if not all_persons: continue

            target_face_embedding = None

            for p_idx, p_data in enumerate(all_persons):
                x1, y1, x2, y2, conf = p_data
                box = (x1, y1, x2, y2)

                # 1. Crop (55% Height Logic)
                head_crop_img, (offset_x, offset_y) = get_head_crop(frame, box)
                if head_crop_img.size == 0: continue

                # 2. Detect Face
                face_obj, _ = detect_face_in_crop(head_crop_img)
                if face_obj is None or face_obj.det_score < 0.55: continue

                # 3. Generate Embedding
                fx1, fy1, fx2, fy2 = map(int, face_obj.bbox)
                face_img_tight = head_crop_img[max(0, fy1):fy2, max(0, fx1):fx2]
                if face_img_tight.size == 0: continue

                emb_vector = generate_embedding(head_crop_img, face_obj)

                # --- TARGET ---
                if p_idx == 0:
                    target_face_embedding = emb_vector
                    save_name = f"{person_label}_face_{img_idx}.jpg"
                    save_path = os.path.join(person_out_dir, save_name)
                    cv2.imwrite(save_path, face_img_tight)

                    target_embeddings_list.append({
                        "embedding": emb_vector.tolist(),
                        "face_crop": save_path
                    })
                    print(f"    -> Saved Target Face: {save_name}")

                # --- UNKNOWNS ---
                else:
                    # CHECK A: Duplicate of Target?
                    if target_face_embedding is not None:
                        if np.dot(emb_vector, target_face_embedding) > DUPLICATE_THRESH:
                            print(f"    [SKIP] Ignored duplicate target")
                            continue

                    # CHECK B: Existing Unknown? (Clustering)
                    found_unknown_idx = -1
                    best_sim = -1.0

                    for u_idx, u_data in enumerate(unknown_registry):
                        sim = np.dot(emb_vector, u_data["ref_embedding"])
                        if sim > best_sim:
                            best_sim = sim
                            if sim > SAME_PERSON_THRESH:
                                found_unknown_idx = u_idx

                    if found_unknown_idx != -1:
                        # Existing Unknown
                        u_label = unknown_registry[found_unknown_idx]["label"]
                        print(f"    -> [MATCH] Found existing {u_label} (Sim: {best_sim:.2f})")

                        u_dir = os.path.join(output_root, f"person_{u_label}")
                        count = len(unknown_registry[found_unknown_idx]["data_entries"])
                        save_name = f"{u_label}_{count}.jpg"
                        save_path = os.path.join(u_dir, save_name)
                        cv2.imwrite(save_path, face_img_tight)

                        unknown_registry[found_unknown_idx]["data_entries"].append({
                            "embedding": emb_vector.tolist(),
                            "face_crop": save_path
                        })

                    else:
                        # New Unknown
                        new_id = len(unknown_registry) + 1
                        u_label = f"Unknown_{new_id}"
                        print(f"    -> [NEW] Created {u_label}")

                        u_dir = os.path.join(output_root, f"person_{u_label}")
                        os.makedirs(u_dir, exist_ok=True)
                        save_name = f"{u_label}_0.jpg"
                        save_path = os.path.join(u_dir, save_name)
                        cv2.imwrite(save_path, face_img_tight)

                        unknown_registry.append({
                            "label": u_label,
                            "ref_embedding": emb_vector,
                            "data_entries": [{
                                "embedding": emb_vector.tolist(),
                                "face_crop": save_path
                            }]
                        })

        if target_embeddings_list:
            db_json.append({
                "label": person_label,
                "embeddings": target_embeddings_list
            })

    # Merge Unknowns
    print(f"\n[INFO] Merging {len(unknown_registry)} unique Unknown groups...")
    for u_data in unknown_registry:
        db_json.append({
            "label": u_data["label"],
            "embeddings": u_data["data_entries"]
        })

    json_path = os.path.join(output_root, "embeddings.json")
    with open(json_path, "w") as f:
        json.dump(db_json, f, indent=4)

    print(f"\n[SUCCESS] Database built at: {json_path}")


if __name__ == "__main__":
    build_database_optimized(output_root="db_1")