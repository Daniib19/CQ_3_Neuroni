import os
import cv2
import json
import numpy as np
import config as cfg

from ultralytics import YOLO
import onnxruntime as ort
from insightface.app import FaceAnalysis

person_model = YOLO(cfg.YOLO_PATH)
face_model   = YOLO(cfg.YOLO_FACE_PATH)
embedd_model = cfg.FACE_EMBEDD_MODEL

app_align = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
app_align.prepare(ctx_id=0)

def detect_and_crop_persons(fullres_img, yolo_model, conf_threshold=0.25):
    h_full, w_full = fullres_img.shape[:2]

    target_w, target_h = 1280, 720
    resized = cv2.resize(fullres_img, (target_w, target_h))
    results = yolo_model.predict(resized, conf=conf_threshold)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return []

    scale_x = w_full / target_w
    scale_y = h_full / target_h

    crops = []

    for box in results[0].boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_full, x2); y2 = min(h_full, y2)

        crop = fullres_img[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)

    return crops


# ============================================================
# FACE DETECTION
# ============================================================

def extract_face_from_person(person_img, face_model):
    results = face_model.predict(person_img, conf=0.05)
    if len(results[0].boxes) == 0:
        return None

    best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    h, w = person_img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    face = person_img[y1:y2, x1:x2]
    return face if face.size > 0 else None


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name, s.get_outputs()[0].name


session, embed_input, embed_output = load_embedding_model(embedd_model)

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    return np.transpose(img, (2, 0, 1))[None]


def get_embedding(face_img, session, input_name, output_name):
    faces = app_align.get(face_img)
    aligned = faces[0].aligned if len(faces) else face_img

    blob = preprocess(aligned)
    emb = session.run([output_name], {input_name: blob})[0][0]
    return emb / np.linalg.norm(emb)


# ============================================================
# DATABASE BUILDER (WITH UNKNOWN PERSON FOLDERS)
# ============================================================

def build_database(db_root="data/dataset/db", output_root="db_test"):
    persons_dict = {}

    # Group images by real person
    for root, _, files in os.walk(db_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                person = f.split("-")[0].strip()
                img_path = os.path.join(root, f)
                persons_dict.setdefault(person, []).append(img_path)

    selected = list(persons_dict.keys())[:10]
    print("[INFO] Selected persons:", selected)

    os.makedirs(output_root, exist_ok=True)

    db_json = []
    unknown_counter = 1
    unknown_person_map = {}  # maps unknown_X -> embeddings list

    # MAIN LOOP
    for person in selected:
        print(f"\n[PERSON] {person}")

        # Create folder for real person
        person_folder = os.path.join(output_root, f"person_{person}")
        os.makedirs(person_folder, exist_ok=True)

        image_paths = persons_dict[person][:2]  # exactly 2 images
        real_embeddings = []

        for idx, img_path in enumerate(image_paths, 1):
            print("  →", img_path)

            img = cv2.imread(img_path)
            if img is None:
                print("  [WARN] Could not read image")
                continue

            persons = detect_and_crop_persons(img, person_model)
            if not persons:
                print("  [WARN] No persons in image")
                continue

            areas = [p.shape[0] * p.shape[1] for p in persons]
            closest = int(np.argmax(areas))

            # === REAL PERSON FACE ===
            real_face = extract_face_from_person(persons[closest], face_model)
            if real_face is not None:
                save_path = os.path.join(person_folder, f"{person}_face_{idx}.jpg")
                cv2.imwrite(save_path, real_face)

                emb = get_embedding(real_face, session, embed_input, embed_output)
                real_embeddings.append({
                    "embedding": emb.tolist(),
                    "face_crop": save_path
                })

            # === UNKNOWN FACES ===
            for u_idx, crop in enumerate(persons):
                if u_idx == closest:
                    continue  # skip real one

                unknown_face = extract_face_from_person(crop, face_model)
                if unknown_face is None:
                    continue

                # Assign unknown label
                label = f"Unknown_{unknown_counter}"
                folder = os.path.join(output_root, f"person_{label}")

                # Create folder only once
                if label not in unknown_person_map:
                    os.makedirs(folder, exist_ok=True)
                    unknown_person_map[label] = []

                # Save face
                face_name = f"{label}_face_{len(unknown_person_map[label])+1}.jpg"
                save_path = os.path.join(folder, face_name)
                cv2.imwrite(save_path, unknown_face)

                emb = get_embedding(unknown_face, session, embed_input, embed_output)

                unknown_person_map[label].append({
                    "embedding": emb.tolist(),
                    "face_crop": save_path
                })

            # Increment unknown ID only after processing a frame
            unknown_counter += 1

        db_json.append({
            "label": person,
            "embeddings": real_embeddings
        })

    # Add unknown people to JSON
    for label, embeds in unknown_person_map.items():
        db_json.append({
            "label": label,
            "embeddings": embeds
        })

    # Save JSON
    json_path = os.path.join(output_root, "embeddings.json")
    with open(json_path, "w") as f:
        json.dump(db_json, f, indent=4)

    print("\n[✔] Database saved to:", json_path)

if __name__ == "__main__":
  build_database(output_root="db_1")