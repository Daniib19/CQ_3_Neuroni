import os
import json
import cv2
import numpy as np
import config as cfg
from ultralytics import YOLO

from extract_face import (
  detect_and_crop_persons,
  extract_face_from_person,
  load_embedding_model,
  get_embedding
)

OUR_THRESHOLD = 0.23
print(f"[INFO] Using cosine threshold = {OUR_THRESHOLD}")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============================================================
# Recognition
# ============================================================
def recognize(embedding, db_entries, threshold=OUR_THRESHOLD):
    scores = []

    for entry in db_entries:
        person_label = entry["label"]

        # compute best score among this person's embeddings
        best_score = -1  

        for item in entry["embeddings"]:
            db_emb = item["embedding"]
            score = cosine(embedding, db_emb)
            best_score = max(best_score, score)

        scores.append((person_label, best_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_label, best_score = scores[0]
    is_match = best_score >= threshold

    return best_label, best_score, is_match, scores[:3]

# ============================================================
# Evaluate ONE image
# ============================================================
def evaluate_test_image(
    img_path,
    db_entries,
    person_model,
    face_model,
    session, input_name, output_name,
    db_json_path,
    faces_output_dir,
    threshold=OUR_THRESHOLD
):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {img_path}")
        return []

    gt_label = os.path.basename(img_path).split("-")[0].strip()

    persons = detect_and_crop_persons(img, person_model)
    if not persons:
        print(f"[WARN] No persons detected: {img_path}")
        return [{
            "gt": gt_label,
            "pred": "NO_PERSON",
            "score": 0,
            "status": "FALSE_NEG",
            "path": img_path
        }]

    results = []

    for idx, person_img in enumerate(persons, start=1):
        face_img = extract_face_from_person(person_img, face_model)

        if face_img is None:
            results.append({
                "gt": gt_label,
                "pred": "NO_FACE",
                "score": 0,
                "status": "FALSE_NEG",
                "path": img_path
            })
            continue

        emb = get_embedding(face_img, session, input_name, output_name)

        # recognition
        best_label, score, is_match, top3 = recognize(emb, db_entries, threshold)

        # ---------------------------------------
        # Case 1 — MATCHED to someone in DB
        # ---------------------------------------
        if is_match:
            status = "TRUE_POS" if best_label == gt_label else "FALSE_POS"

            results.append({
                "gt": gt_label,
                "pred": best_label,
                "score": score,
                "status": status,
                "path": img_path
            })

        # ---------------------------------------
        # Case 2 — NO MATCH → NEW PERSON
        # ---------------------------------------
        else:
            new_label = add_new_person_to_db(
                face_img=face_img,
                embedding=emb,
                source_path=img_path,
                db_json_path=db_json_path,
                faces_output_dir=faces_output_dir
            )

            results.append({
                "gt": gt_label,
                "pred": new_label,
                "score": score,
                "status": "NEW_PERSON_ADDED",
                "path": img_path
            })

            # Update in-memory DB
            db_entries = load_database(db_json_path)

    return results

def evaluate_dataset_with_autoupdate(
    test_root,
    db_json="results_w600k/embeddings.json",
    threshold=OUR_THRESHOLD
):
    # Load or CREATE database
    db_entries = load_database(db_json)

    person_model = YOLO(cfg.YOLO_PATH)
    face_model   = YOLO(cfg.YOLO_FACE_PATH)
    session, input_name, output_name = load_embedding_model(cfg.FACE_EMBEDD_MODEL)

    results = []
    faces_output_dir = os.path.join(os.path.dirname(db_json), "faces")
    os.makedirs(faces_output_dir, exist_ok=True)

    print("\n======= STARTING DATASET EVALUATION =======\n")

    for root, _, files in os.walk(test_root):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, filename)
            print(f"[EVAL] {img_path}")

            res_list = evaluate_test_image(
                img_path,
                db_entries,
                person_model,
                face_model,
                session, input_name, output_name,
                db_json_path=db_json,
                faces_output_dir=faces_output_dir,
                threshold=threshold
            )

            if res_list:
                # extend (because multiple persons per image)
                results.extend(res_list)

                # Always reload DB in case new person was added
                db_entries = load_database(db_json)

    return results

def add_new_person_to_db(face_img, embedding, source_path, db_json_path, faces_output_dir):
    """
    Creates:
        db_root/
            embeddings.json
            New_person1/
                New_person1_1.jpg
    Instead of:
        faces/New_person1/
    """

    # Load existing database
    if os.path.exists(db_json_path):
        with open(db_json_path, "r") as f:
            raw_db = json.load(f)
    else:
        raw_db = []

    # Determine the root directory where JSON is located
    db_root = os.path.dirname(db_json_path)

    # Determine next "New_personX"
    existing = [p["label"] for p in raw_db if p["label"].startswith("New_person")]
    next_id = len(existing) + 1
    new_label = f"New_person{next_id}"

    # Create the folder directly under db_root (NOT faces/)
    person_folder = os.path.join(db_root, new_label)
    os.makedirs(person_folder, exist_ok=True)

    # Save face crop
    face_filename = f"{new_label}_1.jpg"
    face_path = os.path.join(person_folder, face_filename)
    cv2.imwrite(face_path, face_img)

    # New DB entry
    new_entry = {
        "label": new_label,
        "embeddings": [
            {
                "embedding": embedding.tolist(),
                "face_crop": face_path
            }
        ]
    }

    # Append and save JSON
    raw_db.append(new_entry)

    with open(db_json_path, "w") as f:
        json.dump(raw_db, f, indent=4)

    print(f"[NEW PERSON] Added {new_label} → {face_path}")

    return new_label

# ============================================================
# Load DB embeddings
# ============================================================
def load_database(json_path):
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r") as f:
        raw = json.load(f)

    db_entries = []

    for person in raw:
        label = person.get("label")
        embed_list = []

        for item in person.get("embeddings", []):
            emb = np.array(item["embedding"], dtype=np.float32)
            crop = item.get("face_crop", None)
            embed_list.append({"embedding": emb, "face_crop": crop})

        db_entries.append({
            "label": label,
            "embeddings": embed_list
        })

    return db_entries

def annotate_and_save_faces(img, results, save_path="annotated.jpg"):
    annotated = img.copy()

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("[WARN] No face detections to annotate.")
        return None

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text: class + confidence
        label = f"{cls} ({conf:.2f})"

        # Draw text background
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + 120, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(save_path, annotated)
    print("[✔] Annotated image saved:", save_path)

    return save_path

# ============================================================
# Evaluate entire dataset
# ============================================================
def evaluate_dataset(
    test_root,
    db_json="results_w600k/embeddings.json",
    threshold=OUR_THRESHOLD
):
    db_entries = load_database(db_json)

    person_model = YOLO(cfg.YOLO_PATH)
    face_model   = YOLO(cfg.YOLO_FACE_PATH)
    session, input_name, output_name = load_embedding_model(cfg.FACE_EMBEDD_MODEL)

    results = []

    for root, _, files in os.walk(test_root):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, filename)

            print(f"[EVAL] {img_path}")
            res = evaluate_test_image(
                img_path,
                db_entries,
                person_model,
                face_model,
                session, input_name, output_name,
                db_json_path=db_json,
                faces_output_dir=os.path.join(os.path.dirname(db_json), "faces"),
                threshold=threshold
            )

            if res:
                results.append(res)

    # Summary
    TP = sum(r["status"] == "TRUE_POS" for r in results)
    FP = sum(r["status"] == "FALSE_POS" for r in results)
    FN = sum(r["status"] == "FALSE_NEG" for r in results)
    TN = sum(r["status"] == "TRUE_NEG" for r in results)

    print("\n============== SUMMARY ==============")
    print(f"Threshold used       : {threshold}")
    print(f"True Positives       : {TP}")
    print(f"False Positives      : {FP}")
    print(f"False Negatives      : {FN}")
    print(f"True Negatives       : {TN}")
    print("=====================================\n")

    print("\n===== FALSE POSITIVES =====")
    for r in results:
        if r["status"] == "FALSE_POS":
            print(f"{r['path']} → GT={r['gt']} | Pred={r['pred']} | Score={r['score']:.4f}")

    print("\n===== FALSE NEGATIVES =====")
    for r in results:
        if r["status"] == "FALSE_NEG":
            print(f"{r['path']} → GT={r['gt']} | Pred={r['pred']} | Score={r['score']:.4f}")

    print("\n===== TRUE POSITIVES =====")
    for r in results:
        if r["status"] == "TRUE_POS":
            print(f"{r['path']} → {r['gt']} matched correctly ({r['score']:.4f})")

    return results

# MAIN
# ============================================================
if __name__ == "__main__":
    evaluate_dataset_with_autoupdate(test_root="data/dataset_proprietar", db_json="db_test/embeddings.json", threshold=0.23)
