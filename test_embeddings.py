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
        score = cosine(embedding, entry["embedding"])
        scores.append((entry["label"], score))

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

def add_new_person_to_db(face_img, embedding, source_path, db_json_path, faces_output_dir):
    """
    Saves new face crop + embedding into the database.
    Assigns unique label: New Person X
    """

    # Load the DB
    with open(db_json_path, "r") as f:
        db_entries = json.load(f)

    # Determine next "New Person" index
    new_indices = [
        int(e["label"].split()[-1])
        for e in db_entries
        if e["label"].startswith("New Person")
    ]
    next_id = max(new_indices) + 1 if new_indices else 1

    new_label = f"New Person {next_id}"

    # Save new face crop
    face_filename = f"new_face_{next_id}.jpg"
    face_path = os.path.join(faces_output_dir, face_filename)
    cv2.imwrite(face_path, face_img)

    # Build new entry
    new_entry = {
        "label": new_label,
        "embedding": embedding.tolist(),
        "path": source_path,
        "face_crop": face_path
    }

    # Append + save DB
    db_entries.append(new_entry)

    with open(db_json_path, "w") as f:
        json.dump(db_entries, f, indent=4)

    print(f"[INFO] Added {new_label} to DB → {face_path}")

    # Return label for evaluation
    return new_label

# ============================================================
# Load DB embeddings
# ============================================================
def load_database(json_path):
    with open(json_path, "r") as f:
        db_entries = json.load(f)

    for entry in db_entries:
        entry["embedding"] = np.array(entry["embedding"], dtype=np.float32)

    return db_entries


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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
  TEST_ROOT = "data/dataset/test" 
  evaluate_dataset(TEST_ROOT, db_json="results_w600k/embeddings.json", threshold=OUR_THRESHOLD)
