import os
import json
import cv2
import time
import numpy as np
import torch
import warnings
import config as cfg
import onnxruntime as ort
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# --- 0. SUPPRESS WARNINGS & SETUP ---
warnings.filterwarnings("ignore")

# [CRITICAL] 0.13 Permissive Threshold
OUR_THRESHOLD = 0.13
# [CLUSTERING] Threshold to merge two "Unknowns" into one identity
CLUSTERING_THRESHOLD = 0.30

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# --- DIAGNOSTICS ---
print("\n" + "=" * 50)
print("      ACTIVE LEARNING PIPELINE (CATEGORIZED)      ")
print("=" * 50)
print(f"[INFO] Device: {DEVICE.upper()}")
print(f"[INFO] Rec Threshold: {OUR_THRESHOLD}")
print(f"[INFO] Cluster Threshold: {CLUSTERING_THRESHOLD}")
print(f"[INFO] Mode: Auto-Add New Persons + Folder Categorization")

# ============================================================
# 1. MODEL INITIALIZATION
# ============================================================

if not os.path.exists(cfg.YOLO_PATH):
    print(f"CRITICAL: YOLO model not found at {cfg.YOLO_PATH}")
    exit()

# A. YOLO (Body Detection)
person_model = YOLO(cfg.YOLO_PATH)
person_model.to(DEVICE)

# B. FACE DETECTOR (SCRFD)
# Using permissive settings to catch faces
print("[INFO] Loading Face Detector (buffalo_l)...")
face_detector = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'],
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_detector.prepare(ctx_id=0, det_size=(256, 256), det_thresh=0.05)

# [SPEED] ONNX Graph Optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# C. EMBEDDING MODEL
embed_session = ort.InferenceSession(cfg.FACE_EMBEDD_MODEL, sess_options,
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
embed_input_name = embed_session.get_inputs()[0].name
embed_output_name = embed_session.get_outputs()[0].name


# ============================================================
# 2. PIPELINE FUNCTIONS
# ============================================================

def get_best_person(frame):
    """Stage 1: Detect Person (Permissive)."""
    results = person_model.predict(frame, conf=0.15, imgsz=640, half=True, verbose=False, classes=[0])[0]
    valid_persons = []
    if not results.boxes: return None

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        if (x2 - x1) * (y2 - y1) < 500: continue
        valid_persons.append((x1, y1, x2, y2, float(box.conf[0])))

    if not valid_persons: return None
    valid_persons.sort(key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), reverse=True)
    return valid_persons[0][:4]


def get_head_crop(frame, person_box):
    """Stage 2: Digital Zoom (55% Height)."""
    img_h, img_w = frame.shape[:2]
    px1, py1, px2, py2 = person_box
    pw, ph = px2 - px1, py2 - py1

    crop_h = int(ph * 0.55)
    crop_y2 = min(img_h, py1 + crop_h)
    pad_x = int(pw * 0.20)
    crop_x1 = max(0, px1 - pad_x)
    crop_x2 = min(img_w, px2 + pad_x)

    return frame[py1:crop_y2, crop_x1:crop_x2]


def preprocess_batch(imgs):
    resized_imgs = [cv2.resize(img, (112, 112)) for img in imgs]
    batch = np.stack(resized_imgs, axis=0)
    batch = batch[..., ::-1]  # BGR -> RGB
    batch = batch.astype(np.float32)
    batch = (batch - 127.5) / 128.0
    batch = np.transpose(batch, (0, 3, 1, 2))
    return batch


def get_embedding_TTA_Batch(head_crop):
    """Batch TTA with Permissive Gate."""
    faces = face_detector.get(head_crop)
    if not faces: return None, None, 0.0

    face_obj = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]

    # [PERMISSIVE] Allow low quality to catch everything
    if face_obj.det_score < 0.15:
        return None, None, face_obj.det_score

    aligned_face = face_align.norm_crop(head_crop, landmark=face_obj.kps)
    aligned_face_flip = cv2.flip(aligned_face, 1)

    batch_blob = preprocess_batch([aligned_face, aligned_face_flip])
    embeddings = embed_session.run([embed_output_name], {embed_input_name: batch_blob})[0]

    final_emb = np.sum(embeddings, axis=0)
    final_emb = final_emb / np.linalg.norm(final_emb)

    return final_emb, aligned_face, face_obj.det_score


# ============================================================
# 3. DATABASE & CLUSTERING LOGIC
# ============================================================

def prepare_fast_database(json_path):
    """Loads DB and returns both Matrix (for fast search) and Raw List (for appending)."""
    if not os.path.exists(json_path):
        return None, None, []

    with open(json_path, "r") as f:
        db_data = json.load(f)

    emb_list = []
    label_list = []

    for person_entry in db_data:
        label = person_entry["label"]
        for sample in person_entry["embeddings"]:
            emb_list.append(sample["embedding"])
            label_list.append(label)

    if not emb_list:
        return None, None, db_data

    return np.array(emb_list, dtype=np.float32), label_list, db_data


def register_new_person(emb, face_img, source_path, db_data, new_persons_registry):
    """
    CLUSTERING LOGIC:
    1. Check if this 'Unknown' matches a recently added 'New Person'.
    2. If yes -> Merge.
    3. If no -> Create New ID.
    """
    best_sim = -1.0
    best_idx = -1

    # Check against newly created people
    for idx, person in enumerate(new_persons_registry):
        # Compare with the first embedding of that new person (Centroid approx)
        ref_emb = np.array(person["embeddings"][0]["embedding"])
        sim = np.dot(emb, ref_emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    # Decision
    if best_sim > CLUSTERING_THRESHOLD:
        # It's a match to a recently found new person!
        target_person = new_persons_registry[best_idx]
        label = target_person["label"]
        count = len(target_person["embeddings"])

        # Save crop
        save_dir = os.path.join("db_1", f"person_{label}")
        # Ensure dir exists
        os.makedirs(save_dir, exist_ok=True)

        save_name = f"{label}_{count}.jpg"
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, face_img)

        # Append data
        target_person["embeddings"].append({
            "embedding": emb.tolist(),
            "face_crop": save_path,
            "source": source_path
        })
        return label, True  # True = Merged existing

    else:
        # It's a completely new person!
        # Find next available ID
        existing_ids = [int(p["label"].split("_")[-1]) for p in new_persons_registry if "Unknown_" in p["label"]]

        # We must also look through the *entire* existing database (db_data) for existing Unknown IDs
        for entry in db_data:
            if "Unknown_" in entry["label"]:
                try:
                    existing_ids.append(int(entry["label"].split("_")[-1]))
                except ValueError:
                    continue  # Skip if ID suffix isn't an integer

        next_id = max(existing_ids) + 1 if existing_ids else 1

        label = f"Unknown_{next_id}"
        save_dir = os.path.join("db_1", f"person_{label}")
        os.makedirs(save_dir, exist_ok=True)

        save_name = f"{label}_0.jpg"
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, face_img)

        new_entry = {
            "label": label,
            "embeddings": [{
                "embedding": emb.tolist(),
                "face_crop": save_path,
                "source": source_path
            }]
        }

        new_persons_registry.append(new_entry)
        db_data.append(new_entry)  # Add to main DB list for final save
        return label, False  # False = Created new


# ============================================================
# 4. CATEGORIZATION LOGIC
# ============================================================

def determine_category(img_path):
    """
    Classifies image into one of 4 categories based on the FOLDER PATH.
    It looks for 'mask'/'outdoor'/'indoor' in the directory structure.
    FIXED: Defaults to 'nonmasked' if 'mask' is NOT in path.
    """
    path_lower = img_path.lower()

    # 1. Determine Mask Status
    if "mask" in path_lower and "non" not in path_lower and "no_" not in path_lower:
        is_masked = True
    else:
        is_masked = False  # Default to non-masked

    # 2. Determine Environment
    if "outdoor" in path_lower:
        is_outdoor = True
    elif "indoor" in path_lower:
        is_outdoor = False
    else:
        is_outdoor = False  # Default assumption

    # 3. Construct Category Key
    if is_masked and is_outdoor:
        return "masked_outdoor"
    elif is_masked and not is_outdoor:
        return "masked_indoor"
    elif not is_masked and is_outdoor:
        return "nonmasked_outdoor"
    else:
        return "nonmasked_indoor"


# ============================================================
# 5. EVALUATION & ACTIVE LEARNING LOOP
# ============================================================

def process_image(img_path, emb_matrix, label_list, db_data, new_persons_registry):
    filename = os.path.basename(img_path)
    # Ground Truth: Assume the folder structure is '.../Name/...' or filename starts with Name
    # Assuming filename like "Diego - Indoor.jpg" or parent folder is "Diego"
    # To be safe, let's look at the parent folder name if filename doesn't have '-'
    if "-" in filename:
        gt_label = filename.split("-")[0].strip()
    else:
        # Fallback: Use parent directory name
        gt_label = os.path.basename(os.path.dirname(img_path))

    category = determine_category(img_path)  # Pass full path for folder checking

    print(f"Processing: {filename} | Category: {category} | GT: {gt_label}")

    img = cv2.imread(img_path)
    if img is None: return None

    # 1. Detect
    person_box = get_best_person(img)
    if person_box is None: return None  # SKIP (Don't count non-detected)

    # 2. Crop
    head_crop = get_head_crop(img, person_box)

    # 3. Embed
    emb, face_img, score = get_embedding_TTA_Batch(head_crop)
    if emb is None: return None  # SKIP (Don't count non-detected faces)

    # 4. Recognize against STATIC DB
    final_pred = "Unknown"
    final_score = 0.0

    if emb_matrix is not None:
        scores = np.dot(emb_matrix, emb)
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        if best_score >= OUR_THRESHOLD:
            final_pred = label_list[best_idx]
            final_score = best_score

    # 5. Active Learning Logic
    action = "MATCH"
    if final_pred == "Unknown":
        # It's unknown, so we register it
        final_pred, is_merged = register_new_person(emb, face_img, img_path, db_data, new_persons_registry)
        final_score = 1.0  # New registration implies 100% match to self
        action = "MERGED" if is_merged else "CREATED"

    # 6. Status Logic for Metrics
    if final_pred == gt_label:
        status = "TRUE_POS"
    elif "Unknown" in final_pred and ("Unknown" in gt_label or "New Person" in gt_label):
        status = "TRUE_POS"  # Correctly handled unknown
    elif "Unknown" in final_pred and gt_label not in ["Unknown", "New Person"]:
        status = "FALSE_NEG"  # Should have been known, but marked as new unknown
    else:
        status = "FALSE_POS"  # Confused A for B

    return {
        "path": filename,
        "gt": gt_label,
        "pred": final_pred,
        "score": float(final_score),
        "status": status,
        "action": action,
        "category": category
    }


def calculate_metrics_by_category(results):
    """
    Calculates TP, FP, FN and Precision for the whole dataset AND per category.
    """
    categories = ["masked_outdoor", "nonmasked_outdoor", "masked_indoor", "nonmasked_indoor"]

    # Initialize structure
    stats = {cat: {"TP": 0, "FP": 0, "FN": 0, "Total": 0} for cat in categories}
    stats["OVERALL"] = {"TP": 0, "FP": 0, "FN": 0, "Total": 0}

    for r in results:
        cat = r["category"]
        status = r["status"]

        # Increment Category Stats
        if status == "TRUE_POS":
            stats[cat]["TP"] += 1
        elif status == "FALSE_POS":
            stats[cat]["FP"] += 1
        elif status == "FALSE_NEG":
            stats[cat]["FN"] += 1
        stats[cat]["Total"] += 1

        # Increment Overall Stats
        if status == "TRUE_POS":
            stats["OVERALL"]["TP"] += 1
        elif status == "FALSE_POS":
            stats["OVERALL"]["FP"] += 1
        elif status == "FALSE_NEG":
            stats["OVERALL"]["FN"] += 1
        stats["OVERALL"]["Total"] += 1

    # Calculate Derived Metrics (Precision, Recall, F1)
    final_metrics = {}
    for key, data in stats.items():
        tp = data["TP"]
        fp = data["FP"]
        fn = data["FN"]

        # Precision = TP / (TP + FP + FN) [As requested]
        precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        final_metrics[key] = {
            "TP": tp, "FP": fp, "FN": fn, "Total": data["Total"],
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4)
        }

    return final_metrics


def write_results_to_file(metrics, avg_time, threshold, filename="results.txt"):
    """Writes the results to file using the specific visual format requested."""

    avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
    overall = metrics["OVERALL"]

    output = f"""
========================================
Avg Time : {avg_time:.4f} s | Avg FPS : {avg_fps:.2f}
----------------------------------------
Precision: {overall['Precision']:.4f}
Recall   : {overall['Recall']:.4f}
F1 Score : {overall['F1_Score']:.4f}
----------------------------------------
TP: {overall['TP']} | FP: {overall['FP']} | FN: {overall['FN']}
========================================

--- CATEGORY BREAKDOWN ---
"""
    # Add category details below the main block
    categories = ["masked_outdoor", "nonmasked_outdoor", "masked_indoor", "nonmasked_indoor"]
    for cat in categories:
        data = metrics.get(cat, {"TP": 0, "FP": 0, "FN": 0, "Total": 0})
        cat_title = cat.replace("_", " ").upper()
        output += f"{cat_title}: TP={data['TP']} | FP={data['FP']} | FN={data['FN']} (Total: {data['Total']})\n"

    output += "\n"

    # Append to file
    with open(filename, "a") as f:
        f.write(output)

    print(f"\n[INFO] Results appended to {filename}")


if __name__ == "__main__":
    TEST_ROOT = "data/dataset/test"
    DB_PATH = "db_1/embeddings.json"

    print(f"\n[INFO] Loading Database...")
    emb_matrix, label_list, db_data = prepare_fast_database(DB_PATH)

    # Registry to track new people added THIS SESSION
    new_persons_registry = []

    print(f"[START] Active Learning Evaluation...")
    all_results = []
    processing_times = []

    for root, _, files in os.walk(TEST_ROOT):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                path = os.path.join(root, f)
                start_time = time.perf_counter()

                result = process_image(path, emb_matrix, label_list, db_data, new_persons_registry)

                elapsed = time.perf_counter() - start_time

                if result:
                    processing_times.append(elapsed)
                    all_results.append(result)

                    tag = ""
                    if result['action'] == "CREATED":
                        tag = " [NEW ID]"
                    elif result['action'] == "MERGED":
                        tag = " [CLUSTERED]"

                    print(
                        f"[{result['status']}] {result['gt']} -> {result['pred']} ({result['score']:.2f}){tag} | Cat: {result['category']}")

    # Save Updated Database
    if new_persons_registry:
        print(f"\n[DB UPDATE] Saving {len(new_persons_registry)} new identities to {DB_PATH}...")
        with open(DB_PATH, "w") as f:
            json.dump(db_data, f, indent=4)

    if all_results:
        metrics = calculate_metrics_by_category(all_results)
        avg_time = np.mean(processing_times) if processing_times else 0.0

        # Console Output matches the file output format roughly for verification
        print("\n" + "=" * 40)
        print(f"Avg Time : {avg_time:.4f} s | Avg FPS : {1.0 / avg_time if avg_time > 0 else 0:.2f}")
        print("-" * 40)
        print(f"Precision: {metrics['OVERALL']['Precision']}")
        print(f"Recall   : {metrics['OVERALL']['Recall']}")
        print(f"F1 Score : {metrics['OVERALL']['F1_Score']}")
        print("-" * 40)
        print(f"TP: {metrics['OVERALL']['TP']} | FP: {metrics['OVERALL']['FP']} | FN: {metrics['OVERALL']['FN']}")
        print("=" * 40)

        # Write to File
        write_results_to_file(metrics, avg_time, OUR_THRESHOLD)