import config as cfg
import cv2
import os
from ultralytics import YOLO

face_model = YOLO(cfg.YOLO_FACE_PATH)

if __name__ == "__main__":
    image_path = "results/person_1.jpg"

    # Load + resize image for face detector
    img = cv2.imread(image_path)
    results = face_model.predict(img, conf=0.05)

    print(results[0].boxes)
    print(results[0].names)

    # No detections?
    if len(results[0].boxes) == 0:
        print("No face detected.")
        exit()

    # Output folders
    annotated_out = os.path.join(cfg.RESULTS_PATh, "test2.jpg")
    face_crop_dir = os.path.join(cfg.RESULTS_PATh, "faces")
    os.makedirs(face_crop_dir, exist_ok=True)

    # Loop over all faces (usually 1)
    face_count = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        # Save cropped face
        face_count += 1
        face_path = os.path.join(face_crop_dir, f"face_{face_count}.jpg")
        cv2.imwrite(face_path, crop)
        print(f"Saved face: {face_path}")

    # Also save the annotated image (optional)
    results[0].names = {0: "face"}
    annotated = results[0].plot()
    cv2.imwrite(annotated_out, annotated)

    print(f"Saved annotated image to: {annotated_out}")