import config as cfg
import cv2
import os
from ultralytics import YOLO

yolo_model = YOLO(cfg.YOLO_PATH)
face_model = YOLO(cfg.YOLO_FACE_PATH)

if __name__ == "__main__":
  image_path = "data/dataset/test/Indoor/Non-masked/Andres - Indoor - 9C.png"
  results = yolo_model(image_path)[0]

  img = results.orig_img

  person_count = 0

  for box in results.boxes:
    cls = int(box.cls[0])

    if cls == 0:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop from original image
        crop = img[y1:y2, x1:x2]

        # Skip invalid/empty crops
        if crop.size == 0:
            continue

        person_count += 1
        out_path = os.path.join(cfg.RESULTS_PATH, f"person_4.jpg")
        cv2.imwrite(out_path, crop)
        print(f"Saved: {out_path}")

  print(f"\nTotal persons cropped: {person_count}")

  # annotated = results[0].plot()
  # output_path = os.path.join(cfg.RESULTS_PATh, "annotated2.jpg")
  # cv2.imwrite(output_path, annotated)

  # print(f"Saved annotated image to: {output_path}")


  # Input image

  # img = results.orig_img.copy()  # original BGR image from OpenCV

  # person_boxes = []
  # for box in results.boxes:
  #     cls = int(box.cls[0])
      
  #     # YOLO COCO class 0 = person
  #     if cls == 0:
  #         x1, y1, x2, y2 = map(int, box.xyxy[0])
  #         area = (x2 - x1) * (y2 - y1)
          
  #         person_boxes.append({
  #             "coords": (x1, y1, x2, y2),
  #             "conf": float(box.conf[0]),
  #             "area": area
  #         })

  # # If we found no people
  # if not person_boxes:
  #     print("No person detected.")
  #     exit()

  # # Pick the person closest to the camera (largest bbox area)
  # closest = max(person_boxes, key=lambda x: x["area"])
  # x1, y1, x2, y2 = closest["coords"]

  # # Draw only this bounding box
  # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
  # cv2.putText(img, f"Person {closest['conf']:.2f}",
  #             (x1, y1 - 10),
  #             cv2.FONT_HERSHEY_SIMPLEX,
  #             0.8, (0, 255, 0), 2)

  # # Save result
  # output_path = os.path.join(cfg.RESULTS_PATh, "closest_person.jpg")
  # cv2.imwrite(output_path, img)

  # print("Saved:", output_path)