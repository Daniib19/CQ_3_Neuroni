import config as cfg
import onnxruntime as ort
import os
import cv2
import numpy as np

IMAGE_PATH = "results/faces/face_1.jpg"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(img):
  # Resize to 112x112
  img = cv2.resize(img, (112, 112))

  # BGR → RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Convert to float32
  img = img.astype(np.float32)

  # ArcFace standard: subtract 127.5, divide by 128
  img = (img - 127.5) / 128.0

  # HWC → CHW
  img = np.transpose(img, (2, 0, 1))

  # Batch dimension
  img = np.expand_dims(img, axis=0)
  
  return img

def load_embedding_model(path):
  session = ort.InferenceSession(
    path,
    providers=["CPUExecutionProvider"]
  )
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  return session, input_name, output_name

def get_embedding(face_img, session, input_name, output_name):
  blob = preprocess(face_img)
  embedding = session.run([output_name], {input_name: blob})[0][0]
  embedding = embedding / np.linalg.norm(embedding)
  return embedding

def extract_embedding_from_path(img_path, session, input_name, output_name):
  img = cv2.imread(img_path)
  if img is None:
    raise ValueError(f"Failed to load image: {img_path}")

  # Directly use as face (assumes cropped/aligned)
  face = img.copy()

  emb = get_embedding(face, session, input_name, output_name)
  return emb

def compare_images(img_path1, img_path2, session, input_name, output_name):
  emb1 = extract_embedding_from_path(img_path1, session, input_name, output_name)
  emb2 = extract_embedding_from_path(img_path2, session, input_name, output_name)

  sim = cosine_similarity(emb1, emb2)
  return sim

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
  session, input_name, output_name = load_embedding_model(cfg.FACE_EMBEDD_MODEL)

  # img1 = "results/person_1.jpg"
  # img2 = "results/faces/face_2.jpg"

  # similarity = compare_images(img1, img2, session, input_name, output_name)

  # print(f"Cosine similarity: {similarity:.4f}")
  print(1)