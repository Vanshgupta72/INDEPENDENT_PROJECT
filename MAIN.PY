import cv2
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from datetime import datetime


KNOWN_DIR = "KNOWN_IMAGES"
MODELS_DIR = "MODELS"
EXCEL_PATH = "attendance.xlsx"


if not os.path.exists(KNOWN_DIR):
    raise FileNotFoundError(" KNOWN_IMAGES folder nahi mila")
if not os.path.exists(MODELS_DIR):
    raise FileNotFoundError(" MODELS folder nahi mila")


FACE_CONF = 0.30
DIST_THRESHOLD = 1.25

face_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "deploy.prototxt"),
    os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)

facenet_session = ort.InferenceSession(
    os.path.join(MODELS_DIR, "facenet.onnx"),
    providers=["CPUExecutionProvider"]
)

print("Backend models loaded")


if not os.path.exists(EXCEL_PATH):
    pd.DataFrame(columns=["Name", "Timestamp"]).to_excel(EXCEL_PATH, index=False)


def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    best_box = None
    best_conf = 0

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > FACE_CONF and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            best_conf = conf
            best_box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    return best_box


def get_embedding(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (368, 368))
    face = face.astype("float32") / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    emb = facenet_session.run(
        None, {facenet_session.get_inputs()[0].name: face}
    )[0][0]

    return emb / np.linalg.norm(emb)


def mark_attendance(name):
    if name == "Unknown":
        return

    df = pd.read_excel(EXCEL_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    if not ((df["Name"] == name) & df["Timestamp"].astype(str).str.contains(today)).any():
        df.loc[len(df)] = [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        df.to_excel(EXCEL_PATH, index=False)


known_embeddings = {}
PERSON_NAME = "Vansh"

for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(os.path.join(KNOWN_DIR, file))
        box = detect_face(img)
        if box:
            x1, y1, x2, y2 = box
            face = img[y1:y2, x1:x2]
            if face.size > 0:
                emb = get_embedding(face)
                known_embeddings.setdefault(PERSON_NAME, []).append(emb)

for k in known_embeddings:
    known_embeddings[k] = np.mean(known_embeddings[k], axis=0)
