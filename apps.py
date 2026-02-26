import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os

from MAIN import detect_face, get_embedding, known_embeddings, mark_attendance

st.set_page_config(layout="wide")
st.title("Live Face Recognition Attendance System")

if "run" not in st.session_state:
    st.session_state.run = False

if "cap" not in st.session_state:
    st.session_state.cap = None


left, right = st.columns([2, 1])

with left:
    c1, c2 = st.columns(2)

    if c1.button("▶ Start Camera"):
        st.session_state.run = True
        if st.session_state.cap is None:
            
            st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if c2.button("⛔ Stop Camera"):
        st.session_state.run = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

frame_placeholder = left.empty()


# ATTENDANCE TABLE
def load_attendance():
    if os.path.exists("attendance.xlsx"):
        return pd.read_excel("attendance.xlsx")
    return pd.DataFrame(columns=["Name", "Timestamp"])

right.subheader("📋 Attendance Log")
table_placeholder = right.empty()


# VERY LENIENT SETTINGS
DIST_THRESHOLD = 1.40     # Unknown cam
MIN_FACE_SIZE = 45        # small faces allowed
BLUR_KERNEL = (5, 5)      # noise reduction


# LIVE CAMERA LOOP

if st.session_state.run and st.session_state.cap:
    cap = st.session_state.cap

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        name = "Unknown"

        # FACE DETECTION
        box = detect_face(frame)
        if box:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.shape[0] > MIN_FACE_SIZE and face.shape[1] > MIN_FACE_SIZE:
                # 🔥 noise reduction for low-end camera
                face = cv2.GaussianBlur(face, BLUR_KERNEL, 0)

                emb = get_embedding(face)

                min_dist = float("inf")
                for k, known_emb in known_embeddings.items():
                    dist = np.linalg.norm(known_emb - emb)
                    if dist < min_dist:
                        min_dist = dist
                        name = k

                #  VERY LENIENT UNKNOWN CHECK
                if min_dist > DIST_THRESHOLD:
                    name = "Unknown"

                mark_attendance(name)

            # Draw box + name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        #  DISPLAY 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        table_placeholder.dataframe(
            load_attendance(),
            use_container_width=True
        )

        time.sleep(0.03)

else:
    frame_placeholder.info("Camera is OFF")
    table_placeholder.dataframe(
        load_attendance(),
        use_container_width=True
    )