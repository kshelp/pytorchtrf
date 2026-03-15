# demo_image.py
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


# =========================
# Model Load (Streamlit cache)
# =========================
@st.cache_resource
def load_model():
    return YOLO("../models/yolov8m-pose.pt")


# =========================
# Prediction
# =========================
def predict(frame, iou=0.7, conf=0.25):
    results = model(
        source=frame,
        device="0" if torch.cuda.is_available() else "cpu",
        iou=iou,
        conf=conf,
        verbose=False,
    )
    return results[0]


# =========================
# Draw Bounding Boxes
# =========================
def draw_boxes(result, frame):
    if result.boxes is None:
        return frame

    boxes = result.boxes.data.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            1,
        )
    return frame


# =========================
# Draw Keypoints
# =========================
def draw_keypoints(result, frame):
    if result.keypoints is None:
        return frame

    annotator = Annotator(frame, line_width=1)

    # (num_person, num_kpts, 3)
    kps_all = result.keypoints.data.cpu().numpy()

    for person_kps in kps_all:
        # Skeleton
        annotator.kpts(person_kps)

        # Individual keypoints
        for idx, (x, y, score) in enumerate(person_kps):
            if score > 0.5:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    3,
                    (0, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    str(idx),
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

    return annotator.result()


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="YOLOv8 Pose Demo", layout="centered")
st.title("🧍 YOLOv8 Pose Estimation Demo")

model = load_model()

uploaded_file = st.file_uploader(
    "이미지 파일 선택",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    if "image" in uploaded_file.type:
        with st.spinner("포즈 정보 추출 중..."):
            # PIL → OpenCV
            pil_image = Image.open(uploaded_file).convert("RGB")
            np_image = np.asarray(pil_image)
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            # Inference
            result = predict(cv_image)

            # Draw
            image = draw_boxes(result, cv_image)
            image = draw_keypoints(result, image)

            # Show
            st.image(image, channels="BGR", caption="Pose Estimation Result")
