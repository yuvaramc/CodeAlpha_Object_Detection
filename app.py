import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🎯 Object Detection and Tracking")
st.write("Upload a video to detect and track objects using YOLOv8.")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Object Detection + Tracking
        results = model.track(frame, persist=True)

        annotated_frame = results[0].plot()

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()

    st.success("✅ Video processing completed!")