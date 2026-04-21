import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# -------------------------------
# Load YOLO model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# Detection Function
# -------------------------------
def run_detection(frame):
    results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

    if len(results) == 0:
        return frame, []

    r = results[0]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            detections.append({
                "label": model.names.get(cls, str(cls)),
                "confidence": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })

    annotated = r.plot() if hasattr(r, "plot") else frame
    return annotated, detections

# -------------------------------
# WebRTC Video Processor
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        annotated, _ = run_detection(img)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -------------------------------
# UI
# -------------------------------
st.title("😊 Facial Emotion Detection System")

option = st.sidebar.selectbox(
    "Choose Input Type",
    ("Upload Image", "Webcam Snapshot", "Live Webcam")
)

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

        if st.button("Run Detection"):
            annotated, detections = run_detection(image)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detected Image")
            st.json(detections)

# -------------------------------
# SNAPSHOT WEBCAM
# -------------------------------
elif option == "Webcam Snapshot":
    picture = st.camera_input("Capture an image")

    if picture is not None:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if st.button("Run Detection"):
            annotated, detections = run_detection(image)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detected Image")
            st.json(detections)

# -------------------------------
# LIVE WEBCAM
# -------------------------------
elif option == "Live Webcam":
    st.info("Starting live webcam... Click 'Start' below.")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )