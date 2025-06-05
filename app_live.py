# app_live.py
# This is a Streamlit web application that provides real-time face mask detection
# using your webcam feed. It provides a user-friendly interface for mask detection.

import streamlit as st
import av
import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Setup Streamlit web interface
st.title("😷 Live Face Mask Detection")

# Model paths
LOCAL_MODEL_PATH = "mask_detector.h5"
BACKUP_MODEL_URL = "https://huggingface.co/Sooraj-jain/face-mask-detector/resolve/main/mask_detector.h5"

# Try to load local model first
try:
    model = load_model(LOCAL_MODEL_PATH)
    st.success("✅ Loaded local model successfully!")
except Exception as e:
    st.warning("⚠️ Could not load local model. Attempting to download from Hugging Face...")
    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            with st.spinner("📦 Downloading model from Hugging Face..."):
                urllib.request.urlretrieve(BACKUP_MODEL_URL, LOCAL_MODEL_PATH)
        model = load_model(LOCAL_MODEL_PATH)
        st.success("✅ Downloaded and loaded model successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Configuration constants
IMAGE_SIZE = 100  # Size that the model expects
LABELS = ["Mask", "No Mask"]  # Classification labels
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

# Initialize face detection using OpenCV's pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class VideoTransformer(VideoTransformerBase):
    """
    Custom video transformer class that processes webcam frames:
    1. Detects faces in each frame
    2. For each face, predicts if a mask is worn
    3. Draws bounding boxes and labels on the frame
    """
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Extract and preprocess the face region
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # Make prediction using the model
            try:
                result = model.predict(face_input)
                label = np.argmax(result)
                confidence = round(result[0][label] * 100, 2)

                # Draw detection results on the frame
                cv2.rectangle(img, (x, y), (x+w, y+h), COLORS[label], 2)
                cv2.putText(img, f"{LABELS[label]} ({confidence}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[label], 2)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        return img

st.write("Allow camera access and stay in frame to see mask predictions in real time.")

# Add some usage instructions
st.markdown("""
### Instructions:
1. Click the 'START' button below
2. Allow camera access when prompted
3. Position your face in front of the camera
4. The app will show if you're wearing a mask or not
""")

# Initialize the WebRTC video stream
webrtc_streamer(
    key="mask-detect",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
