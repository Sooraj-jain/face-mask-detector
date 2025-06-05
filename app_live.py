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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading

# Setup Streamlit web interface
st.title("😷 Live Face Mask Detection")

# Add prominent loading notice
st.warning("""
⚠️ **Important Notice:**
- After clicking START, please wait for about 1-2 minutes for the video to load
- The green camera light will turn on immediately, but the video feed takes time to initialize
- This is normal for the free version of the application
- Please don't close the tab, your video will appear shortly!
""")

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_face_mask_model():
    LOCAL_MODEL_PATH = "mask_detector.h5"
    BACKUP_MODEL_URL = "https://huggingface.co/Sooraj-jain/face-mask-detector/resolve/main/mask_detector.h5"
    
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
    return model

# Load model
model = load_face_mask_model()

# Configuration constants
IMAGE_SIZE = 100  # Size that the model expects
LABELS = ["Mask", "No Mask"]  # Classification labels
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

# Initialize face detection using OpenCV's pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lock for thread-safe model prediction
prediction_lock = threading.Lock()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.prediction_cache = []  # Prediction history for smoothing
        self.cache_size = 8  # Increased cache size for better smoothing
        
    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Process at full resolution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # More thorough face detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for more accurate detection
            minNeighbors=6,    # More neighbors for more stable detection
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        with prediction_lock:
            for (x, y, w, h) in faces:
                # Add padding to face region for better detection
                padding_x = int(w * 0.1)  # 10% padding
                padding_y = int(h * 0.1)
                
                # Ensure padded coordinates are within image bounds
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(img.shape[1], x + w + padding_x)
                y2 = min(img.shape[0], y + h + padding_y)
                
                # Extract and preprocess the face region with padding
                face = img[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                face_normalized = face_resized / 255.0
                face_input = np.reshape(face_normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

                try:
                    # Get prediction with higher batch size for better accuracy
                    result = model.predict(face_input, verbose=0)
                    label = np.argmax(result)
                    confidence = result[0][label]
                    
                    # Add to prediction cache
                    self.prediction_cache.append((label, confidence))
                    if len(self.prediction_cache) > self.cache_size:
                        self.prediction_cache.pop(0)
                    
                    # Weighted majority voting based on confidence
                    if len(self.prediction_cache) >= 3:  # Wait for minimum predictions
                        # Calculate weighted votes for each class
                        votes = {0: 0.0, 1: 0.0}  # Mask: 0, No Mask: 1
                        for pred_label, pred_conf in self.prediction_cache:
                            votes[pred_label] += pred_conf
                        
                        # Get final prediction
                        final_label = max(votes.items(), key=lambda x: x[1])[0]
                        
                        # Calculate average confidence for the final prediction
                        relevant_confidences = [conf for (lbl, conf) in self.prediction_cache if lbl == final_label]
                        avg_confidence = sum(relevant_confidences) / len(relevant_confidences)
                        confidence_percentage = round(avg_confidence * 100, 2)
                        
                        # Only show high-confidence predictions
                        if confidence_percentage > 70:  # Higher confidence threshold
                            color = COLORS[final_label]
                            label_text = f"{LABELS[final_label]} ({confidence_percentage}%)"
                            
                            # Draw the face region with padding
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            
                            # Add background rectangle for text
                            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(img, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
                            
                            # Add prediction text
                            cv2.putText(img, label_text, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        return img

st.write("Allow camera access and stay in frame to see mask predictions in real time.")

# Add some usage instructions
st.markdown("""
### Instructions:
1. Click the 'START' button below
2. Allow camera access when prompted
3. **Wait for 1-2 minutes** for the video feed to initialize
4. Position your face in front of the camera
5. The app will show if you're wearing a mask or not

> 💡 **Tip:** The initial loading time after clicking START may take up to a minute since this is running on a Free server. 
            Thank you for your patience!
""")

# Initialize the WebRTC video stream with high quality settings
webrtc_streamer(
    key="mask-detect",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={
        "video": {
            "frameRate": {"ideal": 30},  # Maximum frame rate for smooth video
            "width": {"ideal": 1280},    # Higher resolution
            "height": {"ideal": 720},
            "facingMode": "user"
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ],
        "iceTransportPolicy": "all",
        "bundlePolicy": "max-bundle",
        "iceCandidatePoolSize": 1
    },
    video_frame_callback=None,
    mode=WebRtcMode.SENDRECV
)
