# app_live.py
# This is a Streamlit web application that provides real-time face mask detection
# using your webcam feed. It provides a user-friendly interface for mask detection.

import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading
import os
import urllib.request
import sys
from utils.model_utils import load_face_mask_model

# Add error handling for imports
try:
    # Setup Streamlit web interface
    st.title("😷 Live Face Mask Detection")

    # Add clean, consolidated instructions
    st.info("""
    ### Quick Start Guide
    1. Click **START** below and allow camera access when prompted.
    2. The green camera light will turn on immediately, but the video feed takes time (up to a minute) to initialize
    3. Position yourself in front of the camera.
    4. Real-time mask detection will begin automatically.

    💡 Note: Initial loading takes about a minute after you hit *START* as this runs on a free server. 
           Thank you for your patience! 
    """)

    # Load latest model version
    model = load_face_mask_model("latest")

    # Configuration constants
    IMAGE_SIZE = 100  # Size that the model expects
    LABELS = ["Mask", "No Mask"]  # Classification labels
    COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

    # Robust Haar cascade classifier loading for cloud deployment
    @st.cache_resource
    def load_face_cascade():
        """Load Haar cascade classifier with fallback for cloud environments."""
        cascade_file = "haarcascade_frontalface_default.xml"
        
        # Try multiple paths for the cascade file
        possible_paths = [
            cv2.data.haarcascades + cascade_file,
            os.path.join(os.getcwd(), cascade_file),
            cascade_file
        ]
        
        for path in possible_paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    st.success("✅ Face detection model loaded successfully!")
                    return cascade
            except Exception:
                continue
        
        # If local file not found, download it
        st.info("📦 Downloading face detection model...")
        try:
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(cascade_url, cascade_file)
            cascade = cv2.CascadeClassifier(cascade_file)
            if not cascade.empty():
                st.success("✅ Face detection model ready!")
                return cascade
        except Exception as e:
            st.error(f"❌ Error loading face detection model: {str(e)}")
            st.stop()
        
        st.error("❌ Could not load face detection model")
        st.stop()

    # Initialize face detection using OpenCV's pre-trained Haar Cascade classifier
    face_cascade = load_face_cascade()

    # Lock for thread-safe model prediction
    prediction_lock = threading.Lock()

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.prediction_cache = []
            self.cache_size = 4  # Balanced cache size for stability
            self.process_every_n_frames = 1  # Process every frame for accuracy
            
        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Process every frame for better accuracy
            # Convert to slightly smaller size for balanced processing
            scale_factor = 0.75  # Increased from 0.5 for better accuracy
            small_frame = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Balanced face detection parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,     # More accurate detection
                minNeighbors=7,      # Balanced for accuracy and speed
                minSize=(int(40*scale_factor), int(40*scale_factor)),  # Slightly smaller min size
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Scale back the face coordinates
            faces = [(int(x/scale_factor), int(y/scale_factor), 
                     int(w/scale_factor), int(h/scale_factor)) 
                    for (x, y, w, h) in faces]

            with prediction_lock:
                for (x, y, w, h) in faces:
                    # Balanced padding for better face region
                    padding_x = int(w * 0.12)  # 12% padding
                    padding_y = int(h * 0.12)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(img.shape[1], x + w + padding_x)
                    y2 = min(img.shape[0], y + h + padding_y)
                    
                    face = img[y1:y2, x1:x2]
                    face_resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                    face_normalized = face_resized / 255.0
                    face_input = np.reshape(face_normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

                    try:
                        result = model.predict(face_input, verbose=0)
                        label = np.argmax(result)
                        confidence = result[0][label]
                        
                        self.prediction_cache.append((label, confidence))
                        if len(self.prediction_cache) > self.cache_size:
                            self.prediction_cache.pop(0)
                        
                        if len(self.prediction_cache) >= 2:
                            # Weighted voting system for better accuracy
                            votes = {0: 0.0, 1: 0.0}
                            recent_weight = 1.2  # Slight weight to recent predictions
                            
                            for i, (pred_label, pred_conf) in enumerate(self.prediction_cache):
                                weight = recent_weight if i >= len(self.prediction_cache) - 2 else 1.0
                                votes[pred_label] += pred_conf * weight
                            
                            final_label = max(votes.items(), key=lambda x: x[1])[0]
                            relevant_confidences = [conf for (lbl, conf) in self.prediction_cache if lbl == final_label]
                            avg_confidence = sum(relevant_confidences) / len(relevant_confidences)
                            confidence_percentage = round(avg_confidence * 100, 2)
                            
                            if confidence_percentage > 82:  # Balanced threshold
                                color = COLORS[final_label]
                                label_text = f"{LABELS[final_label]} ({confidence_percentage}%)"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                
                                # Add background for better text visibility
                                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                                
                                # Add prediction text
                                cv2.putText(img, label_text, (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
            
            return img

    # Initialize the WebRTC video stream with balanced settings
    webrtc_streamer(
        key="mask-detect",
        video_processor_factory=VideoTransformer,
        media_stream_constraints={
            "video": {
                "frameRate": {"ideal": 20},  # Balanced frame rate
                "width": {"ideal": 800},     # Balanced resolution
                "height": {"ideal": 600},
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
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]}
            ],
            "iceTransportPolicy": "all",
            "bundlePolicy": "max-bundle",
        }
    )

except Exception as e:
    st.error(f"❌ Application Error: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    st.error(f"Error details: {sys.exc_info()}")
    st.stop()
