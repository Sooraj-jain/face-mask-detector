# detect_mask_webcam.py
# This script performs real-time face mask detection using your computer's webcam
# It uses OpenCV for face detection and a trained CNN model for mask classification

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
import urllib.request

# Load the trained mask detection model
#model = load_model("mask_detector.keras")

# Auto-download model from Hugging Face
MODEL_URL = "https://huggingface.co/Sooraj-jain/face-mask-detector/resolve/main/mask_detector.h5"
MODEL_PATH = "mask_detector.h5"

if not os.path.exists(MODEL_PATH):
    print("📦 Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = load_model(MODEL_PATH)

# Initialize face detection using OpenCV's pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Configuration constants
IMAGE_SIZE = 100  # Size that the model expects
LABELS = ["Mask", "No Mask"]  # Classification labels
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract and preprocess the face region
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized / 255.0  # Normalize pixel values
        reshaped = np.reshape(normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # Make prediction using the model
        result = model.predict(reshaped)
        label = np.argmax(result)
        confidence = round(result[0][label] * 100, 2)

        # Draw detection results on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLORS[label], 2)
        text = f"{LABELS[label]} ({confidence}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[label], 2)

    # Display the result
    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
