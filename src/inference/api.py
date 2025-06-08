"""
Face Mask Detection - Model Serving API
-------------------------------------
A FastAPI-based REST API for serving the face mask detection model.
Provides endpoints for:
1. Model information
2. Single image prediction
3. Model health check
4. Performance metrics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import uvicorn
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import Dict, Any
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Mask Detection API",
    description="API for real-time face mask detection using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATHS = {
    'h5': "models/mask_detector.h5",
    'keras': "models/mask_detector.keras"
}
IMAGE_SIZE = 100
LABELS = ["Mask", "No Mask"]

# Performance monitoring
REQUESTS = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "avg_inference_time": 0
}

class ModelResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    confidence: float
    processing_time: float

def load_face_detector():
    """Loads OpenCV's face detector."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return detector

def initialize_model():
    """
    Loads and initializes the model.
    Tries both .h5 and .keras formats.
    """
    for format_type, path in MODEL_PATHS.items():
        try:
            logger.info(f"Attempting to load model from {path}...")
            model = load_model(path)
            # Warm up the model
            model.predict(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)))
            logger.info(f"✅ Successfully loaded model from {path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {str(e)}")
            continue
    
    # If we get here, both formats failed
    error_msg = "Failed to load model from both .h5 and .keras formats"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Initialize components
try:
    model = initialize_model()
    face_detector = load_face_detector()
    logger.info("✅ Model and face detector initialized successfully")
except Exception as e:
    logger.error(f"❌ Initialization failed: {str(e)}")
    raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Face Mask Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": [
            "/predict",
            "/health",
            "/metrics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Verify model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Test inference
        test_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        model.predict(test_input)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics."""
    return {
        "total_requests": REQUESTS["total"],
        "successful_requests": REQUESTS["successful"],
        "failed_requests": REQUESTS["failed"],
        "average_inference_time": REQUESTS["avg_inference_time"],
        "uptime": "N/A"  # TODO: Add uptime tracking
    }

@app.post("/predict", response_model=ModelResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict mask presence in an uploaded image.
    
    Parameters:
    - file: Image file to analyze
    
    Returns:
    - prediction: Mask or No Mask
    - confidence: Prediction confidence score
    - processing_time: Time taken for inference
    """
    REQUESTS["total"] += 1
    start_time = datetime.now()
    
    try:
        # Read and preprocess image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in image")
        
        # Process the first detected face
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Make prediction
        prediction = model.predict(face)[0]
        label_idx = np.argmax(prediction)
        confidence = float(prediction[label_idx])
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        REQUESTS["successful"] += 1
        REQUESTS["avg_inference_time"] = (
            (REQUESTS["avg_inference_time"] * (REQUESTS["successful"] - 1) + processing_time)
            / REQUESTS["successful"]
        )
        
        return ModelResponse(
            prediction=LABELS[label_idx],
            confidence=confidence,
            processing_time=processing_time
        )
        
    except HTTPException:
        REQUESTS["failed"] += 1
        raise
        
    except Exception as e:
        REQUESTS["failed"] += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 