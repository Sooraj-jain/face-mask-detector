#!/bin/bash
# Script to run the FastAPI server

echo "🚀 Starting Face Mask Detection API..."

# Activate virtual environment and set up Python path
source .FaceMaskDetectorEnv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run API from project root (no need to change directory)
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

echo "✅ API server started at: http://localhost:8000"
echo "📖 API documentation available at: http://localhost:8000/docs" 