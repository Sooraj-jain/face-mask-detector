#!/bin/bash
# Script to run model training pipeline

echo "🏋️  Starting model training pipeline..."

# Activate virtual environment and set up Python path
source .FaceMaskDetectorEnv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training from project root
python src/training/train_mask_model.py

echo "✅ Training completed! Check model_artifacts_* directory for results." 