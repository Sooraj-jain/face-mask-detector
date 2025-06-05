# Face Mask Detection System 😷

A production-ready deep learning system for real-time face mask detection, showcasing end-to-end ML engineering from model development to deployment.

## Project Overview

This project demonstrates professional ML engineering practices including:
- Data pipeline development
- Model architecture design and training
- Real-time inference optimization
- Production deployment with monitoring
- Web application integration

### Key Features

- **ML Pipeline**:
  - Automated data preprocessing and augmentation
  - Configurable model architecture
  - Training with validation monitoring
  - Model evaluation and metrics tracking
  - Artifact versioning and management

- **Model Architecture**:
  - Custom CNN optimized for real-time inference
  - BatchNormalization for training stability
  - Dropout layers for regularization
  - Multi-stage convolutional blocks

- **Production Features**:
  - Real-time face detection and mask classification
  - Confidence score estimation
  - Performance monitoring and logging
  - Web-based deployment using Streamlit
  - Optimized inference pipeline

## Technical Architecture

### Data Pipeline
```
Raw Images → Preprocessing → Augmentation → Training/Validation Split
└── Preprocessing: Resize, Normalize, Convert to RGB
└── Augmentation: Rotation, Flip, Brightness Adjustment
└── Split: 80% Training, 20% Validation
```

### Model Architecture
```
Input (100x100x3)
│
├── Conv Block 1
│   ├── Conv2D(32, 3x3, ReLU)
│   ├── BatchNorm
│   └── MaxPool(2x2)
│
├── Conv Block 2
│   ├── Conv2D(64, 3x3, ReLU)
│   ├── BatchNorm
│   └── MaxPool(2x2)
│
├── Conv Block 3
│   ├── Conv2D(128, 3x3, ReLU)
│   ├── BatchNorm
│   └── MaxPool(2x2)
│
├── Flatten
├── Dense(256, ReLU)
├── BatchNorm
├── Dropout(0.5)
└── Dense(2, Softmax)
```

## Project Structure

```
FaceMaskDetector/
├── train_mask_model.py      # Training pipeline
├── app_live.py             # Production web application
├── model_artifacts/        # Training artifacts
│   ├── best_model.h5      # Best model checkpoint
│   ├── config.json        # Training configuration
│   ├── metrics/           # Training metrics
│   └── plots/            # Performance visualizations
├── requirements.txt       # Python dependencies
└── face-mask-dataset/    # Training dataset
    ├── with_mask/
    └── without_mask/
```

## Model Performance

The model achieves the following metrics on the validation set:
- Accuracy: ~98%
- AUC-ROC: ~0.99
- Real-time inference: ~30ms per frame

Detailed metrics and visualizations are generated during training:
- Training/validation accuracy curves
- Loss convergence analysis
- Confusion matrix
- Classification report

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FaceMaskDetector
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline

Train the model with custom configuration:
```bash
python train_mask_model.py
```

The training pipeline:
1. Sets up logging and artifact directories
2. Loads and preprocesses the dataset
3. Builds and trains the model with callbacks:
   - Model checkpointing
   - Early stopping
   - Learning rate scheduling
4. Generates evaluation metrics and visualizations
5. Saves the model and artifacts

### Production Deployment

Launch the web application:
```bash
streamlit run app_live.py
```

The application provides:
1. Real-time video processing
2. Face detection and mask classification
3. Confidence score display
4. Performance monitoring

## Development

### Model Configuration

Key hyperparameters can be modified in `train_mask_model.py`:
```python
CONFIG = {
    "image_size": 100,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "validation_split": 0.2
}
```

### Custom Dataset Training

To train with your own dataset:
1. Organize images in the required directory structure
2. Adjust data augmentation parameters if needed
3. Run the training pipeline
4. Monitor training metrics and artifacts

## Contributing

Contributions welcome! Please check the issues page and follow our contribution guidelines.

## License

[Add your license information here]

## Acknowledgments

- TensorFlow/Keras for the deep learning framework
- OpenCV for computer vision capabilities
- Streamlit for web application deployment 