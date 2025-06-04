# Face Mask Detector 😷

A real-time face mask detection system using Computer Vision and Deep Learning. This project provides both a command-line interface and a web application for detecting whether people are wearing face masks.

## Features

- Real-time face detection
- Mask/No-mask classification with confidence scores
- Color-coded detection boxes (Green: Mask, Red: No Mask)
- Two interface options:
  - Command-line interface with OpenCV window
  - Web application interface using Streamlit
- Pre-trained deep learning model for accurate detection

## Project Structure

```
FaceMaskDetector/
├── app_live.py           # Streamlit web application
├── detect_mask_webcam.py # Command-line interface
├── train_mask_model.py   # Model training script
├── mask_detector.keras   # Trained model
├── requirements.txt      # Python dependencies
└── face-mask-dataset/    # Training dataset directory
    ├── with_mask/
    └── without_mask/
```

## Requirements

- Python 3.10 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FaceMaskDetector
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application Interface (Recommended)

1. Start the Streamlit web application:
```bash
streamlit run app_live.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)
3. Allow camera access when prompted
4. Position yourself in front of the camera to see the mask detection in action

### Command-line Interface

Run the webcam detection script:
```bash
python detect_mask_webcam.py
```

- Press 'q' to quit the application

### Training Your Own Model (Optional)

If you want to train the model with your own dataset:

1. Organize your dataset in the following structure:
```
face-mask-dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. Run the training script:
```bash
python train_mask_model.py
```

## Model Architecture

The face mask detector uses a Convolutional Neural Network (CNN) with the following architecture:
- Input layer (100x100x3)
- 2 Convolutional layers with MaxPooling
- Flatten layer
- Dense layers with dropout for classification
- Output layer (2 classes: mask/no mask)

## Technical Details

- Face detection: OpenCV's Haar Cascade Classifier
- Deep Learning: TensorFlow/Keras
- Web Interface: Streamlit
- Video Processing: OpenCV and streamlit-webrtc
- Image Size: 100x100 pixels
- Training Split: 80% training, 20% testing

## Performance

The model achieves good accuracy in real-time detection with the following characteristics:
- Works with multiple faces in the frame
- Provides confidence scores for predictions
- Real-time processing with minimal lag
- Handles various lighting conditions

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenCV for the face detection implementation
- TensorFlow and Keras for the deep learning framework
- Streamlit for the web interface 