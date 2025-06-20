# 😷 Face Mask Detection - ML Engineering Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.2-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive **Machine Learning Engineering Portfolio Project** demonstrating end-to-end ML pipeline development, from data preprocessing to production deployment. This project showcases real-time face mask detection using deep learning with multiple deployment options.

## 🎯 Project Overview

This project demonstrates a complete **ML Engineering workflow** including:

- **Data Engineering**: Dataset validation, preprocessing, and augmentation
- **Model Development**: CNN architecture design and training pipeline
- **Model Validation**: Cross-validation, robustness testing, and performance evaluation
- **Model Serving**: REST API and real-time web application
- **Production Deployment**: Cloud deployment with monitoring and error handling
- **DevOps**: CI/CD considerations and infrastructure management

## 🏗️ Architecture & Key Concepts

### ML Engineering Pipeline Components

#### 1. **Data Pipeline** (`train_mask_model.py`)
- **Data Validation**: Automated quality checks and integrity validation
- **Data Preprocessing**: Image normalization, resizing, and augmentation
- **Data Quality Monitoring**: Corrupt image detection and handling
- **Class Balance Validation**: Ensuring sufficient samples per class

#### 2. **Model Development Pipeline**
- **Architecture Design**: CNN with BatchNormalization and Dropout for regularization
- **Hyperparameter Management**: Centralized configuration management
- **Training Orchestration**: Callbacks for early stopping, learning rate scheduling
- **Model Versioning**: Support for multiple model versions with metadata

#### 3. **Model Validation & Testing**
- **Cross-Validation**: K-fold validation for model stability assessment
- **Robustness Testing**: Performance under noise and perturbations
- **Performance Metrics**: Accuracy, precision, recall, F1-score, ROC curves
- **Unit Testing**: Comprehensive test suite for pipeline components

#### 4. **Model Serving Infrastructure**
- **REST API** (`api.py`): FastAPI-based service with health checks and metrics
- **Real-time Web App** (`app_live.py`): Streamlit application with WebRTC
- **Model Loading**: Intelligent fallback between local and cloud storage
- **Error Handling**: Graceful degradation and comprehensive logging

#### 5. **Production Deployment**
- **Cloud Deployment**: Streamlit Cloud with SSL and WebRTC support
- **Containerization**: Docker support for consistent environments
- **Monitoring**: Request tracking, performance metrics, and health checks
- **Configuration Management**: Environment-specific configurations

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenCV dependencies (see `packages.txt`)
- Webcam access (for live detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face-mask-detector.git
cd face-mask-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**

**Option A: Live Web Application (Recommended)**
```bash
streamlit run app_live.py
```

**Option B: REST API**
```bash
python api.py
```

**Option C: Local HTTPS Setup**
```bash
chmod +x run_local.sh
./run_local.sh
```

## 📊 Model Performance

### Architecture
- **Input**: 100x100x3 RGB images
- **Architecture**: CNN with Conv2D, MaxPooling2D, BatchNormalization, Dropout
- **Output**: Binary classification (Mask/No Mask)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout (0.5) and BatchNormalization

### Performance Metrics
- **Accuracy**: ~95% on test set
- **Cross-Validation**: 5-fold CV with mean accuracy >94%
- **Inference Time**: <100ms per frame
- **Model Size**: ~14MB (optimized for deployment)

## 🛠️ Development & Training

### Training Pipeline

1. **Data Preparation**
```bash
# Ensure dataset structure:
face-mask-dataset/
├── with_mask/
│   ├── image1.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    └── ...
```

2. **Model Training**
```bash
python train_mask_model.py
```

3. **Run Tests**
```bash
python test_train_pipeline.py
```

### Configuration Management

The project uses centralized configuration in `config/model_config.json`:
```json
{
    "models": {
        "latest": {
            "version": "v2",
            "file": "mask_detector.h5",
            "url": "https://huggingface.co/Sooraj-jain/face-mask-detector/resolve/main/mask_detector.h5",
            "description": "Enhanced model with improved accuracy",
            "date": "2025-06-06"
        }
    }
}
```

## 🌐 API Documentation

### REST API Endpoints

**Base URL**: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/metrics` | GET | Performance metrics |
| `/predict` | POST | Image prediction |

### Example API Usage

```python
import requests

# Upload image for prediction
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Fork/Clone** the repository
2. **Connect** to Streamlit Cloud
3. **Deploy** with automatic dependency resolution

The app includes robust error handling for cloud environments:
- Automatic model downloading from HuggingFace
- Haar cascade classifier fallback mechanisms
- WebRTC configuration for camera access

### Local Deployment with HTTPS

```bash
# Generate SSL certificates and run with HTTPS
./run_local.sh
```

### Docker Deployment

```dockerfile
# Build and run with Docker
docker build -t face-mask-detector .
docker run -p 8501:8501 face-mask-detector
```

## 🧪 Testing

### Test Suite

```bash
# Run all tests
python test_train_pipeline.py

# Test OpenCV setup
streamlit run test_opencv_setup.py
```

### Test Coverage
- Data validation pipeline
- Model architecture and training
- Cross-validation functionality
- Robustness evaluation
- API endpoints

## 📁 Project Structure

```
face-mask-detector/
├── 📄 app_live.py              # Streamlit web application
├── 📄 api.py                   # FastAPI REST service
├── 📄 train_mask_model.py      # Training pipeline
├── 📄 requirements.txt         # Python dependencies
├── 📄 packages.txt            # System dependencies
├── 📄 run_local.sh            # Local deployment script
├── 📁 config/
│   └── 📄 model_config.json   # Model configuration
├── 📁 models/
│   └── 📄 mask_detector.h5    # Trained model
├── 📁 utils/
│   ├── 📄 model_utils.py      # Model loading utilities
│   ├── 📄 check_models.py     # Model validation
│   └── 📄 upload_model.py     # Model upload utilities
├── 📁 src/
│   └── 📁 utils/              # Source utilities
├── 📄 test_train_pipeline.py  # Unit tests
├── 📄 test_opencv_setup.py    # OpenCV diagnostics
└── 📄 DEPLOYMENT_GUIDE.md     # Deployment documentation
```

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH`: Path to model file
- `API_HOST`: API server host
- `API_PORT`: API server port

### Model Configuration
- Image size: 100x100 pixels
- Batch size: 32
- Learning rate: 0.001
- Epochs: 15 (with early stopping)

## 📈 Monitoring & Logging

### Performance Metrics
- Request count and success rate
- Average inference time
- Model prediction confidence
- Error rates and types

### Logging
- Structured logging with timestamps
- Error tracking and reporting
- Performance monitoring
- Deployment status updates

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Submit** a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **TensorFlow/Keras** for deep learning framework
- **Streamlit** for web application framework
- **FastAPI** for REST API development
- **HuggingFace** for model hosting

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/face-mask-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/face-mask-detector/discussions)

---

**Built with ❤️ for ML Engineering Excellence**

*This project demonstrates professional ML engineering practices including data validation, model development, testing, deployment, and monitoring in a production-ready environment.* 