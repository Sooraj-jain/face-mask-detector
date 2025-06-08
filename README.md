# Face Mask Detection System 😷

## 🏗️ Enterprise Architecture

A production-ready, enterprise-level deep learning system for real-time face mask detection. This system showcases professional ML engineering practices with clear separation of concerns, modular architecture, and deployment-ready structure.

## 📁 Project Structure

```
face-mask-detector/
├── src/                          # Source code (main application logic)
│   ├── inference/               # Real-time inference & serving
│   │   ├── app_live.py         # Streamlit web application
│   │   ├── api.py              # FastAPI REST service
│   │   └── __init__.py
│   ├── training/               # Model training & evaluation
│   │   ├── train_mask_model.py # Training pipeline
│   │   ├── test_train_pipeline.py # Testing utilities
│   │   └── __init__.py
│   ├── utils/                  # Shared utilities
│   │   ├── model_utils.py      # Model loading & management
│   │   ├── check_models.py     # Model validation
│   │   ├── upload_model.py     # Model deployment helpers
│   │   └── __init__.py
│   └── __init__.py
├── models/                     # Trained model artifacts
│   └── mask_detector.h5       # Production model
├── config/                     # Configuration files
│   └── model_config.json      # Model versioning config
├── scripts/                    # Deployment & utility scripts
│   ├── run_local.sh           # Local development with HTTPS
│   ├── train_model.sh         # Model training pipeline
│   ├── run_api.sh             # API server startup
│   └── reset_config.sh        # Cloud deployment prep
├── tests/                      # Test suite
│   └── __init__.py
├── docs/                       # Documentation
│   └── README.md              # Detailed project documentation
├── .streamlit/                 # Streamlit configuration
│   ├── config.toml            # Cloud-compatible config
│   └── config_local.toml      # Local HTTPS config
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚀 Quick Start

### Local Development (with Camera Access)
```bash
./scripts/run_local.sh
```
- Automatically sets up HTTPS for camera access
- Access at: https://localhost:8501

### API Server
```bash
./scripts/run_api.sh
```
- Starts FastAPI server
- Access at: http://localhost:8000
- Docs at: http://localhost:8000/docs

### Model Training
```bash
./scripts/train_model.sh
```
- Runs complete training pipeline
- Outputs artifacts to `model_artifacts_*` directory

## 🌐 Deployment

### Streamlit Cloud
1. Reset to cloud configuration:
   ```bash
   ./scripts/reset_config.sh
   ```
2. Push to GitHub
3. Deploy on [share.streamlit.io](https://share.streamlit.io)

### Docker Deployment
*Coming soon - Docker configurations for containerized deployment*

## 🧩 Architecture Components

### Inference Module (`src/inference/`)
- **Web App**: Interactive Streamlit application for real-time detection
- **API Service**: RESTful API for programmatic access
- **Features**: WebRTC streaming, confidence scoring, performance monitoring

### Training Module (`src/training/`)
- **Training Pipeline**: End-to-end model training with validation
- **Testing Suite**: Model evaluation and performance testing
- **Features**: Cross-validation, early stopping, model checkpointing

### Utilities Module (`src/utils/`)
- **Model Management**: Loading, versioning, and deployment utilities
- **Configuration**: Centralized config management
- **Validation**: Model and data integrity checks

## 📊 Model Performance

- **Accuracy**: ~98% on validation set
- **Real-time Performance**: ~30ms per frame
- **Supported Formats**: H5, Keras
- **Model Versioning**: Automatic version management

## 🔧 Development Workflow

1. **Feature Development**: Work in respective modules (`inference/`, `training/`, `utils/`)
2. **Testing**: Add tests in `tests/` directory
3. **Local Testing**: Use `./scripts/run_local.sh` for HTTPS testing
4. **Deployment**: Use `./scripts/reset_config.sh` before cloud deployment

## 📚 API Documentation

The FastAPI service provides:
- `POST /predict`: Image-based mask detection
- `GET /health`: Service health check
- `GET /metrics`: Performance metrics
- `GET /docs`: Interactive API documentation

## 🛠️ Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Web Framework**: Streamlit, FastAPI
- **Video Processing**: WebRTC, streamlit-webrtc
- **Deployment**: Streamlit Cloud, Docker-ready

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for complete dependencies
- Camera access for real-time detection

## 🤝 Contributing

1. Follow the modular architecture
2. Add tests for new features
3. Update documentation
4. Use the provided scripts for testing

## 📄 License

*Add your license information here*

---

**Enterprise-ready** | **Production-tested** | **Deployment-optimized** 