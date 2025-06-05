# train_mask_model.py
"""
Face Mask Detection - Model Training Pipeline
------------------------------------------
This script implements an end-to-end training pipeline for a face mask detection model.
It includes data preprocessing, model architecture definition, training, evaluation,
and model serialization.

Key Components:
- Data loading and preprocessing with augmentation
- CNN architecture optimized for real-time inference
- Training with validation monitoring
- Model evaluation metrics
- Model serialization in multiple formats
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
CONFIG = {
    "image_size": 100,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "dataset_path": "face-mask-dataset",
    "validation_split": 0.2,
    "random_state": 42
}

def setup_training_session():
    """
    Sets up a new training session with logging and model artifacts directory.
    Returns the path to save model artifacts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = f"model_artifacts_{timestamp}"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(artifacts_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    return artifacts_dir

def load_data():
    """
    Data Pipeline Implementation
    ---------------------------
    1. Loads images from the dataset directory
    2. Applies preprocessing:
       - Resizing to standard dimensions
       - Pixel normalization
       - Data augmentation for training
    3. Splits data into training and validation sets
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting data loading and preprocessing...")
    data = []
    labels = []
    
    # Load and preprocess images
    for label, category in enumerate(["with_mask", "without_mask"]):
        folder = os.path.join(CONFIG["dataset_path"], category)
        logger.info(f"Processing {category} images...")
        
        files = os.listdir(folder)
        for file in files:
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (CONFIG["image_size"], CONFIG["image_size"]))
                data.append(img)
                labels.append(label)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays and normalize
    X = np.array(data) / 255.0
    y = to_categorical(np.array(labels), num_classes=2)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["validation_split"],
        random_state=CONFIG["random_state"]
    )
    
    logger.info(f"Dataset split complete. Training samples: {len(X_train)}, Validation samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def build_model():
    """
    Model Architecture Implementation
    -------------------------------
    Creates a CNN optimized for face mask detection:
    - Uses BatchNormalization for better training stability
    - Implements dropout for regularization
    - Optimized for real-time inference
    
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(CONFIG["image_size"], CONFIG["image_size"], 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    logger.info(f"Model architecture built:\n{model.summary()}")
    return model

def plot_training_history(history, artifacts_dir):
    """
    Visualizes training metrics and saves plots.
    """
    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, X_test, y_test, artifacts_dir):
    """
    Performs model evaluation and saves metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    report = classification_report(y_test_classes, y_pred_classes, target_names=['With Mask', 'Without Mask'])
    logger.info(f"\nModel Evaluation:\n{report}")
    
    # Save classification report
    with open(os.path.join(artifacts_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    """
    Main training pipeline
    """
    # Setup training session
    artifacts_dir = setup_training_session()
    logger.info(f"Training artifacts will be saved to: {artifacts_dir}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data()
    
    # Build and train model
    model = build_model()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(artifacts_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    evaluate_model(model, X_test, y_test, artifacts_dir)
    
    # Plot training history
    plot_training_history(history, artifacts_dir)
    
    # Save final model in both formats
    model.save(os.path.join(artifacts_dir, "mask_detector.keras"))
    model.save("mask_detector.h5")  # Save in root for compatibility
    logger.info("✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
