# train_mask_model.py
"""
Face Mask Detection - Model Training Pipeline
------------------------------------------
This script implements an end-to-end training pipeline for a face mask detection model.
It includes data preprocessing, model architecture definition, training, evaluation,
and model serialization.

Key Components:
- Data validation and quality checks
- Data preprocessing and augmentation
- Model architecture with validation
- Training with monitoring
- Evaluation and metrics
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Configuration
CONFIG = {
    "image_size": 100,
    "batch_size": 32,
    "epochs": 15,            # Balanced number of epochs
    "learning_rate": 0.001,
    "dataset_path": "face-mask-dataset",
    "validation_split": 0.2,
    "random_state": 42,
    "min_samples_per_class": 100,
    "required_classes": ["with_mask", "without_mask"],
    "image_channels": 3,
    "cross_validation_folds": 5,
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.2,
    "min_lr": 1e-6
}

class DataValidator:
    """
    Validates dataset quality and structure before training.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_path = Path(config["dataset_path"])
        self.validation_results = {}

    def validate_dataset_structure(self):
        """Validates the dataset directory structure and minimum requirements."""
        logger.info("Validating dataset structure...")
        
        # Check dataset directory exists
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset directory {self.dataset_path} does not exist!")

        # Validate required classes
        for class_name in self.config["required_classes"]:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                raise ValueError(f"Required class directory {class_name} not found!")
            
            # Count samples
            samples = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
            sample_count = len(samples)
            
            if sample_count < self.config["min_samples_per_class"]:
                raise ValueError(
                    f"Insufficient samples for class {class_name}. "
                    f"Found {sample_count}, required {self.config['min_samples_per_class']}"
                )
            
            self.validation_results[f"{class_name}_samples"] = sample_count
            
            # Validate image integrity
            corrupt_images = self._validate_images(samples)
            if corrupt_images:
                logger.warning(f"Found {len(corrupt_images)} corrupt images in {class_name}")
                self._handle_corrupt_images(corrupt_images)

        logger.info("Dataset structure validation completed successfully!")
        return self.validation_results

    def _validate_images(self, image_paths):
        """Validates each image for corruption and correct dimensions."""
        corrupt_images = []
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_images.append(img_path)
                    continue
                
                # Check dimensions and channels
                if len(img.shape) != 3 or img.shape[2] != self.config["image_channels"]:
                    corrupt_images.append(img_path)
                    continue
                
            except Exception as e:
                logger.error(f"Error reading {img_path}: {str(e)}")
                corrupt_images.append(img_path)
                
        return corrupt_images

    def _handle_corrupt_images(self, corrupt_images):
        """Moves corrupt images to a separate directory for inspection."""
        corrupt_dir = self.dataset_path / "corrupt_images"
        corrupt_dir.mkdir(exist_ok=True)
        
        for img_path in corrupt_images:
            try:
                shutil.move(str(img_path), str(corrupt_dir / img_path.name))
                logger.warning(f"Moved corrupt image {img_path.name} to {corrupt_dir}")
            except Exception as e:
                logger.error(f"Failed to move corrupt image {img_path}: {str(e)}")

class ModelValidator:
    """
    Validates model performance and reliability through cross-validation
    and various evaluation metrics.
    """
    def __init__(self, config):
        self.config = config
        self.metrics = {}

    def cross_validate_model(self, X, y, model_builder):
        """Performs k-fold cross-validation of the model."""
        logger.info("Starting cross-validation...")
        
        kfold = KFold(n_splits=self.config["cross_validation_folds"], 
                      shuffle=True, 
                      random_state=self.config["random_state"])
        
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.config['cross_validation_folds']}")
            
            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Build and train model
            model = model_builder()
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=5,  # Reduced epochs for CV
                batch_size=self.config["batch_size"],
                validation_data=(X_val_fold, y_val_fold),
                verbose=0
            )
            
            # Evaluate fold
            metrics = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            metrics_dict = dict(zip(model.metrics_names, metrics))
            
            fold_metrics.append({
                'val_loss': metrics_dict['loss'],
                'val_accuracy': metrics_dict['accuracy']
            })
            
        # Aggregate metrics
        self.metrics['cross_validation'] = {
            'mean_accuracy': np.mean([m['val_accuracy'] for m in fold_metrics]),
            'std_accuracy': np.std([m['val_accuracy'] for m in fold_metrics]),
            'fold_metrics': fold_metrics
        }
        
        logger.info(f"Cross-validation completed. Mean accuracy: {self.metrics['cross_validation']['mean_accuracy']:.4f} "
                   f"(±{self.metrics['cross_validation']['std_accuracy']:.4f})")
        return self.metrics

    def evaluate_model_robustness(self, model, X_test, y_test):
        """Evaluates model robustness through various perturbations."""
        logger.info("Evaluating model robustness...")
        
        # Original performance
        base_predictions = model.predict(X_test)
        base_accuracy = np.mean(np.argmax(base_predictions, axis=1) == np.argmax(y_test, axis=1))
        
        robustness_metrics = {'base_accuracy': base_accuracy}
        
        # Test with noise
        noise_levels = [0.1, 0.2]
        for noise in noise_levels:
            noisy_X = X_test + np.random.normal(0, noise, X_test.shape)
            noisy_X = np.clip(noisy_X, 0, 1)
            noisy_pred = model.predict(noisy_X)
            noisy_acc = np.mean(np.argmax(noisy_pred, axis=1) == np.argmax(y_test, axis=1))
            robustness_metrics[f'noise_{noise}_accuracy'] = noisy_acc
        
        self.metrics['robustness'] = robustness_metrics
        logger.info("Robustness evaluation completed!")
        return robustness_metrics

def setup_training_session():
    """Sets up a new training session with logging and artifact directories."""
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
    Creates a CNN model architecture optimized for face mask detection.
    Balances between complexity and accuracy:
    - Uses proven base architecture
    - Adds minimal but effective regularization
    - Maintains fast inference time
    """
    model = Sequential([
        # First Convolutional Block - Base feature extraction
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(CONFIG["image_size"], CONFIG["image_size"], 3)),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block - Feature enhancement
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block - Fine-grained features
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Proven dropout rate
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']  # Track both accuracy and AUC
    )
    
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
    Production-grade training pipeline with:
    - Data validation
    - Cross-validation for model stability
    - Proper model checkpointing
    - Performance monitoring
    - Model evaluation
    """
    logger.info("Starting model training pipeline...")
    
    # Setup training session
    artifacts_dir = setup_training_session()
    
    # Load and validate data
    data_validator = DataValidator(CONFIG)
    data_validator.validate_dataset_structure()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data()
    
    # Cross-validation for model stability
    model_validator = ModelValidator(CONFIG)
    cv_metrics = model_validator.cross_validate_model(
        X_train, y_train,
        model_builder=build_model
    )
    
    # Build and train final model
    model = build_model()
    
    # Essential callbacks for production training
    callbacks = [
        # Save best model
        ModelCheckpoint(
            os.path.join(artifacts_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        # Adjust learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CONFIG["reduce_lr_factor"],
            patience=CONFIG["reduce_lr_patience"],
            min_lr=CONFIG["min_lr"],
            verbose=1
        )
    ]
    
    # Train with monitoring
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Comprehensive evaluation
    evaluate_model(model, X_test, y_test, artifacts_dir)
    plot_training_history(history, artifacts_dir)
    
    # Save model in both formats with proper error handling
    try:
        model.save("models/mask_detector.keras")
        model.save("models/mask_detector.h5")
        logger.info("✅ Model saved in both .keras and .h5 formats")
    except Exception as e:
        logger.error(f"❌ Error saving model: {str(e)}")
        raise
    
    # Log final metrics
    final_val_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, final_val_metrics))
    logger.info(f"Final Validation Metrics: {metrics_dict}")
    
    logger.info("✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
