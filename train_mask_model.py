# train_mask_model.py
# This script trains a CNN model to classify images into two categories: with_mask and without_mask
# The trained model is saved as 'mask_detector.keras' for later use in detection

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configuration for image processing and data paths
IMAGE_SIZE = 100
DATASET_PATH = "face-mask-dataset"  # Folder with `with_mask/` and `without_mask/`

def load_data():
    """
    Loads and preprocesses the face mask dataset:
    1. Reads images from with_mask and without_mask folders
    2. Resizes all images to IMAGE_SIZE x IMAGE_SIZE
    3. Normalizes pixel values to range [0,1]
    4. Splits data into training and testing sets
    """
    data = []
    labels = []
    for label, category in enumerate(["with_mask", "without_mask"]):
        folder = os.path.join(DATASET_PATH, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                data.append(img)
                labels.append(label)
            except:
                continue
    data = np.array(data) / 255.0  # Normalize pixel values
    labels = to_categorical(np.array(labels), num_classes=2)  # Convert to one-hot encoding
    return train_test_split(data, labels, test_size=0.2, random_state=42)

def build_model():
    """
    Creates a CNN model architecture:
    - 2 Convolutional layers with MaxPooling
    - Flatten layer to convert 2D features to 1D
    - Dense layers for classification
    - Dropout for preventing overfitting
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes: mask/no mask
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_data()
    
    # Create and train the model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save("mask_detector.keras")
    model.save("mask_detector.h5")
    print("✅ Model saved in both .keras and .h5 formats")
