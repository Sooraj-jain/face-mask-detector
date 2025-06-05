from tensorflow.keras.models import load_model

# Load the existing .keras model
model = load_model("mask_detector.keras")

# Save it in .h5 format
model.save("mask_detector.h5", save_format="h5")

print("✅ Converted 'mask_detector.keras' to 'mask_detector.h5'")
