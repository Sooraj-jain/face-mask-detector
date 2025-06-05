from tensorflow.keras.models import load_model

# Load your existing .keras model
model = load_model("mask_detector.keras")

# Save it in .h5 format
model.save("mask_detector.h5")

print("✅ Conversion complete: Saved as mask_detector.h5")
