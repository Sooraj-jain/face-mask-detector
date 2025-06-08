import tensorflow as tf
import json

def load_and_check_model(model_path):
    print(f"\nChecking model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("\nModel Summary:")
    model.summary()
    print("\nModel Size:", round(model.count_params() * 4 / (1024 * 1024), 2), "MB")  # Approximate size in MB

# Load config
with open('config/model_config.json', 'r') as f:
    config = json.load(f)

# Check both models
for version_key, model_info in config['models'].items():
    model_path = model_info['file']
    try:
        load_and_check_model(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {str(e)}") 