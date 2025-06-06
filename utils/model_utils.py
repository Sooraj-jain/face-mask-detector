"""
Utility functions for model loading and version management.
"""

import os
import json
import urllib.request
from tensorflow.keras.models import load_model
import streamlit as st

def load_model_config():
    """Load model configuration from JSON file."""
    config_path = os.path.join('config', 'model_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model config: {str(e)}")
        st.stop()

@st.cache_resource
def load_face_mask_model(version="latest"):
    """
    Load the face mask detection model with version support.
    
    Args:
        version (str): Model version to load ("latest", "v1", "v2", etc.)
    
    Returns:
        model: Loaded Keras model
    """
    config = load_model_config()
    
    if version not in config['models']:
        st.error(f"❌ Invalid model version: {version}")
        st.stop()
    
    model_config = config['models'][version]
    local_model_path = model_config["file"]
    model_url = model_config["url"]
    
    try:
        model = load_model(local_model_path)
        st.success(f"✅ Model {model_config['version']} loaded successfully!")
    except Exception as e:
        st.info(f"📦 Downloading model {model_config['version']} from cloud storage...")
        try:
            if not os.path.exists(local_model_path):
                urllib.request.urlretrieve(model_url, local_model_path)
            model = load_model(local_model_path)
            st.success(f"✅ Model {model_config['version']} ready!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
    
    return model 