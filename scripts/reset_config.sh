#!/bin/bash
# Script to reset configuration for Streamlit Cloud deployment

echo "☁️  Resetting to cloud-compatible configuration..."

# Reset to original cloud config
git checkout .streamlit/config.toml

echo "✅ Configuration reset for Streamlit Cloud deployment"
echo "📤 You can now commit and push to deploy on Streamlit Cloud" 