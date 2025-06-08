#!/bin/bash
# Script to run the Face Mask Detection app locally with HTTPS support

echo "🔧 Setting up local development environment..."

# Check if SSL certificates exist
if [ ! -f ".ssl/cert.pem" ] || [ ! -f ".ssl/key.pem" ]; then
    echo "📄 Generating SSL certificates..."
    mkdir -p .ssl
    openssl req -x509 -newkey rsa:4096 -keyout .ssl/key.pem -out .ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Copy local config
echo "⚙️  Setting up local configuration..."
cp .streamlit/config_local.toml .streamlit/config.toml

# Activate virtual environment and run
echo "🚀 Starting application..."
source .FaceMaskDetectorEnv/bin/activate
streamlit run app_live.py

echo "✅ Access your app at: https://localhost:8501"
echo "⚠️  Accept the security warning in your browser for the self-signed certificate" 