#!/usr/bin/env python3
"""
Test script to verify OpenCV setup and Haar cascade classifier loading
for Streamlit Cloud deployment.
"""

import streamlit as st
import cv2
import os
import urllib.request

def test_opencv_setup():
    """Test OpenCV installation and Haar cascade loading."""
    st.title("🔧 OpenCV Setup Test")
    
    # Test 1: Check OpenCV version
    st.subheader("1. OpenCV Version Check")
    try:
        cv2_version = cv2.__version__
        st.success(f"✅ OpenCV version: {cv2_version}")
    except Exception as e:
        st.error(f"❌ OpenCV version check failed: {str(e)}")
        return False
    
    # Test 2: Check if haarcascades directory exists
    st.subheader("2. Haar Cascades Directory Check")
    try:
        haarcascades_path = cv2.data.haarcascades
        st.info(f"Haar cascades path: {haarcascades_path}")
        
        if os.path.exists(haarcascades_path):
            st.success("✅ Haar cascades directory exists")
        else:
            st.warning("⚠️ Haar cascades directory not found")
    except Exception as e:
        st.error(f"❌ Haar cascades path check failed: {str(e)}")
    
    # Test 3: Try to load the cascade file
    st.subheader("3. Haar Cascade File Loading Test")
    cascade_file = "haarcascade_frontalface_default.xml"
    
    # Try multiple paths
    possible_paths = [
        cv2.data.haarcascades + cascade_file,
        os.path.join(os.getcwd(), cascade_file),
        cascade_file
    ]
    
    cascade_loaded = False
    for i, path in enumerate(possible_paths):
        st.info(f"Trying path {i+1}: {path}")
        try:
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                st.success(f"✅ Cascade loaded successfully from: {path}")
                cascade_loaded = True
                break
            else:
                st.warning(f"⚠️ Cascade file exists but is empty: {path}")
        except Exception as e:
            st.warning(f"⚠️ Failed to load from {path}: {str(e)}")
    
    if not cascade_loaded:
        st.info("📦 Attempting to download cascade file...")
        try:
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(cascade_url, cascade_file)
            cascade = cv2.CascadeClassifier(cascade_file)
            if not cascade.empty():
                st.success("✅ Cascade downloaded and loaded successfully!")
                cascade_loaded = True
            else:
                st.error("❌ Downloaded cascade file is empty")
        except Exception as e:
            st.error(f"❌ Failed to download cascade file: {str(e)}")
    
    # Test 4: Test face detection on a simple image
    if cascade_loaded:
        st.subheader("4. Face Detection Test")
        try:
            # Create a simple test image (just for testing the cascade)
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            st.success("✅ Face detection function works (no faces detected in test image)")
        except Exception as e:
            st.error(f"❌ Face detection test failed: {str(e)}")
    
    return cascade_loaded

if __name__ == "__main__":
    import numpy as np
    success = test_opencv_setup()
    
    if success:
        st.success("🎉 All tests passed! OpenCV setup is working correctly.")
    else:
        st.error("❌ Some tests failed. Please check the error messages above.") 