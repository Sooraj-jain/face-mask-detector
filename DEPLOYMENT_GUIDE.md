# Streamlit Cloud Deployment Guide

## Recent Fixes Applied

### 1. Haar Cascade Classifier Loading Issue
**Problem**: The original code was failing to load the Haar cascade classifier on Streamlit Cloud due to path issues.

**Solution**: Implemented a robust loading mechanism that:
- Tries multiple possible paths for the cascade file
- Downloads the cascade file from GitHub if not found locally
- Uses `@st.cache_resource` for efficient loading
- Provides clear error messages and status updates

### 2. Error Handling Improvements
**Problem**: Generic error messages made debugging difficult.

**Solution**: Added comprehensive error handling:
- Wrapped main application logic in try-catch blocks
- Added detailed error reporting with error types and stack traces
- Implemented graceful fallbacks for missing dependencies

## Files Modified

1. **`app_live.py`**: 
   - Added robust Haar cascade loading function
   - Improved error handling and logging
   - Added import error handling

2. **`test_opencv_setup.py`**: 
   - Created diagnostic script for testing OpenCV setup
   - Helps identify deployment issues

## Troubleshooting Steps

### If you encounter the original error:

1. **Run the test script first**:
   ```bash
   streamlit run test_opencv_setup.py
   ```

2. **Check the logs**:
   - In Streamlit Cloud, click "Manage app" → "Logs"
   - Look for specific error messages

3. **Verify dependencies**:
   - Ensure `packages.txt` contains required system packages
   - Check that `requirements.txt` has correct versions

### Common Issues and Solutions

#### Issue: "AttributeError: This app has encountered an error"
**Cause**: Usually related to missing files or path issues in cloud environment
**Solution**: The new robust loading mechanism should handle this automatically

#### Issue: Model loading fails
**Cause**: Model file not accessible or corrupted
**Solution**: The app will automatically download from HuggingFace if local file is missing

#### Issue: Camera access problems
**Cause**: WebRTC configuration issues
**Solution**: Check browser permissions and ensure HTTPS is used

## Deployment Checklist

- [ ] All required packages are in `requirements.txt`
- [ ] System dependencies are in `packages.txt`
- [ ] Model file exists locally or is downloadable
- [ ] Haar cascade file can be loaded or downloaded
- [ ] Error handling is in place
- [ ] Test script passes all checks

## Testing Locally

Before deploying to Streamlit Cloud, test locally:

```bash
# Test the main app
streamlit run app_live.py

# Test OpenCV setup
streamlit run test_opencv_setup.py
```

## Monitoring Deployment

1. **Check app status** in Streamlit Cloud dashboard
2. **Review logs** for any error messages
3. **Test camera functionality** once deployed
4. **Monitor performance** and adjust settings if needed

## Performance Optimization

- The app uses caching for model and cascade loading
- Frame processing is optimized for cloud deployment
- WebRTC settings are balanced for performance and quality

## Support

If issues persist:
1. Check the test script output
2. Review Streamlit Cloud logs
3. Verify all dependencies are correctly specified
4. Test with a minimal version first 