try:
    from huggingface_hub import HfApi, create_branch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Model upload functionality disabled.")
    print("Install with: pip install huggingface_hub")

import json
import os

def upload_model():
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for model uploading. "
            "Install with: pip install huggingface_hub"
        )
    
    # Load model config
    with open('config/model_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize Hugging Face API
    api = HfApi()
    repo_id = "Sooraj-jain/face-mask-detector"
    
    # Create v1 branch if it doesn't exist
    try:
        api.create_branch(repo_id=repo_id, branch="v1", repo_type="model")
        print("Created v1 branch")
    except Exception as e:
        print(f"Branch v1 might already exist or other error: {str(e)}")
    
    # Upload all model versions
    for version_key, model_info in config['models'].items():
        model_path = model_info['file']
        version = model_info['version']
        
        # Skip if file doesn't exist
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue
            
        print(f"Uploading {model_path} (Version: {version}) to HuggingFace...")
        
        # For v1, upload to v1 branch, for latest upload to main branch
        branch = "v1" if version == "v1" else "main"
        
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,
                repo_id=repo_id,
                repo_type="model",
                revision=branch
            )
            print(f"Upload complete for {version}!")
        except Exception as e:
            print(f"Error uploading {version}: {str(e)}")

if __name__ == "__main__":
    upload_model() 