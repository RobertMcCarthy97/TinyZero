from huggingface_hub import HfApi, create_repo
import shutil
import os
from pathlib import Path

# Initialize the Hugging Face API client
api = HfApi()

def upload_model_to_hf(local_model_path, hf_model_name):
    # Create the repository first
    try:
        print(f"\nCreating repository {hf_model_name}")
        create_repo(
            repo_id=hf_model_name,
            repo_type="model",
            private=True  # Set to False if you want a public repo
        )
    except Exception as e:
        print(f"Repository creation failed or already exists: {e}")

    # Create a temporary directory for uploading
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)

    # Copy all files from local directory to temp directory
    for file in os.listdir(local_model_path):
        print(f"Copying {file} to {temp_dir}")
        src = os.path.join(local_model_path, file)
        dst = os.path.join(temp_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)  # copy2 preserves metadata

    # Upload all files to Hugging Face
    print(f"Uploading {temp_dir} to {hf_model_name}")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=hf_model_name,
        repo_type="model"
    )

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory {temp_dir}")

# Define the base paths and model names
base_local_path = "scratch/checkpoints/TinyZero/overseer_debug"
base_hf_model_name = "rmcc11/your-model-name"

# Loop through both 'actor' and 'critic'
for model_type in ["actor", "critic"]:
    local_model_path = f"{base_local_path}/{model_type}/latest"
    hf_model_name = f"{base_hf_model_name}-{model_type}"
    upload_model_to_hf(local_model_path, hf_model_name)
