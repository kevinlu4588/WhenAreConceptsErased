from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access Hugging Face API token
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
print(hf_access_token)  # Print to confirm loading, remove in production

# Base path to your models directory
MODEL_PATH = "/Users/klu/ErasingDiffusionModels/models/"
hf_username = "kevinlu4588"  # replace with your Hugging Face username or org name

api = HfApi()

def is_repo_empty(repo_id):
    try:
        files = api.list_repo_files(repo_id)
        return len(files) <= 2
    except Exception as e:
        print(f"Error checking repository {repo_id}: {e}")
        return True

def upload_model_folder(subfolder_path):
    model_name = os.path.basename(subfolder_path)
    repo_id = f"{hf_username}/{model_name}"
    
    # Create the repo if it doesn't exist, or skip if it has files
    api.create_repo(repo_id, exist_ok=True, token=hf_access_token)
    if is_repo_empty(repo_id):
        print(f"Uploading {subfolder_path} to {repo_id}...")
        api.upload_folder(folder_path=subfolder_path,
                          repo_id=repo_id,
                          repo_type="model",
                          token=hf_access_token)
        print(f"Successfully uploaded {subfolder_path} to {repo_id}.")
    else:
        print(f"Repository {repo_id} already contains files. Skipping upload.")

def main():
    for subfolder in os.listdir(MODEL_PATH):
        subfolder_path = os.path.join(MODEL_PATH, subfolder)
        if os.path.isdir(subfolder_path):
            upload_model_folder(subfolder_path)

if __name__ == "__main__":
    main()
