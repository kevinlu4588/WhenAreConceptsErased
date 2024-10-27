from huggingface_hub import HfApi, HfFolder, hf_hub_upload
import os
import subprocess

hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
HfFolder.save_token(hf_access_token)

MODEL_PATH = '/raid/aag/scasper/models/'

def run_cleanup():
    try:
        subprocess.run(["python", "clean_folder.py"], check=True)
        print("Cleanup completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running cleanup script: {e}")
    except FileNotFoundError:
        print("clean_folder.py not found in the current directory.")

def is_repo_empty(repo_id):
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        return len(files) <= 2
    except Exception as e:
        print(f"An error occurred: {e}")
        return True

def upload_pt_file(subdir, model_name, ckpt):
    file_path = f"{MODEL_PATH}{subdir}/{model_name}{ckpt}.pt"
    repo_id = f"LLM-GAT/{model_name}{ckpt}"
    if is_repo_empty(repo_id):
        hf_hub_upload(repo_id=repo_id, path_or_fileobj=file_path, token=hf_access_token)
        print(f"Uploaded {file_path} to {repo_id}")
    else:
        print(f"Repository {repo_id} is not empty. Skipping upload.")

subdirs = ['gd']
model_names = ['llama3_gd_lora-256-128_beta-14_bs-32_lr-1e-04_checkpoint-']
ckpts = list(range(1, 9))

def main():
    missing_models = []
    for subdir, model in zip(subdirs, model_names):
        for ckpt in ckpts:
            file_path = f"{MODEL_PATH}{subdir}/{model}{ckpt}.pt"
            if os.path.isfile(file_path):
                print(f"Uploading {file_path}...")
                upload_pt_file(subdir, model, ckpt)
            else:
                missing_models.append(file_path)
            run_cleanup()
    print('MISSING MODELS:', missing_models)

if __name__ == "__main__":
    main()