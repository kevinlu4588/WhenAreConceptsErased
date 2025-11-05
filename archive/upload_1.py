#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uploads two SD1.4 pipeline models to Hugging Face:
- stereo_church
- rece_church
"""
import os
from huggingface_hub import HfApi, create_repo, whoami

ORG = "DiffusionConceptErasure"

PIPELINES = [
    {
        "repo_name": "stereo_garbage_truck",
        "local_path": "/share/u/kevin/ErasingDiffusionModels/final_models/stereo_garbage_truck",
    },
    {
        "repo_name": "rece_garbage_truck",
        "local_path": "/share/u/kevin/ErasingDiffusionModels/final_models/rece_garbage_truck",
    },
]

def main():
    api = HfApi()
    
    # Verify authentication
    try:
        me = whoami()
        print(f"🔐 Logged in as: {me['name']} ({me.get('type','user')})")
    except Exception as e:
        raise SystemExit("You must huggingface-cli login or set HF_TOKEN") from e
    
    for pipeline in PIPELINES:
        repo_id = f"{ORG}/{pipeline['repo_name']}"
        local_path = pipeline['local_path']
        
        print(f"\n=== Uploading pipeline: {repo_id} ===")
        
        # Check if local path exists
        if not os.path.exists(local_path):
            print(f"❌ Path not found: {local_path}")
            continue
        
        # Create repository
        print(f"📦 Creating repo: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        
        # Upload entire folder
        print(f"⬆️  Uploading folder: {local_path} → {repo_id}")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "*.tmp"],
        )
        
        print(f"✅ Successfully uploaded {repo_id}")
    
    print("\n🎉 Done! Check your models at https://huggingface.co/DiffusionConceptErasure")

if __name__ == "__main__":
    main()