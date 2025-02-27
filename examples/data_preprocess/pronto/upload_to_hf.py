from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json

# Login to Hugging Face (you'll need to get a token from huggingface.co/settings/tokens)
api = HfApi()
# api.set_access_token("your_token_here")

# Create a new repository (dataset)
repo_name = "pronto-2-hop-8-names"  # choose your desired name
# api.create_repo(repo_id=repo_name, repo_type="dataset")

# Upload the JSON file
api.upload_file(
    path_or_fileobj="/root/pronto-experiments/quesitons_pronto_fixed_limited_names/finetune_2hop_1k.json",
    path_in_repo="finetune_2hop_8_names_1k.json",
    repo_id=f"rmcc11/{repo_name}",
    repo_type="dataset"
)