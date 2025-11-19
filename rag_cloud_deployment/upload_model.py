from huggingface_hub import HfApi
import os


local_model_folder = os.path.join(os.path.dirname(__file__), 'model')  

repo_id = "EJ4U/WHO-rag-model" 
repo_type = "model"  



api = HfApi()
token = os.getenv('HF_TOKEN')
api.create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)



api.upload_folder(
    folder_path=local_model_folder,  
    path_in_repo="",                
    repo_id=repo_id,
    repo_type=repo_type,
    token=token
)

print(f"Uploaded {local_model_folder} to https://huggingface.co/{repo_id}")
