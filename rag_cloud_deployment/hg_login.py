from huggingface_hub import login
import os

login(token=os.getenv('HF_TOKEN'))