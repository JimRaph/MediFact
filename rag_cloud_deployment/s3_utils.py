from typing import Dict, List, Optional
import boto3
import os
import json
import logging
# from botocore.exceptions import NoCredentialsError, ClientError

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_s3_client():
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logging.warning("AWS credentials not found in environment. Using default config.")
        return boto3.client('s3', region_name=AWS_REGION)
    
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

def download_chroma_folder_from_s3(s3_prefix: str, local_dir: str):
    """
    Downloads all files under s3_prefix from S3 to local_dir,
    preserving the folder structure for ChromaDB.
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                rel_path = os.path.relpath(s3_key, s3_prefix)
                local_path = os.path.join(local_dir, rel_path)

                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                with open(local_path, "wb") as f:
                    s3.download_fileobj(Bucket=S3_BUCKET_NAME, Key=s3_key, Fileobj=f)

        logging.info(f"ChromaDB folder downloaded from S3 to {local_dir} successfully.")

    except Exception as e:
        logging.error(f"Failed to download ChromaDB folder from S3: {e}")
