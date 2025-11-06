from typing import Dict, List, Optional
import boto3
import os
import json
import logging
from botocore.exceptions import NoCredentialsError, ClientError

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

def upload_to_s3(file_content: str, object_key: str, isByte:bool = False) -> bool:
    logging.info(f"Attempting to upload data to s3://{S3_BUCKET_NAME}/{object_key}")
    try:
        s3 = get_s3_client()
        
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key,
            Body=file_content if isByte else file_content.encode('utf-8'),
            ContentType='application/json'
        )
        logging.info(f"Successfully uploaded data to {object_key}")
        return True
    except ClientError as e:
        logging.error(f"S3 Client Error during upload: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during upload: {e}")
        return False

def download_from_s3(object_key: str, isByte:bool = False) -> Optional[str]:
    logging.info(f"Attempting to download s3://{S3_BUCKET_NAME}/{object_key}")
    try:
        s3 = get_s3_client()
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=object_key)
        if isByte:
            content = response['Body'].read()
        else:    
            content = response['Body'].read().decode('utf-8')
        logging.info(f"Successfully downloaded {object_key}")
        return content
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logging.warning(f"S3 object not found: {object_key}")
            return None
        logging.error(f"S3 Client Error during download: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during download: {e}")
        return None

def download_all_files(prefix: str) -> List[Dict]:
    s3 = get_s3_client()
    all_files = []
    
    try:
        response = s3.list_objects_v2(
            Bucket = S3_BUCKET_NAME,
            Prefix = prefix 
        )
        
        if "Contents" not in response:
            logging.warning(f"No objects under the prefix: {prefix}")
            return all_files
        
        logging.info(f"Successfully loaded data")
        return response 
    except ClientError as e:
        logging.error(f"S3 Client Error during list operation: {e}")
        return []
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return []
    
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
        
def upload_chroma_folder_to_s3(local_dir: str, s3_prefix: str):
    """
    Uploads all files under local_dir to S3 under s3_prefix,
    preserving folder structure and streaming files to avoid memory issues.
    """
    if not os.path.exists(local_dir):
        logging.warning(f"Local ChromaDB folder {local_dir} does not exist. Nothing to upload.")
        return

    if not os.listdir(local_dir):
        logging.warning(f"Local ChromaDB folder {local_dir} is empty. Nothing to upload.")
        return

    s3 = get_s3_client()
    try:
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, local_dir)
                s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/") 

                logging.info(f"uploading {s3_key} at {file_path}/{rel_path}")
                with open(file_path, "rb") as f:
                    s3.upload_fileobj(f, Bucket=S3_BUCKET_NAME, Key=s3_key)

        logging.info(f"ChromaDB folder {local_dir} uploaded to S3 at {s3_prefix} successfully.")

    except Exception as e:
        logging.error(f"Failed to upload ChromaDB folder to S3: {e}")
