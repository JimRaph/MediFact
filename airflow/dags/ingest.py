from datetime import datetime
import os
import logging
import sys

sys.path.append(os.path.join(os.environ['AIRFLOW_HOME'], 'scripts'))
from airflow.decorators import dag, task
from data_processor import (
    topic_data_and_save_to_s3,
    fetch_content_and_save_to_s3,
    process_and_upload_to_chroma,
    embed_and_upload_chroma,
)

WHO_INDEX_URL = "https://www.who.int/news-room/fact-sheets"
PREFIX = 'raw_content/'

@dag(
    dag_id='who_data_rag_ingestion_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['rag', 'chromadb', 's3', 'WHO'],
)
def rag_ingestion_pipeline():

    @task(task_id="index_scraper")
    def scrape_and_index_data(base_url: str) -> None:
        topic_data_and_save_to_s3(base_url) 
    
    @task(task_id="content_scraper")
    def scrape_and_stage_content() -> None:
        fetch_content_and_save_to_s3()

    @task(task_id="chroma_processor")
    def process_and_load_chroma(prefix) -> None:
        process_and_upload_to_chroma(prefix)
    
    @task(task_id="chroma_embedder")
    def embed_and_upload():
        embed_and_upload_chroma() 
        
    scrape_index_task = scrape_and_index_data(WHO_INDEX_URL) 
    
    scrape_content_task = scrape_and_stage_content()
    
    load_chroma_task = process_and_load_chroma(PREFIX)
    
    embed_chroma = embed_and_upload()
 
    scrape_index_task >> scrape_content_task >> load_chroma_task >> embed_chroma


ingestion_dag = rag_ingestion_pipeline()
