
ARG AIRFLOW_IMAGE_NAME="apache/airflow:2.8.1-python3.9"
FROM ${AIRFLOW_IMAGE_NAME}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
