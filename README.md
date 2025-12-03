# MediFact — Full-Stack Medical RAG System

MediFact is a full-stack Retrieval-Augmented Generation (RAG) system that provides **up-to-date, factual medical answers** derived from **WHO Factsheets**.
The system enforces strict **safety and guardrails** to prevent harmful, discriminatory, racist, or non-medical queries.

This repository contains the complete implementation of MediFact, including the **frontend**, **backend**, and a **data pipeline** orchestrated with Airflow (dev) and GitHub Actions (prod).

---

##  Features

* **Medical-grade RAG pipeline** built on WHO factsheets
* **Multi-service architecture** (Airflow, API service, and Cloud RAG deployment)
* **Structured data pipeline** from scraping → cleaning → processing → vectorizing
* **Vector search** using ChromaDB
* **GPU-powered LLM inference** (Modal deployment)
* **Strict guardrails** for safety-compliant medical responses
* **Fully containerized** with Docker Compose
* **Automatic production scheduling** via GitHub Actions

---

##  System Architecture

```
WHO Factsheets ➝ Scraper ➝ Raw Data (S3)
          ▼
Data Cleaning + Structuring ➝ Processed Data (S3)
          ▼
Chunking + Embeddings ➝ Vector DB (ChromaDB, persisted to S3)
          ▼
RAG Backend API (Modal/Flask)
          ▼
Next.js Frontend UI
```

The project is composed of **three services** (see `docker-compose.yml`):

1. **airflow** — Orchestrates the data pipeline in development
2. **app** — Backend RAG API
3. **rag_cloud_deployment** — Cloud deployment logic (Modal)

---

##  Tech Stack

### **Frontend**

* Next.js
* Prisma
* TanStack Query
* Zustand

### **Backend**

* Flask
* Modal
* Hugging Face Models
* ChromaDB
* AWS S3

### **Data Pipeline**

* Apache Airflow (development automation)
* AWS S3 for storage
* GitHub Actions (production scheduler)

---

##  Deployment

* **Frontend:** Vercel
* **Backend:** Modal (serverless GPU deployment)

---

##  Performance Notes

To conserve Modal free GPU credits, the backend runs on a **low-end GPU**:

*  **Expect 2–3 minutes latency per query**
*  **Rate-limited to ~3 queries per user**

---

##  Live Demo

 **MediFact App:** [https://medifact.vercel.app/](https://medifact.vercel.app/)

---

##  Local Development (Docker)

Clone the repo and run:

```bash
git clone <repo-url>
cd <project-folder>
docker compose up --build
```

This launches all three services (Airflow, app API, RAG cloud deployment).

---

##  Project Pipeline Summary

1. Scrape WHO factsheet pages
2. Store raw data on AWS S3
3. Retrieve & process data
4. Save processed data back to S3
5. Retrieve processed data
6. Chunk & vectorize
7. Store vectors in ChromaDB
8. Persist vector DB to S3
9. Serve RAG answers via Modal backend
10. Render results on Next.js frontend

---

## License

This project is for educational and demonstration purposes.
Please consult real medical professionals for personal medical advice.

---

##  Need More Info?

If you'd like sections such as **Screenshots**, **API Routes**, **Folder Structure**, or **Setup Environment Variables**, just let me know!
