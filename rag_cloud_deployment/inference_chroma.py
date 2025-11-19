from __future__ import annotations
import asyncio
import gc
import os
import json
import logging
import torch
from typing import List, Dict, Tuple, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding 
from s3_utils import download_chroma_folder_from_s3

import chromadb
from chromadb.api import Collection
from chromadb import PersistentClient
from chromadb.api.types import QueryResult
import time
from llama_cpp import Llama 

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR")
CHROMA_DIR_INF = "/" + CHROMA_DIR
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
CHROMA_CACHE_COLLECTION = os.getenv("CHROMA_CACHE_COLLECTION", "semantic_cache")

LLM_MODEL_CPU_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
LLM_MODEL_GPU_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHROMA_DB_FILENAME = os.getenv("CHROMA_DB_FILENAME")
SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", 1000))
SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", 120))
# SEMANTIC_CACHE_DIST_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIM_THRESHOLD", 0.1))

RETRIEVE_TOP_K_CPU = int(os.getenv("RETRIEVE_TOP_K_CPU", 3))
RETRIEVE_TOP_K_GPU = int(os.getenv("RETRIEVE_TOP_K_GPU", 8))
MAX_NEW_TOKENS_CPU = int(os.getenv("MAX_NEW_TOKENS_CPU", 256))
MAX_NEW_TOKENS_GPU = int(os.getenv("MAX_NEW_TOKENS_GPU", 1024))
# GPU_MIN_FREE_HOURS_THRESHOLD = float(os.getenv("GPU_MIN_FREE_HOURS_THRESHOLD", 0.5))
GPU_MIN_FREE_HOURS_THRESHOLD = 11

# LLAMA_GGUF_PATH = os.getenv("LLAMA_GGUF_PATH", "/model/tinyllama-reasoning.Q4_K_M.gguf")
LLM_TOKENIZER_ID = "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"

TINYLAMA_CONTEXT_WINDOW = 2048 
LLAMA_3_CONTEXT_WINDOW = 8192
SAFETY_BUFFER = 50 
# MAX_INPUT_TOKENS = TINYLAMA_CONTEXT_WINDOW - MAX_NEW_TOKENS - SAFETY_BUFFER 

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% if loop.last and message['role'] == 'user' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_ID = "EJ4U/WHO-rag-model"
FILENAME = "tinyllama-reasoning.Q4_K_M.gguf"

try:
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
    logger.info("Cross-encoder model loaded successfully.")
except Exception as e:
    logger.warning("Cross-encoder model error: %s", e)

LLAMA_GGUF_PATH = hf_hub_download(
    repo_id=MODEL_ID,
    filename=FILENAME,
    cache_dir="model" 
)

EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
_ = list(EMBEDDING_MODEL.embed(["warmup"])) 
logger.info("FastEmbed model warmup complete.")


def initialize_cpp_llm(gguf_path: str, n_ctx: int = TINYLAMA_CONTEXT_WINDOW, n_threads: int = 4) -> Llama:
    """
    Initialize a quantized GGUF model via llama-cpp (llama_cpp.Llama).
    This replaces the HF AutoModel pipeline for CPU inference.
    """
    logger.info(f"Initializing llama.cpp model from GGUF: {gguf_path}")
    if not os.path.exists(gguf_path):
        logger.error(f"GGUF model not found at {gguf_path}. Make sure the file exists.")
        raise RuntimeError(f"GGUF model not found at {gguf_path}")

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=256,
        use_mmap=True,  # memory-map weights for faster cold-start
        n_gpu_layers=0 
    )
    logger.info("llama.cpp model loaded successfully.")
    return llm


def initialize_llm_pipeline(model_id: str, device: str) -> Pipeline:
    """Initializes a Hugging Face transformers pipeline for GPU."""
    logger.info(f"Initializing HF Pipeline for model: {model_id} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tokenizer, "chat_template", None):
        logger.info("Applying Llama-3 chat template to tokenizer.")
        tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device
    )
    logger.info(f"HF Pipeline for {model_id} loaded successfully.")
    return pipe


def initialize_chroma_client() -> chromadb.PersistentClient:
    """Initializes Chroma client and loads the index from S3/disk."""
    logger.info(f"Initializing Chroma client from persistence directory: {CHROMA_DIR_INF}")
    
    local_db = os.path.exists(CHROMA_DIR_INF) and os.listdir(CHROMA_DIR_INF)
    db_exist = os.path.join(CHROMA_DIR_INF, "chroma.sqlite3")
    
    if not local_db or not os.path.exists(db_exist):
        logger.warning(f"local chroma directory {CHROMA_DIR_INF} is missing or empty. "
                       f"Attempting download from s3.")
        
        if CHROMA_DIR:
            download_chroma_folder_from_s3(
                s3_prefix=CHROMA_DIR,
                local_dir=CHROMA_DIR_INF
            )
            logger.info("Chroma data downloaded from S3.")
        else:
            logger.error("CHROMA_DIR is not set. Cannot retrieve chroma index")
            raise RuntimeError("Chroma index failed to load")
    else:
        logger.info(f"Local chroma data found at {CHROMA_DIR_INF}.")
    logger.info(f"Initializing chroma client from persistence directory: {CHROMA_DIR_INF}")
    try:
        client = PersistentClient(path=CHROMA_DIR_INF, settings=chromadb.Settings(allow_reset=False))
        logger.info(" Chroma client initialized successfully.")
    except Exception as e:
        logger.error(f" Failed to load Chroma index: {e}")
        raise RuntimeError("Chroma index failed to load.")

    return client



async def load_cpu_pipeline() -> Tuple[Optional[object], str, int, int, int]:

    if getattr(app.state, 'cpu_pipeline', None) is not None:
        if isinstance(app.state.cpu_pipeline, Llama):
            return app.state.cpu_pipeline, "cpu_gguf", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU
        return app.state.cpu_pipeline, "hf_gpu" if torch.cuda.is_available() else "cpu_hf", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU

    if getattr(app.state, 'tokenizer', None) is None:
        try:
            logger.info(f"Loading tokenizer from {LLM_TOKENIZER_ID}")
            app.state.tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_ID, use_fast=False)
            if not getattr(app.state.tokenizer, "chat_template", None):
                app.state.tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Failed to load tokenizer: {e}")

    if torch.cuda.is_available():
        try:
            logger.info(f"GPU detected. Attempting to load HF GPU model {LLM_MODEL_GPU_ID}...")
            app.state.cpu_pipeline = await asyncio.to_thread(
                initialize_llm_pipeline,
                LLM_MODEL_GPU_ID,
                "cuda"
            )
            logger.info("HF GPU model loaded successfully.")
            return app.state.cpu_pipeline, "hf_gpu", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU
        except Exception as e:
            logger.warning(f"Failed to load HF GPU model: {e}. Falling back to CPU...")

    if LLAMA_GGUF_PATH and os.path.exists(LLAMA_GGUF_PATH):
        try:
            logger.info("Loading TinyLlama GGUF (CPU)...")
            logger.info(f"Model: {LLAMA_GGUF_PATH}")
            app.state.cpu_pipeline = await asyncio.to_thread(
                initialize_cpp_llm,
                LLAMA_GGUF_PATH,
                TINYLAMA_CONTEXT_WINDOW,
                max(1, os.cpu_count() - 1)
            )
            logger.info("TinyLlama GGUF loaded successfully.")
            return app.state.cpu_pipeline, "cpu_gguf", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU
        except Exception as e:
            logger.warning(f"Failed to load GGUF CPU model: {e}")

    try:
        logger.info(f"Loading HF CPU model {LLM_MODEL_CPU_ID}...")
        app.state.cpu_pipeline = await asyncio.to_thread(
            initialize_llm_pipeline,
            LLM_MODEL_CPU_ID,
            "cpu"
        )
        logger.info("HF CPU model loaded successfully.")
        return app.state.cpu_pipeline, "cpu_hf", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU
    except Exception as e:
        logger.error(f"FATAL: Failed to load any CPU model: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to load any model: {e}")


async def get_pipeline_for_runtime() -> Tuple[Optional[object], str, int, int, int]:
    """
    Determines runtime, lazily loads the correct pipeline (GPU/CPU),
    and returns the pipeline and its associated settings.
    NOTE: Return type is Optional[object] to handle both Pipeline and Llama
    """
    if await gpu_hours_available():
        logger.info("GPU hours available. Attempting to load GPU pipeline.")
        if getattr(app.state, 'gpu_pipeline', None) is None:
            logger.info("Lazy-loading Llama-3.1-8B (GPU)...")
            try:
                if getattr(app.state, 'cpu_pipeline', None):
                    # NOTE: Clear both the Llama object and the separate tokenizer
                    del app.state.cpu_pipeline
                    app.state.cpu_pipeline = None
                    if getattr(app.state, 'llm_cpp', None):
                        del app.state.llm_cpp
                        app.state.llm_cpp = None
                    if getattr(app.state, 'tokenizer', None):
                        del app.state.tokenizer
                        app.state.tokenizer = None
                    gc.collect() 
                    logger.info("Cleared CPU pipeline (Llama) and tokenizer from memory.")
                    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                app.state.gpu_pipeline = await asyncio.to_thread(
                    initialize_llm_pipeline, LLM_MODEL_GPU_ID, "cuda"
                )
                logger.info("GPU pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load GPU pipeline: {e}. Falling back to CPU.", exc_info=True)
                return await load_cpu_pipeline()
                
        return app.state.gpu_pipeline, "gpu", LLAMA_3_CONTEXT_WINDOW, MAX_NEW_TOKENS_GPU, RETRIEVE_TOP_K_GPU
    else:
        logger.info("GPU hours exhausted or unavailable. Loading CPU pipeline.")
        return await load_cpu_pipeline()
    
async def gpu_hours_available() -> bool:

    force_gpu = False
    if force_gpu:
        return True

    remaining_hours = 10
    return remaining_hours > GPU_MIN_FREE_HOURS_THRESHOLD

app = FastAPI(title="RAG Inference API (Chroma + Llama 3)", version="1.0.0")

@app.on_event("startup")
async def load_models():
    try:
        logger.info("Starting FastAPI model loading...")
        
        client = await asyncio.to_thread(initialize_chroma_client) 

        if not CHROMA_COLLECTION:
            raise RuntimeError("CHROMA_COLLECTION variable not set in env")
        app.state.chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
        if(app.state.chroma_collection):
            app.state.chroma_ready = app.state.chroma_collection is not None
        logger.info(f" Loaded collection: {CHROMA_COLLECTION} (Documents: {app.state.chroma_collection.count()})")


        app.state.cache_collection = client.get_or_create_collection(name=CHROMA_CACHE_COLLECTION)
        logger.info(f"Loaded Cache collection: {CHROMA_CACHE_COLLECTION} ({app.state.cache_collection.count()} items)")
        
        app.state.gpu_pipeline: Optional[Pipeline] = None  # type: ignore
        app.state.cpu_pipeline: Optional[object] = None  # type: ignore
        app.state.llm_cpp: Optional[Llama] = None # type: ignore
        app.state.tokenizer: Optional[AutoTokenizer] = None # type: ignore
                
        if not app.state.chroma_ready:
            raise RuntimeError("ChromaDB critical component failed to load.")


        await load_cpu_pipeline()
        logger.info("FastAPI models loaded successfully (CPU pipeline pre-warmed).")
        
    except Exception as e:
        app.state.chroma_ready = False
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    
class HistoryMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's latest message.")
    history: List[HistoryMessage] = Field(default_factory=list, description="The previous turns of the conversation.")
    stream: bool = Field(False) 

class RAGResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The final answer generated by the LLM.")
    sources: List[str] = Field(..., description="Unique source URLs used for the answer.")
    context_chunks: List[str] = Field(..., description="The final context chunks (text only) sent to the LLM.")
    expanded_queries: List[str] = Field(..., description="Queries used for retrieval.")


def call_llm_pipeline(pipe_like: Optional[object],
                      prompt_text: str,
                      deterministic=False,
                      max_new_tokens: int = MAX_NEW_TOKENS_CPU,
                      is_expansion: bool = False
                      ) -> str:
    """
    Unified caller for LLM:
    - Handles llama_cpp.Llama instances (CPU)
    - Handles transformers.Pipeline instances (GPU)
    """
    
    logging.info(f"model used: {pipe_like}")

    if pipe_like is None:
        raise HTTPException(status_code=503, detail="LLM pipeline is not available.")

    if deterministic:
        temp = 0.0
    elif is_expansion:
        temp = 0.1 
    else:
        temp = 0.6 
    tokenizer = getattr(app.state, "tokenizer", None)
    if tokenizer is None and isinstance(pipe_like, Pipeline):
        tokenizer = getattr(pipe_like, "tokenizer", None)
        
    try:
        if tokenizer:
            input_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=True))
            logger.info(f"LLM Input Token Count: {input_token_count}.")
    except Exception:
        logger.debug("Token counting failed, continuing without token count.")

    try:
        
        if isinstance(pipe_like, Llama): 
            llm = pipe_like 
            with torch.inference_mode(): 
                resp = llm(
                    prompt_text, 
                    max_tokens=max_new_tokens, 
                    temperature=temp,
                    stop=["<|eot_id|>", "<|start_header_id|>", "<|end_of_text|>"] 
                )
            text = resp.get("choices", [{}])[0].get("text", "").strip()
            return text
        
        elif isinstance(pipe_like, Pipeline):
            pipe = pipe_like 
            
            with torch.inference_mode():
                outputs = pipe(
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temp if temp > 0.0 else None, 
                    do_sample=True if temp > 0.0 else False,
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    return_full_text=False 
                )
            
            text = outputs[0]['generated_text'].strip()
            
            if '<|eot_id|>' in text:
                 text = text.split('<|eot_id|>')[0].strip()
            if '<|end_of_text|>' in text:
                text = text.split('<|end_of_text|>')[0].strip()
                
            return text
        
        else:
            logger.error(f"Unknown pipeline type: {type(pipe_like)}")
            raise TypeError(f"Unknown pipeline type: {type(pipe_like)}")

    except Exception as e:
        logger.error(f"Error calling LLM pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


async def expand_query_with_llm(pipe: Optional[object], 
                                  user_query: str,
                                  summary: str,
                                  history: Optional[List[HistoryMessage]]
                                  ) -> List[str]:
    """
    Implements the robust two-mode query strategy: expansion or rewriting.
    """
    
    messages = []
    expanded_queries: List[str] = []
    
    
    if not history or len(history) == 0:
        system_prompt = "You are a specialized query expansion engine."
        user_prompt = f"""
        Generate 3 alternative search queries similar to the user query below.
        The goal is to maximize retrieval relevance based on the user query.
        Return only the queries, one per line, without numbers or extra text.
        If user query is a greeting, don't reply with a greeting too and ask how you can help.
        If user query is gibberish

        User Query:
        {user_query}
        """
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    else:
        
        messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant who expands user queries into multiple search queries based on conversation history and user intent."
    },
    {
        "role": "user",
        "content": f"""
Given the conversation summary below and the user query, expand the user query into three queries that best reflect the conversation history, intent, and user needs.
Return only the queries, one per line, without numbers, preamble, or other text.

Conversation Summary:
{summary}

User Query:
{user_query}

Queries:
"""
    }
]

    tokenizer = getattr(app.state, "tokenizer", None)
    if tokenizer is None and isinstance(pipe, Pipeline):
        tokenizer = getattr(pipe, "tokenizer", None)

    if tokenizer:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        logger.warning("No tokenizer found for expansion, using simple join.")
        prompt_text = "\n".join([m["content"] for m in messages])

    logger.info(f"Query Expansion/Rewrite Prompt: {prompt_text}")
    start = time.time()

    llm_output = await asyncio.to_thread(
        call_llm_pipeline, pipe, prompt_text, deterministic=True, is_expansion=True, max_new_tokens=150
    )
    end = time.time()
    logger.info(f"Query Expansion/Rewrite Output: {llm_output} (Time: {end-start:.2f}s)")

    if not history or len(history) <= 0:
        expanded_queries = [
            q.strip() for q in llm_output.split('\n') 
            if q.strip() and "engine" not in q.lower() and "task" not in q.lower() and "search queries" not in q.lower()
        ]
    else:
        expanded_queries = [llm_output.strip()]

    expanded_queries.append(user_query)
        
    return list(set(q for q in expanded_queries if q))


def retrieve_context(queries: List[str], collection: Collection) -> Tuple[List[Dict], List[str]]:
    """Retrieves context from ChromaDB based on query embeddings."""
    try:
        embeddings_list = [
            [float(x) for x in emb] 
            for emb in EMBEDDING_MODEL.embed(queries, batch_size=8)
        ]
    except Exception as e:
        logger.error(f"Failed to generate embeddings for retrieval: {e}", exc_info=True)
        return [], []

    try:
        n_results_to_fetch = max(10, RETRIEVE_TOP_K_CPU * len(queries))
        start = time.time()
        results = collection.query(
            query_embeddings=embeddings_list,
            n_results=n_results_to_fetch, 
            include=['documents', 'metadatas']
        )
        end = time.time()
        logger.info(f'RETRIEVING TOOK:  {end-start:.2f}s')
    except Exception as e:
        logger.error(f"Chroma query failed: {e}")
        return [], []

    context_data = []
    source_urls = set()
    seen_texts = set()

    if results.get("documents") and results.get("metadatas"):
        for docs_list, metadatas_list in zip(results["documents"], results["metadatas"]):
            for doc, metadata in zip(docs_list, metadatas_list):
                if doc and metadata and doc not in seen_texts:
                    context_data.append({'text': doc, 'url': metadata.get('source')})
                    if metadata.get("source"):
                        source_urls.add(metadata.get('source'))
                    seen_texts.add(doc)

    return context_data, list(source_urls)


def rerank_documents(query: str, context: List[Dict], top_k: int) -> List[Dict]:
    """
    Re-ranks context documents using a cross-encoder.
    Returns the top-k most relevant documents.
    """
    if not context or not cross_encoder:
        logger.warning("Skipping reranking (no context or cross-encoder not loaded).")
        return context[:top_k]
    
    top_k = min(top_k, len(context))
    pairs = [(query, doc['text']) for doc in context]

    try:
        start = time.time()
        scores = cross_encoder.predict(pairs)
        end = time.time()
        logger.info(f'RERANKING TOOK {end-start:.2f}s')
    except Exception as e:
        logger.error(f"Cross-encoder prediction failed: {e}. Returning non-reranked results.", exc_info=True)
        return context[:top_k]

    for doc, score in zip(context, scores):
        doc['score'] = score

    ranked_docs = sorted(context, key=lambda x: x['score'], reverse=True)

    return ranked_docs[:top_k]

async def summarize_history(history: List[HistoryMessage], pipe: Optional[object]) -> str:
    """
    Summarizes the conversation history using the correct LLM (via call_llm_pipeline).
    """
    if not history:
        return ''
    
    history_text = "\n".join([f"{h.role}: {h.content}" for h in history[-8:]])
    
    tokenizer = getattr(app.state, "tokenizer", None)
    if tokenizer is None and isinstance(pipe, Pipeline):
        tokenizer = getattr(pipe, "tokenizer", None)
        
    history_tokens = len(tokenizer.encode(history_text)) if tokenizer else len(history_text.split())

    if history_tokens < 150: 
        return ""

    summarizer_prompt = f"""
        You are an intelligent agent who summarizes conversations. 
        Concisely summarize the key topics and entities discussed in the
        conversation history between a user and an assistant.
        The summary should be a few sentences long.
        
        CONVERSATION HISTORY:
        {history_text}
        
        CONCISE SUMMARY:
        """

    logger.info("Generating conversation summary...")
    start = time.time()
    summary = await asyncio.to_thread(
        call_llm_pipeline, pipe, summarizer_prompt, deterministic=True, max_new_tokens=150, is_expansion=False
    )
    end = time.time()
    logger.info(f"HISTORY SUMMARIZATION: {summary} (Time: {end-start:.2f}s)") 
    return summary


def build_prompt(user_query: str, context: List[Dict], summary: str) -> List[Dict]:
    """
    Builds the final list of messages for the chat template, including the RAG context.
    """
    
    messages = []
    context_text = "\n---\n".join([c['text'] for c in context]) if context else "No relevant context found."

    rag_system_prompt = f"""
        You are a helpful medical assistant with a friendly, conversational tone.
        Use the retrieved context to answer the user's query accurately. 
        If the context is missing, clearly state that the WHO factsheets don't contain the information.
        Do not repeat the user's question or the context in your response. Do not answer racist, harmful, discriminatory, non-health question.
        Formulate a direct, conversational answer using only the provided context as definitive truth.

        ---
        retrieved context: 
        {context_text}
        ---
        conversation history summary: 
        {summary if summary else "No summary available."}
        ---
    """
    messages.append({"role": "system", "content": rag_system_prompt})
    
    messages.append({"role": "user", "content": user_query})
    
    return messages


async def prune_messages_to_fit_context(messages: List[Dict], 
                                        final_context: List[Dict], 
                                        summary: str,
                                        max_input_tokens: int, 
                                        pipe: Optional[object]
                                        ) -> Tuple[List[Dict], List[Dict], int]:
    """
    Ensures the total prompt fits within max_input_tokens.
    Prunes retrieved context chunks and compresses summary if needed.
    Uses app.state.tokenizer (CPU) or pipe.tokenizer (GPU).
    """
    
    tokenizer = getattr(app.state, "tokenizer", None)
    if tokenizer is None and isinstance(pipe, Pipeline):
        tokenizer = getattr(pipe, "tokenizer", None)

    if not tokenizer:
        logger.error("Tokenizer not loaded for pruning.")
        return messages, final_context, 0

    def get_token_count(msg_list: List[Dict]) -> int:
        prompt_text = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(prompt_text, add_special_tokens=False))

    current_context = final_context[:]
    current_messages = messages[:]
    tok_length = 0
    token_count = get_token_count(current_messages)
    base_user_query = messages[-1]["content"]
    logger.info(f"Pre-pruning token count: {token_count}. Max: {max_input_tokens}")

    if token_count <= max_input_tokens:
        tok_length = max_input_tokens - token_count
        return current_messages, current_context, tok_length

    current_summary = summary
    if len(summary.split()) > 50:
        logger.warning(f"Context overflow ({token_count} > {max_input_tokens}). Compressing conversation summary.")
        compression_prompt = f"""
        The following conversation summary is too long for the LLM's context window.
        Rewrite it to be half its length, retaining only the essential topics.
        Do not add preamble or commentary.

        Current summary:
        {summary}

        Compressed summary:
        """
        start = time.time()
        new_summary_text = await asyncio.to_thread(
            call_llm_pipeline, pipe, compression_prompt, deterministic=True, is_expansion=False, max_new_tokens=75
        )
        end = time.time()
        current_summary = new_summary_text.strip()
        logger.info(f"SUMMARY COMPRESSED {end - start:.2f}s. New summary: {current_summary}")


        token_count = get_token_count(current_messages)
        logger.info(f"Token count after summary compression: {token_count}")

        if token_count <= max_input_tokens:
            tok_length = max_input_tokens - token_count
            return current_messages, current_context, tok_length

    logger.warning(f"Context still overflowing ({token_count} > {max_input_tokens}). Pruning context chunks.")
    while token_count > max_input_tokens and len(current_context) > 1:
        removed_chunk = current_context.pop()
        logger.warning(f"Removing last context chunk: {removed_chunk['text'][:50]}...")

        current_messages = build_prompt(base_user_query, current_context, current_summary)
        token_count = get_token_count(current_messages)
        logger.info(f"Token count after removing a chunk: {token_count}")
        
        if token_count <= max_input_tokens:
                    tok_length = max_input_tokens - token_count
                    return current_messages, current_context, tok_length
        logger.warning(f"Context still overflowing ({token_count} > {max_input_tokens}). Aggressively dropping least-relevant chunks.")
                
    while token_count > max_input_tokens and len(current_context) > 1: 
        removed_chunk = current_context.pop()
        logger.warning(f"Removing last context chunk: {removed_chunk['text'][:50]}...")
        
        current_messages = build_prompt(base_user_query, current_context, current_summary)
        token_count = get_token_count(current_messages)
        logger.info(f"Token count after removing a chunk: {token_count}")
        
    if token_count <= max_input_tokens:
        tok_length = max_input_tokens - token_count
        return current_messages, current_context, tok_length

    if token_count > max_input_tokens and current_context:
        logger.error(f"FATAL: Prompt still exceeds limit ({token_count}) with only 1 chunk remaining. Token-based truncation required.")
        
        tokens_without_chunk = get_token_count(build_prompt(base_user_query, [], current_summary)) 
        
        max_chunk_tokens = max_input_tokens - tokens_without_chunk - 5
        
        final_chunk = current_context[0]
        
        if max_chunk_tokens > 50:
            encoded_chunk = tokenizer.encode(final_chunk['text'])
            truncated_tokens = encoded_chunk[:max_chunk_tokens]
            
            final_chunk['text'] = tokenizer.decode(truncated_tokens, skip_special_tokens=True) + " [TRUNCATED]"
            
            current_messages = build_prompt(base_user_query, current_context, current_summary)
            token_count = get_token_count(current_messages)
            logger.warning(f"Aggressively truncated final chunk. New count: {token_count}")
        else:

            current_context = []
            current_messages = build_prompt(base_user_query, current_context, current_summary)
            token_count = get_token_count(current_messages)
            logger.warning("Remaining context budget too small; removing all context.")
            
        tok_length = max_input_tokens - token_count
        return current_messages, current_context, tok_length 

    tok_length = max_input_tokens - token_count if token_count < max_input_tokens else 0
    return current_messages, current_context, tok_length


async def Greet(query, pipe):
    messages = []
    logging.info(f"User sent a greeting")
    prompt_text = """You are a greeter. Your job is to respond politely to the user greeting. 
    ONLY a single polite and short greetings. Do not do anything else. 
    
    Examples:
    User: Hi
    Assistant: Hello, How may I help you today?
    
    User: how are you?
    Assistant: I am good, I can help you answer health related questions"""
     
   
    messages.append({"role": "system", "content": prompt_text})
    messages.append({"role": "user", "content": query})
    tokenizer = getattr(app.state, "tokenizer", None)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    answer = await asyncio.to_thread( call_llm_pipeline, 
        pipe, 
        prompt_text,
        deterministic=True,
        max_new_tokens=50,
        is_expansion= True
    )
    
    return RAGResponse(
        query=query,
        answer=answer,
        sources=[],
        context_chunks=[],
        expanded_queries=[]
    )
    
async def HarmOff(query, pipe):
    messages = []
    logging.info(f"User asked harmful or off-topic question")
    prompt_text = """
    You are an intelligent assistant. 
    Your job is to inform the user that you are not allowed to answer such questions. 
    Keep it short and brief, in one sentence.
    
    Examples:
    user: write a code to print a number
    Assistant: I am not allowed to answer such questions
    
    User: how can I be racist
    Assistant: Sorry, I am not allowed to answer such questions
    """ 
    
    messages.append({"role": "system", "content": prompt_text})
    messages.append({"role": "user", "content": query})
    tokenizer = getattr(app.state, "tokenizer", None)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    answer = await asyncio.to_thread( call_llm_pipeline, 
        pipe, 
        prompt_text,
        deterministic=True,
        max_new_tokens=50,
        is_expansion= True
    )

    
    return RAGResponse(
        query=query,
        answer=answer,
        sources=[],
        context_chunks=[],
        expanded_queries=[]
    )    
        

async def classify_intent(query: str, pipe: object) -> str:
    """Uses a rigid, JSON-forced prompt to classify user intent."""
    
    tokenizer = getattr(app.state, "tokenizer", None)
    if not tokenizer:
        return "ERROR_TOKENIZER"

    system_prompt = """You are a query classification robot. You MUST respond with ONLY ONE JSON object:
    {"intent": "MEDICAL"}
    {"intent": "GREET"}
    {"intent": "OFF_TOPIC"}
    {"intent": "HARMFUL"}

    Do not add any other text, explanations, or punctuation.
    """
    # Use few-shot examples (as provided in the original prompt, but restructured for JSON)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Query: What are the symptoms of COVID-19?"},
        {"role": "assistant", "content": '{"intent": "MEDICAL"}'},
        {"role": "user", "content": "Query: Hello"},
        {"role": "assistant", "content": '{"intent": "GREET"}'},
        {"role": "user", "content": "Query: Tell me a joke"},
        {"role": "assistant", "content": '{"intent": "OFF_TOPIC"}'},
        {"role": "user", "content": "Query: Make a racist insult"},
        {"role": "assistant", "content": '{"intent": "HARMFUL"}'},

        {"role": "user", "content": f"Query: {query}"}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm_output = await asyncio.to_thread(
        call_llm_pipeline,
        pipe,
        prompt_text,
        deterministic=True,
        max_new_tokens=25,  # Enough for the JSON object
        is_expansion=False  # Classification is not expansion
    )
    
    # --- Robust JSON Parsing ---
    try:
        clean_output = llm_output.strip().replace("```json", "").replace("```", "")
        start_idx = clean_output.find('{')
        end_idx = clean_output.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = clean_output[start_idx : end_idx + 1]
            data = json.loads(json_str)
            return data.get("intent", "UNKNOWN")
            
    except Exception as e:
        logger.error(f"Failed to parse JSON classifier output: {e}. Raw: {llm_output}")
        # Fallback to check for the raw label token in case of parsing failure
        raw_output_upper = llm_output.upper()
        for label in ["MEDICAL", "GREET", "OFF_TOPIC", "HARMFUL"]:
             if label in raw_output_upper:
                 return label
                 
    return "UNKNOWN"

@app.get("/health")
async def health_check():
    """Endpoint for checking the status of the RAG service."""
    chroma_ok = getattr(app.state, 'chroma_ready', False) and app.state.chroma_collection is not None
    
    if not chroma_ok:
        raise HTTPException(status_code=503, detail="Vector DB is not loaded.")
    
    llm_status = "not_loaded"
    if getattr(app.state, 'gpu_pipeline', None) or getattr(app.state, 'cpu_pipeline', None):
        llm_status = "loaded"
    else:
        try:
            await load_cpu_pipeline() 
            llm_status = "lazy_loaded_cpu_ok"
        except Exception as e:
            logger.error(f"Health check failed to load CPU model: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Chroma is loaded, but failed to load fallback LLM: {e}")

    return {"status": "ok", "service": "rag-service", "chroma": "loaded", "llm": llm_status}


@app.post("/rag", response_model=RAGResponse)
async def rag_handler(request: QueryRequest):


    start = time.time()
    try:
        pipe, runtime_env, max_context, max_gen, top_k = await load_cpu_pipeline()
        logger.info(f"Using model: {pipe}")
        
    except HTTPException as e:
        logger.error(f"Failed to get LLM pipeline: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unhandled error getting pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to load LLM model: {str(e)}")
        
    if not getattr(app.state, 'chroma_ready', False) or not app.state.chroma_collection:
        raise HTTPException(status_code=503, detail="Service is initializing or failed to load Vector DB.")
        
    try:
        answer = await classify_intent(request.query, pipe)
        end_time = time.time()
        logger.info(f"answer directly by model: {answer}, TIME: {end_time-start:.2f}s")
        
        if answer == 'GREET':
            response = await Greet(request.query, pipe)
            end_time = time.time()
            logger.info(f"Query handled directly by model: {request.query}, TIME: {end_time-start:.2f}s")
            logger.info(f"answer directly by model: {response}, TIME: {end_time-start:.2f}s")
            return response
        if answer == "HARMFUL"  or answer == "OFF_TOPIC":
            response = await HarmOff(request.query, pipe)
            end_time = time.time()
            logger.info(f"Query handled directly by model: {request.query}, TIME: {end_time-start:.2f}s")
            logger.info(f"answer directly by model: {response}, TIME: {end_time-start:.2f}s")
            return response 

        logger.info("Classifier returned RETRIEVE. Starting RAG pipeline.")
        
        summary = await summarize_history(request.history, pipe) 
        
        expanded_queries = await expand_query_with_llm(pipe, request.query, summary, request.history)
        
        context_data, all_sources = await asyncio.to_thread(retrieve_context, expanded_queries, app.state.chroma_collection)
        
        final_context = await asyncio.to_thread(rerank_documents, request.query, context_data, top_k=top_k)
        final_sources = list({c['url'] for c in final_context if c.get('url')})

        if not final_context:
            final_answer = "I could not find relevant documents in the knowledge base to answer your question. I can help you if you have another question."
            context_chunks_text = [] 
        else:
            initial_messages = build_prompt(request.query, final_context, summary)
            
            max_input_tokens = max_context - max_gen - SAFETY_BUFFER
            logger.info(
                f"Runtime: {runtime_env}, Max Context: {max_context}, "
                f"Max Gen: {max_gen}, Max Input: {max_input_tokens}"
            )
            
            final_messages, final_context_pruned, tok_length = await prune_messages_to_fit_context(
                initial_messages, 
                final_context,
                summary,
                max_input_tokens,
                pipe 
            )
            
            context_chunks_text = [c['text'] for c in final_context_pruned]
            
            tokenizer = getattr(app.state, "tokenizer", None)
            if tokenizer:
                prompt_text = tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
            else:
                logger.warning("Tokenizer not found for final prompt, using simple join.")
                prompt_text = "\n".join([m["content"] for m in final_messages])

            final_answer = await asyncio.to_thread(
                call_llm_pipeline, 
                pipe, 
                prompt_text, 
                deterministic=False, 
                max_new_tokens=max(max_gen, tok_length)
            )

        end_time = time.time()
        logger.info(
            json.dumps({
                "query": request.query,
                "latency_sec": round(end_time - start, 2),
                "runtime_env": runtime_env,
                "num_sources": len(final_sources),
                "num_context_chunks_sent": len(context_chunks_text),
                "expanded_queries": expanded_queries,
                "final_answer": final_answer,
                "retrieved_context_snippets": [c[:50] + "..." for c in context_chunks_text] 
            })
        )

        return RAGResponse(
            query=request.query,
            answer=final_answer,
            sources=final_sources, 
            context_chunks=context_chunks_text,
            expanded_queries=expanded_queries
        )
            
    except Exception as e:
        logger.error(f"Unhandled exception in RAG handler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")