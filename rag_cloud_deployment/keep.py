# from __future__ import annotations
# import asyncio
# import gc
# import os
# import json
# import logging
# import torch
# from typing import List, Dict, Tuple, Optional, Literal
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
# from sentence_transformers import CrossEncoder
# from fastembed import TextEmbedding 
# from s3_utils import download_chroma_folder_from_s3

# import chromadb
# from chromadb.api import Collection
# from chromadb import PersistentClient
# from chromadb.api.types import QueryResult
# import time
# from llama_cpp import Llama 

# logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
# logger = logging.getLogger(__name__)

# CHROMA_DIR = os.getenv("CHROMA_DIR")
# CHROMA_DIR_INF = "/" + CHROMA_DIR
# CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
# CHROMA_CACHE_COLLECTION = os.getenv("CHROMA_CACHE_COLLECTION", "semantic_cache")

# LLM_MODEL_CPU_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
# LLM_MODEL_GPU_ID = "meta-llama/Llama-3.1-8B-Instruct"
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# CHROMA_DB_FILENAME = os.getenv("CHROMA_DB_FILENAME")
# SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", 1000))
# SUMMARY_TARGET_TOKENS = int(os.getenv("SUMMARY_TARGET_TOKENS", 120))
# # SEMANTIC_CACHE_DIST_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIM_THRESHOLD", 0.1))

# RETRIEVE_TOP_K_CPU = int(os.getenv("RETRIEVE_TOP_K_CPU", 3))
# RETRIEVE_TOP_K_GPU = int(os.getenv("RETRIEVE_TOP_K_GPU", 8))
# MAX_NEW_TOKENS_CPU = int(os.getenv("MAX_NEW_TOKENS_CPU", 256))
# MAX_NEW_TOKENS_GPU = int(os.getenv("MAX_NEW_TOKENS_GPU", 1024))
# # GPU_MIN_FREE_HOURS_THRESHOLD = float(os.getenv("GPU_MIN_FREE_HOURS_THRESHOLD", 0.5))
# GPU_MIN_FREE_HOURS_THRESHOLD = 11

# LLAMA_GGUF_PATH = os.getenv("LLAMA_GGUF_PATH", "/model/tinyllama-reasoning.Q4_K_M.gguf")
# LLM_TOKENIZER_ID = "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"

# TINYLAMA_CONTEXT_WINDOW = 2048 
# LLAMA_3_CONTEXT_WINDOW = 8192
# SAFETY_BUFFER = 50             
# # MAX_INPUT_TOKENS = TINYLAMA_CONTEXT_WINDOW - MAX_NEW_TOKENS - SAFETY_BUFFER 

# LLAMA_3_CHAT_TEMPLATE = (
#     "{% for message in messages %}"
#     "{% if message['role'] == 'user' %}"
#     "{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
#     "{% elif message['role'] == 'assistant' %}"
#     "{{ '<|start_header_id|>assistant<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
#     "{% elif message['role'] == 'system' %}"
#     "{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
#     "{% endif %}"
#     "{% if loop.last and message['role'] == 'user' %}"
#     "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
#     "{% endif %}"
#     "{% endfor %}"
# )

# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# try:
#     cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
#     logger.info("Cross-encoder model loaded successfully.")
# except Exception as e:
#     logger.warning("Cross-encoder model error: %s", e)

# EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
# _ = list(EMBEDDING_MODEL.embed(["warmup"])) 
# logger.info("FastEmbed model warmup complete.")


# def initialize_llm_pipeline(model_id: str, device: str) -> Pipeline:
#     logger.info("Initializing llama model for inference")
 
#     try:
#         if device == "cuda" and torch.cuda.is_available():
#             dtype = torch.bfloat16
#             device_map: str = "auto"
#             low_cpu_mem_usage = True
#         else:
#             dtype = torch.bfloat16
#             device_map: str = "cpu"
#             low_cpu_mem_usage = False
            
        
#         # app.state.tokenizer = tokenizer 
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
        
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=device_map,
#             dtype=dtype,
#             low_cpu_mem_usage=low_cpu_mem_usage
#         )
        
#         model.eval()
        
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#         )
        
#         if not pipe.tokenizer.chat_template:
#             logger.info(f"Applying custom llama 3 chat template to {model_id}")
#             pipe.tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
            
#         logger.info(f"Model initialization successful for {model_id} on {device}.")
        
#         return pipe 
    
#     except Exception as e:
#         logger.error(f"Failed to initialize LLM pipeline {model_id}: {e}", exc_info=True)
#         raise 


# def initialize_chroma_client() -> chromadb.PersistentClient:
#     """Initializes Chroma client and loads the index from S3/disk."""
#     logger.info(f"Initializing Chroma client from persistence directory: {CHROMA_DIR_INF}")
    
#     local_db = os.path.exists(CHROMA_DIR_INF) and os.listdir(CHROMA_DIR_INF)
#     db_exist = os.path.join(CHROMA_DIR_INF, "chroma.sqlite3")
    
#     if not local_db or not os.path.exists(db_exist):
#         logger.warning(f"local chroma directory {CHROMA_DIR_INF} is missing or empty.\
#             Attempting download from s3.")
        
#         if CHROMA_DIR:
#             download_chroma_folder_from_s3(
#                 s3_prefix=CHROMA_DIR,
#                 local_dir=CHROMA_DIR_INF
#             )
#             logger.info("Chroma data downloaded from S3.")
#         else:
#             logger.error("CHROMA_DIR is not set. Cannot retrieve chroma index")
#             raise RuntimeError("Chroma index failed to load")
#     else:
#         logger.info(f"Local chroma data found at {CHROMA_DIR_INF}.")
#     logger.info(f"Initializing chroma client from persistence directory: {CHROMA_DIR_INF}")
#     # logger.info(f"Directory contents: {os.listdir(CHROMA_DIR_INF)}")
#     try:
#         client = PersistentClient(path=CHROMA_DIR_INF, settings =chromadb.Settings(allow_reset=False))
#         logger.info(" Chroma client initialized successfully.")
#     except Exception as e:
#         logger.error(f" Failed to load Chroma index: {e}")
#         raise RuntimeError("Chroma index failed to load.")

#     return client

# async def load_cpu_pipeline() -> Tuple[Pipeline, str, int, int, int]:
#     """
#     Lazily loads the CPU pipeline, clearing the GPU pipeline if it exists.
#     """
#     if getattr(app.state, 'cpu_pipeline', None) is None:
#         logger.info("Lazy-loading TinyLlama (CPU)...")
#         try:
            
#             if getattr(app.state, 'gpu_pipeline', None):
#                 del app.state.gpu_pipeline
#                 app.state.gpu_pipeline = None
#                 gc.collect()
#                 logger.info("Cleared GPU pipeline from memory.")
                
#             app.state.cpu_pipeline = await asyncio.to_thread(
#                 initialize_llm_pipeline, LLM_MODEL_CPU_ID, "cpu"
#             )
#             logger.info("CPU pipeline loaded successfully.")
#         except Exception as e:
#             logger.error(f"FATAL: Failed to load CPU pipeline: {e}", exc_info=True)
#             raise HTTPException(status_code=503, detail=f"Failed to load fallback CPU model: {e}")
            
#     return app.state.cpu_pipeline, "cpu", TINYLAMA_CONTEXT_WINDOW, MAX_NEW_TOKENS_CPU, RETRIEVE_TOP_K_CPU


# async def get_pipeline_for_runtime() -> Tuple[Pipeline, str, int, int, int]:
#     """
#     Determines runtime, lazily loads the correct pipeline (GPU/CPU),
#     and returns the pipeline and its associated settings.
#     """
#     if await gpu_hours_available():
#         logger.info("GPU hours available. Attempting to load GPU pipeline.")
#         if getattr(app.state, 'gpu_pipeline', None) is None:
#             logger.info("Lazy-loading Llama-3.1-8B (GPU)...")
#             try:
#                 if getattr(app.state, 'cpu_pipeline', None):
#                     del app.state.cpu_pipeline
#                     app.state.cpu_pipeline = None
#                     gc.collect() 
#                     logger.info("Cleared CPU pipeline from memory.")
                    
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
                    
#                 app.state.gpu_pipeline = await asyncio.to_thread(
#                     initialize_llm_pipeline, LLM_MODEL_GPU_ID, "cuda"
#                 )
#                 logger.info("GPU pipeline loaded successfully.")
#             except Exception as e:
#                 logger.error(f"Failed to load GPU pipeline: {e}. Falling back to CPU.", exc_info=True)
#                 return await load_cpu_pipeline()
                
#         return app.state.gpu_pipeline, "gpu", LLAMA_3_CONTEXT_WINDOW, MAX_NEW_TOKENS_GPU, RETRIEVE_TOP_K_GPU
#     else:
#         logger.info("GPU hours exhausted or unavailable. Loading CPU pipeline.")
#         return await load_cpu_pipeline()
    
# async def gpu_hours_available() -> bool:

#     force_gpu = False
#     if force_gpu:
#         return True

#     remaining_hours = 10
#     return remaining_hours > GPU_MIN_FREE_HOURS_THRESHOLD

# app = FastAPI(title="RAG Inference API (Chroma + Llama 3)", version="1.0.0")

# @app.on_event("startup")
# async def load_models():
#     try:
#         logger.info("Starting FastAPI model loading...")
        
#         client = await asyncio.to_thread(initialize_chroma_client)
#         # app.state.llm_pipeline = await asyncio.to_thread(initialize_llm_pipeline)
        

#         if not CHROMA_COLLECTION:
#             raise RuntimeError("CHROMA_COLLECTION variable not set in env")
#         app.state.chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
#         if(app.state.chroma_collection):
#             app.state.chroma_ready = app.state.chroma_collection is not None
#         logger.info(f" Loaded collection: {CHROMA_COLLECTION} (Documents: {app.state.chroma_collection.count()})")


#         app.state.cache_collection = client.get_or_create_collection(name=CHROMA_CACHE_COLLECTION)
#         logger.info(f"Loaded Cache collection: {CHROMA_CACHE_COLLECTION} ({app.state.cache_collection.count()} items)")
        
#         app.state.gpu_pipeline: Optional[Pipeline] = None   # pyright: ignore[reportInvalidTypeForm]
#         app.state.cpu_pipeline: Optional[Pipeline] = None    # pyright: ignore[reportInvalidTypeForm]
             
#         # app.state.llm_loaded = app.state.llm_pipeline is not None
        
#         if not app.state.chroma_ready:
#             raise RuntimeError("ChromaDB critical component failed to load.")
        
#         # if not app.state.llm_loaded:
#         #     raise RuntimeError("One or more critical components (LLM/Chroma) failed to load.")

#         logger.info("FastAPI models loaded successfully.")
#     except Exception as e:
#         app.state.chroma_ready = False
#         logger.error(f"Error during startup: {e}", exc_info=True)
#         raise
    
# class HistoryMessage(BaseModel):
#     role: Literal['user', 'assistant']
#     content: str

# class QueryRequest(BaseModel):
#     query: str = Field(..., description="The user's latest message.")
#     # conversation_id: Optional[str] = Field(None)
#     history: List[HistoryMessage] = Field(default_factory=list, description="The previous turns of the conversation.")
#     stream: bool = Field(False) 

# class RAGResponse(BaseModel):
#     query: str = Field(..., description="The original user query.")
#     answer: str = Field(..., description="The final answer generated by the LLM.")
#     sources: List[str] = Field(..., description="Unique source URLs used for the answer.")
#     context_chunks: List[str] = Field(..., description="The final context chunks (text only) sent to the LLM.")
#     expanded_queries: List[str] = Field(..., description="Queries used for retrieval.")


# def call_llm_pipeline(pipe: Pipeline, 
#                       prompt_text: str, 
#                       deterministic=False, 
#                       max_new_tokens: int = MAX_NEW_TOKENS_CPU, 
#                       is_expansion: bool = False
#                       ) -> str:

#     if not pipe:
#         raise HTTPException(status_code=503, detail="LLM pipeline is not available.")
    
#     # if not app.state.llm_loaded or not app.state.llm_pipeline:
#     #     raise HTTPException(status_code=503, detail="LLM model failed to load at startup.")

#     temp = 0.1 if is_expansion and deterministic else 0.6
    
#     try:
#         input_token_count = len(pipe.tokenizer.encode(prompt_text, add_special_tokens=True))
#         logger.info(f"LLM Input Token Count: {input_token_count}. Max Input: {MAX_NEW_TOKENS_CPU}")

#         with torch.inference_mode():
#             outputs = pipe(
#                 prompt_text,
#                 max_new_tokens=max_new_tokens,
#                 do_sample= not deterministic,
#                 temperature=temp,
#                 return_full_text=False,
#                 eos_token_id=pipe.tokenizer.eos_token_id 
#             )

#         generated_text = outputs[0]['generated_text'].strip()
        
#         if '<|eot_id|>' in generated_text:
#             generated_text = generated_text.split('<|eot_id|>')[0].strip()
        
#         return generated_text

#     except Exception as e:
#         logger.error(f"Error calling LLM pipeline: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")



# async def expand_query_with_llm(pipe: Pipeline,
#                                 user_query: str,
#                                 summary: str,
#                                 history: Optional[List[HistoryMessage]]
#                                 ) -> List[str]:
#     """
#     Implements the robust two-mode query strategy: expansion or rewriting.
#     """
    
#     messages = []

#     expanded_queries: List[str] = []
    
    
#     if not history or len(history) == 0:
#         system_prompt = "You are a specialized query expansion engine."
#         user_prompt = f"""
#         Generate 3 alternative search queries similar to the user query below.
#         The goal is to maximize retrieval relevance based on the user query.
#         Return only the queries, one per line, without numbers or extra text.
#         If user query is a greeting, don't reply with a greeting too and ask how you can help.
#         If user query is gibberish

#         User Query:
#         {user_query}
#         """
#         messages.append({"role": "system", "content": system_prompt})
#         messages.append({"role": "user", "content": user_prompt})

#     else:
        
#         messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant who expands user queries into multiple search queries based on conversation history and user intent."
#     },
#     {
#         "role": "user",
#         "content": f"""
# Given the conversation summary below and the user query, expand the user query into three queries that best reflect the conversation history, intent, and user needs.
# Return only the queries, one per line, without numbers, preamble, or other text.

# Conversation Summary:
# {summary}

# User Query:
# {user_query}

# Queries:
# """
#     }
# ]


#     tokenizer = getattr(app.state, "tokenizer", None)
#     if tokenizer:
#         prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     else:
#         # fallback simple join
#         prompt_text = "\n".join([m["content"] for m in messages])

#     logger.info(f"Query Expansion/Rewrite Prompt: {prompt_text}")
#     start = time.time()

#     llm_output = await asyncio.to_thread(call_llm_pipeline, app.state.llm_cpp, prompt_text, deterministic=True, is_expansion=True, max_new_tokens=150)
#     end = time.time()
#     logger.info(f"Query Expansion/Rewrite Output: {llm_output} (Time: {end-start:.2f}s)")

#     if not history or len(history) <= 0:
#         expanded_queries = [
#             q.strip() for q in llm_output.split('\n') 
#             if q.strip() and "engine" not in q.lower() and "task" not in q.lower() and "search queries" not in q.lower()
#         ]
#     else:
#         expanded_queries = [llm_output.strip()]

#     expanded_queries.append(user_query)
#     return list(set(q for q in expanded_queries if q))

    
    
#     # prompt_text = pipe.tokenizer.apply_chat_template(
#     #     messages, tokenize=False, add_generation_prompt=True
#     # )
        
#     # logger.info(F"Query Expansion/Rewrite Prompt: {prompt_text}")
#     # start = time.time()
    
#     # llm_output = await asyncio.to_thread(
#     #     call_llm_pipeline, pipe, prompt_text, deterministic=True, is_expansion=True , max_new_tokens=150
#     # )
#     # end = time.time()
#     # logger.info(f"Query Expansion/Rewrite Output: {llm_output} (Time: {end-start:.2f}s)")
    
#     # if not history or len(history) <= 0:
#     #     expanded_queries = [
#     #         q.strip() for q in llm_output.split('\n') 
#     #         if q.strip() and "engine" not in q.lower() and "task" not in q.lower() and "search queries" not in q.lower()
#     #     ]
#     # else:
#     #     expanded_queries = [llm_output.strip()]
        
#     # expanded_queries.append(user_query)
    
#     # return list(set(q for q in expanded_queries if q))


# def retrieve_context(queries: List[str], collection: Collection) -> Tuple[List[Dict], List[str]]:
#     """Retrieves context from ChromaDB based on query embeddings."""

#     try:
#         embeddings_list = [
#             [float(x) for x in emb] 
#             for emb in EMBEDDING_MODEL.embed(queries, batch_size=8)
#         ]
#     except Exception as e:
#         logger.error(f"Failed to generate embeddings for retrieval: {e}", exc_info=True)
#         return [], []
    
#     try:
#         n_results_to_fetch = max(10, RETRIEVE_TOP_K_GPU * len(queries))
        
#         start = time.time()
#         results = collection.query(
#             query_embeddings=embeddings_list,
#             n_results=n_results_to_fetch, 
#             include=['documents', 'metadatas']
#         )
#         end = time.time()
#         logger.info(f'RETREIVING TOOK:  {end-start:.2f}s')
#     except Exception as e:
#         logger.error(f"Chroma query failed: {e}")
#         return [], []
    

#     context_data = []
#     source_urls = set()
#     seen_texts = set()

#     if results.get("documents") and results.get("metadatas"):
#         for docs_list, metadatas_list in zip(results["documents"], results["metadatas"]):
#             for doc, metadata in zip(docs_list, metadatas_list):
#                 if doc and metadata and doc not in seen_texts:
#                     context_data.append({'text': doc, 'url': metadata.get('source')})
#                     if metadata.get("source"):
#                         source_urls.add(metadata.get('source'))
#                     seen_texts.add(doc)
    
#     return context_data, list(source_urls)



# def rerank_documents(query: str, context: List[Dict], top_k: int) -> List[Dict]:
#     """
#     Re-ranks context documents using a cross-encoder.
#     Returns the top-k most relevant documents.
#     """
#     if not context or not cross_encoder:
#         logger.warning("Skipping reranking (no context or cross-encoder not loaded).")
#         return context[:top_k]
    
#     top_k = min(top_k, len(context))
#     pairs = [(query, doc['text']) for doc in context]

#     try:
#         start = time.time()
#         scores = cross_encoder.predict(pairs)
#         end = time.time()
#         logger.info(f'RERANKING TOOK {end-start:.2f}s')
#     except Exception as e:
#         logger.error(f"Cross-encoder prediction failed: {e}. Returning non-reranked results.", exc_info=True)
#         return context[:top_k]

#     for doc, score in zip(context, scores):
#         doc['score'] = score

#     ranked_docs = sorted(context, key=lambda x: x['score'], reverse=True)

#     return ranked_docs[:top_k]


# async def summarize_history(history: List[HistoryMessage], pipe: Pipeline) -> str:
#     """
#     Summarizes the conversation history using the LLM. 
#     """
#     if not history:
#         return ''
    
#     history_text = "\n".join([f"{h.role}: {h.content}" for h in history[-8:]])
    
#     history_tokens = len(pipe.tokenizer.encode(history_text))
    
#     if history_tokens < 150: 
#         return ""

#     summarizer_prompt = f"""
#         You are an intelligent agent who summarizes conversations. 
#         Concisely summarize the key topics and entities discussed in the
#         conversation history between a user and an assistant.
#         The summary should be a few sentences long.
        
#         CONVERSATION HISTORY:
#         {history_text}
        
#         CONCISE SUMMARY:
#         """
        
#     logger.info("Generating conversation summary...")
#     start = time.time()
#     summary =  await asyncio.to_thread(
#         call_llm_pipeline, pipe, summarizer_prompt, deterministic=True, max_new_tokens=150, is_expansion=False 
#     )
#     end = time.time()
#     logger.info(f"HISTORY SUMMARIZATION: {summary} (Time: {end-start:.2f}s)") 
#     return summary

# def build_prompt(user_query: str, context: List[Dict], summary: str) -> List[Dict]:
#     """
#     Builds the final list of messages for the chat template, including the RAG context.
#     """
    
#     messages = []
#     context_text = "\n---\n".join([c['text'] for c in context]) if context else "No relevant context found."

#     rag_system_prompt = f"""
#         You are a helpful medical assistant with a friendly, conversational tone.
#         Use the retrieved context to answer the user's query accurately. 
#         If the context is missing, clearly state that the WHO factsheets don't contain the information.
#         Do not repeat the user's question or the context in your response.
#         Formulate a direct, conversational answer using only the provided context as definitive truth.

#         ---
#         retrieved context: 
#         {context_text}
#         ---
#         conversation history summary: 
#         {summary if summary else "No summary available."}
#         ---
#     """
#     messages.append({"role": "system", "content": rag_system_prompt})
    
#     messages.append({"role": "user", "content": user_query})
    
#     return messages


# async def prune_messages_to_fit_context(messages: List[Dict], 
#                                         final_context: List[Dict], 
#                                         summary: str,
#                                         max_input_tokens: int, 
#                                         pipe: Pipeline 
#                                         ) -> Tuple[List[Dict], List[Dict]]:
#     """
#     Ensures the total prompt fits within max_input_tokens.
#     Prunes retrieved context chunks and compresses summary if needed.
#     """

#     def get_token_count(msg_list: List[Dict]) -> int:
#         """Count tokens for all messages using the chat template."""
#         prompt_text = pipe.tokenizer.apply_chat_template(
#             msg_list, tokenize=False, add_generation_prompt=True
#         )
#         return len(pipe.tokenizer.encode(prompt_text, add_special_tokens=False))

#     current_context = final_context[:]
#     current_messages = messages[:]

#     token_count = get_token_count(current_messages)
#     logger.info(f"Pre-pruning token count: {token_count}. Max: {max_input_tokens}")

#     if token_count <= max_input_tokens:
#         return current_messages, current_context

#     current_summary = summary
#     if len(summary.split()) > 50: 
#         logger.warning(f"Context overflow ({token_count} > {max_input_tokens}). Compressing conversation summary.")

#         compression_prompt = f"""
#         The following conversation summary is too long for the LLM's context window.
#         Rewrite it to be half its length, retaining only the essential topics.
#         Do not add preamble or commentary.
        
#         Current summary:
#         {summary}
        
#         Compressed summary:
#         """

#         start = time.time()
#         new_summary_text = await asyncio.to_thread(
#             call_llm_pipeline,
#             pipe,
#             compression_prompt,
#             deterministic=True,
#             is_expansion=False,
#             max_new_tokens=75, 
#         )
#         end = time.time()
#         current_summary = new_summary_text.strip()
#         logger.info(f"SUMMARY COMPRESSED {end - start:.2f}s. New summary: {current_summary}")

#         from copy import deepcopy
#         base_user_query = deepcopy(current_messages[-1]["content"])
#         current_messages = build_prompt(base_user_query, current_context, current_summary)

#         token_count = get_token_count(current_messages)
#         logger.info(f"Token count after summary compression: {token_count}")

#         if token_count <= max_input_tokens:
#             return current_messages, current_context

#     logger.warning(f"Context still overflowing ({token_count} > {max_input_tokens}). Pruning context chunks.")
    
#     while token_count > max_input_tokens and len(current_context) > 1:
#         removed_chunk = current_context.pop()
#         logger.warning(f"Removing last context chunk: {removed_chunk['text'][:50]}...")

#         from copy import deepcopy
#         base_user_query = deepcopy(current_messages[-1]["content"])
#         current_messages = build_prompt(base_user_query, current_context, current_summary)

#         token_count = get_token_count(current_messages)
#         logger.info(f"Token count after removing a chunk: {token_count}")

#     if token_count > max_input_tokens:
#         logger.error(f"FATAL: Prompt still exceeds limit ({token_count}) even with 1 chunk. Proceeding, but output may be truncated.")
#         if current_context:
#             final_chunk_text = current_context[0]['text']
#             excess_tokens = token_count - max_input_tokens
#             estimated_chars_to_cut = excess_tokens * 4 + 200 
            
#             if len(final_chunk_text) > estimated_chars_to_cut:
#                 current_context[0]['text'] = final_chunk_text[:-estimated_chars_to_cut] + " [TRUNCATED]"
                
#             from copy import deepcopy
#             base_user_query = deepcopy(current_messages[-1]["content"])
#             current_messages = build_prompt(base_user_query, current_context, current_summary)
#             logger.warning(f"Aggressively truncated final chunk. New count: {get_token_count(current_messages)}")

#     return current_messages, current_context




# @app.get("/health")
# async def health_check():
#     """Endpoint for checking the status of the RAG service."""
#     chroma_ok = getattr(app.state, 'chroma_ready', False) and app.state.chroma_collection is not None
    
#     if not chroma_ok:
#         raise HTTPException(status_code=503, detail="Vector DB is not loaded.")
    
#     llm_status = "not_loaded"
#     if getattr(app.state, 'gpu_pipeline', None) or getattr(app.state, 'cpu_pipeline', None):
#         llm_status = "loaded"
#     else:
#         try:
#             await load_cpu_pipeline()
#             llm_status = "lazy_loaded_cpu_ok"
#         except Exception as e:
#             logger.error(f"Health check failed to load CPU model: {e}", exc_info=True)
#             raise HTTPException(status_code=503, detail=f"Chroma is loaded, but failed to load fallback LLM: {e}")

#     return {"status": "ok", "service": "rag-service", "chroma": "loaded", "llm": llm_status}

# @app.post("/rag", response_model=RAGResponse) 
#   async def rag_handler(request: QueryRequest):
#     """
#     Main RAG endpoint: performs query expansion, retrieval, and generation
#     using the dynamic GPU/CPU pipeline.
#     """
#     start = time.time()
    
#     try:
#         pipe, runtime_env, max_context, max_gen, top_k = await get_pipeline_for_runtime()
#     except HTTPException as e:
#         logger.error(f"Failed to get LLM pipeline: {e.detail}", exc_info=True)
#         raise e
#     except Exception as e:
#         logger.error(f"Unhandled error getting pipeline: {e}", exc_info=True)
#         raise HTTPException(status_code=503, detail=f"Failed to load LLM model: {str(e)}")
        
#     if not getattr(app.state, 'chroma_ready', False) or not app.state.chroma_collection:
#         raise HTTPException(status_code=503, detail="Service is initializing or failed to load Vector DB.")
        
#     try:
#         GREETINGS = ['hi', 'hello', 'hey', 'greetings', 'thanks', 'thank you', 'tysm']
#         query_lower = request.query.lower().strip().replace('!', '').replace('.', '').replace(',', '')
        
#         if query_lower in GREETINGS or not query_lower:
#             final_answer = "Hello! I'm here to help you with health information. What is your question?"
#             logger.info(f"Trivial query filtered: {request.query}")
#             return RAGResponse(
#                 query=request.query, answer=final_answer, sources=[], 
#                 context_chunks=[], expanded_queries=[]
#             )
            
#         summary = await summarize_history(request.history, pipe)
        
#         expanded_queries = await expand_query_with_llm(pipe, request.query, summary, request.history)
        
#         context_data, all_sources = await asyncio.to_thread(
#             retrieve_context, expanded_queries, app.state.chroma_collection
#         )
        
#         final_context = rerank_documents(request.query, context_data, top_k=top_k)
#         final_sources = list({c['url'] for c in final_context if c.get('url')})

#         if not final_context:
#             final_answer = "I could not find relevant documents in the knowledge base to answer your question. I can help you if you have another question."
#             context_chunks_text = [] 
#         else:
#             initial_messages = build_prompt(request.query, final_context, summary)
            
#             max_input_tokens = max_context - max_gen - SAFETY_BUFFER
#             logger.info(
#                 f"Runtime: {runtime_env}, Max Context: {max_context}, "
#                 f"Max Gen: {max_gen}, Max Input: {max_input_tokens}"
#             )
            
#             final_messages, final_context_pruned = await prune_messages_to_fit_context(
#                 initial_messages, 
#                 final_context,
#                 summary,
#                 max_input_tokens,
#                 pipe              
#             )
            
#             context_chunks_text = [c['text'] for c in final_context_pruned]
            
#             prompt_text = pipe.tokenizer.apply_chat_template(
#                 final_messages, tokenize=False, add_generation_prompt=True
#             )
            
#             final_answer = await asyncio.to_thread(
#                 call_llm_pipeline, 
#                 pipe, 
#                 prompt_text, 
#                 deterministic=False,
#                 max_new_tokens=max_gen 
#             )

#         end_time = time.time()
#         logger.info(
#             json.dumps({
#                 "query": request.query,
#                 "latency_sec": round(end_time - start, 2),
#                 "runtime_env": runtime_env, # NEW
#                 "num_sources": final_sources,
#                 "num_context_chunks_sent": len(context_chunks_text),
#                 "expanded_queries": expanded_queries,
#                 "final_answer": final_answer,
#                 "retrieved_context_snippets": [c[:50] + "..." for c in context_chunks_text] 
#             })
#         )

#         return RAGResponse(
#             query=request.query,
#             answer=final_answer,
#             sources=final_sources, 
#             context_chunks=context_chunks_text,
#             expanded_queries=expanded_queries
#         )
            
#     except Exception as e:
#         logger.error(f"Unhandled exception in RAG handler: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")