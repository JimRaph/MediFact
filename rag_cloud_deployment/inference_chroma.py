import asyncio
import os
import json
import logging
import torch
from typing import List, Dict, Tuple, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding 
from utils.s3_utils import download_chroma_folder_from_s3

import chromadb
from chromadb.api import Collection
from chromadb import PersistentClient
from chromadb.api.types import QueryResult
import time

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR")
CHROMA_DIR_INF = "/" + CHROMA_DIR
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHROMA_DB_FILENAME = os.getenv("CHROMA_DB_FILENAME")


TINYLAMA_CONTEXT_WINDOW = 2048 
MAX_NEW_TOKENS = 256           
SAFETY_BUFFER = 50             
MAX_INPUT_TOKENS = TINYLAMA_CONTEXT_WINDOW - MAX_NEW_TOKENS - SAFETY_BUFFER 

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
try:
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
    logger.info("Cross-encoder model loaded successfully.")
except Exception as e:
    logger.warning("Cross-encoder model error: %s", e)

EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
_ = list(EMBEDDING_MODEL.embed(["warmup"])) 
logger.info("FastEmbed model warmup complete.")


def initialize_llm_pipeline():
    logger.info("Initializing llama model for inference")
 
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    
    app.state.tokenizer = tokenizer 
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        top_p=0.95,
        max_new_tokens=MAX_NEW_TOKENS 
    )
    
    if not pipe.tokenizer.chat_template:
        pipe.tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    logger.info("Model initialization successful.")
    
    return pipe

def initialize_chroma_client() -> chromadb.PersistentClient:
    """Initializes Chroma client and loads the index from S3/disk."""
    logger.info(f"Initializing Chroma client from persistence directory: {CHROMA_DIR_INF}")
    
    local_db = os.path.exists(CHROMA_DIR_INF) and os.listdir(CHROMA_DIR_INF)
    db_exist = os.path.join(CHROMA_DIR_INF, "chroma.sqlite3")
    
    if not local_db or not os.path.exists(db_exist):
        logger.warning(f"local chroma directory {CHROMA_DIR_INF} is missing or empty.\
            Attempting download from s3.")
        
        if CHROMA_DIR:
            download_chroma_folder_from_s3(
                s3_prefix=CHROMA_DIR,
                local_dir=CHROMA_DIR_INF
            )
        else:
            logger.error("CHROMA_DIR is not set. Cannot retrieve chroma index")
            raise RuntimeError("Chroma index failed to load")
    else:
        logger.info(f"Local chroma data found at {CHROMA_DIR_INF}.")
    logger.info(f"Initializing chroma client from persistence directory: {CHROMA_DIR_INF}")
    logger.info(f"Directory contents: {os.listdir(CHROMA_DIR_INF)}")
    try:
        client = PersistentClient(path=CHROMA_DIR_INF, settings =chromadb.Settings(allow_reset=False))
        logger.info(" Chroma client initialized successfully.")
    except Exception as e:
        logger.error(f" Failed to load Chroma index: {e}")
        raise RuntimeError("Chroma index failed to load.")

    try:
        if not CHROMA_COLLECTION:
            raise RuntimeError("CHROMA_COLLECTION variable not set in env")
        collection = client.get_collection(name=CHROMA_COLLECTION)
        logger.info(f" Loaded collection: {CHROMA_COLLECTION} (Documents: {collection.count()})")
    except Exception as e:
        logger.error(f"Failed to get or create Chroma collection: {e}")
        raise
    
    return collection

app = FastAPI(title="RAG Inference API (Chroma + Llama 3)", version="1.0.0")

@app.on_event("startup")
async def load_models():
    try:
        logger.info("Starting FastAPI model loading...")
        
        app.state.chroma_collection = await asyncio.to_thread(initialize_chroma_client)
        app.state.llm_pipeline = await asyncio.to_thread(initialize_llm_pipeline)
        
        app.state.llm_loaded = app.state.chroma_collection is not None and app.state.llm_pipeline is not None
        
        if not app.state.llm_loaded:
            raise RuntimeError("One or more critical components (LLM/Chroma) failed to load.")

        logger.info("FastAPI models loaded successfully.")
    except Exception as e:
        app.state.llm_loaded = False
        logger.error(f"Error during startup: {e}")
        raise
    
class HistoryMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's latest message.")
    conversation_id: Optional[str] = Field(None)
    history: List[HistoryMessage] = Field(default_factory=list, description="The previous turns of the conversation.")
    stream: bool = Field(False) 

class RAGResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The final answer generated by the LLM.")
    sources: List[str] = Field(..., description="Unique source URLs used for the answer.")
    context_chunks: List[str] = Field(..., description="The final context chunks (text only) sent to the LLM.")
    expanded_queries: List[str] = Field(..., description="Queries used for retrieval.")


def call_llm_pipeline(prompt_text: str, deterministic=False, max_new_tokens: int = MAX_NEW_TOKENS, is_expansion: bool = False) -> str:
    """
    Calls the local LLM pipeline for generation, integrating history for conversational context.
    The 'prompt' parameter here contains the RAG prompt (context + final query).
    """
    if not app.state.llm_loaded or not app.state.llm_pipeline:
        raise HTTPException(status_code=503, detail="LLM model failed to load at startup.")

    temp = 0.1 if is_expansion and deterministic else 0.6
    
    try:
        input_token_count = len(app.state.tokenizer.encode(prompt_text, add_special_tokens=True))
        logger.info(f"LLM Input Token Count: {input_token_count}. Max Input: {MAX_INPUT_TOKENS}")

        with torch.inference_mode():
            outputs = app.state.llm_pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample= not deterministic,
                return_full_text=False,
                eos_token_id=app.state.llm_pipeline.tokenizer.eos_token_id 
            )

        generated_text = outputs[0]['generated_text'].strip()
        generated_text = generated_text.split('<|eot_id|>')[0].strip()
        return generated_text

    except Exception as e:
        logger.error(f"Error calling LLM pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

async def expand_query_with_llm(user_query: str, history: Optional[List[HistoryMessage]]) -> List[str]:
    """
    Implements the robust two-mode query strategy: expansion or rewriting.
    """
    
    messages = [
        # {"role": "system", "content": "You are a specialized query reformulation and expansion engine."},
    ]

    expanded_queries: List[str] = []
    
    if not history or len(history) <= 0:
        
        prompt = f"""
        You are a specialized query expansion engine. You are to generate a list of 2-3 
        alternative search queries that are similar to the user'squestion. 
        The goal is to maximize the chance of finding relevant documents.
        Return only the queries, one per line, without numbers or preamble.
        User Question: {user_query}
        """
        
    #     prompt = (
    #     "You are a Query Decomposer. Your task is to analyze the user's complex "
    #     "request and generate 2 to 3 concise, specific, and high-signal search "
    #     "queries that will maximize the chances of retrieving relevant documents. "
    #     "Respond ONLY with a JSON list of strings, like this: "
    #     '["query 1", "query 2"]'
    # )
        
        messages.append({"role": "system", "content": prompt})

        
        prompt_text = app.state.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(F"PROMPT TEXT: {prompt_text}")
        start = time.time()
        llm_output = await asyncio.to_thread(
            call_llm_pipeline, prompt_text, deterministic=True, is_expansion=True, max_new_tokens=150
        )
        end = time.time()
        logger.info(F"QUERY EXPANSION: {llm_output} \n TIME: {end-start}")
        expanded_queries = [
                    q.strip() for q in llm_output.split('\n') 
                    if q.strip() and "engine" not in q.lower() and "task" not in q.lower() and "search queries" not in q.lower()
                ]     
        expanded_queries.append(user_query)
        

    else:
        history_text = "\n".join([f"{h.role}: {h.content}" for h in history[-5:]])
        
        prompt = f"""
        Given the conversation history between a user and assistant, and the latest user query below, 
        think over the conversation history and then rewrite the user query into a concise, clear 
        query taking into account the conversation history.
        
        E.g
        user: what is covid-19
        assistant: Covid-19 is a virus that went viral in 2020, leading to closure of business and schools.
        user: How was it contained
        
        The conversation history helped the assistant to know that 'it' referred to the covid-19.
        So you could rewrite or rephrase the question as 'How was covid-19 contained'

        Conversation History:
        {history_text}
        
        Latest User Query: {user_query}
        """
        messages.append({"role": "system", "content": prompt})
        prompt_text = app.state.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        
        logger.info(F"HISTORY PROMPT TEXT: {prompt_text}")
        start = time.time()
        rewritten_query = await asyncio.to_thread(
            call_llm_pipeline, prompt_text, deterministic=True, is_expansion=True, max_new_tokens=100
        )
        end = time.time()
        logger.info(F"HISTORY QUERY EXPANSION: {rewritten_query} \n TIME: {end-start}")
        expanded_queries = [rewritten_query.strip(), user_query.strip()]
        
    return list(set(q for q in expanded_queries if q))


def retrieve_context(queries: List[str], collection: Collection) -> Tuple[List[Dict], List[str]]:
    """Retrieves context from ChromaDB based on query embeddings."""

    embeddings_list = [
        [float(x) for x in emb] 
        for emb in EMBEDDING_MODEL.embed(queries, batch_size=8)
        ]
    
    try:
        results = collection.query(
        query_embeddings=embeddings_list,
        n_results=10 * len(queries), 
        include=['documents', 'metadatas']
        )
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

def rerank_documents(query: str, context: List[Dict], top_k: int = 4) -> List[Dict]:
    """
    Re-ranks context documents using a cross-encoder.
    Returns the top-k most relevant documents.
    """
    if not context:
        return []
    
    top_k = min(top_k, len(context))
    pairs = [(query, doc['text']) for doc in context]

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(context, scores):
        doc['score'] = score

    ranked_docs = sorted(context, key=lambda x: x['score'], reverse=True)

    return ranked_docs[:top_k]

async def summarize_history(history: List[HistoryMessage]) -> str:
    """
    Summarizes the conversation history using the LLM. 
    """
    
    if not history:
        return ''
    
    history_text = "\n".join([f"{h.role}: {h.content}" for h in history[-5:]])

    summarizer_prompt = f"""
        You are an intelligent agent who summarizes conversations. 
        Summarize the conversation history between user and assistant
        into a smaller size that captures the intent and purpose of the conversation. 
        Your summary will be further passed into a tiny llm as part of the context
        to further conversation with the user.
        
        CONVERSATION HISTORY: {history_text}
        
        """
        
    logger.info(F"HISTORY PROMPT TEXT: {summarizer_prompt}")
    start = time.time()
    summary =  await asyncio.to_thread(
            call_llm_pipeline, summarizer_prompt, deterministic=True, is_expansion=False, max_new_tokens=100
        )
    end = time.time()
    logger.info(F"HISTORY SUMMARY : {summary} \n TIME: {end-start}") 
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
        Do not repeat the user's question or the context in your response.
        Formulate a direct, conversational answer using only the provided context as definitive truth.

        retrieved context: 
        {context_text}
        conversation history summary: 
        {summary}
    """
    messages.append({"role": "system", "content": rag_system_prompt})
    
    # if summary:
    #     messages.append({"role": "system", "content": f"Conversation Summary: {summary}"})
    

    # combined_query = (
    #     f"CONTEXT:\n{context_text}\n\n"
    #     f"QUESTION: {user_query}"
    # )
    # messages.append({"role": "user", "content": combined_query})
    # messages.append({
    #     "role": "system",
    #     "content": f"Reference context for answering:\n{context_text}"
    # })

    # User query as its own message
    messages.append({"role": "user", "content": user_query})
    
    return messages 


async def prune_messages_to_fit_context(
    messages: List[Dict], 
    final_context: List[Dict], 
    summary: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Ensures the total prompt (system + user) fits within MAX_INPUT_TOKENS.
    Prunes retrieved context chunks and compresses summary if needed.
    """

    if not app.state.tokenizer:
        logger.error("Tokenizer not loaded for pruning.")
        return messages, final_context

    def get_token_count(msg_list: List[Dict]) -> int:
        """Count tokens for all messages using the chat template."""
        prompt_text = app.state.tokenizer.apply_chat_template(
            msg_list, tokenize=False, add_generation_prompt=True
        )
        return len(app.state.tokenizer.encode(prompt_text, add_special_tokens=False))

    current_context = final_context[:]
    current_messages = messages[:]

    token_count = get_token_count(current_messages)
    logger.info(f"Pre-pruning token count: {token_count}. Max: {MAX_INPUT_TOKENS}")

    if token_count <= MAX_INPUT_TOKENS:
        return current_messages, current_context

    if len(summary.split()) > 50:
        logger.warning("Prompt too large — compressing conversation summary.")

        compression_prompt = f"""
        The following conversation summary is too long for the LLM's context window.
        Rewrite it to be half its length, retaining only the essential topics.
        Do not add preamble or commentary.
        
        Current summary:
        {summary}
        """

        start = time.time()
        new_summary_text = await asyncio.to_thread(
            call_llm_pipeline,
            compression_prompt,
            deterministic=True,
            is_expansion=False,
            max_new_tokens=50,
        )
        end = time.time()

        logger.info(f"Summary compressed in {end - start:.2f}s. New summary: {new_summary_text}")

        summary = new_summary_text.strip()
        from copy import deepcopy
        base_user_query = deepcopy(current_messages[-1]["content"])
        current_messages = build_prompt(base_user_query, current_context, summary)

        token_count = get_token_count(current_messages)
        logger.info(f"Token count after summary compression: {token_count}")

        if token_count <= MAX_INPUT_TOKENS:
            return current_messages, current_context

    while token_count > MAX_INPUT_TOKENS and len(current_context) > 1:
        removed_chunk = current_context.pop()
        logger.warning(f"Context overflow. Removing last chunk: {removed_chunk['text'][:50]}...")

        from copy import deepcopy
        base_user_query = deepcopy(current_messages[-1]["content"])
        current_messages = build_prompt(base_user_query, current_context, summary)

        token_count = get_token_count(current_messages)
        logger.info(f"Token count after removing a chunk: {token_count}")

    if token_count > MAX_INPUT_TOKENS:
        logger.error(f"FATAL: Prompt still exceeds limit! Final count: {token_count}")

    return current_messages, current_context



@app.get("/health")
async def health_check():
    """Endpoint for checking the status of the RAG service."""
    if app.state.llm_loaded and app.state.chroma_collection:
        return {"status": "ok", "service": "rag-service"}
    else:
        raise HTTPException(status_code=503, detail="RAG service is initializing or failed to load critical components (LLM/Vector DB).")


@app.post("/rag", response_model=RAGResponse)
async def rag_handler(request: QueryRequest):
    """
    Main RAG endpoint: performs query expansion, retrieval, and generation.
    """
    start = time.time()
    if not app.state.llm_loaded or not app.state.chroma_collection:
        raise HTTPException(status_code=503, detail="Service is initializing or failed to load LLM/Vector DB.")
        
    try:
        GREETINGS = ['hi', 'hello', 'hey', 'greetings', 'thanks', 'thank you', 'tysm']
        query_lower = request.query.lower().strip().replace('!', '').replace('.', '').replace(',', '')
        
        if query_lower in GREETINGS or not query_lower:
            final_answer = "Hello! I'm here to help you with health information. What is your question?"
            
            logger.info(
                json.dumps({
                    "query": request.query,
                    "latency_sec": round(time.time() - start, 2),
                    "result": "Trivial Query Filtered"
                })
            )
            return RAGResponse(
                query=request.query,
                answer=final_answer,
                sources=[], 
                context_chunks=[],
                expanded_queries=[]
            )
            
        expanded_queries = await expand_query_with_llm(request.query, request.history)
        
        context_data, all_sources = await asyncio.to_thread(
            retrieve_context, expanded_queries, app.state.chroma_collection
        )
        
        if cross_encoder: 
            final_context = rerank_documents(request.query, context_data, top_k=3)
            final_sources = list({c['url'] for c in final_context if c.get('url')})
        else:
            final_context = context_data[:3] 
            final_sources = all_sources

        if not final_context:
            final_answer = "I could not find relevant documents in the knowledge \
            base to answer your question. I can help you if you have another question."
            context_chunks_text = [] 
        else:
            summary = await summarize_history( request.history)
            
            initial_messages = build_prompt(request.query, final_context, summary)
            
            final_messages, final_context_pruned = await prune_messages_to_fit_context(
                initial_messages, 
                final_context,
                summary
            )
            
            context_chunks_text = [c['text'] for c in final_context_pruned]
            
            prompt_text = app.state.llm_pipeline.tokenizer.apply_chat_template(
                final_messages, tokenize=False, add_generation_prompt=True
            )
            
            final_answer = await asyncio.to_thread(
                call_llm_pipeline, prompt_text, deterministic=False
            )

        logger.info(
            json.dumps({
                "query": request.query,
                "latency_sec": round(time.time() - start, 2),
                "num_sources": len(final_sources),
                "num_context_chunks_sent": len(context_chunks_text),
                "expanded_queries": expanded_queries,
                "final answer: ": final_answer,
                "retrieved: ": context_chunks_text
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