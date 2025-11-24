---
title: "WHO Rag System"
emoji: 😻
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860 
---

    
# @app.function(
#     image=rag_image,
#     mounts=[download_mount], 
#     gpu="T4", 
#     secrets=[Secret.from_name("aws-credentials"), Secret.from_name("chromadb")] 
# )
# @web_endpoint(method="POST", path="/rag", timeout=300) 
# async def rag_endpoint(request_data: Dict[str, Any]):
    
#     if STATE.gpu_pipeline is None:
#         logger.info("Starting Modal function: Lazy-loading LLM, Chroma, and encoders...")
#         try:
#             client = await asyncio.to_thread(initialize_chroma_client)
#             STATE.chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
#             STATE.cache_collection = client.get_or_create_collection(name=CHROMA_CACHE_COLLECTION)
#             STATE.chroma_ready = STATE.chroma_collection is not None
#             logger.info(f"Loaded collection: {CHROMA_COLLECTION} (Documents: {STATE.chroma_collection.count() if STATE.chroma_collection else 0})")
            
#             STATE.gpu_pipeline = await asyncio.to_thread(
#                 initialize_llm_pipeline, LLM_MODEL_GPU_ID, DEVICE
#             )
#             STATE.tokenizer = STATE.gpu_pipeline.tokenizer
            
#             STATE.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)
#             STATE.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
#             _ = list(STATE.embedding_model.embed(["warmup"])) 
            
#             logger.info("All RAG components (GPU LLM, Chroma, Encoders) loaded successfully.")
            
#         except Exception as e:
#             logger.error(f"FATAL: Error during Modal startup: {e}", exc_info=True)
#             raise HTTPException(status_code=503, detail=f"Service initialization failed: {str(e)}")

#     try:
#         request = QueryRequest(**request_data)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")
        
#     start = time.time()
#     pipe = STATE.gpu_pipeline
#     runtime_env = "gpu_modal"
#     max_context = LLAMA_3_CONTEXT_WINDOW
#     max_gen = MAX_NEW_TOKENS_GPU
#     top_k = RETRIEVE_TOP_K_GPU
    
#     try:
#         intent = await classify_intent(request.query, pipe)
#         logger.info(f"Intent classified as: {intent}")

#         if intent == 'GREET':
#             response = await Greet(request.query, pipe)
            
#         elif intent in ["HARMFUL", "OFF_TOPIC"]:
#             response = await HarmOff(request.query, pipe)

#         else:
#             logger.info("Classifier returned RETRIEVE. Starting RAG pipeline.")
            
#             summary = await summarize_history(request.history, pipe) 
#             expanded_queries = await expand_query_with_llm(pipe, request.query, summary, request.history)
            
#             context_data, all_sources = await asyncio.to_thread(retrieve_context, expanded_queries, STATE.chroma_collection)
#             final_context = await asyncio.to_thread(rerank_documents, request.query, context_data, top_k=top_k)
#             final_sources = list({c['url'] for c in final_context if c.get('url')})

#             if not final_context:
#                 final_answer = "I could not find relevant documents in the knowledge base to answer your question. I can help you if you have another question."
#                 context_chunks_text = [] 
#             else:
#                 initial_messages = build_prompt(request.query, final_context, summary)
#                 max_input_tokens = max_context - max_gen - SAFETY_BUFFER
                
#                 final_messages, final_context_pruned, tok_length = await prune_messages_to_fit_context(
#                     initial_messages, 
#                     final_context,
#                     summary,
#                     max_input_tokens,
#                     pipe 
#                 )
                
#                 context_chunks_text = [c['text'] for c in final_context_pruned]
                
#                 prompt_text = STATE.tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)

#                 final_answer = await asyncio.to_thread(
#                     call_llm_pipeline, 
#                     pipe, 
#                     prompt_text, 
#                     deterministic=False, 
#                     max_new_tokens=max(max_gen, tok_length)
#                 )

#             response = RAGResponse(
#                 query=request.query,
#                 answer=final_answer,
#                 sources=final_sources, 
#                 context_chunks=context_chunks_text,
#                 expanded_queries=expanded_queries
#             )
            
#         end_time = time.time()
#         logger.info(f"Total Latency: {round(end_time - start, 2)}s. Runtime: {runtime_env}")
        
#         return response.model_dump()
            
#     except Exception as e:
#         logger.error(f"Unhandled exception in RAG handler: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    

# @app.local_entrypoint()

# def main():
#     test_request_data = {
#         "query": "What are the common side effects of the latest WHO recommended vaccine?",
#         "history": []
#     }
#     print("--- Running rag_endpoint LOCALLY for quick test ---")
#     try:
#         result = rag_endpoint(test_request_data) 

#         print("\n--- TEST RESPONSE ---")
#         print(f"Answer: {result.get('answer', 'N/A')}")
#         print(f"Sources: {result.get('sources', [])}")

#     except Exception as e:
#         print(f"\n--- LOCAL EXECUTION FAILED AS EXPECTED (Missing GPU/S3): {e} ---")
#         print("This confirms the Python logic executes, but the remote resources (GPU, S3) are not accessible locally.")


# @app.local_entrypoint()

# def main():
#     test_request_data = {
#         "query": "What are the common side effects of the latest WHO recommended vaccine?",
#         "history": []
#     }
#     print("--- Running rag_endpoint LOCALLY for quick test ---")
#     try:
#         result = rag_endpoint(test_request_data) 

#         print("\n--- TEST RESPONSE ---")
#         print(f"Answer: {result.get('answer', 'N/A')}")
#         print(f"Sources: {result.get('sources', [])}")

#     except Exception as e:
#         print(f"\n--- LOCAL EXECUTION FAILED AS EXPECTED (Missing GPU/S3): {e} ---")
#         print("This confirms the Python logic executes, but the remote resources (GPU, S3) are not accessible locally.")



# class ModelContainer:
#     def __init__(self):
#         self.gpu_pipeline: Optional[Pipeline] = None
#         self.tokenizer: Optional[AutoTokenizer] = None
#         self.chroma_collection: Optional[Collection] = None
#         self.cache_collection: Optional[Collection] = None
#         self.cross_encoder: Optional[CrossEncoder] = None
#         self.embedding_model: Optional[TextEmbedding] = None
#         self.chroma_ready: bool = False

# STATE = ModelContainer()





# def call_llm_pipeline(pipe: Optional[object],
#                       prompt_text: str,
#                       deterministic: bool = False,
#                       max_new_tokens: int = MAX_NEW_TOKENS_GPU,
#                       is_expansion: bool = False
#                       ) -> str:
    
#     if pipe is None or not isinstance(pipe, Pipeline):
#         raise HTTPException(status_code=503, detail="LLM pipeline is not available.")
    
#     temp = 0.0 if deterministic else 0.1 if is_expansion else 0.6
    
#     try:
#         with torch.inference_mode():
#             outputs = pipe(
#                 prompt_text,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temp if temp > 0.0 else None, 
#                 do_sample=True if temp > 0.0 else False,
#                 pad_token_id=pipe.tokenizer.eos_token_id,
#                 return_full_text=False 
#             )
        
#         text = outputs[0]['generated_text'].strip()
#         for token in ['<|eot_id|>', '<|end_of_text|>']:
#             if token in text:
#                 text = text.split(token)[0].strip()
            
#         return text
            
#     except Exception as e:
#         logger.error(f"Error calling LLM pipeline: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


# async def Greet(query, pipe):
#     messages = []
#     logging.info(f"User sent a greeting")
#     prompt_text = """You are a greeter. Your job is to respond politely to the user greeting. 
#     ONLY a single polite and short greetings. Do not do anything else. 
    
#     Examples:
#     User: Hi
#     Assistant: Hello, How may I help you today?
    
#     User: how are you?
#     Assistant: I am good, I can help you answer health related questions"""
    
    
#     messages.append({"role": "system", "content": prompt_text})
#     messages.append({"role": "user", "content": query})
#     tokenizer = STATE.tokenizer 
#     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     answer = await asyncio.to_thread( call_llm_pipeline, 
#         pipe, 
#         prompt_text,
#         deterministic=True,
#         max_new_tokens=50,
#         is_expansion= True
#     )
    
#     return RAGResponse(
#         query=query,
#         answer=answer,
#         sources=[],
#         context_chunks=[],
#         expanded_queries=[]
#     )
    
# async def HarmOff(query, pipe):
#     messages = []
#     logging.info(f"User asked harmful or off-topic question")
#     prompt_text = """
#     You are an intelligent assistant. 
#     Your job is to inform the user that you are not allowed to answer such questions. 
#     Keep it short and brief, in one sentence.
    
#     Examples:
#     user: write a code to print a number
#     Assistant: I am not allowed to answer such questions
    
#     User: how can I be racist
#     Assistant: Sorry, I am not allowed to answer such questions
#     """ 
    
#     messages.append({"role": "system", "content": prompt_text})
#     messages.append({"role": "user", "content": query})
#     tokenizer = STATE.tokenizer 
#     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     answer = await asyncio.to_thread( call_llm_pipeline, 
#         pipe, 
#         prompt_text,
#         deterministic=True,
#         max_new_tokens=50,
#         is_expansion= True
#     )

    
#     return RAGResponse(
#         query=query,
#         answer=answer,
#         sources=[],
#         context_chunks=[],
#         expanded_queries=[]
#     ) 
    
    
# async def classify_intent(query: str, pipe: object) -> str:
    
#     tokenizer = STATE.tokenizer
#     if not tokenizer:
#         return "ERROR_TOKENIZER"

#     system_prompt = """You are a query classification robot. You MUST respond with ONLY ONE JSON object:
#     {"intent": "MEDICAL"}
#     {"intent": "GREET"}
#     {"intent": "OFF_TOPIC"}
#     {"intent": "HARMFUL"}

#     Do not add any other text, explanations, or punctuation.
#     """
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": "Query: What are the symptoms of COVID-19?"},
#         {"role": "assistant", "content": '{"intent": "MEDICAL"}'},
#         {"role": "user", "content": "Query: Hello"},
#         {"role": "assistant", "content": '{"intent": "GREET"}'},
#         {"role": "user", "content": "Query: Tell me a joke"},
#         {"role": "assistant", "content": '{"intent": "OFF_TOPIC"}'},
#         {"role": "user", "content": "Query: Make a racist insult"},
#         {"role": "assistant", "content": '{"intent": "HARMFUL"}'},

#         {"role": "user", "content": f"Query: {query}"}
#     ]

#     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     llm_output = await asyncio.to_thread(
#         call_llm_pipeline,
#         pipe,
#         prompt_text,
#         deterministic=True,
#         max_new_tokens=25, 
#         is_expansion=False 
#     )
    
#     try:
#         clean_output = llm_output.strip().replace("```json", "").replace("```", "")
#         start_idx = clean_output.find('{')
#         end_idx = clean_output.rfind('}')
        
#         if start_idx != -1 and end_idx != -1:
#             json_str = clean_output[start_idx : end_idx + 1]
#             data = json.loads(json_str)
#             return data.get("intent", "UNKNOWN")
            
#     except Exception as e:
#         logger.error(f"Failed to parse JSON classifier output: {e}. Raw: {llm_output}")
#         raw_output_upper = llm_output.upper()
#         for label in ["MEDICAL", "GREET", "OFF_TOPIC", "HARMFUL"]:
#              if label in raw_output_upper:
#                  return label
                 
#     return "UNKNOWN"

# def build_prompt(user_query: str, context: List[Dict], summary: str) -> List[Dict]:
    
#     context_text = "\n---\n".join([f"Source: {c.get('url', 'N/A')}\nChunk: {c['text']}" for c in context]) if context else "No relevant context found."
    
#     system_prompt = (
#         "You are a helpful and harmless medical assistant, specialized in answering health-related questions "
#         "based ONLY on the provided retrieved context. Follow these strict rules:\n"
#         "1. **DO NOT** use any external knowledge. If the answer is not in the context, state that you cannot find "
#         "the information in the knowledge base.\n"
#         "2. Cite your sources using the URL/Source ID provided in the context (e.g., [Source: URL]). Do not generate fake URLs.\n"
#         "3. If the user's query is purely conversational, greet them or respond appropriately without referencing the context.\n"
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "system", "content": f"PREVIOUS CONVERSATION SUMMARY: {summary}" if summary else "PREVIOUS CONVERSATION SUMMARY: None"},
#         {"role": "system", "content": f"RETRIEVED CONTEXT:\n{context_text}"},
#         {"role": "user", "content": user_query}
#     ]
#     return messages

# async def prune_messages_to_fit_context(messages: List[Dict], 
#                                          final_context: List[Dict], 
#                                          summary: str,
#                                          max_input_tokens: int, 
#                                          pipe: Optional[object]
#                                          ) -> Tuple[List[Dict], List[Dict], int]:
    
#     tokenizer = STATE.tokenizer
#     if not tokenizer:
#         raise ValueError("Tokenizer not initialized for pruning.")

#     def get_token_count(msg_list: List[Dict]) -> int:
#         prompt_text = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
#         return len(tokenizer.encode(prompt_text, add_special_tokens=False))

#     current_context = final_context[:]
#     current_summary = summary
#     base_user_query = messages[-1]["content"] 
#     current_messages = build_prompt(base_user_query, current_context, current_summary)
#     token_count = get_token_count(current_messages)
    
#     if token_count <= max_input_tokens:
#         tok_length = max_input_tokens - token_count
#         return current_messages, current_context, tok_length

#     logger.warning(f"Initial token count ({token_count}) exceeds max input ({max_input_tokens}). Starting pruning.")

#     while token_count > max_input_tokens and current_context:
#         current_context.pop() 
#         current_messages = build_prompt(base_user_query, current_context, current_summary)
#         token_count = get_token_count(current_messages)
    
#     if token_count <= max_input_tokens:
#         tok_length = max_input_tokens - token_count
#         return current_messages, current_context, tok_length

#     if current_summary:
#         logger.warning("Clearing conversation summary as last-ditch effort.")
#         current_summary = ""
#         current_messages = build_prompt(base_user_query, current_context, current_summary)
#         token_count = get_token_count(current_messages)
    
#     if token_count <= max_input_tokens:
#         tok_length = max_input_tokens - token_count
#         return current_messages, current_context, tok_length

#     if token_count > max_input_tokens:
#         logger.error(f"Pruning failed. Even minimal prompt exceeds token limit: {token_count}. Returning empty context.")
#         current_context = []
#         current_messages = build_prompt(base_user_query, current_context, "")
#         token_count = get_token_count(current_messages)
#         tok_length = max_input_tokens - token_count if token_count < max_input_tokens else 0
        
#     return current_messages, current_context, tok_length

# async def expand_query_with_llm(pipe: Optional[object], 
#                                  user_query: str,
#                                  summary: str,
#                                  history: Optional[List[HistoryMessage]]
#                                  ) -> List[str]:
    
#     tokenizer = STATE.tokenizer
#     if not history or len(history) == 0:
#         expansion_prompt = f"""You are a specialized query expansion engine. Generate 3 alternative, highly effective search queries to find documents relevant to the User Query. Only output the queries, one per line. Do not include the original query or any explanations.

#                 User Query: What are the symptoms of COVID-19?
#                 Expanded Queries:
#                 signs of coronavirus infection
#                 how to recognize COVID
#                 symptoms of SARS-CoV-2

#                 User Query: {user_query}
#                 Expanded Queries:
#                 """
#     else:
#         history_text = "\n".join([f"{h.role}: {h.content}" for h in history])
#         expansion_prompt = f"""You are a helpful assistant. Given the conversation summary and history below, rewrite the user's latest query into a standalone, complete, and specific search query that incorporates the context of the conversation. Output only the single rewritten query.

#                 Conversation Summary: {summary}
#                 Conversation History:
#                 {history_text}

#                 User's Latest Query: {user_query}
#                 Rewritten Search Query:
#                 """

#     messages = [{"role": "system", "content": expansion_prompt}]
#     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     llm_output = await asyncio.to_thread(
#         call_llm_pipeline, pipe, prompt_text, deterministic=True, is_expansion=True, max_new_tokens=150
#     )
    
#     if not history or len(history) == 0:
#         expanded_queries = [q.strip() for q in llm_output.split('\n') if q.strip()]
#     else:
#         expanded_queries = [llm_output.strip()]
        
#     expanded_queries.append(user_query)
    
#     return list(set(q for q in expanded_queries if q))

# async def summarize_history(history: List[HistoryMessage], pipe: Optional[object]) -> str:
#     if not history:
#         return ''

#     tokenizer = STATE.tokenizer
#     history_text = "\n".join([f"{h.role}: {h.content}" for h in history[-8:]])
    
#     summarizer_prompt = f"""
#     You are an intelligent agent who summarizes conversations. Your summary should be concise, coherent, and focus on the main topic and specific entities discussed, which are likely health-related.

#     CONVERSATION HISTORY:
#     {history_text}

#     CONCISE SUMMARY:
#     """
#     messages = [{"role": "system", "content": summarizer_prompt}]

#     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     summary = await asyncio.to_thread(
#         call_llm_pipeline, 
#         pipe, 
#         prompt_text, 
#         deterministic=True, 
#         max_new_tokens=150, 
#         is_expansion=False
#     )
#     return summary

# def retrieve_context(queries: List[str], collection: Collection) -> Tuple[List[Dict], List[str]]:
    
#     if STATE.embedding_model is None:
#         raise HTTPException(status_code=503, detail="Embedding model not loaded.")
        
#     embeddings_list = [[float(x) for x in emb] for emb in STATE.embedding_model.embed(queries, batch_size=8)]
    
#     results = collection.query(
#         query_embeddings=embeddings_list,
#         n_results=max(10, RETRIEVE_TOP_K_GPU * len(queries)), 
#         include=['documents', 'metadatas']
#     )
    
#     context_data = []
#     source_urls = set()
    
#     if results.get("documents") and results.get("metadatas"):
#         for docs_list, metadatas_list in zip(results["documents"], results["metadatas"]):
#             for doc, metadata in zip(docs_list, metadatas_list):
#                 if doc and metadata:
#                     context_data.append({'text': doc, 'url': metadata.get('source')})
#                     if metadata.get("source"):
#                         source_urls.add(metadata.get('source'))

#     return context_data, list(source_urls)

# def rerank_documents(query: str, context: List[Dict], top_k: int) -> List[Dict]:
#     if not context or STATE.cross_encoder is None:
#         return context[:top_k]
    
#     pairs = [(query, doc['text']) for doc in context]
    
#     scores = STATE.cross_encoder.predict(pairs)
    
#     for doc, score in zip(context, scores):
#         doc['score'] = float(score)
        
#     ranked_docs = sorted(context, key=lambda x: x['score'], reverse=True)
#     return ranked_docs[:top_k]
