
export interface InferenceAPIRequest {
  query: string
  conversation_id?: string
  history?: Array<{
    role: 'user' | 'assistant'
    content: string
  }>
  stream?: boolean
}

export interface InferenceAPIResponse {
    query: string;
    answer: string;
    sources: string[]; 
    context_chunks: string[]; 
    expanded_queries: string[];
    time?: Date
}

export type RagResponse = {
    answer: string;
    sources: Array<{ title: string; url: string }> | string; 
    responseTime?: number;
};

export interface APIError {
  error: string
  code?: string
  details?: unknown
}

export interface RegisterData {
  name: string
  email:string 
  password: string 
}

export interface LoginData {
  email: string
  password: string
}