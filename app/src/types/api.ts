/**
 * API Request/Response Types
 * These ensure type safety when communicating with external APIs
 */

// Request to your Railway inference API
export interface InferenceAPIRequest {
  query: string
  conversation_id?: string
  history?: Array<{
    role: 'user' | 'assistant'
    content: string
  }>
  stream?: boolean
}

// Response from your Railway inference API  
export interface InferenceAPIResponse {
    query: string;
    answer: string;
    sources: string[]; // List of source URLs (strings)
    context_chunks: string[]; // List of context text (strings)
    expanded_queries: string[];
    time?: Date
}

export type RagResponse = {
    answer: string;
    // Client UI requires objects, not just strings, for the 'sources' display
    sources: Array<{ title: string; url: string }>; 
    responseTime?: number;
};

// Error responses
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