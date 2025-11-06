

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  conversationId?: string  
  createdAt: Date | string
}

export interface Conversation {
  id: string
  title: string
  userId: string
  createdAt: Date
  updatedAt: Date
  messages: Message[]
}

export interface ChatRequest {
  message: string
  conversationId?: string
  history?: Message[]
}

export interface ChatResponse {
  conversationId?: string,
  query: string
  answer: string
  sources: string[]
  context_chunks: string[]
  expanded_queries: string[]
}