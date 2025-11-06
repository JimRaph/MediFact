'use client'

import { useState, useCallback } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Conversation, Message, ChatRequest, ChatResponse } from '@/types/chat'



interface UseChatProps {
  currentConversation: Conversation | null
  userId: string
  onNewConversationCreated?: (newId: string) => void
}

interface UseChatReturn {
  messages: Message[]
  input: string
  setInput: (value: string) => void
  handleSubmit: (e: React.FormEvent) => void
  isSending: boolean
  isLoadingHistory: boolean
}


export function useChat({ currentConversation, userId, onNewConversationCreated }: UseChatProps): UseChatReturn {
  const [input, setInput] = useState('')
  const queryClient = useQueryClient()

  const { data: messages = [], isLoading: isLoadingHistory } = useQuery({
    queryKey: ['messages', currentConversation?.id],
    queryFn: async (): Promise<Message[]> => {
      const response = await fetch(`/api/chat/${currentConversation!.id}`) 
      if (!response.ok) {
        throw new Error('Failed to fetch messages')
      }
      return response.json()
    },
    enabled: !!currentConversation?.id, 
    initialData: []
  })

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string): Promise<ChatResponse> => {
      const requestBody: ChatRequest = {
        message,
        conversationId: currentConversation?.id,
        history: messages,
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        console.log('response: ', response)
        throw new Error('Failed to send message')
      }
      return response.json()
    },
    
    onMutate: async (message) => {
      const newMessage: Message = {
        id: crypto.randomUUID(), 
        content: message,
        role: 'user',
        createdAt: new Date().toISOString(),
      }

      await queryClient.cancelQueries({ queryKey: ['messages', currentConversation?.id] })

      const queryKey = ['messages', currentConversation?.id]
      const previousMessages = queryClient.getQueryData<Message[]>(queryKey)

      queryClient.setQueryData<Message[]>(queryKey, (old) => {
        return [...(old || []), newMessage]
      })

      setInput('') 
      return { previousMessages }
    },
    
    onSuccess: (data) => {
      if (data.conversationId && !currentConversation) {
        onNewConversationCreated?.(data.conversationId);
        queryClient.invalidateQueries({ queryKey: ['conversations', userId] })
      }
      
      const finalConversationId = data.conversationId || currentConversation?.id;
      if (finalConversationId) {
        queryClient.invalidateQueries({ queryKey: ['messages', finalConversationId] })
      }
    },
    
    onError: (err, newMessage, context) => {
      console.error('Message Send Error:', err)
      const queryKey = ['messages', currentConversation?.id]
      queryClient.setQueryData(queryKey, context?.previousMessages)
    },
  })

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || sendMessageMutation.isPending) return
    
    sendMessageMutation.mutate(input)
  }, [input, sendMessageMutation])


  return {
    messages,
    input,
    setInput,
    handleSubmit,
    isSending: sendMessageMutation.isPending,
    isLoadingHistory,
  }
}
