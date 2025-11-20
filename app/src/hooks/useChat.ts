'use client'

import { useState, useCallback, useEffect } from 'react'
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
  cancelSending: () => void
  pendingConversationId: string | null
}

export function useChat({ currentConversation, userId, onNewConversationCreated }: UseChatProps): UseChatReturn {
  const [input, setInput] = useState('')
  const [tempMessages, setTempMessages] = useState<Message[]>([])
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null)

  const queryClient = useQueryClient()

  if (!(globalThis as any).__CHAT_CONTROLLERS__) {
    ;(globalThis as any).__CHAT_CONTROLLERS__ = new Map<string, AbortController>()
  }

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
    initialData: [],
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
  })

  const sendMessageMutation = useMutation({
    
    mutationFn: async (message: string): Promise<ChatResponse> => {
      const ctrl = new AbortController()
      const map: Map<string, AbortController> = (globalThis as any).__CHAT_CONTROLLERS__
      const mapKey = currentConversation?.id || 'temp'
      map.set(mapKey, ctrl)

      const requestBody: ChatRequest = {
        message,
        conversationId: currentConversation?.id,
        history: messages,
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: ctrl.signal,
      })

      map.delete(mapKey)

      if (!response.ok) throw new Error('Failed to send message')
      return response.json()
    },

    onMutate: async (message) => {
      const newMessage: Message = {
        id: crypto.randomUUID(),
        content: message,
        role: 'user',
        createdAt: new Date().toISOString(),
      }

      const convKey = currentConversation?.id ?? 'temp'
      setPendingConversationId(convKey)

      if (!currentConversation?.id) {
        setTempMessages((prev) => [...prev, newMessage])
      } else {
        queryClient.setQueryData<Message[]>(
          ['messages', currentConversation.id],
          (old = []) => [...old, newMessage]
        )
      }

      setInput('')
      return { convKey }
    },

    onSuccess: async (data, _variables, context) => {
      const convId = data.conversationId || currentConversation?.id 
      if (!convId) {
        setPendingConversationId(null)
        return
      }

      if (data.conversationId && !currentConversation) {
        onNewConversationCreated?.(data.conversationId)
        queryClient.setQueryData<Message[]>(['messages', data.conversationId], (old = []) => [
          ...tempMessages,
          ...old,
        ])
        setTempMessages([])
        queryClient.invalidateQueries({ queryKey: ['conversations', userId] })
      }

      if (data.response) {
        const assistant: Message = {
          id: crypto.randomUUID(),
          content: data.response,
          role: 'assistant',
          createdAt: new Date().toISOString(),
        }
        queryClient.setQueryData<Message[]>(['messages', convId], (old = []) => [...old, assistant])
      }

      setPendingConversationId(null)
      const map: Map<string, AbortController> = (globalThis as any).__CHAT_CONTROLLERS__
      map.delete(convId)
    },

    
    onError: (err, _variables, context) => {
      console.error('Message Send Error:', err)
      const conv = currentConversation?.id ?? (context as any)?.convKey ?? null
      if (conv && conv !== 'temp') {
        queryClient.invalidateQueries({ queryKey: ['messages', conv] })
      } else {
        setTempMessages([])
      }
      setPendingConversationId(null)
      const map: Map<string, AbortController> = (globalThis as any).__CHAT_CONTROLLERS__
      if ((context as any)?.convKey) map.delete((context as any).convKey)
    },
  })

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (!input.trim() || sendMessageMutation.isPending) return
      sendMessageMutation.mutate(input)
    },
    [input, sendMessageMutation]
  )

  useEffect(() => {
    const handler = () => {
      const map: Map<string, AbortController> = (globalThis as any).__CHAT_CONTROLLERS__
      if (map) {
        for (const [, ctrl] of map) {
          try {
            ctrl.abort()
          } catch (e) {
             console.log('Error occurred: ', e)
          }
        }
        map.clear()
      }
      try {
        sendMessageMutation.reset()
      } catch (e) {
        console.log('Error occurred: ', e)
      }
      setPendingConversationId(null)
      setTempMessages([])
    }
    window.addEventListener('cancel-active-mutation', handler)
    return () => window.removeEventListener('cancel-active-mutation', handler)
  }, [sendMessageMutation])

  const allMessages = currentConversation?.id ? messages : tempMessages

  return {
    messages: allMessages,
    input,
    setInput,
    handleSubmit,
    isSending: sendMessageMutation.isPending,
    isLoadingHistory,
    cancelSending: sendMessageMutation.reset,
    pendingConversationId,
  }
}
