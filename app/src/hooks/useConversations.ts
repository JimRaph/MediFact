'use client'

import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Conversation } from '@/types/chat' 

interface UseConversationsReturn {
    conversations: Conversation[]
    currentConversation: Conversation | null
    selectConversation: (conversation: Conversation | null) => void
    createNewConversation: () => void
    isLoadingConversations: boolean
    isCreatingConversation: boolean
    deleteConversation: (conversationId: string) => void 
    isDeletingConversation: boolean
    conversationDeleteError: string | null
    responseMsg: string | null,
    clearResponseMessages: () => void 
}



export function useConversations(userId: string): UseConversationsReturn {
 const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
 const [conversationDeleteError, setConversationDeleteError] = useState<string | null>(null);
 const [responseMsg, setResponseMsg] = useState<string | null>(null);
 const queryClient = useQueryClient()


 const { data: conversations = [], isLoading: isLoadingConversations } = useQuery({
    queryKey: ['conversations', userId],
    queryFn: async (): Promise<Conversation[]> => {
    const response = await fetch('/api/conversations')
    if (!response.ok) {
        throw new Error('Failed to fetch conversations')
    }
    return response.json()
},

    staleTime: 5 * 60 * 1000, 
 })

 const deleteConversationMutation = useMutation({
    mutationFn: async (conversationId: string) => {
        const response = await fetch(`/api/conversations/${conversationId}`,{
            method: 'DELETE',
        })
        if (!response.ok){
            throw new Error('Failed to delete conversation')
        }

        const responseMessage = await response.text()
        return responseMessage
    },

    onSuccess: (responseMessage, conversationId) => {
        setResponseMsg(responseMessage)
        queryClient.invalidateQueries({queryKey: ['conversations', userId]})
        if (currentConversation?.id === conversationId) {
            setCurrentConversation(null)
        }
        queryClient.removeQueries({queryKey: ['messages', conversationId]})
    },

    onError: (error) => {
        setConversationDeleteError('Error deleting conversation, try again later')
        console.error('Error deleting conversation: ', error)
    }
 })


 const selectConversation = useCallback((conversation: Conversation | null) => {
    setCurrentConversation(conversation)
 }, [])

 const createNewConversation = useCallback(() => {
    setCurrentConversation(null); 
 }, [])

 const deleteConversation = useCallback((conversationId: string) => {
    deleteConversationMutation.mutate(conversationId)
 }, [deleteConversationMutation])

 const clearResponseMessages = () => {
    setResponseMsg(null);
    setConversationDeleteError(null);
};

 return {
  conversations,
  currentConversation,
  selectConversation,
  createNewConversation,
  isLoadingConversations,
  isCreatingConversation: false,
  deleteConversation,
  isDeletingConversation: deleteConversationMutation.isPending,
  conversationDeleteError,
  responseMsg,
  clearResponseMessages
 }
}