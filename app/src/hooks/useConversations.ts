'use client'

import { useState, useCallback, useEffect } from 'react'
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

const storedconversationkey = 'stored:conversation'

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

 useEffect(()=>{
    if(conversations.length > 0) {  
        const storedConversationId = typeof window !== 'undefined' ? localStorage.getItem(storedconversationkey) : null 
        if(!storedConversationId) return 

        if(currentConversation) return 
        console.log('conversations test: ', conversations)
        const isFound = conversations.find((conv) => conv.id === storedConversationId)
        if(isFound){
            console.log('isfound: ', storedConversationId, isFound)
            setCurrentConversation(isFound)
        } else {
            localStorage.removeItem(storedconversationkey)
            console.log('removed: ', storedConversationId)
            console.log('isfound here: ', isFound)
        }
    } 
    
 }, [conversations])


 //Original intention here was to sync conversation across tabs
 // but a user might want to have two separate conversation simultaneouly
 // I feel like this is a better user experience than syncing.
//  useEffect(()=>{
//     if(conversations.length < 0) return
        
//         const onStorage = (e:StorageEvent) => {
//         if (e.key !== storedconversationkey) return 
//         const newId = e.newValue
//         if(!newId){
//             setCurrentConversation(null)
//             return 
//         }

//         const isFound = conversations.find((conv) => conv.id === newId)
//         if(isFound){
//             setCurrentConversation(isFound)
//         } else {

//         }
//     }
    

//     window.addEventListener('storage', onStorage)
//     return () => window.removeEventListener('storage', onStorage)
//  },[conversations])


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
            localStorage.removeItem(storedconversationkey)
        }

        queryClient.removeQueries({queryKey: ['messages', conversationId]})
        window.dispatchEvent(new CustomEvent('cancel-active-mutation'))
    },

    onError: (error) => {
        setConversationDeleteError('Error deleting conversation, try again later')
        console.error('Error deleting conversation: ', error)
    }
 })


 const selectConversation = useCallback((conversation: Conversation | null) => {
    setCurrentConversation(conversation)
    console.log('convo ', conversation)
    if(conversation?.id){
        try{
            localStorage.setItem(storedconversationkey, conversation.id)
        } catch(e) {

        }
    } else {
        try {
            localStorage.removeItem(storedconversationkey)
        } catch(e) {

        }
    }
 }, [])

 const createNewConversation = useCallback(() => {
    setCurrentConversation(null); 

    try {
        localStorage.removeItem(storedconversationkey)
    } catch (e) {
        
    }

    queryClient.removeQueries({queryKey: ['messages']})
    queryClient.cancelQueries({queryKey: ['messages']})
    setResponseMsg(null)
    setConversationDeleteError(null)
 }, [queryClient])

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