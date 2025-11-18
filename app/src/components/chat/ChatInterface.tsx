'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import { useChat } from '@/hooks/useChat'
import { useConversations } from '@/hooks/useConversations'
import { ChatInput } from './ChatInput'
import { MessageBubble } from './MessageBubble'
import { ConversationSidebar } from './ConversationSidebar'
import { ChatInterfaceProps } from '@/types/user'
import { toast } from 'react-toastify'
// import { Bars3Icon, H3Icon, XMarkIcon } from '@heroicons/react/24/solid'
import { Conversation } from '@/types/chat' 
// import { useAuthStore } from '@/stores/authStore'
import { useComp } from '@/stores/compStore'


export function ChatInterface({user}: ChatInterfaceProps) {

   console.log('session1: ', user)
  const setSidebarOpen = useComp((state) => state.setSidebarOpen)
  const sidebarOpen = useComp((state) => state.sidebarOpen)
  
  
  const { conversations,currentConversation,selectConversation,createNewConversation,
    isCreatingConversation, deleteConversation,isDeletingConversation,
    conversationDeleteError,responseMsg,clearResponseMessages} = useConversations(user.id)
  
  const handleNewConversationCreated = useCallback((newId: string) => {
    // const newConversation = conversations.find(c => c.id === newId);
    // if (newConversation) {
    //     selectConversation(newConversation); 
    // }
    selectConversation({id: newId} as Conversation)
    
}, [conversations, selectConversation]);

  const handleSelectConversation = useCallback((conv:Conversation ) => {
    selectConversation(conv)
    if(sidebarOpen){
      setSidebarOpen(false)
    }
  }, [selectConversation, sidebarOpen])

  const {
    messages,
    input,
    setInput,
    handleSubmit,
    isSending, 
    isLoadingHistory,
    cancelSending,
  } = useChat({
    currentConversation,
    userId: user.id,
    onNewConversationCreated:handleNewConversationCreated
  })

  const isLoading = isSending || isLoadingHistory || isCreatingConversation;

  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(()=>{
    if(responseMsg){
      toast(responseMsg)
      clearResponseMessages()
    }
    if(conversationDeleteError){
      toast(conversationDeleteError)
      clearResponseMessages()
    }
  }, [responseMsg, conversationDeleteError,clearResponseMessages])

  useEffect(() => {
    if (isDeletingConversation || isCreatingConversation) {
      cancelSending()
    }
  }, [isDeletingConversation, isCreatingConversation])

  return (

    <div className="flex h-[calc(100vh-4rem)] bg-gray-50 relative overflow-hidden md:overflow-visible ">


     <div className={`
            fixed inset-y-0 left-0 z-20 w-80 bg-gray-50 border-r border-gray-200 
            flex-col shrink-0 transition-transform duration-300 ease-in-out
            ${sidebarOpen ? 'translate-x-0 shadow-xl' : '-translate-x-full shadow-none'}
            lg:static lg:flex lg:translate-x-0
        `}
    > 
    

      <ConversationSidebar
        conversations={conversations}
        currentConversation={currentConversation}
        onSelectConversation={handleSelectConversation} 
        onCreateNewConversation={createNewConversation}
        isCreatingConversation={isCreatingConversation}
        onDeleteConversation={deleteConversation}
        isDeletingConversation={isDeletingConversation}
        userId = {user.id}
        setInput = {setInput}
      /> 

      

    </div> 

    {sidebarOpen && (
        <div 
            className="fixed inset-0 bg-black opacity-30 z-10 lg:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-hidden="true"
        />
    )}
      
      <div className="flex-1 flex flex-col relative max-w-2xl mx-auto border-gray-150 border-r border-l shadow shadow-gray-300">


        <div className="flex-1  p-4 overflow-auto space-y-4 scrollbar-thin">
          {messages.length === 0 && !isLoadingHistory && !isSending && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <h3 className="text-lg font-medium">Welcome to Health Info Hub</h3>
                <p className="mt-2">Ask a health-related question to get started.</p>
              </div>
            </div>
          )}
          

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}


          {isSending && (
            <div className="flex justify-start">
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          )}
          

          {isLoadingHistory && (
             <div className="text-center text-gray-500 p-4">Loading conversation history...</div>
          )}

          <div ref={messagesEndRef} />

          
        </div>
        

        <ChatInput 
          input={input}
          setInput={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading} 
        />
      </div>

    </div>


 )
}