'use client'

import { Conversation } from '@/types/chat'
import { PlusIcon } from '@heroicons/react/24/outline'
import { formatDate } from '@/lib/utils'
import { ConversationItem } from '../ui/ConversationItem'

interface ConversationSidebarProps {
  conversations: Conversation[]
  currentConversation: Conversation | null
  onSelectConversation: (conversation: Conversation) => void 
  onCreateNewConversation: () => void
  isCreatingConversation: boolean 
  onDeleteConversation: (conversationId: string) => void 
  isDeletingConversation: boolean
  userId: string
  setInput: (value: string ) => void
}

export function ConversationSidebar({
  conversations,
  currentConversation,
  onSelectConversation,
  onCreateNewConversation,
  isCreatingConversation,
  onDeleteConversation,
  isDeletingConversation,
  userId,
  setInput
}: ConversationSidebarProps) {

  console.log('Conversation: ', conversations)

  return (
    <div className="w-80 bg-gray-50 border-r border-gray-200 flex flex-col h-full">

      <div className="p-4 border-b border-gray-200">
        <button
          onClick={()=>{
            setInput('')
            onCreateNewConversation()
          }
        }
          disabled={isCreatingConversation} 
          className={`flex items-center justify-center w-full px-4 py-2 text-sm font-medium
             text-white rounded-md transition-colors focus:outline-none focus:ring-2 
             focus:ring-offset-2 ${
            isCreatingConversation 
              ? 'bg-blue-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
          }`}
        >
          <PlusIcon className="w-4 h-4 mr-2" />
          {isCreatingConversation ? 'Creating...' : 'New Conversation'}
        </button>
      </div>

 
      <div className="flex-1 overflow-y-scroll scrollbar-thin">
        <div className="p-2 space-y-1">
          {conversations.map((conversation) => (
            <ConversationItem 
              key={conversation.id}
              conversation={conversation}
              isSelected={currentConversation?.id === conversation.id}
              onSelect={onSelectConversation}
              onDelete={onDeleteConversation}
              isDisabled={isCreatingConversation || isDeletingConversation}
            />
          ))}
        </div>
      </div>
    </div>
  )

}