'use client'

import { useState } from 'react';
import { Message } from '@/types/chat' 
import { formatDate } from '@/lib/utils'
import { UserIcon, ComputerDesktopIcon, DocumentTextIcon, ChevronDownIcon } from '@heroicons/react/24/outline'



interface MessageBubbleProps {
 message: Message & { sources?: string[] } 
}

export function MessageBubble({ message }: MessageBubbleProps) {
 const isUser = message.role === 'user'
 
 console.log('messsage ', message)
 const hasSources = !isUser && message.sources && message.sources.length > 0
 const [isSourcesExpanded, setIsSourcesExpanded] = useState(false)

 return (
  <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start space-x-3`}>

   <div className={`shrink-0 ${isUser ? 'order-2' : 'order-1'}`}>
    {isUser ? (
     <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
      <UserIcon className="w-4 h-4 text-white" />
     </div>
    ) : (
     <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
      <ComputerDesktopIcon className="w-4 h-4 text-white" />
     </div>
    )}
   </div>


   <div className={`flex flex-col ${isUser ? 'order-1 items-end' : 'order-2 items-start'} max-w-[70%]`}>
    

    <div
     className={`rounded-lg px-4 py-2 ${
      isUser
       ? 'bg-blue-600 text-white'
       : 'bg-gray-100 text-gray-900 border border-gray-200'
     }`}
    >
     <div className="whitespace-pre-wrap wrap-break-word">{message.content}</div>
    </div>
    

    {hasSources && (
     <div className="mt-2 w-full max-w-full">
      <button
       onClick={() => setIsSourcesExpanded(!isSourcesExpanded)}
       className="flex items-center text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors focus:outline-none"
      >
       <DocumentTextIcon className="w-3 h-3 mr-1 text-blue-500" />
       {isSourcesExpanded ? 'Hide Citations' : `Show ${message.sources!.length} Source${message.sources!.length > 1 ? 's' : ''}`}
       <ChevronDownIcon 
        className={`w-3 h-3 ml-1 transform transition-transform ${isSourcesExpanded ? 'rotate-180' : 'rotate-0'}`} 
       />
      </button>

      {isSourcesExpanded && (
       <div className="mt-1 space-y-1 p-2 bg-white rounded-lg border border-gray-200 shadow-inner max-h-40 overflow-y-auto">
        {message.sources!.map((url, index) => (
         <a 
          key={index} 
          href={url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="block text-xs text-blue-600 hover:underline truncate transition-colors"
         >

          🔗 {url.replace(/^(https?:\/\/)?(www\.)?/, '').split('/')[0]}... (Source {index + 1})
         </a>
        ))}
       </div>
      )}
     </div>
    )}
    

    <div
     className={`text-xs mt-1 ${
      isUser ? 'text-gray-500' : 'text-gray-400'
     }`}
    >
     {formatDate(message.createdAt)}
    </div>
   </div>
  </div>
 )
}