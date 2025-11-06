import { create } from 'zustand'



interface ChatState {
  isSidebarOpen: boolean
  input: string
  isStreaming: boolean
  
  setSidebarOpen: (open: boolean) => void
  setInput: (input: string) => void
  setStreaming: (streaming: boolean) => void
  resetChatState: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  isSidebarOpen: true,
  input: '',
  isStreaming: false,
  
  setSidebarOpen: (isSidebarOpen) => set({ isSidebarOpen }),
  setInput: (input) => set({ input }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  resetChatState: () => set({ 
    input: '', 
    isStreaming: false 
  }),
}))