import {create} from 'zustand' 


interface CompState {
  sidebarOpen: boolean;
  setSidebarOpen: (value: boolean) => void;
}

export const useComp = create<CompState>((set) => (
    {
    sidebarOpen: false,

    setSidebarOpen: (value: boolean) => set((state) => ({
        sidebarOpen: value
    })),
})

) 

