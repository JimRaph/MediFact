import {create} from 'zustand' 


interface CompState {
  sidebarOpen: boolean;
  setSidebarOpen: (value: boolean) => void;
}

export const useComp = create<CompState>((set) => (
    {
    sidebarOpen: false,

    setSidebarOpen: (value: boolean) => set((_state) => ({
        sidebarOpen: value
    })),
})

) 

