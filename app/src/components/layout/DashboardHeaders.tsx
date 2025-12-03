'use client'

import { signOut } from 'next-auth/react'
import { User } from 'next-auth'
import { Bars3Icon } from '@heroicons/react/24/solid'
// import { useState } from 'react'
import { useComp } from '@/stores/compStore'

interface DashboardHeaderProps {
  user: User
}

export function DashboardHeader({ user }: DashboardHeaderProps) {

  const setSidebarOpen = useComp((state) => state.setSidebarOpen)

  return (
    <header className="bg-white shadow-sm border-b border-gray-200 ">
      <div className=" mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="flex justify-between h-13">

              
          
          <div className="flex items-center">
            <div  className='lg:hidden p-2 '>
                <button className="p-2 rounded-md text-gray-700 hover: bg-gray-200 " 
                  onClick={() => setSidebarOpen(true)}
                  aria-label = "Open sidebar"
                >
                  <Bars3Icon className="w-6 h-6" />
                </button>
              </div>
            <h1 className="text-md font-semibold text-gray-900">MediFact</h1>
          </div>
         
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-700 flex items-center ">Welcome
              <span 
              className='rounded-full bg-gray-300 text-gray-700 items-center justify-center flex w-6 h-6 ml-1'>
                {(user.name || user.email)?.charAt(0) }
              </span>
            </span>
            <span>|</span>
            <button
              onClick={() => signOut({ callbackUrl: '/login' })}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Sign out
            </button>
          </div>

        </div>
        
      </div>
      
    </header>
  )
}