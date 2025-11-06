'use client'

import { signOut } from 'next-auth/react'
import { User } from 'next-auth'

interface DashboardHeaderProps {
  user: User
}

export function DashboardHeader({ user }: DashboardHeaderProps) {
   console.log('🔍 DashboardHeader - User prop:', user)

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">

      <div className=" mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="flex justify-between h-16">
          
          <div className="flex items-center">
            <h1 className="text-xl font-semibold text-gray-900">HealthInfo</h1>
          </div>
         
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-700">Welcome, {user.name || user.email}</span>
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