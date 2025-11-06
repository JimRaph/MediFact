'use client'

import { useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useAuthStore } from '@/stores/authStore'

export default function AuthSync() {
  const { data: session, status } = useSession()
  const setUser = useAuthStore((s) => s.setUser)
  const setLoading = useAuthStore((s) => s.setLoading)

  useEffect(() => {
    if (status === 'loading') {
      setLoading(true)
      return
    }
    setLoading(false)
    setUser(session?.user ?? null)
  }, [session, status, setUser, setLoading])

  return null
}
