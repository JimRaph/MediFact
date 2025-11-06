import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { DashboardHeader } from '@/components/layout/DashboardHeaders'


interface DashboardLayoutProps {
  children: React.ReactNode
}

export default async function DashboardLayout({ children }: DashboardLayoutProps) {

  const session = await getServerSession(authOptions)

    console.log('🔍 DashboardLayout - Session:', session)
  console.log('🔍 DashboardLayout - User:', session?.user)

  if (!session) {
    redirect('/login')
  }

  return (
    <div className="flex flex-col max-h-screen overflow-hidden">
      <DashboardHeader user={session.user} />
      <div className="flex flex-1">
        {children}
      </div>
    </div>
  )
}