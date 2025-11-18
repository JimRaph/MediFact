
import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { DashboardHeader } from '@/components/layout/DashboardHeaders'
import {checkUser} from '@/lib/db'


interface DashboardLayoutProps {
  children: React.ReactNode
}

export default async function DashboardLayout({ children }: DashboardLayoutProps) {

  const session = await getServerSession(authOptions)


  if (!session) {
    redirect('/login')
  }

  const userId = session.user.id;
  const userExists = await checkUser(userId);

  return (
    <div className="flex flex-col max-h-screen overflow-hidden">
      <DashboardHeader user={session.user} />
      <div className="flex flex-1">
        {children}
      </div>
    </div>
  )
}