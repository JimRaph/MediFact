import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { ChatInterface } from '@/components/chat/ChatInterface'



export default async function DashboardPage() {
  const session = await getServerSession(authOptions)

  if (!session) {
    redirect('/login')
  }

  return (
    <div className="flex-1 ">
      <ChatInterface user={session.user} />
    </div>
  )
}