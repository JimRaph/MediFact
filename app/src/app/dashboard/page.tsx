import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { redirect } from 'next/navigation'
import { ChatInterface } from '@/components/chat/ChatInterface'


export default async function Dashboard() {
  const session = await getServerSession(authOptions)
  
  if (!session) {
    redirect('/login'); 
  }

  return (
    <div className='w-screen'>
         <ChatInterface user = {session?.user} />  
    </div>
  )
}