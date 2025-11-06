import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { redirect } from 'next/navigation'
// import {ToastContainer} from 'react-toastify'
// import 'react-toastify/dist/ReactToastify.css';

export default async function Home() {
  const session = await getServerSession(authOptions)

  if (!session) {
    redirect('/login')
  }

  if(session){
    redirect('/dashboard')
  }

  return (
    <div>
           
    </div>
  )
}