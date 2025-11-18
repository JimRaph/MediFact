import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { SignInForm } from '@/components/auth/SignInForm'
// import SignInForm  from '@/components/auth/SignInForm'

export default async function LoginPage() {
  const session = await getServerSession(authOptions)
  if (session) redirect('/')

  return (
    <div className="min-h-screen bg-sky-300 overflow-clip">
      <SignInForm />
    </div>
  )
}
