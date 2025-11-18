'use client'

import { motion } from 'framer-motion'
import { HeartIcon } from '@heroicons/react/24/solid'
import { signIn } from 'next-auth/react'
import { useForm } from 'react-hook-form'
import Image from 'next/image'
import { useAuthStore } from '@/stores/authStore'
import { RegisterData } from '@/types/api'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export function RegisterForm() {
    const { register, handleSubmit } = useForm<RegisterData>()
    const { isLoading, setLoading, setError, error } = useAuthStore()
    const router = useRouter()
    
    const onSubmit = async (data: RegisterData) => {
        try {
        setError(null)
        setLoading(true)

        const res = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        })

        const response = await res.json()

        if (response.error){
          setError(response.error)
        }

        const loginRes = await signIn('credentials', {
            redirect: false,
            email: data.email,
            password: data.password
        })

        if (loginRes?.error) {
            setError('Login failed.')
        } else{
          router.push('/')
        }
        } catch (err) {
        console.error(err)
        setError('Something went wrong.')
        } finally {
        setLoading(false)
        }
    }

    const handleGoogleSignIn = async () => {
        setLoading(true);
        try {
        await signIn("google", { callbackUrl: "/" });
        } catch (err: unknown) {
        setError(`Google sign-in failed: ${err}`);
        } finally {
        setLoading(false);
        }
    }

  useEffect(() => {
    if (error) {
      const clearError = setTimeout(() => {
        setError(null);
      }, 3000);

      return () => clearTimeout(clearError);
    }
  }, [error, setError]);


return (
    <div className="flex min-h-screen">
      <motion.div
        initial={{ x: -50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="hidden lg:flex flex-col justify-center items-center 
        w-1/2 relative bg-linear-to-br from-emerald-100 via-emerald-200
        to-emerald-300 text-white overflow-hidden"
      >
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="mb-6"
        >
          <HeartIcon className="w-16 h-16 " />
        </motion.div>

        <motion.h2
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-4xl font-semibold mb-3 text-emerald-900"
        >
          Join Our Health Network 
        </motion.h2>

        <div className="text-emerald-800 max-w-md text-center leading-relaxed">
          <p>Create your account and start your AI-powered health information journey.</p>
        </div>

        <svg
          className="absolute bottom-0 left-0 w-full opacity-50"
          viewBox="0 0 800 400"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M0,320 C200,280 600,360 800,320 L800,400 L0,400 Z"
            fill="white"
          />
        </svg>
      </motion.div>

      <motion.div
        initial={{ x: 50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="flex w-full lg:w-1/2 items-center justify-center p-8 bg-white"
      >
        <div className="w-full max-w-md space-y-8">
          <div className="text-center">
            <h2 className="text-3xl font-semibold text-gray-600">
              Create your account
            </h2>
            <p className="mt-2 text-gray-500">
              Sign up with your details or continue with Google.
            </p>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="text-gray-700">
            <div className="mb-4">
              <label
                htmlFor="name"
                className="block text-sm font-medium text-gray-700"
              >
                Full Name
              </label>
              <input
                id="name"
                {...register('name')}
                type="text"
                required
                className="mt-2 block w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-emerald-500 focus:ring-emerald-500 outline-none"
              />
            </div>

            <div className="mb-4">
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email address
              </label>
              <input
                id="email"
                {...register('email')}
                type="email"
                required
                className="mt-2 block w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-emerald-500 focus:ring-emerald-500 outline-none"
              />
            </div>

            <div className="mb-4">
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <input
                id="password"
                {...register('password')}
                type="password"
                required
                className="mt-2 block w-full rounded-lg border border-gray-300 px-4 py-2 
                focus:border-emerald-500 focus:ring-emerald-500 outline-none"
              />
            </div>

            {error && <p className='text-red-800'>{error}</p>}
            
            <button
              type="submit"
              disabled={isLoading }
              className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium 
              py-2 mt-2 rounded-lg transition"
            >
              {isLoading  ? 'Creating account...' : 'Register'}
            </button>
          </form>

          <div className="flex items-center justify-center">
            <div className="h-px bg-gray-300 w-1/4"></div>
            <span className="text-gray-500 text-sm mx-3">or</span>
            <div className="h-px bg-gray-300 w-1/4"></div>
          </div>

          <button
            type="button"
            onClick={handleGoogleSignIn}
            className="flex items-center justify-center gap-2 w-full border border-gray-300
             hover:bg-gray-50 rounded-lg py-2 transition text-gray-700"
          >
            <Image
              src="https://www.svgrepo.com/show/475656/google-color.svg"
              alt=""
              width={18}
              height={18}
            />
            Continue with Google
          </button>
        </div>
      </motion.div>
    </div>
  )
}
