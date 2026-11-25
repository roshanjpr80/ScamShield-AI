'use client'

import { useState, useCallback } from 'react'
import { Shield, Mail, Lock, Unlock, User, Eye, EyeOff, Loader2, AlertCircle, CheckCircle, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { apiClient } from '@/lib/api'

interface AuthState {
  email: string
  password: string
  rememberMe: boolean
  isLogin: boolean
  loading: boolean
  error: string | null
  showPassword: boolean
}

interface TokenResponse {
  access_token: string
  token_type: string
  expires_in: number
  user: {
    id: string
    name: string
    email: string
    role: string
  }
}

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const redirectTo = searchParams.get('redirect') || '/'

  const [state, setState] = useState<AuthState>({
    email: '',
    password: '',
    rememberMe: false,
    isLogin: true,
    loading: false,
    error: null,
    showPassword: false,
  })

  const handleInputChange = useCallback((field: keyof AuthState, value: any) => {
    setState(prev => ({ ...prev, [field]: value, error: null }))
  }, [])

  const toggleMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      isLogin: !prev.isLogin,
      error: null,
    }))
  }, [])

  const togglePassword = useCallback(() => {
    setState(prev => ({ ...prev, showPassword: !prev.showPassword }))
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))

      // In production, this would call the actual auth API
      // const response = await apiClient.auth.login({ email, password })
      // or apiClient.auth.register({ email, password, name })

      // Mock successful authentication
      const mockToken: TokenResponse = {
        access_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFkbWluIFVzZXIiLCJpYXQiOjE1MTYyMzkwMjJ9.mock_token',
        token_type: 'Bearer',
        expires_in: 3600,
        user: {
          id: 'user_001',
          name: state.isLogin ? 'Admin User' : 'New User',
          email: state.email,
          role: 'Administrator',
        },
      }

      // Store token in localStorage (in production, use httpOnly cookies)
      if (state.rememberMe) {
        localStorage.setItem('authToken', mockToken.access_token)
        localStorage.setItem('user', JSON.stringify(mockToken.user))
      } else {
        sessionStorage.setItem('authToken', mockToken.access_token)
        sessionStorage.setItem('user', JSON.stringify(mockToken.user))
      }

      // Set auth header for API client
      apiClient.setAuthToken(mockToken.access_token)

      // Redirect to intended page
      router.push(redirectTo)
      router.refresh()
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: state.isLogin ? 'Invalid email or password' : 'Registration failed. Please try again.',
        loading: false,
      }))
    }
  }

  const handleDemoLogin = async () => {
    setState(prev => ({
      ...prev,
      email: 'admin@scamshield.ai',
      password: 'demo1234',
      loading: true,
      error: null,
    }))

    try {
      await new Promise(resolve => setTimeout(resolve, 1000))

      const mockToken: TokenResponse = {
        access_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFkbWluIFVzZXIiLCJpYXQiOjE1MTYyMzkwMjJ9.demo_token',
        token_type: 'Bearer',
        expires_in: 3600,
        user: {
          id: 'user_001',
          name: 'Admin User',
          email: 'admin@scamshield.ai',
          role: 'Administrator',
        },
      }

      localStorage.setItem('authToken', mockToken.access_token)
      localStorage.setItem('user', JSON.stringify(mockToken.user))
      apiClient.setAuthToken(mockToken.access_token)

      router.push(redirectTo)
      router.refresh()
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: 'Demo login failed',
        loading: false,
      }))
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-gray-50 flex items-center justify-center p-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary-100 rounded-full blur-3xl opacity-50" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-100 rounded-full blur-3xl opacity-50" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2">
            <Shield className="h-12 w-12 text-primary-600" />
            <span className="font-bold text-2xl text-gray-900">ScamShield</span>
          </Link>
          <p className="text-gray-500 mt-2">Sign in to your fraud detection dashboard</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
          {/* Tabs */}
          <div className="flex mb-6 border-b border-gray-200">
            <button
              onClick={() => setState(prev => ({ ...prev, isLogin: true, error: null }))}
              className={cn(
                'flex-1 py-3 text-sm font-medium transition-colors',
                state.isLogin
                  ? 'text-primary-600 border-b-2 border-primary-600'
                  : 'text-gray-500 hover:text-gray-700'
              )}
            >
              Sign In
            </button>
            <button
              onClick={toggleMode}
              className={cn(
                'flex-1 py-3 text-sm font-medium transition-colors',
                !state.isLogin
                  ? 'text-primary-600 border-b-2 border-primary-600'
                  : 'text-gray-500 hover:text-gray-700'
              )}
            >
              Sign Up
            </button>
          </div>

          {state.error && (
            <div className="mb-6 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center gap-3 animate-shake">
              <AlertCircle className="h-5 w-5 text-danger-600 flex-shrink-0" />
              <span className="text-danger-700 text-sm">{state.error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {/* Email */}
            <div>
              <label htmlFor="email" className="label">
                Email Address
              </label>
              <div className="relative mt-1">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  id="email"
                  type="email"
                  value={state.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  className="input pl-10"
                  placeholder="admin@scamshield.ai"
                  required
                  autoComplete="email"
                  disabled={state.loading}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="label">
                  Password
                </label>
                {!state.isLogin && (
                  <span className="text-xs text-gray-500">Min. 8 characters</span>
                )}
              </div>
              <div className="relative mt-1">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  id="password"
                  type={state.showPassword ? 'text' : 'password'}
                  value={state.password}
                  onChange={(e) => handleInputChange('password', e.target.value)}
                  className="input pl-10 pr-10"
                  placeholder="••••••••"
                  required
                  autoComplete={state.isLogin ? 'current-password' : 'new-password'}
                  disabled={state.loading}
                />
                <button
                  type="button"
                  onClick={togglePassword}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  aria-label={state.showPassword ? 'Hide password' : 'Show password'}
                >
                  {state.showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* Additional fields for registration */}
            {!state.isLogin && (
              <div className="space-y-4">
                <div>
                  <label htmlFor="name" className="label">
                    Full Name
                  </label>
                  <div className="relative mt-1">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      id="name"
                      type="text"
                      className="input pl-10"
                      placeholder="John Doe"
                      required
                      autoComplete="name"
                      disabled={state.loading}
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="confirmPassword" className="label">
                    Confirm Password
                  </label>
                  <div className="relative mt-1">
                    <Unlock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      id="confirmPassword"
                      type={state.showPassword ? 'text' : 'password'}
                      className="input pl-10"
                      placeholder="••••••••"
                      required
                      autoComplete="new-password"
                      disabled={state.loading}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Remember me & Forgot password */}
            {state.isLogin && (
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={state.rememberMe}
                    onChange={(e) => handleInputChange('rememberMe', e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm text-gray-700">Remember me</span>
                </label>
                <Link href="/forgot-password" className="text-sm text-primary-600 hover:underline">
                  Forgot password?
                </Link>
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              className={cn(
                'btn-primary w-full py-3',
                state.loading && 'opacity-50 cursor-not-allowed'
              )}
              disabled={state.loading}
            >
              {state.loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  {state.isLogin ? 'Signing in...' : 'Creating account...'}
                </>
              ) : (
                <>
                  {state.isLogin ? 'Sign In' : 'Create Account'}
                  <ArrowRight className="h-5 w-5 ml-2" />
                </>
              )}
            </button>
          </form>

          {/* Demo login */}
          {state.isLogin && (
            <div className="mt-6 pt-6 border-t border-gray-100">
              <button
                onClick={handleDemoLogin}
                disabled={state.loading}
                className="btn-secondary w-full py-2.5"
              >
                <Shield className="h-4 w-4 mr-2" />
                Demo Login (admin@scamshield.ai / demo1234)
              </button>
              <p className="text-xs text-gray-500 text-center mt-3">
                Use demo credentials for quick access to the dashboard
              </p>
            </div>
          )}

          {/* Footer */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-500">
              {state.isLogin ? "Don't have an account?" : 'Already have an account?'}
              {' '}
              <button
                onClick={toggleMode}
                className="text-primary-600 font-medium hover:underline"
              >
                {state.isLogin ? 'Sign up' : 'Sign in'}
              </button>
            </p>
            <p className="text-xs text-gray-400 mt-4">
              By continuing, you agree to our{' '}
              <Link href="/terms" className="underline hover:text-gray-600">Terms of Service</Link>{' '}
              and{' '}
              <Link href="/privacy" className="underline hover:text-gray-600">Privacy Policy</Link>
            </p>
          </div>
        </div>

        {/* Version info */}
        <div className="text-center mt-6 text-xs text-gray-400">
          ScamShield AI v1.0.0 &copy; 2024
        </div>
      </div>
    </div>
  )
}