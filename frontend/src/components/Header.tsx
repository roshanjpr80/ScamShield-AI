'use client'

import { cn } from '@/lib/utils'
import { Bell, Sun, Moon, Menu, LogOut, User, Shield, Settings } from 'lucide-react'
import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { apiClient } from '@/lib/api'

export function Header({ sidebarCollapsed }: { sidebarCollapsed: boolean }) {
  const [notificationsOpen, setNotificationsOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const [darkMode, setDarkMode] = useState(false)
  const router = useRouter()

  const notifications = [
    { id: '1', title: 'High fraud rate detected', message: 'Fraud rate exceeded 0.5% in last hour', time: '5 min ago', read: false, severity: 'high' },
    { id: '2', title: 'Model latency spike', message: 'P99 latency reached 65ms on fraud_classifier', time: '12 min ago', read: false, severity: 'medium' },
    { id: '3', title: 'New scam pattern added', message: 'Fake UPI refund scam added to library', time: '1 hour ago', read: true, severity: 'info' },
    { id: '4', title: 'Scheduled maintenance', message: 'Database maintenance scheduled for 2 AM UTC', time: '3 hours ago', read: true, severity: 'info' },
  ]

  const unreadCount = notifications.filter((n) => !n.read).length

  const handleLogout = () => {
    // Clear auth token from API client
    apiClient.clearAuthToken()
    // Clear from localStorage and sessionStorage
    localStorage.removeItem('authToken')
    localStorage.removeItem('user')
    sessionStorage.removeItem('authToken')
    sessionStorage.removeItem('user')
    // Close user menu
    setUserMenuOpen(false)
    // Redirect to login page
    router.push('/login')
    router.refresh()
  }

  return (
    <header
      className={cn(
        'fixed top-0 right-0 z-30 h-16 bg-white/80 backdrop-blur-sm border-b border-gray-200 transition-all duration-300',
        sidebarCollapsed ? 'left-16' : 'left-64'
      )}
    >
      <div className="flex h-full items-center justify-between px-4">
        {/* Left side - Mobile menu button */}
        <div className="flex items-center gap-4">
          <button
            className="lg:hidden p-2 rounded-lg text-gray-500 hover:bg-gray-100"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
        </div>

        {/* Right side - Actions */}
        <div className="flex items-center gap-3">
          {/* Theme toggle */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setNotificationsOpen(!notificationsOpen)}
              className="relative p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
              aria-label="Notifications"
              aria-expanded={notificationsOpen}
            >
              <Bell className="h-5 w-5" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-danger-500 text-xs text-white">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            {notificationsOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-lg border border-gray-200 py-2 animate-fade-in">
                <div className="px-4 py-2 border-b border-gray-100 flex items-center justify-between">
                  <h3 className="font-semibold text-gray-900">Notifications</h3>
                  {unreadCount > 0 && (
                    <button className="text-sm text-primary-600 hover:underline">Mark all read</button>
                  )}
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={cn(
                        'px-4 py-3 hover:bg-gray-50 border-l-4 transition-colors',
                        !notification.read && 'bg-blue-50 border-primary-500',
                        notification.read && 'border-transparent'
                      )}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className={cn('text-sm font-medium text-gray-900', !notification.read && 'font-semibold')}>
                            {notification.title}
                          </p>
                          <p className="text-sm text-gray-600 mt-0.5">{notification.message}</p>
                          <p className="text-xs text-gray-400 mt-1">{notification.time}</p>
                        </div>
                        <span
                          className={cn(
                            'badge flex-shrink-0',
                            notification.severity === 'high' && 'badge-danger',
                            notification.severity === 'medium' && 'badge-warning',
                            notification.severity === 'info' && 'badge-info'
                          )}
                        >
                          {notification.severity}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="px-4 py-2 border-t border-gray-100 text-center">
                  <Link href="/alerts" className="text-sm text-primary-600 hover:underline">
                    View all alerts
                  </Link>
                </div>
              </div>
            )}
          </div>

          {/* User menu */}
          <div className="relative">
            <button
              onClick={() => setUserMenuOpen(!userMenuOpen)}
              className="flex items-center gap-2 p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="User menu"
              aria-expanded={userMenuOpen}
            >
              <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                <Shield className="h-5 w-5 text-primary-600" />
              </div>
              <span className="hidden sm:block text-sm font-medium text-gray-700">Admin</span>
            </button>

            {userMenuOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-lg border border-gray-200 py-2 animate-fade-in">
                <div className="px-4 py-2 border-b border-gray-100">
                  <p className="font-medium text-gray-900">Admin User</p>
                  <p className="text-sm text-gray-500">admin@scamshield.ai</p>
                </div>
                <Link
                  href="/settings"
                  className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                  onClick={() => setUserMenuOpen(false)}
                >
                  <User className="h-4 w-4" />
                  Profile
                </Link>
                <Link
                  href="/settings"
                  className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                  onClick={() => setUserMenuOpen(false)}
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </Link>
                <hr className="my-2 border-gray-100" />
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 w-full px-4 py-2 text-sm text-danger-600 hover:bg-gray-50"
                >
                  <LogOut className="h-4 w-4" />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}