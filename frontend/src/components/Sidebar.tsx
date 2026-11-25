'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Shield,
  Search,
  BarChart3,
  BookOpen,
  Settings,
  Bell,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  Sparkles,
} from 'lucide-react'
import { useState } from 'react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Live Scoring', href: '/scoring', icon: Shield },
  { name: 'Investigations', href: '/investigations', icon: Search },
  { name: 'Genable AI', href: '/genable', icon: Sparkles },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Scam Library', href: '/scams', icon: BookOpen },
  { name: 'Models', href: '/models', icon: Settings },
  { name: 'Alerts', href: '/alerts', icon: Bell },
  { name: 'Documentation', href: '/docs', icon: HelpCircle },
]

export function Sidebar({ collapsed = false, onCollapseChange }: { collapsed?: boolean; onCollapseChange?: (v: boolean) => void }) {
  const pathname = usePathname()
  const [internalCollapsed, setInternalCollapsed] = useState(false)

  const isCollapsed = collapsed ?? internalCollapsed
  const setCollapsed = (v: boolean) => {
    setInternalCollapsed(v)
    onCollapseChange?.(v)
  }

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen bg-white border-r border-gray-200 transition-all duration-300',
        isCollapsed ? 'w-16' : 'w-64'
      )}
    >
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div
          className={cn(
            'flex h-16 items-center justify-between px-4 border-b border-gray-200 transition-all duration-300',
            isCollapsed && 'justify-center'
          )}
        >
          {!isCollapsed && (
            <Link href="/" className="flex items-center gap-2">
              <Shield className="h-8 w-8 text-primary-600" />
              <span className="font-bold text-xl text-gray-900">ScamShield</span>
            </Link>
          )}
          {isCollapsed && (
            <Link href="/" className="flex items-center justify-center">
              <Shield className="h-8 w-8 text-primary-600" />
            </Link>
          )}
          <button
            onClick={() => setCollapsed(!isCollapsed)}
            className={cn(
              'p-1.5 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors',
              isCollapsed && 'ml-auto'
            )}
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1" aria-label="Main navigation">
          {navigation.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`)
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'sidebar-link',
                  isActive && 'sidebar-link-active',
                  isCollapsed && 'justify-center px-2'
                )}
                title={isCollapsed ? item.name : undefined}
              >
                <item.icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
                {!isCollapsed && <span>{item.name}</span>}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div
          className={cn(
            'p-4 border-t border-gray-200 transition-all duration-300',
            isCollapsed && 'hidden'
          )}
        >
          <div className="text-xs text-gray-500 text-center">
            ScamShield AI v1.0.0
          </div>
        </div>
      </div>
    </aside>
  )
}