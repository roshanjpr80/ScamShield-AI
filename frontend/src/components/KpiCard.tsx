'use client'

import { cn } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface KpiCardProps {
  title: string
  label?: string  // alias for title
  value: string | number
  change?: number
  changeLabel?: string
  icon?: React.ReactNode
  iconColor?: string
  trend?: 'up' | 'down' | 'neutral'
  variant?: 'default' | 'success' | 'warning' | 'danger'
  loading?: boolean
}

export function KpiCard({ title, label, value, change, changeLabel, icon, iconColor = 'primary', trend, variant, loading }: KpiCardProps) {
  const displayTitle = title || label
  const isPositive = change && change > 0
  const isNegative = change && change < 0

  return (
    <div className="kpi-card card-hover">
      <div className="flex-1 min-w-0">
        <p className="kpi-label">{displayTitle}</p>
        <p className="kpi-value mt-1">{value}</p>
        {change !== undefined && (
          <div className="flex items-center gap-1 mt-2">
            {isPositive && <TrendingUp className="h-4 w-4 kpi-change-positive" />}
            {isNegative && <TrendingDown className="h-4 w-4 kpi-change-negative" />}
            {!isPositive && !isNegative && <Minus className="h-4 w-4 text-gray-400" />}
            <span className={cn('text-sm font-medium', isPositive && 'kpi-change-positive', isNegative && 'kpi-change-negative')}>
              {isPositive ? '+' : ''}{change.toFixed(1)}%
            </span>
            {changeLabel && <span className="text-sm text-gray-500">{changeLabel}</span>}
          </div>
        )}
      </div>
      {icon && (
        <div className={cn('h-12 w-12 rounded-xl flex items-center justify-center', `bg-${iconColor}-100 text-${iconColor}-600`)}>
          {icon}
        </div>
      )}
    </div>
  )
}