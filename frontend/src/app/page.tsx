'use client'

import { useState, useEffect } from 'react'
import { Shield, TrendingUp, TrendingDown, AlertTriangle, Users, DollarSign, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { KpiCard } from '@/components/KpiCard'
import { TrendChart, DecisionPie, LatencyChart, FraudTypeBar, GeographyChart } from '@/components/Charts'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'

interface KpiData {
  label: string
  value: string | number
  change?: number
  changeLabel?: string
  icon: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'danger'
}

interface RecentTransaction {
  id: string
  payer: string
  payee: string
  amount: number
  riskScore: number
  decision: 'APPROVE' | 'REVIEW' | 'DECLINE'
  timestamp: string
}

interface Alert {
  id: string
  title: string
  message: string
  severity: 'high' | 'medium' | 'low'
  time: string
}

export default function DashboardPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [kpis, setKpis] = useState<KpiData[]>([])
  const [recentTransactions, setRecentTransactions] = useState<RecentTransaction[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState('24h')

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000)
    return () => clearInterval(interval)
  }, [timeRange])

  const fetchDashboardData = async () => {
    try {
      // Mock data for now - replace with real API calls
      setKpis([
        { label: 'Total Transactions', value: '1,234,567', change: 12.5, changeLabel: 'vs last 24h', icon: <Shield className="h-6 w-6" /> },
        { label: 'Fraud Detected', value: '2,847', change: -8.2, changeLabel: 'vs last 24h', icon: <AlertTriangle className="h-6 w-6" />, variant: 'danger' },
        { label: 'Approval Rate', value: '98.7%', change: 0.3, changeLabel: 'vs last 24h', icon: <CheckCircle className="h-6 w-6" />, variant: 'success' },
        { label: 'Avg Latency', value: '23ms', change: -2.1, changeLabel: 'vs last 24h', icon: <Clock className="h-6 w-6" />, variant: 'success' },
        { label: 'Active Users', value: '45,672', change: 5.7, changeLabel: 'vs last 24h', icon: <Users className="h-6 w-6" /> },
        { label: 'Amount Protected', value: '$2.4B', change: 15.3, changeLabel: 'vs last 24h', icon: <DollarSign className="h-6 w-6" />, variant: 'success' },
      ])

      setRecentTransactions([
        { id: 'txn_001', payer: 'user1@upi', payee: 'merchant1@upi', amount: 5000, riskScore: 0.12, decision: 'APPROVE', timestamp: '2024-01-15T10:30:00Z' },
        { id: 'txn_002', payer: 'user2@upi', payee: 'merchant2@upi', amount: 75000, riskScore: 0.87, decision: 'DECLINE', timestamp: '2024-01-15T10:29:45Z' },
        { id: 'txn_003', payer: 'user3@upi', payee: 'merchant3@upi', amount: 12000, riskScore: 0.45, decision: 'REVIEW', timestamp: '2024-01-15T10:29:30Z' },
        { id: 'txn_004', payer: 'user4@upi', payee: 'merchant4@upi', amount: 3000, riskScore: 0.05, decision: 'APPROVE', timestamp: '2024-01-15T10:29:15Z' },
        { id: 'txn_005', payer: 'user5@upi', payee: 'merchant5@upi', amount: 150000, riskScore: 0.92, decision: 'DECLINE', timestamp: '2024-01-15T10:29:00Z' },
      ])

      setAlerts([
        { id: '1', title: 'High fraud rate detected', message: 'Fraud rate exceeded 0.5% in last hour', severity: 'high', time: '5 min ago' },
        { id: '2', title: 'Model latency spike', message: 'P99 latency reached 65ms on fraud_classifier', severity: 'medium', time: '12 min ago' },
        { id: '3', title: 'New scam pattern added', message: 'Fake UPI refund scam added to library', severity: 'low', time: '1 hour ago' },
      ])

      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
      setLoading(false)
    }
  }

  const getDecisionBadge = (decision: string) => {
    const variants = {
      APPROVE: 'badge-success',
      REVIEW: 'badge-warning',
      DECLINE: 'badge-danger',
    }
    const icons = {
      APPROVE: <CheckCircle className="h-3 w-3 mr-1" />,
      REVIEW: <AlertCircle className="h-3 w-3 mr-1" />,
      DECLINE: <XCircle className="h-3 w-3 mr-1" />,
    }
    return (
      <span className={cn('badge', variants[decision as keyof typeof variants])}>
        {icons[decision as keyof typeof icons]}
        {decision}
      </span>
    )
  }

  const getSeverityColor = (severity: string) => {
    const colors = {
      high: 'text-danger-600 bg-danger-50 border-danger-200',
      medium: 'text-warning-600 bg-warning-50 border-warning-200',
      low: 'text-primary-600 bg-primary-50 border-primary-200',
    }
    return colors[severity as keyof typeof colors] || colors.low
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar onCollapseChange={setSidebarCollapsed} collapsed={sidebarCollapsed} />
      <Header sidebarCollapsed={sidebarCollapsed} />

      <main className={cn('pt-16 transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-64')}>
        <div className="p-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-gray-500 mt-1">Real-time fraud detection overview</p>
            </div>
            <div className="flex items-center gap-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="input w-auto"
              >
                <option value="1h">Last 1 hour</option>
                <option value="24h">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
              </select>
              <button className="btn-primary">
                <svg className="h-4 w-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export Report
              </button>
            </div>
          </div>

          {/* KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-6">
            {kpis.map((kpi, index) => (
              <KpiCard
                key={index}
                title={kpi.label}
                value={kpi.value}
                change={kpi.change}
                changeLabel={kpi.changeLabel}
                icon={kpi.icon}
                variant={kpi.variant}
                loading={loading}
              />
            ))}
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Transaction Volume</h2>
                <span className="badge badge-info">Real-time</span>
              </div>
              <TrendChart
                data={[
                  { timestamp: '2024-01-01T00:00:00', transactions: 1200, fraud_detected: 45, avg_latency_ms: 12 },
                  { timestamp: '2024-01-01T04:00:00', transactions: 800, fraud_detected: 30, avg_latency_ms: 8 },
                  { timestamp: '2024-01-01T08:00:00', transactions: 3200, fraud_detected: 85, avg_latency_ms: 22 },
                  { timestamp: '2024-01-01T12:00:00', transactions: 5400, fraud_detected: 120, avg_latency_ms: 35 },
                  { timestamp: '2024-01-01T16:00:00', transactions: 4800, fraud_detected: 95, avg_latency_ms: 28 },
                  { timestamp: '2024-01-01T20:00:00', transactions: 3800, fraud_detected: 75, avg_latency_ms: 20 },
                ]}
              />
            </div>

            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Risk Score Distribution</h2>
                <span className="badge badge-info">Live</span>
              </div>
              <FraudTypeBar
                data={[
                  { type: '0.0-0.2', count: 45000 },
                  { type: '0.2-0.4', count: 32000 },
                  { type: '0.4-0.6', count: 18000 },
                  { type: '0.6-0.8', count: 8500 },
                  { type: '0.8-1.0', count: 3200 },
                ]}
              />
            </div>
          </div>

          {/* Bottom Row: Recent Transactions & Alerts */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Recent Transactions */}
            <div className="xl:col-span-2 card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Recent Transactions</h2>
                <button className="btn-ghost text-sm">View All</button>
              </div>
              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Transaction ID</th>
                      <th>Payer</th>
                      <th>Payee</th>
                      <th className="text-right">Amount</th>
                      <th>Risk Score</th>
                      <th>Decision</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTransactions.map((txn) => (
                      <tr key={txn.id}>
                        <td className="font-mono text-sm">{txn.id}</td>
                        <td>{txn.payer}</td>
                        <td>{txn.payee}</td>
                        <td className="text-right font-medium">₹{txn.amount.toLocaleString()}</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className={cn('h-full rounded-full transition-all', txn.riskScore > 0.7 ? 'bg-danger-500' : txn.riskScore > 0.4 ? 'bg-warning-500' : 'bg-success-500')}
                                style={{ width: `${txn.riskScore * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium">{Math.round(txn.riskScore * 100)}%</span>
                          </div>
                        </td>
                        <td>{getDecisionBadge(txn.decision)}</td>
                        <td className="text-sm text-gray-500">
                          {new Date(txn.timestamp).toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Alerts Panel */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Active Alerts</h2>
                <button className="btn-ghost text-sm">View All</button>
              </div>
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={cn('p-4 rounded-lg border', getSeverityColor(alert.severity))}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900">{alert.title}</p>
                        <p className="text-sm text-gray-600 mt-0.5">{alert.message}</p>
                        <p className="text-xs text-gray-400 mt-1">{alert.time}</p>
                      </div>
                      <span
                        className={cn(
                          'badge flex-shrink-0',
                          alert.severity === 'high' && 'badge-danger',
                          alert.severity === 'medium' && 'badge-warning',
                          alert.severity === 'low' && 'badge-info'
                        )}
                      >
                        {alert.severity}
                      </span>
                    </div>
                  </div>
                ))}
                {alerts.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <Shield className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                    <p>No active alerts</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}