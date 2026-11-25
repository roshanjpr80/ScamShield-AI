'use client'

import { useState, useEffect } from 'react'
import { Calendar, Download, TrendingUp, TrendingDown, Filter, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { TrendChart, DecisionPie, LatencyChart, FraudTypeBar, GeographyChart } from '@/components/Charts'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'

interface AnalyticsData {
  volume: Array<{ date: string; approved: number; declined: number; flagged: number; total: number }>
  riskDistribution: Array<{ range: string; count: number }>
  topFraudTypes: Array<{ type: string; count: number; percentage: number }>
  merchantRisk: Array<{ merchant: string; riskScore: number; volume: number }>
  deviceRisk: Array<{ deviceType: string; riskScore: number; count: number }>
  hourlyPattern: Array<{ hour: number; volume: number; fraudRate: number }>
  velocityStats: Array<{ period: string; avgVelocity: number; maxVelocity: number }>
  modelPerformance: Array<{ model: string; accuracy: number; precision: number; recall: number; f1: number; latency: number }>
}

export default function AnalyticsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [timeRange, setTimeRange] = useState('7d')
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAnalyticsData()
  }, [timeRange])

  const fetchAnalyticsData = async () => {
    setLoading(true)
    try {
      // Mock data - replace with real API
      await new Promise(resolve => setTimeout(resolve, 500))

      const mockData: AnalyticsData = {
        volume: [
          { date: '2024-01-09', approved: 45200, declined: 1200, flagged: 340, total: 46740 },
          { date: '2024-01-10', approved: 48100, declined: 1350, flagged: 410, total: 49860 },
          { date: '2024-01-11', approved: 52300, declined: 1100, flagged: 280, total: 53680 },
          { date: '2024-01-12', approved: 55800, declined: 1420, flagged: 390, total: 57610 },
          { date: '2024-01-13', approved: 51200, declined: 980, flagged: 250, total: 52430 },
          { date: '2024-01-14', approved: 49800, declined: 1560, flagged: 450, total: 51810 },
          { date: '2024-01-15', approved: 53400, declined: 1280, flagged: 320, total: 55000 },
        ],
        riskDistribution: [
          { range: '0.0-0.1', count: 125000 },
          { range: '0.1-0.2', count: 89000 },
          { range: '0.2-0.3', count: 45000 },
          { range: '0.3-0.4', count: 28000 },
          { range: '0.4-0.5', count: 15000 },
          { range: '0.5-0.6', count: 8200 },
          { range: '0.6-0.7', count: 4100 },
          { range: '0.7-0.8', count: 2100 },
          { range: '0.8-0.9', count: 1100 },
          { range: '0.9-1.0', count: 450 },
        ],
        topFraudTypes: [
          { type: 'Velocity Fraud', count: 1240, percentage: 32.5 },
          { type: 'Amount Anomaly', count: 980, percentage: 25.7 },
          { type: 'Device Spoofing', count: 650, percentage: 17.0 },
          { type: 'Merchant Collusion', count: 420, percentage: 11.0 },
          { type: 'Account Takeover', count: 310, percentage: 8.1 },
          { type: 'Synthetic Identity', count: 220, percentage: 5.7 },
        ],
        merchantRisk: [
          { merchant: 'crypto_exchange@upi', riskScore: 0.85, volume: 12400 },
          { merchant: 'gaming_site@upi', riskScore: 0.72, volume: 8900 },
          { merchant: 'new_merchant@upi', riskScore: 0.68, volume: 5600 },
          { merchant: 'electronics@upi', riskScore: 0.45, volume: 23400 },
          { merchant: 'fashion_store@upi', riskScore: 0.32, volume: 18200 },
        ],
        deviceRisk: [
          { deviceType: 'Android (Rooted)', riskScore: 0.78, count: 2100 },
          { deviceType: 'iOS (Jailbroken)', riskScore: 0.72, count: 890 },
          { deviceType: 'Android (VPN)', riskScore: 0.55, count: 5600 },
          { deviceType: 'Web (Tor)', riskScore: 0.89, count: 340 },
          { deviceType: 'iOS (Normal)', riskScore: 0.12, count: 45200 },
          { deviceType: 'Android (Normal)', riskScore: 0.18, count: 38900 },
        ],
        hourlyPattern: Array.from({ length: 24 }, (_, i) => ({
          hour: i,
          volume: Math.floor(Math.random() * 5000) + 1000,
          fraudRate: Math.random() * 0.05,
        })),
        velocityStats: [
          { period: '5 min', avgVelocity: 2.3, maxVelocity: 45 },
          { period: '1 hour', avgVelocity: 12.5, maxVelocity: 280 },
          { period: '24 hour', avgVelocity: 45.8, maxVelocity: 1200 },
        ],
        modelPerformance: [
          { model: 'XGBoost', accuracy: 0.962, precision: 0.941, recall: 0.923, f1: 0.932, latency: 8 },
          { model: 'LightGBM', accuracy: 0.958, precision: 0.935, recall: 0.918, f1: 0.926, latency: 6 },
          { model: 'Transformer', accuracy: 0.971, precision: 0.952, recall: 0.938, f1: 0.945, latency: 22 },
          { model: 'GNN', accuracy: 0.945, precision: 0.921, recall: 0.905, f1: 0.913, latency: 15 },
          { model: 'Ensemble', accuracy: 0.978, precision: 0.961, recall: 0.948, f1: 0.954, latency: 23 },
        ],
      }
      setData(mockData)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch analytics:', error)
      setLoading(false)
    }
  }

  if (!data) return null

  const totalVolume = data.volume.reduce((sum, d) => sum + d.total, 0)
  const totalDeclined = data.volume.reduce((sum, d) => sum + d.declined, 0)
  const totalFlagged = data.volume.reduce((sum, d) => sum + d.flagged, 0)
  const fraudRate = ((totalDeclined + totalFlagged) / totalVolume * 100).toFixed(2)
  const approvalRate = (totalVolume - totalDeclined - totalFlagged) / totalVolume * 100

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar onCollapseChange={setSidebarCollapsed} collapsed={sidebarCollapsed} />
      <Header sidebarCollapsed={sidebarCollapsed} />

      <main className={cn('pt-16 transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-64')}>
        <div className="p-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
              <p className="text-gray-500 mt-1">Deep insights into fraud patterns and model performance</p>
            </div>
            <div className="flex items-center gap-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="input w-auto"
              >
                <option value="1d">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 90 days</option>
              </select>
              <button
                onClick={fetchAnalyticsData}
                disabled={loading}
                className="btn-secondary"
              >
                <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
                Refresh
              </button>
              <button className="btn-primary">
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </button>
            </div>
          </div>

          {/* Summary KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Transactions</p>
                  <p className="text-2xl font-bold text-gray-900">{totalVolume.toLocaleString()}</p>
                </div>
                <TrendingUp className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Approval Rate</p>
                  <p className="text-2xl font-bold text-success-600">{approvalRate.toFixed(1)}%</p>
                </div>
                <TrendingUp className="h-8 w-8 text-success-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Fraud Rate</p>
                  <p className="text-2xl font-bold text-danger-600">{fraudRate}%</p>
                </div>
                <TrendingDown className="h-8 w-8 text-danger-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Avg Risk Score</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {data.riskDistribution.reduce((sum, d) => {
                      const mid = parseFloat(d.range.split('-')[0]) + 0.05
                      return sum + mid * d.count
                    }, 0) / data.riskDistribution.reduce((sum, d) => sum + d.count, 0) * 100}
                    %
                  </p>
                </div>
                <TrendingDown className="h-8 w-8 text-gray-600" />
              </div>
            </div>
          </div>

          {/* Charts Row 1: Volume & Risk Distribution */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Transaction Volume Trend</h2>
                <span className="badge badge-info">{timeRange}</span>
              </div>
              <TrendChart
                data={data.volume.map(v => ({ timestamp: v.date, transactions: v.total, fraud_detected: v.flagged, avg_latency_ms: 0 }))}
              />
            </div>

            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Risk Score Distribution</h2>
                <span className="badge badge-info">All transactions</span>
              </div>
              <FraudTypeBar
                data={data.riskDistribution.map(r => ({ type: r.range, count: r.count }))}
              />
            </div>
          </div>

          {/* Charts Row 2: Hourly Pattern & Fraud Types */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Hourly Transaction Pattern</h2>
                <span className="badge badge-info">24-hour cycle</span>
              </div>
              <LatencyChart
                data={data.hourlyPattern.map(h => ({ time: `${h.hour}:00`, p50: h.volume * 0.5, p95: h.volume * 0.8, p99: h.volume }))}
              />
            </div>

            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Top Fraud Types</h2>
                <span className="badge badge-info">By count</span>
              </div>
              <FraudTypeBar
                data={data.topFraudTypes.map(f => ({ type: f.type, count: f.count }))}
              />
            </div>
          </div>

          {/* Charts Row 3: Merchant Risk & Device Risk */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Top Risky Merchants</h2>
                <span className="badge badge-info">By risk score</span>
              </div>
              <div className="space-y-3">
                {data.merchantRisk.map((merchant, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="w-8 text-center text-sm font-medium text-gray-500">{idx + 1}</span>
                        <div>
                          <p className="font-medium text-gray-900 truncate max-w-[200px]">{merchant.merchant}</p>
                          <p className="text-sm text-gray-500">Volume: {merchant.volume.toLocaleString()}</p>
                        </div>
                      </div>
                      <span className={cn('badge font-medium',
                        merchant.riskScore > 0.7 ? 'badge-danger' :
                        merchant.riskScore > 0.4 ? 'badge-warning' : 'badge-success')}>
                        {Math.round(merchant.riskScore * 100)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={cn('h-full rounded-full transition-all',
                          merchant.riskScore > 0.7 ? 'bg-danger-500' :
                          merchant.riskScore > 0.4 ? 'bg-warning-500' : 'bg-success-500')}
                        style={{ width: `${merchant.riskScore * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Device Risk Profile</h2>
                <span className="badge badge-info">By device type</span>
              </div>
              <div className="space-y-3">
                {data.deviceRisk.map((device, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="w-8 text-center text-sm font-medium text-gray-500">{idx + 1}</span>
                        <div>
                          <p className="font-medium text-gray-900">{device.deviceType}</p>
                          <p className="text-sm text-gray-500">Count: {device.count.toLocaleString()}</p>
                        </div>
                      </div>
                      <span className={cn('badge font-medium',
                        device.riskScore > 0.7 ? 'badge-danger' :
                        device.riskScore > 0.4 ? 'badge-warning' : 'badge-success')}>
                        {Math.round(device.riskScore * 100)}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={cn('h-full rounded-full transition-all',
                          device.riskScore > 0.7 ? 'bg-danger-500' :
                          device.riskScore > 0.4 ? 'bg-warning-500' : 'bg-success-500')}
                        style={{ width: `${device.riskScore * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Model Performance Table */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Comparison</h2>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Latency (ms)</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.modelPerformance.map((model, idx) => (
                    <tr key={idx} className={model.model === 'Ensemble' ? 'bg-primary-50 font-medium' : ''}>
                      <td className="font-medium">{model.model}</td>
                      <td>{(model.accuracy * 100).toFixed(1)}%</td>
                      <td>{(model.precision * 100).toFixed(1)}%</td>
                      <td>{(model.recall * 100).toFixed(1)}%</td>
                      <td>{(model.f1 * 100).toFixed(1)}%</td>
                      <td className={cn('font-medium', model.latency > 20 ? 'text-warning-600' : 'text-success-600')}>
                        {model.latency}ms
                      </td>
                      <td>
                        <span className={cn('badge',
                          model.model === 'Ensemble' ? 'badge-primary' :
                          model.latency <= 20 ? 'badge-success' : 'badge-warning')}>
                          {model.model === 'Ensemble' ? 'Production' : model.latency <= 20 ? 'Active' : 'Monitoring'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Velocity Statistics */}
          <div className="card mt-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Velocity Statistics</h2>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Period</th>
                    <th>Average Velocity</th>
                    <th>Maximum Velocity</th>
                    <th>Threshold</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.velocityStats.map((stat, idx) => (
                    <tr key={idx}>
                      <td className="font-medium">{stat.period}</td>
                      <td>{stat.avgVelocity.toFixed(1)} txns</td>
                      <td>{stat.maxVelocity} txns</td>
                      <td>{stat.period === '5 min' ? '20' : stat.period === '1 hour' ? '100' : '500'}</td>
                      <td>
                        <span className={cn('badge',
                          stat.maxVelocity > (stat.period === '5 min' ? 20 : stat.period === '1 hour' ? 100 : 500) ? 'badge-danger' : 'badge-success')}>
                          {stat.maxVelocity > (stat.period === '5 min' ? 20 : stat.period === '1 hour' ? 100 : 500) ? 'Exceeded' : 'Normal'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}