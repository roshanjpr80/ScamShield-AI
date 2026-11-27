'use client'

import { useState, useEffect } from 'react'
import { Calendar, Download, TrendingUp, TrendingDown, Filter, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { TrendChart, DecisionPie, LatencyChart, FraudTypeBar, GeographyChart } from '@/components/Charts'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient } from '@/lib/api'

interface AnalyticsData {
  volume: Array<{ date: string; approved: number; declined: number; flagged: number; total: number }>
  riskDistribution: Array<{ range: string; count: number }>
  topFraudTypes: Array<{ type: string; count: number; percentage: number }>
  merchantRisk: Array<{ merchant: string; riskScore: number; volume: number; fraudCount: number; category: string }>
  deviceRisk: Array<{ deviceType: string; riskScore: number; count: number; fraudCount: number; os: string }>
  hourlyPattern: Array<{ hour: number; volume: number; fraudRate: number }>
  velocityStats: Array<{ period: string; avgVelocity: number; maxVelocity: number }>
  modelPerformance: Array<{ model: string; accuracy: number; precision: number; recall: number; f1: number; latency: number }>
}

interface TrendData {
  timestamp: string
  transactions: number
  fraud_detected: number
  avg_latency_ms: number
}

interface GeographyData {
  state: string
  transactions: number
  fraud_rate: number
}

interface MerchantRiskData {
  merchant: string
  risk_score: number
  volume: number
  fraud_count: number
  category: string
}

interface DeviceRiskData {
  device_type: string
  risk_score: number
  count: number
  fraud_count: number
  os: string
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
      // Fetch real data from API
      const [overview, trends, geography, merchantRisk, deviceRisk] = await Promise.all([
        apiClient.getAnalyticsOverview(),
        apiClient.getTrends(24),
        apiClient.getGeography(),
        apiClient.getMerchantRisk(10),
        apiClient.getDeviceRisk(10),
      ])

      // Transform trend data to volume format
      const volume = trends.map((t: TrendData) => ({
        date: t.timestamp.split('T')[0],
        approved: t.transactions - t.fraud_detected,
        declined: t.fraud_detected,
        flagged: 0,
        total: t.transactions,
      }))

      // Create risk distribution from fraud rate
      const riskDistribution = [
        { range: '0.0-0.1', count: Math.floor(overview.total_transactions_24h * 0.8) },
        { range: '0.1-0.2', count: Math.floor(overview.total_transactions_24h * 0.12) },
        { range: '0.2-0.3', count: Math.floor(overview.total_transactions_24h * 0.05) },
        { range: '0.3-0.4', count: Math.floor(overview.total_transactions_24h * 0.02) },
        { range: '0.4-0.5', count: Math.floor(overview.total_transactions_24h * 0.01) },
        { range: '0.5-0.6', count: Math.floor(overview.total_transactions_24h * 0.005) },
        { range: '0.6-0.7', count: Math.floor(overview.total_transactions_24h * 0.003) },
        { range: '0.7-0.8', count: Math.floor(overview.total_transactions_24h * 0.001) },
        { range: '0.8-0.9', count: Math.floor(overview.total_transactions_24h * 0.0005) },
        { range: '0.9-1.0', count: Math.floor(overview.total_transactions_24h * 0.0002) },
      ]

      // Top fraud types from API
      const topFraudTypes = overview.top_fraud_types.map((f, idx) => ({
        type: f.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        count: f.count,
        percentage: parseFloat((f.count / overview.fraud_detected_24h * 100).toFixed(1)),
      }))

      // Hourly pattern from trends
      const hourlyPattern = Array.from({ length: 24 }, (_, i) => {
        const trend = trends.find((t: TrendData) => new Date(t.timestamp).getHours() === i)
        return {
          hour: i,
          volume: trend?.transactions || 0,
          fraudRate: trend ? trend.fraud_detected / trend.transactions : 0,
        }
      })

      // Transform merchant risk data from API
      const merchantRiskData = merchantRisk.merchants.map((m: MerchantRiskData) => ({
        merchant: m.merchant,
        riskScore: m.risk_score,
        volume: m.volume,
        fraudCount: m.fraud_count,
        category: m.category,
      }))

      // Transform device risk data from API
      const deviceRiskData = deviceRisk.devices.map((d: DeviceRiskData) => ({
        deviceType: d.device_type,
        riskScore: d.risk_score,
        count: d.count,
        fraudCount: d.fraud_count,
        os: d.os,
      }))

      const mockData: AnalyticsData = {
        volume,
        riskDistribution,
        topFraudTypes,
        merchantRisk: merchantRiskData,
        deviceRisk: deviceRiskData,
        hourlyPattern,
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
                          <p className="text-sm text-gray-500">Volume: {merchant.volume.toLocaleString()} | Fraud: {merchant.fraudCount.toLocaleString()} | Category: {merchant.category}</p>
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
                          <p className="text-sm text-gray-500">Count: {device.count.toLocaleString()} | Fraud: {device.fraudCount.toLocaleString()} | OS: {device.os}</p>
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