'use client'

import { useState, useEffect, useCallback } from 'react'
import { Shield, Search, RefreshCw, AlertTriangle, CheckCircle, XCircle, AlertCircle, Eye, Download, Filter, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient, ScoreResponse, UPITransaction } from '@/lib/api'

interface LiveTransaction {
  txn_id: string
  payer_vpa: string
  payee_vpa: string
  amount: number
  currency: string
  txn_type: string
  timestamp: string
  device_id?: string
  client_ip?: string
  merchant_category?: string
  status: 'pending' | 'scored'
  response?: ScoreResponse
}

export default function LiveScoringPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [transactions, setTransactions] = useState<LiveTransaction[]>([])
  const [scoring, setScoring] = useState<string | null>(null)
  const [autoScore, setAutoScore] = useState(false)
  const [selectedTxn, setSelectedTxn] = useState<LiveTransaction | null>(null)
  const [filterDecision, setFilterDecision] = useState<string>('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Mock transaction templates for adding new transactions
  const transactionTemplates = [
    {
      txn_id: 'TXN_' + Date.now(),
      payer_vpa: 'user@upi',
      payee_vpa: 'merchant@upi',
      amount: 5000,
      currency: 'INR',
      txn_type: 'P2M',
      timestamp: new Date().toISOString(),
      device_id: 'device_123',
      client_ip: '192.168.1.1',
      merchant_category: 'retail',
    },
    {
      txn_id: 'TXN_' + Date.now() + 1,
      payer_vpa: 'customer@upi',
      payee_vpa: 'shop@upi',
      amount: 75000,
      currency: 'INR',
      txn_type: 'P2M',
      timestamp: new Date().toISOString(),
      device_id: 'device_456',
      client_ip: '10.0.0.1',
      merchant_category: 'electronics',
    },
    {
      txn_id: 'TXN_' + Date.now() + 2,
      payer_vpa: 'buyer@upi',
      payee_vpa: 'seller@upi',
      amount: 120000,
      currency: 'INR',
      txn_type: 'P2P',
      timestamp: new Date().toISOString(),
      device_id: 'device_789',
      client_ip: '172.16.0.1',
      merchant_category: 'peer_transfer',
    },
  ]

  const scoreTransaction = useCallback(async (txn: LiveTransaction) => {
    setScoring(txn.txn_id)
    setError(null)
    try {
      const response = await apiClient.scoreTransaction({
        transaction: txn,
      })

      setTransactions(prev => prev.map(t =>
        t.txn_id === txn.txn_id ? { ...t, status: 'scored', response: response } : t
      ))
    } catch (err: any) {
      setError(err.message || 'Scoring failed')
      console.error('Scoring failed:', err)
    } finally {
      setScoring(null)
    }
  }, [])

  const scoreBatch = useCallback(async () => {
    const pendingTxns = transactions.filter(t => t.status === 'pending')
    if (pendingTxns.length === 0) return

    setLoading(true)
    setError(null)
    try {
      const response = await apiClient.scoreBatch(
        pendingTxns.map(t => ({ transaction: t })),
        { includeExplanation: true, includeFeatures: false }
      )

      const resultsMap = new Map(response.results.map((r: any) => [r.txn_id, r]))
      setTransactions(prev => prev.map(t => {
        const result = resultsMap.get(t.txn_id) as ScoreResponse | undefined
        if (result) {
          return { ...t, status: 'scored' as const, response: result }
        }
        return t
      }))
    } catch (err: any) {
      setError(err.message || 'Batch scoring failed')
      console.error('Batch scoring failed:', err)
    } finally {
      setLoading(false)
    }
  }, [transactions])

  const addMockTransaction = () => {
    const template = transactionTemplates[Math.floor(Math.random() * transactionTemplates.length)]
    const newTxn: LiveTransaction = {
      ...template,
      txn_id: 'TXN_' + Date.now(),
      timestamp: new Date().toISOString(),
      status: 'pending',
    }
    setTransactions(prev => [newTxn, ...prev.slice(0, 49)])
  }

  const addHighRiskTransaction = () => {
    const newTxn: LiveTransaction = {
      txn_id: 'TXN_' + Date.now(),
      payer_vpa: 'suspicious@upi',
      payee_vpa: 'newmerchant@upi',
      amount: 199999,
      currency: 'INR',
      txn_type: 'P2M',
      timestamp: new Date().toISOString(),
      device_id: 'device_vpn',
      client_ip: '45.12.34.56',
      merchant_category: 'crypto',
      status: 'pending',
    }
    setTransactions(prev => [newTxn, ...prev.slice(0, 49)])
  }

  useEffect(() => {
    if (autoScore) {
      const interval = setInterval(() => {
        addMockTransaction()
        // Auto-score the latest pending
        setTransactions(prev => {
          const pending = prev.find(t => t.status === 'pending')
          if (pending) scoreTransaction(pending)
          return prev
        })
      }, 3000)
      return () => clearInterval(interval)
    }
  }, [autoScore, scoreTransaction])

  const filteredTransactions = transactions.filter(txn => {
    if (filterDecision === 'all') return true
    return txn.response?.decision === filterDecision
  })

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

  const getRiskLevelColor = (level: string) => {
    const colors = {
      LOW: 'text-success-600 bg-success-50',
      MEDIUM: 'text-warning-600 bg-warning-50',
      HIGH: 'text-danger-600 bg-danger-50',
      CRITICAL: 'text-danger-600 bg-danger-50 font-bold',
    }
    return colors[level as keyof typeof colors] || colors.LOW
  }

  const handleExport = () => {
    const scoredTxns = transactions.filter(t => t.status === 'scored' && t.response)
    if (scoredTxns.length === 0) return

    const csvContent = [
      ['Transaction ID', 'Payer VPA', 'Payee VPA', 'Amount', 'Type', 'Risk Score', 'Decision', 'Risk Level', 'Latency (ms)', 'Timestamp'].join(','),
      ...scoredTxns.map(t => [
        t.txn_id, t.payer_vpa, t.payee_vpa, t.amount, t.txn_type,
        Math.round(t.response!.risk_score * 100) + '%', t.response!.decision, t.response!.risk_level,
        t.response!.processing_time_ms, t.timestamp
      ].map(v => `"${v}"`).join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `scoring-results-${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
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
              <h1 className="text-2xl font-bold text-gray-900">Live Scoring</h1>
              <p className="text-gray-500 mt-1">Real-time fraud detection for UPI transactions</p>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoScore}
                  onChange={(e) => setAutoScore(e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  disabled={loading}
                />
                <span className="text-sm text-gray-700">Auto-score new transactions</span>
              </label>
              <button
                onClick={addMockTransaction}
                disabled={scoring !== null || loading}
                className="btn-primary"
              >
                <Shield className="h-4 w-4 mr-2" />
                Add Normal Transaction
              </button>
              <button
                onClick={addHighRiskTransaction}
                disabled={scoring !== null || loading}
                className="btn-danger"
              >
                <AlertTriangle className="h-4 w-4 mr-2" />
                Add High-Risk Transaction
              </button>
              <button
                onClick={scoreBatch}
                disabled={transactions.filter(t => t.status === 'pending').length === 0 || loading}
                className="btn-secondary"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Score All Pending
              </button>
              <button
                onClick={handleExport}
                disabled={transactions.filter(t => t.status === 'scored').length === 0 || loading}
                className="btn-primary"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Results
              </button>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-4 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center justify-between">
              <span className="text-danger-700">{error}</span>
              <button onClick={() => setError(null)} className="text-danger-500 hover:text-danger-700">
                <XCircle className="h-5 w-5" />
              </button>
            </div>
          )}

          {/* Stats Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Scored</p>
                  <p className="text-2xl font-bold text-gray-900">{transactions.filter(t => t.status === 'scored').length}</p>
                </div>
                <Shield className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Approved</p>
                  <p className="text-2xl font-bold text-success-600">
                    {transactions.filter(t => t.response?.decision === 'APPROVE').length}
                  </p>
                </div>
                <CheckCircle className="h-8 w-8 text-success-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Under Review</p>
                  <p className="text-2xl font-bold text-warning-600">
                    {transactions.filter(t => t.response?.decision === 'REVIEW').length}
                  </p>
                </div>
                <AlertCircle className="h-8 w-8 text-warning-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Declined</p>
                  <p className="text-2xl font-bold text-danger-600">
                    {transactions.filter(t => t.response?.decision === 'DECLINE').length}
                  </p>
                </div>
                <XCircle className="h-8 w-8 text-danger-600" />
              </div>
            </div>
          </div>

          {/* Filter Tabs */}
          <div className="flex gap-2 mb-4">
            {['all', 'APPROVE', 'REVIEW', 'DECLINE'].map((decision) => (
              <button
                key={decision}
                onClick={() => setFilterDecision(decision)}
                className={cn(
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  filterDecision === decision
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-100'
                )}
              >
                {decision === 'all' ? 'All' : decision}
              </button>
            ))}
          </div>

          {/* Transactions Table */}
          <div className="card">
            <div className="overflow-x-auto">
              <table className="table w-full">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Payer → Payee</th>
                    <th className="text-right">Amount</th>
                    <th>Type</th>
                    <th>Risk Score</th>
                    <th>Decision</th>
                    <th>Risk Level</th>
                    <th>Latency</th>
                    <th>Time</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTransactions.map((txn) => (
                    <tr key={txn.txn_id} className={txn.status === 'pending' ? 'bg-yellow-50' : ''}>
                      <td className="font-mono text-sm">{txn.txn_id}</td>
                      <td>
                        <div>
                          <p className="font-medium">{txn.payer_vpa}</p>
                          <p className="text-sm text-gray-500">→ {txn.payee_vpa}</p>
                        </div>
                      </td>
                      <td className="text-right font-medium">₹{txn.amount.toLocaleString()}</td>
                      <td>
                        <span className="badge badge-info">{txn.txn_type}</span>
                      </td>
                      <td>
                        {txn.status === 'pending' ? (
                          <div className="flex items-center gap-2">
                            <Loader2 className="h-4 w-4 animate-spin text-primary-600" />
                            <span className="text-sm text-gray-500">Scoring...</span>
                          </div>
                        ) : txn.response ? (
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className={cn('h-full rounded-full transition-all',
                                  txn.response.risk_score > 0.7 ? 'bg-danger-500' :
                                  txn.response.risk_score > 0.4 ? 'bg-warning-500' : 'bg-success-500')}
                                style={{ width: `${txn.response.risk_score * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium">{Math.round(txn.response.risk_score * 100)}%</span>
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td>
                        {txn.response ? getDecisionBadge(txn.response.decision) : (
                          <span className="badge badge-info animate-pulse">Pending</span>
                        )}
                      </td>
                      <td>
                        {txn.response ? (
                          <span className={cn('badge', getRiskLevelColor(txn.response.risk_level))}>
                            {txn.response.risk_level}
                          </span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="text-sm text-gray-500">
                        {txn.response ? `${txn.response.processing_time_ms}ms` : '-'}
                      </td>
                      <td className="text-sm text-gray-500">
                        {new Date(txn.timestamp).toLocaleTimeString()}
                      </td>
                      <td>
                        <div className="flex items-center gap-1">
                          {txn.response && (
                            <button
                              onClick={() => setSelectedTxn(txn)}
                              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-900 transition-colors"
                              aria-label="View details"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                          )}
                          {txn.status === 'pending' && scoring !== txn.txn_id && !loading && (
                            <button
                              onClick={() => scoreTransaction(txn)}
                              className="p-2 rounded-lg text-primary-600 hover:bg-primary-50 transition-colors"
                              aria-label="Score now"
                            >
                              <RefreshCw className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                  {filteredTransactions.length === 0 && !loading && (
                    <tr>
                      <td colSpan={10} className="text-center py-12 text-gray-500">
                        <Shield className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                        <p>No transactions found</p>
                        <p className="text-sm mt-1">Click "Add Transaction" to start scoring</p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>

      {/* Detail Modal */}
      {selectedTxn && selectedTxn.response && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 animate-fade-in">
          <div className="bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto animate-slide-up">
            <div className="p-6 border-b border-gray-100 flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900">Transaction Details</h2>
              <button
                onClick={() => setSelectedTxn(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6">
              {/* Summary */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="card p-4">
                  <p className="text-sm text-gray-500">Risk Score</p>
                  <p className="text-3xl font-bold text-gray-900">{Math.round(selectedTxn.response.risk_score * 100)}%</p>
                </div>
                <div className="card p-4">
                  <p className="text-sm text-gray-500">Decision</p>
                  <p className="text-xl font-bold">{getDecisionBadge(selectedTxn.response.decision)}</p>
                </div>
                <div className="card p-4">
                  <p className="text-sm text-gray-500">Risk Level</p>
                  <p className="text-xl font-bold">
                    <span className={cn('badge', getRiskLevelColor(selectedTxn.response.risk_level))}>
                      {selectedTxn.response.risk_level}
                    </span>
                  </p>
                </div>
                <div className="card p-4">
                  <p className="text-sm text-gray-500">Processing Time</p>
                  <p className="text-xl font-bold text-gray-900">{selectedTxn.response.processing_time_ms}ms</p>
                </div>
              </div>

              {/* Transaction Info */}
              <div className="card mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Transaction Information</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Transaction ID</p>
                    <p className="font-mono text-sm">{selectedTxn.txn_id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Payer VPA</p>
                    <p className="font-medium">{selectedTxn.payer_vpa}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Payee VPA</p>
                    <p className="font-medium">{selectedTxn.payee_vpa}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Amount</p>
                    <p className="font-bold text-gray-900">₹{selectedTxn.amount.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Type</p>
                    <p className="font-medium">{selectedTxn.txn_type}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Device ID</p>
                    <p className="font-mono text-sm">{selectedTxn.device_id || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Client IP</p>
                    <p className="font-mono text-sm">{selectedTxn.client_ip || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Merchant Category</p>
                    <p className="font-medium">{selectedTxn.merchant_category || 'N/A'}</p>
                  </div>
                </div>
              </div>

              {/* Top Features */}
              {selectedTxn.response.explanation?.top_features && (
                <div className="card mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Contributing Features</h3>
                  <div className="space-y-3">
                    {selectedTxn.response.explanation.top_features.map((feat: any, idx: number) => (
                      <div key={idx} className="flex items-center gap-4">
                        <span className="w-10 text-center text-sm font-medium text-gray-500">{idx + 1}</span>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-900">{feat.feature}</p>
                          <p className="text-xs text-gray-500">Value: {feat.value}</p>
                        </div>
                        <div className="w-32 h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary-600 rounded-full transition-all"
                            style={{ width: `${feat.importance * 100}%` }}
                          />
                        </div>
                        <span className="w-12 text-right text-sm font-medium text-gray-900">
                          {Math.round(feat.importance * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Rule Results */}
              {selectedTxn.response.explanation?.rule_results && (
                <div className="card mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Rule Engine Results</h3>
                  <div className="table-container">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Rule ID</th>
                          <th>Triggered</th>
                          <th>Weight</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedTxn.response.explanation.rule_results.map((rule: any, idx: number) => (
                          <tr key={idx}>
                            <td className="font-mono text-sm">{rule.rule_id}</td>
                            <td>
                              {rule.triggered ? (
                                <span className="badge badge-danger">Yes</span>
                              ) : (
                                <span className="badge badge-success">No</span>
                              )}
                            </td>
                            <td>{rule.weight}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Threat Intelligence */}
              {selectedTxn.response.explanation?.threat_intel && (
                <div className="card">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Threat Intelligence</h3>
                  <div className="space-y-3">
                    {selectedTxn.response.explanation.threat_intel.map((intel: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900 capitalize">{intel.source.replace('_', ' ')}</p>
                          <p className="text-sm text-gray-500">Risk Score: {Math.round(intel.risk_score * 100)}%</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className={cn('h-full rounded-full',
                                intel.risk_score > 0.7 ? 'bg-danger-500' :
                                intel.risk_score > 0.3 ? 'bg-warning-500' : 'bg-success-500')}
                              style={{ width: `${intel.risk_score * 100}%` }}
                            />
                          </div>
                          {intel.labels.length > 0 && (
                            <span className="badge badge-info">{intel.labels.join(', ')}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}