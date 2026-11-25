'use client'

import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, ChevronDown, ChevronUp, AlertTriangle, CheckCircle, XCircle, AlertCircle, Eye, MoreVertical, Flag, Download, Loader2, Calendar } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient } from '@/lib/api'

interface InvestigationCase {
  id: string
  txn_id: string
  payer_vpa: string
  payee_vpa: string
  amount: number
  risk_score: number
  decision: 'APPROVE' | 'REVIEW' | 'DECLINE'
  status: 'open' | 'investigating' | 'resolved' | 'escalated'
  priority: 'low' | 'medium' | 'high' | 'critical'
  assigned_to?: string
  created_at: string
  updated_at: string
  tags: string[]
  notes: string[]
}

interface FilterState {
  status: string
  priority: string
  decision: string
  dateRange: string
  search: string
}

export default function InvestigationsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [cases, setCases] = useState<InvestigationCase[]>([])
  const [selectedCase, setSelectedCase] = useState<InvestigationCase | null>(null)
  const [filters, setFilters] = useState<FilterState>({
    status: 'all',
    priority: 'all',
    decision: 'all',
    dateRange: '7d',
    search: '',
  })
  const [showFilters, setShowFilters] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchCases = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params: any = { limit: 100 }
      if (filters.status !== 'all') params.status = filters.status
      if (filters.priority !== 'all') params.priority = filters.priority
      if (filters.decision !== 'all') params.decision = filters.decision
      const response = await apiClient.listCases(params)
      // Backend returns { cases: [...] } or just [...]
      const data = Array.isArray(response) ? response : (response.cases || [])
      // Transform backend case format to frontend format
      const transformed = data.map((c: any) => ({
        id: c.case_id || c.id,
        txn_id: c.txn_id,
        payer_vpa: c.payer_vpa,
        payee_vpa: c.payee_vpa,
        amount: c.amount,
        risk_score: c.risk_score,
        decision: c.decision,
        status: c.status,
        priority: c.priority,
        assigned_to: c.assigned_to,
        created_at: c.created_at,
        updated_at: c.updated_at || c.created_at,
        tags: c.tags || [],
        notes: c.notes || [],
      }))
      setCases(transformed)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch cases')
      console.error('Failed to fetch cases:', err)
    } finally {
      setLoading(false)
    }
  }, [filters.status, filters.priority, filters.decision])

  useEffect(() => {
    fetchCases()
  }, [fetchCases])

  const filteredCases = cases.filter(case_ => {
    if (filters.search) {
      const search = filters.search.toLowerCase()
      return (
        case_.id.toLowerCase().includes(search) ||
        case_.txn_id.toLowerCase().includes(search) ||
        case_.payer_vpa.toLowerCase().includes(search) ||
        case_.payee_vpa.toLowerCase().includes(search)
      )
    }
    return true
  })

  const getStatusBadge = (status: string) => {
    const variants = {
      open: 'badge-info',
      investigating: 'badge-warning',
      resolved: 'badge-success',
      escalated: 'badge-danger',
    }
    return (
      <span className={cn('badge', variants[status as keyof typeof variants])}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  const getPriorityBadge = (priority: string) => {
    const variants = {
      low: 'badge-success',
      medium: 'badge-info',
      high: 'badge-warning',
      critical: 'badge-danger',
    }
    return (
      <span className={cn('badge', variants[priority as keyof typeof variants])}>
        {priority.charAt(0).toUpperCase() + priority.slice(1)}
      </span>
    )
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

  const updateCaseStatus = async (caseId: string, newStatus: InvestigationCase['status']) => {
    setLoading(true)
    try {
      await apiClient.updateCase(caseId, { status: newStatus })
      setCases(prev => prev.map(c => c.id === caseId ? { ...c, status: newStatus, updated_at: new Date().toISOString() } : c))
      if (selectedCase?.id === caseId) {
        setSelectedCase(prev => prev ? { ...prev, status: newStatus, updated_at: new Date().toISOString() } : null)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to update case status')
      console.error('Failed to update case status:', err)
    } finally {
      setLoading(false)
    }
  }

  const assignCase = async (caseId: string, assignee: string) => {
    try {
      await apiClient.updateCase(caseId, { assigned_to: assignee })
      setCases(prev => prev.map(c => c.id === caseId ? { ...c, assigned_to: assignee, updated_at: new Date().toISOString() } : c))
      if (selectedCase?.id === caseId) {
        setSelectedCase(prev => prev ? { ...prev, assigned_to: assignee, updated_at: new Date().toISOString() } : null)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to assign case')
      console.error('Failed to assign case:', err)
    }
  }

  const addNote = async (caseId: string, note: string) => {
    try {
      await apiClient.addCaseNote(caseId, { content: note, type: 'investigation' })
      setCases(prev => prev.map(c => c.id === caseId ? { ...c, notes: [...c.notes, note], updated_at: new Date().toISOString() } : c))
      if (selectedCase?.id === caseId) {
        setSelectedCase(prev => prev ? { ...prev, notes: [...prev.notes, note], updated_at: new Date().toISOString() } : null)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to add note')
      console.error('Failed to add note:', err)
    }
  }

  const stats = {
    open: cases.filter(c => c.status === 'open').length,
    investigating: cases.filter(c => c.status === 'investigating').length,
    resolved: cases.filter(c => c.status === 'resolved').length,
    escalated: cases.filter(c => c.status === 'escalated').length,
    critical: cases.filter(c => c.priority === 'critical').length,
    high: cases.filter(c => c.priority === 'high').length,
  }

  const handleExport = () => {
    const csvContent = [
      ['Case ID', 'Transaction ID', 'Payer VPA', 'Payee VPA', 'Amount', 'Risk Score', 'Decision', 'Status', 'Priority', 'Assigned To', 'Created At', 'Tags'].join(','),
      ...cases.map(c => [
        c.id, c.txn_id, c.payer_vpa, c.payee_vpa, c.amount, c.risk_score, c.decision, c.status, c.priority, c.assigned_to || '', c.created_at, c.tags.join(';')
      ].map(v => `"${v}"`).join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `cases-${new Date().toISOString().split('T')[0]}.csv`
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
              <h1 className="text-2xl font-bold text-gray-900">Investigations</h1>
              <p className="text-gray-500 mt-1">Manage and track fraud investigation cases</p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={cn('btn-secondary', showFilters && 'bg-primary-50 text-primary-700')}
                disabled={loading}
              >
                <Filter className="h-4 w-4 mr-2" />
                Filters
                <ChevronDown className="h-4 w-4 ml-1" />
              </button>
              <button
                onClick={handleExport}
                className="btn-primary"
                disabled={loading || cases.length === 0}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
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

          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Open</p>
                  <p className="text-2xl font-bold text-primary-600">{stats.open}</p>
                </div>
                <AlertCircle className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Investigating</p>
                  <p className="text-2xl font-bold text-warning-600">{stats.investigating}</p>
                </div>
                <Search className="h-8 w-8 text-warning-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Resolved</p>
                  <p className="text-2xl font-bold text-success-600">{stats.resolved}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-success-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Escalated</p>
                  <p className="text-2xl font-bold text-danger-600">{stats.escalated}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-danger-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Critical Priority</p>
                  <p className="text-2xl font-bold text-danger-600">{stats.critical}</p>
                </div>
                <Flag className="h-8 w-8 text-danger-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">High Priority</p>
                  <p className="text-2xl font-bold text-warning-600">{stats.high}</p>
                </div>
                <Flag className="h-8 w-8 text-warning-600" />
              </div>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="card mb-6 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4 p-4">
                <div>
                  <label className="label">Status</label>
                  <select
                    value={filters.status}
                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                    className="input"
                    disabled={loading}
                  >
                    <option value="all">All Statuses</option>
                    <option value="open">Open</option>
                    <option value="investigating">Investigating</option>
                    <option value="resolved">Resolved</option>
                    <option value="escalated">Escalated</option>
                  </select>
                </div>
                <div>
                  <label className="label">Priority</label>
                  <select
                    value={filters.priority}
                    onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
                    className="input"
                    disabled={loading}
                  >
                    <option value="all">All Priorities</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div>
                  <label className="label">Decision</label>
                  <select
                    value={filters.decision}
                    onChange={(e) => setFilters({ ...filters, decision: e.target.value })}
                    className="input"
                    disabled={loading}
                  >
                    <option value="all">All Decisions</option>
                    <option value="APPROVE">Approve</option>
                    <option value="REVIEW">Review</option>
                    <option value="DECLINE">Decline</option>
                  </select>
                </div>
                <div>
                  <label className="label">Date Range</label>
                  <select
                    value={filters.dateRange}
                    onChange={(e) => setFilters({ ...filters, dateRange: e.target.value })}
                    className="input"
                    disabled={loading}
                  >
                    <option value="1d">Last 24 hours</option>
                    <option value="7d">Last 7 days</option>
                    <option value="30d">Last 30 days</option>
                    <option value="90d">Last 90 days</option>
                  </select>
                </div>
                <div>
                  <label className="label">Search</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Case ID, TXN ID, VPA..."
                      value={filters.search}
                      onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                      className="input pl-10"
                      disabled={loading}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Cases Table */}
          <div className="card">
            <div className="overflow-x-auto">
              <table className="table w-full">
                <thead>
                  <tr>
                    <th className="w-8"></th>
                    <th>Case ID</th>
                    <th>Transaction</th>
                    <th>Amount</th>
                    <th>Risk Score</th>
                    <th>Decision</th>
                    <th>Status</th>
                    <th>Priority</th>
                    <th>Assigned To</th>
                    <th>Created</th>
                    <th className="w-32">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredCases.map((case_) => (
                    <tr key={case_.id} className={selectedCase?.id === case_.id ? 'bg-primary-50' : ''}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedCase?.id === case_.id}
                          onChange={() => setSelectedCase(selectedCase?.id === case_.id ? null : case_)}
                          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                      </td>
                      <td className="font-mono text-sm font-medium">{case_.id}</td>
                      <td>
                        <div>
                          <p className="font-medium">{case_.txn_id}</p>
                          <p className="text-sm text-gray-500">{case_.payer_vpa} → {case_.payee_vpa}</p>
                        </div>
                      </td>
                      <td className="font-medium">₹{(case_.amount ?? 0).toLocaleString()}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className={cn('h-full rounded-full transition-all',
                                case_.risk_score > 0.7 ? 'bg-danger-500' :
                                case_.risk_score > 0.4 ? 'bg-warning-500' : 'bg-success-500')}
                              style={{ width: `${case_.risk_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium">{Math.round(case_.risk_score * 100)}%</span>
                        </div>
                      </td>
                      <td>{getDecisionBadge(case_.decision)}</td>
                      <td>{getStatusBadge(case_.status)}</td>
                      <td>{getPriorityBadge(case_.priority)}</td>
                      <td className="text-sm">
                        {case_.assigned_to ? (
                          <span className="font-medium">{case_.assigned_to}</span>
                        ) : (
                          <span className="text-gray-400">Unassigned</span>
                        )}
                      </td>
                      <td className="text-sm text-gray-500">
                        {new Date(case_.created_at).toLocaleDateString()}
                      </td>
                      <td>
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => setSelectedCase(case_)}
                            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-900 transition-colors"
                            aria-label="View details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <div className="relative group">
                            <button
                              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
                              aria-label="More actions"
                            >
                              <MoreVertical className="h-4 w-4" />
                            </button>
                            <div className="absolute right-0 mt-1 w-40 bg-white rounded-lg shadow-lg border border-gray-200 py-1 hidden group-hover:block z-10">
                              <button
                                onClick={() => updateCaseStatus(case_.id, 'investigating')}
                                disabled={case_.status !== 'open' || loading}
                                className="w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                              >
                                Start Investigation
                              </button>
                              <button
                                onClick={() => updateCaseStatus(case_.id, 'resolved')}
                                disabled={case_.status === 'resolved' || loading}
                                className="w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                              >
                                Mark Resolved
                              </button>
                              <button
                                onClick={() => updateCaseStatus(case_.id, 'escalated')}
                                disabled={case_.status === 'escalated' || loading}
                                className="w-full px-4 py-2 text-sm text-danger-600 hover:bg-gray-50 disabled:opacity-50"
                              >
                                Escalate
                              </button>
                              <hr className="my-1 border-gray-100" />
                              <button
                                onClick={() => assignCase(case_.id, 'current_user')}
                                className="w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                              >
                                Assign to me
                              </button>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {filteredCases.length === 0 && !loading && (
                    <tr>
                      <td colSpan={11} className="text-center py-12 text-gray-500">
                        <Search className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                        <p>No cases found</p>
                        <p className="text-sm mt-1">Try adjusting your filters</p>
                      </td>
                    </tr>
                  )}
                  {loading && (
                    <tr>
                      <td colSpan={11} className="text-center py-12 text-gray-500">
                        <Loader2 className="h-8 w-8 mx-auto animate-spin text-primary-600 mb-2" />
                        <p>Loading cases...</p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>

      {/* Case Detail Panel */}
      {selectedCase && (
        <div className="fixed inset-0 z-50 flex items-end justify-end p-4 lg:p-0">
          <div className="bg-white rounded-t-xl lg:rounded-xl shadow-xl w-full lg:w-96 max-h-[90vh] lg:max-h-[calc(100vh-4rem)] overflow-y-auto animate-slide-in-right">
            <div className="p-4 lg:p-6 border-b border-gray-100 flex items-center justify-between">
              <h2 className="text-lg lg:text-xl font-bold text-gray-900">Case Details</h2>
              <button
                onClick={() => setSelectedCase(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 lg:p-6 space-y-6">
              {/* Header Info */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <p className="font-mono text-sm text-gray-500">{selectedCase.id}</p>
                  <p className="text-sm text-gray-500 mt-1">{selectedCase.txn_id}</p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  {getStatusBadge(selectedCase.status)}
                  {getPriorityBadge(selectedCase.priority)}
                </div>
              </div>

              {/* Transaction Details */}
              <div className="card p-4">
                <h3 className="font-semibold text-gray-900 mb-3">Transaction Details</h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-500">Payer VPA</p>
                    <p className="font-medium">{selectedCase.payer_vpa}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Payee VPA</p>
                    <p className="font-medium">{selectedCase.payee_vpa}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Amount</p>
                    <p className="font-bold text-gray-900">₹{(selectedCase.amount ?? 0).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Risk Score</p>
                    <p className="font-bold text-gray-900">{Math.round(selectedCase.risk_score * 100)}%</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Decision</p>
                    <p className="font-medium">{getDecisionBadge(selectedCase.decision)}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Created</p>
                    <p className="font-medium">{new Date(selectedCase.created_at).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Updated</p>
                    <p className="font-medium">{new Date(selectedCase.updated_at).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Assigned To</p>
                    <p className="font-medium">{selectedCase.assigned_to || 'Unassigned'}</p>
                  </div>
                </div>
              </div>

              {/* Tags */}
              {selectedCase.tags.length > 0 && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Tags</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedCase.tags.map((tag, idx) => (
                      <span key={idx} className="badge badge-info">{tag}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Status Actions */}
              <div className="card p-4">
                <h3 className="font-semibold text-gray-900 mb-3">Actions</h3>
                <div className="space-y-2">
                  <button
                    onClick={() => updateCaseStatus(selectedCase.id, 'investigating')}
                    disabled={selectedCase.status !== 'open' || loading}
                    className={cn('btn-primary w-full justify-start', selectedCase.status !== 'open' && 'opacity-50')}
                  >
                    <Search className="h-4 w-4 mr-2" />
                    Start Investigation
                  </button>
                  <button
                    onClick={() => updateCaseStatus(selectedCase.id, 'resolved')}
                    disabled={selectedCase.status === 'resolved' || loading}
                    className={cn('btn-success w-full justify-start', selectedCase.status === 'resolved' && 'opacity-50')}
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Mark Resolved
                  </button>
                  <button
                    onClick={() => updateCaseStatus(selectedCase.id, 'escalated')}
                    disabled={selectedCase.status === 'escalated' || loading}
                    className={cn('btn-danger w-full justify-start', selectedCase.status === 'escalated' && 'opacity-50')}
                  >
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    Escalate
                  </button>
                </div>
              </div>

              {/* Notes */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Investigation Notes</h3>
                <div className="space-y-3 mb-4">
                  {selectedCase.notes.length > 0 ? (
                    selectedCase.notes.map((note, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                        <p className="text-sm text-gray-900">{note}</p>
                        <p className="text-xs text-gray-400 mt-1">Added by analyst</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-gray-500 text-sm">No notes yet</p>
                  )}
                </div>
                <AddNoteForm caseId={selectedCase.id} onAddNote={addNote} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {selectedCase && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSelectedCase(null)} />
      )}
    </div>
  )
}

function AddNoteForm({ caseId, onAddNote }: { caseId: string; onAddNote: (caseId: string, note: string) => void }) {
  const [note, setNote] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (note.trim()) {
      onAddNote(caseId, note.trim())
      setNote('')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card p-4">
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Add investigation note..."
        className="input min-h-[80px] resize-y mb-2"
        rows={3}
      />
      <button type="submit" disabled={!note.trim()} className="btn-primary w-full sm:w-auto">
        Add Note
      </button>
    </form>
  )
}