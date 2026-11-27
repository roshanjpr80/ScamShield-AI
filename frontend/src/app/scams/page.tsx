'use client'

import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, ChevronDown, Shield, AlertTriangle, BookOpen, Plus, Eye, Edit, Trash2, Download, Tag, Clock, Users, DollarSign, TrendingUp, ExternalLink, RefreshCw, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient } from '@/lib/api'

interface ScamPattern {
  id: string
  name: string
  description: string
  category: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  indicators: string[]
  examples: string[]
  detectionRules: string[]
  mitigation: string
  reportedCount: number
  lastSeen: string
  status: 'active' | 'monitoring' | 'archived'
  createdAt: string
  updatedAt: string
  fullDescription?: string
  variants?: string[]
  preventionTips?: string[]
}

interface FilterState {
  category: string
  severity: string
  status: string
  search: string
}

interface ScamListResponse {
  scams: ScamEntry[]
  total: number
}

interface ScamEntry {
  id: string
  name: string
  category: string
  description: string
  indicators: string[]
  severity: string
  reported_count: number
  last_seen: string
}

interface ScamDetail extends ScamEntry {
  full_description: string
  variants: string[]
  prevention_tips: string[]
}

const categories = ['all', 'job_scam', 'sms_phishing', 'upi_fraud', 'Social Engineering', 'QR Code Fraud', 'Impersonation', 'Financial Fraud', 'Employment Fraud', 'Lending Fraud', 'Account Takeover']

function mapApiScamToPattern(apiScam: ScamEntry | ScamDetail): ScamPattern {
  return {
    id: apiScam.id,
    name: apiScam.name,
    description: apiScam.description,
    category: apiScam.category,
    severity: apiScam.severity as 'low' | 'medium' | 'high' | 'critical',
    indicators: apiScam.indicators || [],
    examples: 'variants' in apiScam ? apiScam.variants : [],
    detectionRules: [],
    mitigation: 'prevention_tips' in apiScam ? apiScam.prevention_tips.join(' ') : '',
    reportedCount: apiScam.reported_count,
    lastSeen: apiScam.last_seen,
    status: 'active',
    createdAt: apiScam.last_seen,
    updatedAt: apiScam.last_seen,
    fullDescription: 'full_description' in apiScam ? apiScam.full_description : undefined,
    variants: 'variants' in apiScam ? apiScam.variants : undefined,
    preventionTips: 'prevention_tips' in apiScam ? apiScam.prevention_tips : undefined,
  }
}

export default function ScamLibraryPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [patterns, setPatterns] = useState<ScamPattern[]>([])
  const [selectedPattern, setSelectedPattern] = useState<ScamPattern | null>(null)
  const [filters, setFilters] = useState<FilterState>({
    category: 'all',
    severity: 'all',
    status: 'all',
    search: '',
  })
  const [showFilters, setShowFilters] = useState(false)
  const [showAddModal, setShowAddModal] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [totalCount, setTotalCount] = useState(0)

  const fetchScams = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiClient.listScams(50, 0, filters.category !== 'all' ? filters.category : undefined)
      const mappedPatterns = response.scams.map(mapApiScamToPattern)
      setPatterns(mappedPatterns)
      setTotalCount(response.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch scam patterns')
      console.error('Failed to fetch scams:', err)
    } finally {
      setLoading(false)
    }
  }, [filters.category])

  useEffect(() => {
    fetchScams()
  }, [fetchScams])

  const filteredPatterns = patterns.filter(pattern => {
    if (filters.severity !== 'all' && pattern.severity !== filters.severity) return false
    if (filters.status !== 'all' && pattern.status !== filters.status) return false
    if (filters.search) {
      const search = filters.search.toLowerCase()
      return (
        pattern.name.toLowerCase().includes(search) ||
        pattern.description.toLowerCase().includes(search) ||
        pattern.category.toLowerCase().includes(search) ||
        pattern.indicators.some(i => i.toLowerCase().includes(search))
      )
    }
    return true
  })

  const getSeverityBadge = (severity: string) => {
    const variants = {
      low: 'badge-success',
      medium: 'badge-warning',
      high: 'badge-danger',
      critical: 'badge-danger',
    }
    const icons = {
      low: <Shield className="h-3 w-3 mr-1" />,
      medium: <AlertTriangle className="h-3 w-3 mr-1" />,
      high: <AlertTriangle className="h-3 w-3 mr-1" />,
      critical: <AlertTriangle className="h-3 w-3 mr-1 text-red-600" />,
    }
    return (
      <span className={cn('badge font-medium', variants[severity as keyof typeof variants])}>
        {icons[severity as keyof typeof icons]}
        {severity.charAt(0).toUpperCase() + severity.slice(1)}
      </span>
    )
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      active: 'badge-danger',
      monitoring: 'badge-warning',
      archived: 'badge-info',
    }
    return (
      <span className={cn('badge', variants[status as keyof typeof variants])}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  const stats = {
    total: totalCount || patterns.length,
    active: patterns.filter(p => p.status === 'active').length,
    critical: patterns.filter(p => p.severity === 'critical').length,
    totalReports: patterns.reduce((sum, p) => sum + p.reportedCount, 0),
  }

  const handleRefresh = () => {
    fetchScams()
  }

  const handleExportAll = () => {
    const dataStr = JSON.stringify(patterns, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `scam_library_${new Date().toISOString().split('T')[0]}.json`
    a.click()
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
              <h1 className="text-2xl font-bold text-gray-900">Scam Library</h1>
              <p className="text-gray-500 mt-1">Comprehensive database of known scam patterns and fraud techniques</p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={cn('btn-secondary', showFilters && 'bg-primary-50 text-primary-700')}
              >
                <Filter className="h-4 w-4 mr-2" />
                Filters
              </button>
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="btn-secondary"
              >
                <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
                Refresh
              </button>
              <button
                onClick={handleExportAll}
                disabled={patterns.length === 0}
                className="btn-secondary"
              >
                <Download className="h-4 w-4 mr-2" />
                Export All
              </button>
              <button
                onClick={() => { setShowAddModal(true); setSelectedPattern(null) }}
                className="btn-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Pattern
              </button>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
              <div className="flex items-center gap-2 text-red-800">
                <AlertTriangle className="h-5 w-5" />
                <span>Error: {error}</span>
              </div>
              <button onClick={handleRefresh} className="btn-secondary btn-sm">Retry</button>
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Patterns</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <BookOpen className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Active Threats</p>
                  <p className="text-2xl font-bold text-danger-600">{stats.active}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-danger-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Critical Severity</p>
                  <p className="text-2xl font-bold text-danger-600">{stats.critical}</p>
                </div>
                <Shield className="h-8 w-8 text-danger-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Reports</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalReports.toLocaleString()}</p>
                </div>
                <Users className="h-8 w-8 text-primary-600" />
              </div>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="card mb-6 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4">
                <div>
                  <label className="label">Category</label>
                  <select
                    value={filters.category}
                    onChange={(e) => setFilters({ ...filters, category: e.target.value })}
                    className="input"
                  >
                    {categories.map(cat => (
                      <option key={cat} value={cat}>{cat === 'all' ? 'All Categories' : cat}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Severity</label>
                  <select
                    value={filters.severity}
                    onChange={(e) => setFilters({ ...filters, severity: e.target.value })}
                    className="input"
                  >
                    <option value="all">All Severities</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div>
                  <label className="label">Status</label>
                  <select
                    value={filters.status}
                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                    className="input"
                  >
                    <option value="all">All Statuses</option>
                    <option value="active">Active</option>
                    <option value="monitoring">Monitoring</option>
                    <option value="archived">Archived</option>
                  </select>
                </div>
                <div>
                  <label className="label">Search</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search patterns, indicators..."
                      value={filters.search}
                      onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                      className="input pl-10"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Patterns Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {loading ? (
              <>
                {[1, 2, 3, 4, 5, 6].map(i => (
                  <div key={i} className="card animate-pulse">
                    <div className="h-6 bg-gray-200 rounded w-3/4 mb-3" />
                    <div className="h-4 bg-gray-200 rounded w-full mb-2" />
                    <div className="h-4 bg-gray-200 rounded w-2/3 mb-2" />
                    <div className="h-4 bg-gray-200 rounded w-1/2 mb-3" />
                    <div className="flex gap-2">
                      <div className="h-5 bg-gray-200 rounded-full w-20" />
                      <div className="h-5 bg-gray-200 rounded-full w-24" />
                    </div>
                  </div>
                ))}
              </>
            ) : filteredPatterns.map((pattern) => (
              <div
                key={pattern.id}
                className={cn('card card-hover', selectedPattern?.id === pattern.id && 'ring-2 ring-primary-500')}
                onClick={() => setSelectedPattern(pattern)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 mb-1">{pattern.name}</h3>
                    <span className="badge badge-info text-xs">{pattern.category}</span>
                  </div>
                  {getSeverityBadge(pattern.severity)}
                </div>

                <p className="text-sm text-gray-600 mb-3 line-clamp-2">{pattern.description}</p>

                <div className="flex items-center gap-2 mb-3">
                  {getStatusBadge(pattern.status)}
                  <span className="text-xs text-gray-500">{pattern.reportedCount} reports</span>
                </div>

                <div className="flex flex-wrap gap-1 mb-3">
                  {pattern.indicators.slice(0, 3).map((indicator, idx) => (
                    <span key={idx} className="badge badge-info text-xs">{indicator}</span>
                  ))}
                  {pattern.indicators.length > 3 && (
                    <span className="badge badge-info text-xs">+{pattern.indicators.length - 3} more</span>
                  )}
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                  <span className="text-xs text-gray-500">Last seen: {new Date(pattern.lastSeen).toLocaleDateString()}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setSelectedPattern(pattern) }}
                    className="text-primary-600 hover:underline text-sm"
                  >
                    View Details
                  </button>
                </div>
              </div>
            ))}
            {!loading && filteredPatterns.length === 0 && (
              <div className="col-span-full text-center py-12">
                <Search className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                <p className="text-gray-500">No patterns found</p>
                <p className="text-sm text-gray-400 mt-1">Try adjusting your filters</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Detail Panel */}
      {selectedPattern && (
        <div className="fixed inset-0 z-50 flex items-end justify-end p-4 lg:p-0">
          <div className="bg-white rounded-t-xl lg:rounded-xl shadow-xl w-full lg:w-[600px] max-h-[90vh] lg:max-h-[calc(100vh-4rem)] overflow-y-auto animate-slide-in-right">
            <div className="p-4 lg:p-6 border-b border-gray-100 flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h2 className="text-lg lg:text-xl font-bold text-gray-900">{selectedPattern.name}</h2>
                  {getSeverityBadge(selectedPattern.severity)}
                </div>
                <p className="text-sm text-gray-500">{selectedPattern.category} • {getStatusBadge(selectedPattern.status)}</p>
              </div>
              <button
                onClick={() => setSelectedPattern(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 lg:p-6 space-y-6">
              {/* Description */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
                <p className="text-gray-700">{selectedPattern.fullDescription || selectedPattern.description}</p>
              </div>

              {/* Indicators */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Key Indicators</h3>
                <div className="space-y-2">
                  {selectedPattern.indicators.map((indicator, idx) => (
                    <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                      <span className="w-6 h-6 rounded-full bg-primary-100 text-primary-600 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <p className="text-sm text-gray-900">{indicator}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Examples */}
              {(selectedPattern.variants && selectedPattern.variants.length > 0) || (selectedPattern.examples && selectedPattern.examples.length > 0) ? (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Real-World Examples</h3>
                  <div className="space-y-2">
                    {(selectedPattern.variants || selectedPattern.examples).map((example, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 rounded-lg border-l-4 border-primary-500">
                        <p className="text-sm text-gray-700">&ldquo;{example}&rdquo;</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* Detection Rules */}
              {selectedPattern.detectionRules && selectedPattern.detectionRules.length > 0 && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Detection Rules</h3>
                  <div className="space-y-2">
                    {selectedPattern.detectionRules.map((rule, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                        <code className="text-sm font-mono text-gray-900">{rule}</code>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Mitigation */}
              {selectedPattern.mitigation && (
                <div className="p-4 bg-success-50 border border-success-200 rounded-lg">
                  <h3 className="font-semibold text-success-800 mb-2 flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Mitigation Advice
                  </h3>
                  <p className="text-success-700">{selectedPattern.preventionTips?.join(' ') || selectedPattern.mitigation}</p>
                </div>
              )}

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
                <div>
                  <p className="text-xs text-gray-500">Reports</p>
                  <p className="font-medium text-gray-900">{selectedPattern.reportedCount.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Last Seen</p>
                  <p className="font-medium text-gray-900">{new Date(selectedPattern.lastSeen).toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Created</p>
                  <p className="font-medium text-gray-900">{new Date(selectedPattern.createdAt).toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Updated</p>
                  <p className="font-medium text-gray-900">{new Date(selectedPattern.updatedAt).toLocaleDateString()}</p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3 pt-4 border-t border-gray-100">
                <button className="btn-primary flex-1">
                  <Edit className="h-4 w-4 mr-2" />
                  Edit Pattern
                </button>
                <button
                  onClick={() => {
                    const dataStr = JSON.stringify(selectedPattern, null, 2)
                    const blob = new Blob([dataStr], { type: 'application/json' })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `${selectedPattern.id}_${selectedPattern.name.replace(/\s+/g, '_')}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="btn-secondary flex-1"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </button>
                <button className="btn-danger flex-1">
                  <Trash2 className="h-4 w-4 mr-2" />
                  Archive
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {selectedPattern && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSelectedPattern(null)} />
      )}

      {/* Add/Edit Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 animate-fade-in">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto animate-slide-up">
            <div className="p-6 border-b border-gray-100 flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900">{selectedPattern ? 'Edit Pattern' : 'Add New Pattern'}</h2>
              <button
                onClick={() => { setShowAddModal(false); setSelectedPattern(null) }}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <AddEditPatternForm
              pattern={selectedPattern}
              onClose={() => { setShowAddModal(false); setSelectedPattern(null) }}
              onSave={async (pattern) => {
                try {
                  if (selectedPattern) {
                    // Update would need a PUT endpoint
                    await apiClient.addScam({ ...selectedPattern, ...pattern, updatedAt: new Date().toISOString() })
                    setPatterns(prev => prev.map(p => p.id === selectedPattern.id ? { ...p, ...pattern, updatedAt: new Date().toISOString() } : p))
                  } else {
                    const response = await apiClient.addScam(pattern)
                    const newPattern: ScamPattern = {
                      ...pattern,
                      id: response.id,
                      reportedCount: 0,
                      lastSeen: new Date().toISOString(),
                      createdAt: new Date().toISOString(),
                      updatedAt: new Date().toISOString(),
                    }
                    setPatterns(prev => [newPattern, ...prev])
                  }
                  setShowAddModal(false)
                  setSelectedPattern(null)
                } catch (err) {
                  console.error('Failed to save pattern:', err)
                  alert('Failed to save pattern')
                }
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function AddEditPatternForm({ pattern, onClose, onSave }: { pattern: ScamPattern | null; onClose: () => void; onSave: (pattern: ScamPattern) => void }) {
  const [formData, setFormData] = useState<{
    name: string
    description: string
    category: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    status: 'active' | 'monitoring' | 'archived'
    indicators: string
    examples: string
    detectionRules: string
    mitigation: string
  }>({
    name: pattern?.name || '',
    description: pattern?.description || '',
    category: pattern?.category || 'job_scam',
    severity: pattern?.severity || 'medium',
    status: pattern?.status || 'active',
    indicators: pattern?.indicators.join('\n') || '',
    examples: pattern?.examples.join('\n') || '',
    detectionRules: pattern?.detectionRules.join('\n') || '',
    mitigation: pattern?.mitigation || '',
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const basePattern = {
      ...formData,
      indicators: formData.indicators.split('\n').filter(i => i.trim()),
      examples: formData.examples.split('\n').filter(i => i.trim()),
      detectionRules: formData.detectionRules.split('\n').filter(i => i.trim()),
    }
    if (pattern) {
      onSave({ ...pattern, ...basePattern, updatedAt: new Date().toISOString() })
    } else {
      onSave({
        ...basePattern,
        id: 'SCAM_' + Date.now(),
        reportedCount: 0,
        lastSeen: new Date().toISOString(),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      })
    }
  }

  return (
    <form onSubmit={handleSubmit} className="p-6 space-y-6">
      <div>
        <label className="label">Pattern Name *</label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          className="input"
          placeholder="e.g., Fake UPI Refund Scam"
          required
        />
      </div>

      <div>
        <label className="label">Description *</label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          className="input min-h-[100px] resize-y"
          rows={3}
          required
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="label">Category *</label>
          <select
            value={formData.category}
            onChange={(e) => setFormData({ ...formData, category: e.target.value })}
            className="input"
            required
          >
            <option value="job_scam">Job Scam</option>
            <option value="sms_phishing">SMS Phishing</option>
            <option value="upi_fraud">UPI Fraud</option>
            <option value="Social Engineering">Social Engineering</option>
            <option value="QR Code Fraud">QR Code Fraud</option>
            <option value="Impersonation">Impersonation</option>
            <option value="Financial Fraud">Financial Fraud</option>
            <option value="Employment Fraud">Employment Fraud</option>
            <option value="Lending Fraud">Lending Fraud</option>
            <option value="Account Takeover">Account Takeover</option>
          </select>
        </div>
        <div>
          <label className="label">Severity *</label>
          <select
            value={formData.severity}
            onChange={(e) => setFormData({ ...formData, severity: e.target.value as 'low' | 'medium' | 'high' | 'critical' })}
            className="input"
            required
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div>
          <label className="label">Status *</label>
          <select
            value={formData.status}
            onChange={(e) => setFormData({ ...formData, status: e.target.value as 'active' | 'monitoring' | 'archived' })}
            className="input"
            required
          >
            <option value="active">Active</option>
            <option value="monitoring">Monitoring</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>

      <div>
        <label className="label">Key Indicators (one per line) *</label>
        <textarea
          value={formData.indicators}
          onChange={(e) => setFormData({ ...formData, indicators: e.target.value })}
          className="input min-h-[100px] resize-y font-mono text-sm"
          rows={4}
          placeholder="Incoming collect request from unknown VPA&#10;Message claiming refund/cashback pending&#10;Urgency pressure to approve quickly"
          required
        />
      </div>

      <div>
        <label className="label">Real-World Examples (one per line)</label>
        <textarea
          value={formData.examples}
          onChange={(e) => setFormData({ ...formData, examples: e.target.value })}
          className="input min-h-[80px] resize-y font-mono text-sm"
          rows={3}
          placeholder='User receives SMS: "Your refund of ₹5000 is pending..."&#10;Call from "bank support" asking to approve collect request...'
        />
      </div>

      <div>
        <label className="label">Detection Rules (one per line)</label>
        <textarea
          value={formData.detectionRules}
          onChange={(e) => setFormData({ ...formData, detectionRules: e.target.value })}
          className="input min-h-[80px] resize-y font-mono text-sm"
          rows={3}
          placeholder="VEL_001: High velocity collect requests&#10;AMT_002: Round amount collect requests&#10;DEV_003: New device + collect request"
        />
      </div>

      <div>
        <label className="label">Mitigation Advice *</label>
        <textarea
          value={formData.mitigation}
          onChange={(e) => setFormData({ ...formData, mitigation: e.target.value })}
          className="input min-h-[80px] resize-y"
          rows={3}
          required
        />
      </div>

      <div className="flex gap-3 justify-end pt-4 border-t border-gray-100">
        <button type="button" onClick={onClose} className="btn-secondary">
          Cancel
        </button>
        <button type="submit" className="btn-primary">
          {pattern ? 'Update Pattern' : 'Add Pattern'}
        </button>
      </div>
    </form>
  )
}