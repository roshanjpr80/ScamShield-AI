'use client'

import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, ChevronDown, Bell, BellOff, AlertTriangle, CheckCircle, XCircle, Clock, User, Mail, Phone, Zap, Settings, Download, Upload, Plus, Edit, Trash2, Eye, MoreVertical, Flag, Shield, RotateCcw, History, BarChart2, Calendar, RefreshCw, Link2, ExternalLink, Loader2, Save, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient } from '@/lib/api'

interface AlertRule {
  id: string
  name: string
  description: string
  condition: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  enabled: boolean
  created_at: string
  updated_at: string
}

interface Alert {
  id: string
  title: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'active' | 'acknowledged' | 'resolved' | 'silenced'
  source: string
  rule_id: string | null
  created_at: string
  updated_at: string
  acknowledged_at: string | null
  resolved_at: string | null
  metadata: Record<string, any>
}

interface NotificationChannel {
  id: string
  name: string
  type: 'email' | 'slack' | 'pagerduty' | 'webhook'
  config: Record<string, any>
  enabled: boolean
  created_at: string
  updated_at: string
  last_tested?: string
}

interface FilterState {
  severity: string
  status: string
  search: string
}

interface AlertListResponse {
  alerts: Alert[]
  total: number
}

export default function AlertsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [activeTab, setActiveTab] = useState<'alerts' | 'rules' | 'channels'>('alerts')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [rules, setRules] = useState<AlertRule[]>([])
  const [channels, setChannels] = useState<NotificationChannel[]>([])
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)
  const [selectedRule, setSelectedRule] = useState<AlertRule | null>(null)
  const [filters, setFilters] = useState<FilterState>({
    severity: 'all',
    status: 'all',
    search: '',
  })
  const [showFilters, setShowFilters] = useState(false)
  const [showCreateRuleModal, setShowCreateRuleModal] = useState(false)
  const [showCreateChannelModal, setShowCreateChannelModal] = useState(false)
  const [loading, setLoading] = useState<Record<string, boolean>>({
    alerts: false,
    rules: false,
    channels: false,
  })
  const [error, setError] = useState<string | null>(null)

  const statuses = ['all', 'active', 'acknowledged', 'resolved', 'silenced']
  const severities = ['all', 'critical', 'high', 'medium', 'low']

  // Fetch alerts from API
  const fetchAlerts = useCallback(async () => {
    setLoading(prev => ({ ...prev, alerts: true }))
    setError(null)
    try {
      const response = await apiClient.listAlerts({
        severity: filters.severity === 'all' ? undefined : filters.severity,
        status: filters.status === 'all' ? undefined : filters.status,
        limit: 50,
        offset: 0,
      })
      setAlerts(response.alerts || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch alerts')
      console.error('Failed to fetch alerts:', err)
    } finally {
      setLoading(prev => ({ ...prev, alerts: false }))
    }
  }, [filters.severity, filters.status])

  // Fetch rules from API
  const fetchRules = useCallback(async () => {
    setLoading(prev => ({ ...prev, rules: true }))
    setError(null)
    try {
      const response = await apiClient.listAlertRules()
      setRules(response.rules || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch rules')
      console.error('Failed to fetch rules:', err)
    } finally {
      setLoading(prev => ({ ...prev, rules: false }))
    }
  }, [])

  // Fetch channels from API
  const fetchChannels = useCallback(async () => {
    setLoading(prev => ({ ...prev, channels: true }))
    setError(null)
    try {
      const response = await apiClient.listNotificationChannels()
      setChannels(response.channels || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch channels')
      console.error('Failed to fetch channels:', err)
    } finally {
      setLoading(prev => ({ ...prev, channels: false }))
    }
  }, [])

  // Load data when tab changes
  useEffect(() => {
    if (activeTab === 'alerts') {
      fetchAlerts()
    } else if (activeTab === 'rules') {
      fetchRules()
    } else if (activeTab === 'channels') {
      fetchChannels()
    }
  }, [activeTab, fetchAlerts, fetchRules, fetchChannels])

  // Also fetch alerts when filters change
  useEffect(() => {
    if (activeTab === 'alerts') {
      fetchAlerts()
    }
  }, [filters.severity, filters.status, fetchAlerts])

  const getSeverityBadge = (severity: string) => {
    const variants = {
      critical: 'badge-danger',
      high: 'badge-danger',
      medium: 'badge-warning',
      low: 'badge-info',
    }
    const icons = {
      critical: <AlertTriangle className="h-3 w-3 mr-1 text-red-600" />,
      high: <AlertTriangle className="h-3 w-3 mr-1" />,
      medium: <Flag className="h-3 w-3 mr-1" />,
      low: <Bell className="h-3 w-3 mr-1" />,
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
      acknowledged: 'badge-warning',
      resolved: 'badge-success',
      silenced: 'badge-info',
    }
    const icons = {
      active: <Zap className="h-3 w-3 mr-1" />,
      acknowledged: <CheckCircle className="h-3 w-3 mr-1" />,
      resolved: <CheckCircle className="h-3 w-3 mr-1" />,
      silenced: <BellOff className="h-3 w-3 mr-1" />,
    }
    return (
      <span className={cn('badge font-medium', variants[status as keyof typeof variants])}>
        {icons[status as keyof typeof icons]}
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  const getChannelStatusBadge = (enabled: boolean) => {
    return (
      <span className={cn('badge', enabled ? 'badge-success' : 'badge-info')}>
        {enabled ? 'Enabled' : 'Disabled'}
      </span>
    )
  }

  const stats = {
    active: alerts.filter(a => a.status === 'active').length,
    acknowledged: alerts.filter(a => a.status === 'acknowledged').length,
    resolved: alerts.filter(a => a.status === 'resolved').length,
    critical: alerts.filter(a => a.severity === 'critical' && a.status !== 'resolved').length,
  }

  // Handler functions
  const handleAcknowledge = async (alertId: string) => {
    try {
      await apiClient.acknowledgeAlert(alertId)
      fetchAlerts()
      if (selectedAlert?.id === alertId) {
        setSelectedAlert(prev => prev ? { ...prev, status: 'acknowledged', acknowledged_at: new Date().toISOString() } : null)
      }
    } catch (err) {
      console.error('Failed to acknowledge alert:', err)
      alert('Failed to acknowledge alert')
    }
  }

  const handleResolve = async (alertId: string) => {
    try {
      await apiClient.resolveAlert(alertId)
      fetchAlerts()
      if (selectedAlert?.id === alertId) {
        setSelectedAlert(prev => prev ? { ...prev, status: 'resolved', resolved_at: new Date().toISOString() } : null)
      }
    } catch (err) {
      console.error('Failed to resolve alert:', err)
      alert('Failed to resolve alert')
    }
  }

  const handleSilence = async (alertId: string) => {
    try {
      await apiClient.silenceAlert(alertId, 60) // Default 60 minutes
      fetchAlerts()
      if (selectedAlert?.id === alertId) {
        setSelectedAlert(prev => prev ? { ...prev, status: 'silenced' } : null)
      }
    } catch (err) {
      console.error('Failed to silence alert:', err)
      alert('Failed to silence alert')
    }
  }

  const handleTestRule = async (ruleId: string) => {
    try {
      const result = await apiClient.testAlertRule(ruleId)
      alert(result.message || 'Rule test completed')
      fetchRules()
    } catch (err) {
      console.error('Failed to test rule:', err)
      alert('Failed to test rule')
    }
  }

  const handleToggleRule = async (ruleId: string) => {
    const rule = rules.find(r => r.id === ruleId)
    if (!rule) return
    try {
      await apiClient.toggleAlertRule(ruleId, !rule.enabled)
      fetchRules()
    } catch (err) {
      console.error('Failed to toggle rule:', err)
      alert('Failed to toggle rule')
    }
  }

  const handleDeleteRule = async (ruleId: string) => {
    if (!confirm('Are you sure you want to delete this rule?')) return
    try {
      await apiClient.deleteAlertRule(ruleId)
      fetchRules()
      if (selectedRule?.id === ruleId) {
        setSelectedRule(null)
      }
    } catch (err) {
      console.error('Failed to delete rule:', err)
      alert('Failed to delete rule')
    }
  }

  const handleTestChannel = async (channelId: string) => {
    try {
      const result = await apiClient.testNotificationChannel(channelId)
      alert(result.message || 'Channel test completed')
      fetchChannels()
    } catch (err) {
      console.error('Failed to test channel:', err)
      alert('Failed to test channel')
    }
  }

  const handleToggleChannel = async (channelId: string) => {
    const channel = channels.find(c => c.id === channelId)
    if (!channel) return
    try {
      await apiClient.toggleNotificationChannel(channelId, !channel.enabled)
      fetchChannels()
    } catch (err) {
      console.error('Failed to toggle channel:', err)
      alert('Failed to toggle channel')
    }
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
              <h1 className="text-2xl font-bold text-gray-900">Alert Management</h1>
              <p className="text-gray-500 mt-1">Monitor, manage, and respond to fraud detection alerts</p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={cn('btn-secondary', showFilters && 'bg-primary-50 text-primary-700')}
              >
                <Filter className="h-4 w-4 mr-2" />
                Filters
              </button>
              {(activeTab === 'rules' || activeTab === 'channels') && (
                <button className="btn-primary" onClick={() => {
                  if (activeTab === 'rules') {
                    setShowCreateRuleModal(true)
                  } else {
                    setShowCreateChannelModal(true)
                  }
                }}>
                  <Plus className="h-4 w-4 mr-2" />
                  {activeTab === 'rules' ? 'Create Rule' : 'Create Channel'}
                </button>
              )}
              {activeTab === 'rules' && showCreateRuleModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 animate-fade-in">
                  <div className="bg-white rounded-xl shadow-xl w-full max-w-md max-h-[90vh] overflow-y-auto animate-slide-up">
                    <div className="p-6 border-b border-gray-100 flex items-center justify-between">
                      <h2 className="text-xl font-bold text-gray-900">Create New Alert Rule</h2>
                      <button
                        onClick={() => { setShowCreateRuleModal(false) }}
                        className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
                      >
                        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <div className="p-6 space-y-6">
                      <form
                        onSubmit={(e) => {
                          e.preventDefault()
                          const formData = new FormData(e.target as HTMLFormElement)
                          const ruleData: any = {}
                          ;(formData as any).forEach((value: any, key: any) => {
                            ruleData[key] = value
                          })
                          apiClient.createAlertRule(ruleData).then(result => {
                            setShowCreateRuleModal(false)
                            fetchRules()
                            alert('Rule created successfully')
                          }).catch(err => {
                            console.error('Failed to create rule:', err)
                            alert('Failed to create rule')
                          })
                        }}
                        className="space-y-6"
                      >
                        <div>
                          <label className="label">Rule Name *</label>
                          <input
                            type="text"
                            required
                            className="input"
                            placeholder="e.g., High Velocity Collect Requests"
                          />
                        </div>
                        <div>
                          <label className="label">Description *</label>
                          <textarea
                            required
                            className="input min-h-[100px] resize-y"
                            rows={3}
                            placeholder="Description of when this rule should trigger"
                          />
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div>
                            <label className="label">Severity *</label>
                            <select className="input" required>
                              <option value="low">Low</option>
                              <option value="medium">Medium</option>
                              <option value="high">High</option>
                              <option value="critical">Critical</option>
                            </select>
                          </div>
                          <div>
                            <label className="label">Condition *</label>
                            <textarea
                              required
                              className="input min-h-[80px] resize-y font-mono text-sm"
                              rows={3}
                              placeholder="VEL_001: High velocity collect requests"
                            />
                          </div>
                          <div>
                            <label className="label">Enabled *</label>
                            <select className="input" required>
                              <option value="true">Enabled</option>
                              <option value="false">Disabled</option>
                            </select>
                          </div>
                        </div>
                        <div className="flex gap-3 justify-end pt-4 border-t border-gray-100">
                          <button type="button" onClick={() => { setShowCreateRuleModal(false) }} className="btn-secondary">Cancel</button>
                          <button type="submit" className="btn-primary">Create Rule</button>
                        </div>
                      </form>
                    </div>
                  </div>
                </div>
              )}
              {activeTab === 'channels' && showCreateChannelModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 animate-fade-in">
                  <div className="bg-white rounded-xl shadow-xl w-full max-w-md max-h-[90vh] overflow-y-auto animate-slide-up">
                    <div className="p-6 border-b border-gray-100 flex items-center justify-between">
                      <h2 className="text-xl font-bold text-gray-900">Create New Notification Channel</h2>
                      <button
                        onClick={() => { setShowCreateChannelModal(false) }}
                        className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
                      >
                        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <div className="p-6 space-y-6">
                      <form
                        onSubmit={(e) => {
                          e.preventDefault()
                          const formData = new FormData(e.target as HTMLFormElement)
                          const channelData: any = {}
                          ;(formData as any).forEach((value: any, key: any) => {
                            channelData[key] = value
                          })
                          apiClient.createNotificationChannel(channelData).then(result => {
                            setShowCreateChannelModal(false)
                            fetchChannels()
                            alert('Channel created successfully')
                          }).catch(err => {
                            console.error('Failed to create channel:', err)
                            alert('Failed to create channel')
                          })
                        }}
                        className="space-y-6"
                      >
                        <div>
                          <label className="label">Channel Name *</label>
                          <input
                            type="text"
                            required
                            className="input"
                            placeholder="e.g., Email Alerts"
                          />
                        </div>
                        <div>
                          <label className="label">Type *</label>
                          <select required className="input">
                            <option value="email">Email</option>
                            <option value="slack">Slack</option>
                            <option value="pagerduty">PagerDuty</option>
                            <option value="webhook">Webhook</option>
                          </select>
                        </div>
                        <div>
                          <label className="label">Config *</label>
                          <textarea
                            required
                            className="input min-h-[100px] resize-y"
                            rows={3}
                            placeholder="Email address or webhook URL"
                          />
                        </div>
                        <div className="flex gap-3 justify-end pt-4 border-t border-gray-100">
                          <button type="button" onClick={() => { setShowCreateChannelModal(false) }} className="btn-secondary">Cancel</button>
                          <button type="submit" className="btn-primary">Create Channel</button>
                        </div>
                      </form>
                    </div>
                  </div>
                </div>
              )}
              <button className="btn-secondary" onClick={() => {
                if (activeTab === 'alerts') fetchAlerts()
                else if (activeTab === 'rules') fetchRules()
                else fetchChannels()
              }} disabled={loading[activeTab]}>
                <RefreshCw className={cn('h-4 w-4 mr-2', loading[activeTab] && 'animate-spin')} />
                Refresh
              </button>
            </div>
          </div>

          {/* Stats - Only show for Alerts tab */}
          {activeTab === 'alerts' && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Active</p>
                    <p className="text-2xl font-bold text-danger-600">{stats.active}</p>
                  </div>
                  <Zap className="h-8 w-8 text-danger-600" />
                </div>
              </div>
              <div className="card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Acknowledged</p>
                    <p className="text-2xl font-bold text-warning-600">{stats.acknowledged}</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-warning-600" />
                </div>
              </div>
              <div className="card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Resolved (24h)</p>
                    <p className="text-2xl font-bold text-success-600">{stats.resolved}</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-success-600" />
                </div>
              </div>
              <div className="card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Critical Active</p>
                    <p className="text-2xl font-bold text-danger-600">{stats.critical}</p>
                  </div>
                  <AlertTriangle className="h-8 w-8 text-danger-600" />
                </div>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="border-b border-gray-200 mb-6">
            <nav className="flex gap-1" aria-label="Alert management tabs">
              {[
                { id: 'alerts', label: 'Alerts', icon: <Bell className="h-4 w-4" /> },
                { id: 'rules', label: 'Rules', icon: <Settings className="h-4 w-4" /> },
                { id: 'channels', label: 'Channels', icon: <Mail className="h-4 w-4" /> },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id as any);
                    setSelectedAlert(null);
                    setSelectedRule(null)
                  }}
                  className={cn(
                    'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  )}
                >
                  {tab.icon}
                  {tab.label}
                  {tab.id === 'alerts' && stats.active > 0 && (
                    <span className={cn('badge badge-danger ml-2')}>{stats.active}</span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Filters Panel - Only for Alerts tab */}
          {activeTab === 'alerts' && showFilters && (
            <div className="card mb-6 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4">
                <div>
                  <label className="label">Severity</label>
                  <select
                    value={filters.severity}
                    onChange={(e) => setFilters({ ...filters, severity: e.target.value })}
                    className="input"
                  >
                    {severities.map(s => (
                      <option key={s} value={s}>{s === 'all' ? 'All Severities' : s.charAt(0).toUpperCase() + s.slice(1)}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Status</label>
                  <select
                    value={filters.status}
                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                    className="input"
                  >
                    {statuses.map(s => (
                      <option key={s} value={s}>{s === 'all' ? 'All Statuses' : s.charAt(0).toUpperCase() + s.slice(1)}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Search</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search alerts..."
                      value={filters.search}
                      onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                      className="input pl-10"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center justify-between">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">
                <XCircle className="h-5 w-5" />
              </button>
            </div>
          )}

          {/* Alerts Tab Content */}
          {activeTab === 'alerts' && (
            <div className="space-y-4">
              {loading.alerts ? (
                <div className="text-center py-12">
                  <Loader2 className="h-8 w-8 mx-auto text-primary-600 animate-spin mb-2" />
                  <p className="text-gray-500">Loading alerts...</p>
                </div>
              ) : alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={cn('card card-hover p-4 flex items-start justify-between gap-4', selectedAlert?.id === alert.id && 'ring-2 ring-primary-500', alert.status === 'active' && 'border-l-4 border-danger-500', alert.status === 'acknowledged' && 'border-l-4 border-warning-500', alert.status === 'resolved' && 'border-l-4 border-success-500', alert.status === 'silenced' && 'border-l-4 border-info-500')}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <h4 className="font-semibold text-gray-900 truncate">{alert.title}</h4>
                          {getSeverityBadge(alert.severity)}
                          {getStatusBadge(alert.status)}
                        </div>
                        <p className="text-sm text-gray-600 truncate">{alert.description}</p>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-500 flex-shrink-0">
                        <span>{new Date(alert.created_at).toLocaleString()}</span>
                        {alert.acknowledged_at && (
                          <>
                            <span>•</span>
                            <span className="text-warning-600">Ack: {new Date(alert.acknowledged_at).toLocaleTimeString()}</span>
                          </>
                        )}
                        {alert.resolved_at && (
                          <>
                            <span>•</span>
                            <span className="text-success-600">Resolved: {new Date(alert.resolved_at).toLocaleTimeString()}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span className="flex items-center gap-1">
                        <Shield className="h-3 w-3" />
                        Rule: {alert.rule_id || 'N/A'}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        Source: {alert.source}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {alert.status === 'active' && (
                      <button className="btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); handleAcknowledge(alert.id) }}>
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Acknowledge
                      </button>
                    )}
                    {alert.status === 'acknowledged' && (
                      <button className="btn-success btn-sm" onClick={(e) => { e.stopPropagation(); handleResolve(alert.id) }}>
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Resolve
                      </button>
                    )}
                    <button className="btn-ghost btn-sm p-2" onClick={(e) => { e.stopPropagation(); setSelectedAlert(alert) }} title="View Details">
                      <Eye className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
              {!loading.alerts && alerts.length === 0 && (
                <div className="text-center py-12">
                  <Bell className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                  <p className="text-gray-500">No alerts found</p>
                  <p className="text-sm text-gray-400 mt-1">Try adjusting your filters</p>
                </div>
              )}
            </div>
          )}

          {/* Rules Tab Content */}
          {activeTab === 'rules' && (
            <div className="space-y-4">
              {loading.rules ? (
                <div className="text-center py-12">
                  <Loader2 className="h-8 w-8 mx-auto text-primary-600 animate-spin mb-2" />
                  <p className="text-gray-500">Loading rules...</p>
                </div>
              ) : rules.map((rule) => (
                <div
                  key={rule.id}
                  className={cn('card card-hover p-4 flex items-center justify-between gap-4', selectedRule?.id === rule.id && 'ring-2 ring-primary-500')}
                  onClick={() => setSelectedRule(rule)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h4 className="font-semibold text-gray-900">{rule.name}</h4>
                      {getSeverityBadge(rule.severity)}
                      {getChannelStatusBadge(rule.enabled)}
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{rule.description}</p>
                    <div className="flex flex-wrap items-center gap-3 text-sm text-gray-500">
                      <span className="font-mono bg-gray-100 px-2 py-0.5 rounded">{rule.condition}</span>
                      <span>Created: {new Date(rule.created_at).toLocaleDateString()}</span>
                      <span>Updated: {new Date(rule.updated_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <button className="btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); handleTestRule(rule.id) }} disabled={loading.rules}>
                      <Zap className="h-3 w-3 mr-1" />
                      Test
                    </button>
                    <button className={cn('btn-secondary btn-sm', rule.enabled ? 'text-danger-600 hover:text-danger-700' : 'text-success-600 hover:text-success-700')} onClick={(e) => { e.stopPropagation(); handleToggleRule(rule.id) }} disabled={loading.rules}>
                      {rule.enabled ? (
                        <>
                          <BellOff className="h-3 w-3 mr-1" />
                          Disable
                        </>
                      ) : (
                        <>
                          <Bell className="h-3 w-3 mr-1" />
                          Enable
                        </>
                      )}
                    </button>
                    <button className="btn-ghost btn-sm p-2" onClick={(e) => { e.stopPropagation(); setSelectedRule(rule) }} title="View Details">
                      <Eye className="h-4 w-4" />
                    </button>
                    <button className="btn-ghost btn-sm p-2 text-danger-600 hover:text-danger-700" onClick={(e) => { e.stopPropagation(); handleDeleteRule(rule.id) }} title="Delete" disabled={loading.rules}>
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
              {!loading.rules && rules.length === 0 && (
                <div className="text-center py-12">
                  <Settings className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                  <p className="text-gray-500">No rules found</p>
                  <p className="text-sm text-gray-400 mt-1">Create your first alert rule</p>
                </div>
              )}
            </div>
          )}

          {/* Channels Tab Content */}
          {activeTab === 'channels' && (
            <div className="space-y-4">
              {loading.channels ? (
                <div className="text-center py-12">
                  <Loader2 className="h-8 w-8 mx-auto text-primary-600 animate-spin mb-2" />
                  <p className="text-gray-500">Loading channels...</p>
                </div>
              ) : channels.map((channel) => (
                <div key={channel.id} className="card card-hover p-4 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-4 flex-1 min-w-0">
                    <div className={cn('p-3 rounded-xl', channel.type === 'slack' && 'bg-green-100', channel.type === 'pagerduty' && 'bg-red-100', channel.type === 'email' && 'bg-blue-100', channel.type === 'webhook' && 'bg-gray-100')}>
                      {channel.type === 'slack' && <Mail className="h-5 w-5 text-green-600" />}
                      {channel.type === 'pagerduty' && <Zap className="h-5 w-5 text-red-600" />}
                      {channel.type === 'email' && <Mail className="h-5 w-5 text-blue-600" />}
                      {channel.type === 'webhook' && <Link2 className="h-5 w-5 text-gray-600" />}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-gray-900">{channel.name}</h4>
                        {getChannelStatusBadge(channel.enabled)}
                      </div>
                      <p className="text-sm text-gray-500 truncate capitalize">{channel.type}</p>
                      {channel.last_tested && (
                        <p className="text-xs text-gray-400">Last tested: {new Date(channel.last_tested).toLocaleString()}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <button className="btn-secondary btn-sm" onClick={() => handleTestChannel(channel.id)} disabled={loading.channels}>
                      <RotateCcw className="h-3 w-3 mr-1" />
                      Test
                    </button>
                    <button className={cn('btn-secondary btn-sm', channel.enabled ? 'text-danger-600 hover:text-danger-700' : 'text-success-600 hover:text-success-700')} onClick={() => handleToggleChannel(channel.id)} disabled={loading.channels}>
                      {channel.enabled ? (
                        <>
                          <BellOff className="h-3 w-3 mr-1" />
                          Disable
                        </>
                      ) : (
                        <>
                          <Bell className="h-3 w-3 mr-1" />
                          Enable
                        </>
                      )}
                    </button>
                    <button className="btn-ghost btn-sm p-2" title="Edit" disabled={loading.channels}>
                      <Settings className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
              {!loading.channels && channels.length === 0 && (
                <div className="text-center py-12">
                  <Mail className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                  <p className="text-gray-500">No channels found</p>
                  <p className="text-sm text-gray-400 mt-1">Create your first notification channel</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Alert Detail Panel */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 flex items-end justify-end p-4 lg:p-0">
          <div className="bg-white rounded-t-xl lg:rounded-xl shadow-xl w-full lg:w-[700px] max-h-[90vh] lg:max-h-[calc(100vh-4rem)] overflow-y-auto animate-slide-in-right">
            <div className="p-4 lg:p-6 border-b border-gray-100 flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <h2 className="text-lg lg:text-xl font-bold text-gray-900">{selectedAlert.title}</h2>
                  {getSeverityBadge(selectedAlert.severity)}
                  {getStatusBadge(selectedAlert.status)}
                </div>
                <p className="text-sm text-gray-500">Rule: {selectedAlert.rule_id || 'N/A'} • Fired: {new Date(selectedAlert.created_at).toLocaleString()}</p>
              </div>
              <button
                onClick={() => setSelectedAlert(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 lg:p-6 space-y-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
                <p className="text-gray-700">{selectedAlert.description}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Metadata</h3>
                  <div className="space-y-2 text-sm">
                    {Object.entries(selectedAlert.metadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-500 capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="font-medium text-gray-900 font-mono">{typeof value === 'object' ? JSON.stringify(value) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Details</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-500">Source: </span>
                      <span className="font-medium text-gray-900">{selectedAlert.source}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Rule ID: </span>
                      <span className="font-medium text-gray-900">{selectedAlert.rule_id || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-100">
                <div>
                  <p className="text-xs text-gray-500">Status</p>
                  <p className="font-medium text-gray-900 capitalize">{selectedAlert.status}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Acknowledged</p>
                  <p className="font-medium text-gray-900">{selectedAlert.acknowledged_at ? new Date(selectedAlert.acknowledged_at).toLocaleString() : 'Not acknowledged'}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Resolved</p>
                  <p className="font-medium text-gray-900">{selectedAlert.resolved_at ? new Date(selectedAlert.resolved_at).toLocaleString() : 'Not resolved'}</p>
                </div>
              </div>

              <div className="flex gap-3 pt-4 border-t border-gray-100">
                {selectedAlert.status === 'active' && (
                  <button className="btn-primary flex-1" onClick={() => handleAcknowledge(selectedAlert.id)}>
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Acknowledge
                  </button>
                )}
                {selectedAlert.status === 'acknowledged' && (
                  <button className="btn-success flex-1" onClick={() => handleResolve(selectedAlert.id)}>
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Resolve
                  </button>
                )}
                {selectedAlert.status !== 'silenced' && (
                  <button className="btn-secondary flex-1" onClick={() => handleSilence(selectedAlert.id)}>
                    <BellOff className="h-4 w-4 mr-2" />
                    Silence
                  </button>
                )}
                <button className="btn-ghost flex-1" onClick={() => setSelectedAlert(null)}>
                  <XCircle className="h-4 w-4 mr-2" />
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {selectedAlert && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSelectedAlert(null)} />
      )}

      {/* Rule Detail Panel */}
      {selectedRule && (
        <div className="fixed inset-0 z-50 flex items-end justify-end p-4 lg:p-0">
          <div className="bg-white rounded-t-xl lg:rounded-xl shadow-xl w-full lg:w-[600px] max-h-[90vh] lg:max-h-[calc(100vh-4rem)] overflow-y-auto animate-slide-in-right">
            <div className="p-4 lg:p-6 border-b border-gray-100 flex items-start justify-between">
              <div className="flex-1">
                <h2 className="text-lg lg:text-xl font-bold text-gray-900">{selectedRule.name}</h2>
                <p className="text-sm text-gray-500 mt-1">{getSeverityBadge(selectedRule.severity)} • {selectedRule.enabled ? 'Enabled' : 'Disabled'}</p>
              </div>
              <button
                onClick={() => setSelectedRule(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 lg:p-6 space-y-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
                <p className="text-gray-700">{selectedRule.description}</p>
              </div>

              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Condition</h3>
                <code className="font-mono text-sm bg-gray-100 px-3 py-2 rounded-lg block">{selectedRule.condition}</code>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Parameters</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span className="text-gray-500">Created</span><span className="font-medium">{new Date(selectedRule.created_at).toLocaleDateString()}</span></div>
                    <div className="flex justify-between"><span className="text-gray-500">Updated</span><span className="font-medium">{new Date(selectedRule.updated_at).toLocaleDateString()}</span></div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 pt-4 border-t border-gray-100">
                <button className="btn-primary flex-1" onClick={() => handleTestRule(selectedRule.id)} disabled={loading.rules}>
                  <Zap className="h-4 w-4 mr-2" />
                  Test Rule
                </button>
                <button className={cn('btn-secondary flex-1', selectedRule.enabled ? 'text-danger-600' : 'text-success-600')} onClick={() => handleToggleRule(selectedRule.id)} disabled={loading.rules}>
                  {selectedRule.enabled ? (
                    <>
                      <BellOff className="h-4 w-4 mr-2" />
                      Disable
                    </>
                  ) : (
                    <>
                      <Bell className="h-4 w-4 mr-2" />
                      Enable
                    </>
                  )}
                </button>
                <button className="btn-danger flex-1" onClick={() => handleDeleteRule(selectedRule.id)} disabled={loading.rules}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {selectedRule && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSelectedRule(null)} />
      )}
    </div>
  )
}