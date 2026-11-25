'use client'

import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, ChevronDown, ChevronRight, ChevronLeft, Download, Upload, Play, Pause, Trash2, Edit, Eye, GitBranch, Cloud, Server, Database, Zap, BarChart2, Settings, AlertTriangle, CheckCircle, XCircle, Clock, RotateCcw, History, Tag, Link2, ExternalLink, Brain, Target, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { TrendChart, DecisionPie, LatencyChart, FraudTypeBar, GeographyChart } from '@/components/Charts'
import { apiClient } from '@/lib/api'

interface ModelVersion {
  id: string
  version: string
  modelType: 'xgboost' | 'lightgbm' | 'transformer' | 'gnn' | 'ensemble'
  status: 'production' | 'staging' | 'archived' | 'training' | 'failed'
  accuracy: number
  precision: number
  recall: number
  f1: number
  latencyMs: number
  trainingDate: string
  deployedDate?: string
  artifacts: {
    onnxPath: string
    configPath: string
    metricsPath: string
  }
  metadata: {
    trainingDataSize: number
    features: number
    hyperparameters: Record<string, any>
    gitCommit: string
    trainedBy: string
  }
}

interface Model {
  id: string
  name: string
  type: 'xgboost' | 'lightgbm' | 'transformer' | 'gnn' | 'ensemble'
  description: string
  currentVersion: string
  versions: ModelVersion[]
  tags: string[]
  owner: string
  createdAt: string
  updatedAt: string
}

interface FilterState {
  type: string
  status: string
  search: string
}

const modelTypes = ['all', 'xgboost', 'lightgbm', 'transformer', 'gnn', 'ensemble']
const statuses = ['all', 'production', 'staging', 'archived', 'training', 'failed']

export default function ModelsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [selectedVersion, setSelectedVersion] = useState<ModelVersion | null>(null)
  const [filters, setFilters] = useState<FilterState>({
    type: 'all',
    status: 'all',
    search: '',
  })
  const [showFilters, setShowFilters] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'versions' | 'metrics' | 'deployment'>('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showRegisterModel, setShowRegisterModel] = useState(false)
  const [registerModelData, setRegisterModelData] = useState<{
    name: string
    modelType: 'xgboost' | 'lightgbm' | 'transformer' | 'gnn' | 'ensemble'
    description: string
    tags: string[]
  }>({
    name: '',
    modelType: 'xgboost',
    description: '',
    tags: [],
  })

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiClient.listModels()
      // Transform API response to match our Model interface
      const transformedModels: Model[] = Object.entries(response).map(([name, versions]: [string, any]) => {
        const versionList = versions.map((v: any, idx: number) => ({
          id: `ver_${name}_${idx}`,
          version: v.version,
          modelType: v.model_type as ModelVersion['modelType'],
          status: v.stage as ModelVersion['status'],
          accuracy: v.metrics?.accuracy || 0,
          precision: v.metrics?.precision || 0,
          recall: v.metrics?.recall || 0,
          f1: v.metrics?.f1 || 0,
          latencyMs: v.latency_ms || 0,
          trainingDate: v.training_date,
          deployedDate: v.stage === 'production' ? new Date().toISOString() : undefined,
          artifacts: {
            onnxPath: `models/${name}/${v.version}/model.onnx`,
            configPath: `models/${name}/${v.version}/config.json`,
            metricsPath: `models/${name}/${v.version}/metrics.json`,
          },
          metadata: {
            trainingDataSize: v.metrics?.training_samples || 0,
            features: v.metrics?.features || 0,
            hyperparameters: v.metrics?.hyperparameters || {},
            gitCommit: v.metrics?.git_commit || '',
            trainedBy: 'ml-pipeline',
          },
        }))

        const latestVersion = versionList[0]?.version || 'v1.0.0'
        return {
          id: `model_${name}`,
          name: name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' '),
          type: versionList[0]?.modelType || 'xgboost',
          description: `${name} model for fraud detection`,
          currentVersion: latestVersion,
          versions: versionList,
          tags: [versionList[0]?.status || 'staging', 'ml'],
          owner: 'ml-team',
          createdAt: versionList[versionList.length - 1]?.trainingDate || new Date().toISOString(),
          updatedAt: versionList[0]?.trainingDate || new Date().toISOString(),
        }
      })
      setModels(transformedModels)
    } catch (err) {
      setError('Failed to load models')
      console.error('Failed to fetch models:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  const handlePromoteVersion = async (modelName: string, version: string) => {
    try {
      await apiClient.promoteModel(modelName, version, 'production')
      fetchModels() // Refresh
    } catch (err) {
      setError('Failed to promote model')
      console.error('Failed to promote model:', err)
    }
  }

  const handleDownloadArtifacts = async (version: ModelVersion) => {
    try {
      // In a real app, this would download from the API
      // For now, create a mock download
      const content = JSON.stringify({
        version: version.version,
        modelType: version.modelType,
        status: version.status,
        metrics: {
          accuracy: version.accuracy,
          precision: version.precision,
          recall: version.recall,
          f1: version.f1,
          latencyMs: version.latencyMs,
        },
        artifacts: version.artifacts,
        metadata: version.metadata,
      }, null, 2)

      const blob = new Blob([content], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${version.version}_artifacts.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Failed to download artifacts:', err)
    }
  }

  const filteredModels = models.filter(model => {
    if (filters.type !== 'all' && model.type !== filters.type) return false
    if (filters.search) {
      const search = filters.search.toLowerCase()
      return (
        model.name.toLowerCase().includes(search) ||
        model.description.toLowerCase().includes(search) ||
        model.tags.some(t => t.toLowerCase().includes(search))
      )
    }
    return true
  })

  const getStatusBadge = (status: string) => {
    const variants = {
      production: 'badge-success',
      staging: 'badge-warning',
      archived: 'badge-info',
      training: 'badge-primary',
      failed: 'badge-danger',
    }
    const icons = {
      production: <CheckCircle className="h-3 w-3 mr-1" />,
      staging: <Clock className="h-3 w-3 mr-1" />,
      archived: <Database className="h-3 w-3 mr-1" />,
      training: <RotateCcw className="h-3 w-3 mr-1 animate-spin" />,
      failed: <XCircle className="h-3 w-3 mr-1" />,
    }
    return (
      <span className={cn('badge font-medium', variants[status as keyof typeof variants])}>
        {icons[status as keyof typeof icons]}
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  const getTypeIcon = (type: string) => {
    const icons = {
      xgboost: <Zap className="h-5 w-5 text-yellow-600" />,
      lightgbm: <BarChart2 className="h-5 w-5 text-green-600" />,
      transformer: <Brain className="h-5 w-5 text-purple-600" />,
      gnn: <GitBranch className="h-5 w-5 text-blue-600" />,
      ensemble: <Server className="h-5 w-5 text-indigo-600" />,
    }
    return icons[type as keyof typeof icons] || <Zap className="h-5 w-5" />
  }

  const stats = {
    total: models.length,
    production: models.filter(m => m.versions.some(v => v.status === 'production')).length,
    staging: models.filter(m => m.versions.some(v => v.status === 'staging')).length,
    archived: models.filter(m => m.versions.some(v => v.status === 'archived')).length,
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
              <h1 className="text-2xl font-bold text-gray-900">Model Registry</h1>
              <p className="text-gray-500 mt-1">Manage ML models, versions, and deployments</p>
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
                className="btn-primary" disabled={loading}
                onClick={() => setShowRegisterModel(true)}
              >
                <Upload className="h-4 w-4 mr-2" />
                Register Model
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Models</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <Server className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">In Production</p>
                  <p className="text-2xl font-bold text-success-600">{stats.production}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-success-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Staging</p>
                  <p className="text-2xl font-bold text-warning-600">{stats.staging}</p>
                </div>
                <Clock className="h-8 w-8 text-warning-600" />
              </div>
            </div>
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Archived</p>
                  <p className="text-2xl font-bold text-gray-600">{stats.archived}</p>
                </div>
                <Database className="h-8 w-8 text-gray-600" />
              </div>
            </div>
          </div>

          {error && (
            <div className="card mb-6 border-danger-200 bg-danger-50">
              <p className="text-danger-700">{error}</p>
            </div>
          )}

          {/* Register Model Modal */}
          {showRegisterModel && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 lg:p-0">
              <div className="bg-white rounded-xl shadow-xl w-full lg:w-[500px] max-w-full max-h-[90vh] overflow-y-auto animate-slide-in">
                <div className="p-6 lg:p-8 border-b border-gray-100 flex items-start justify-between">
                  <div>
                    <h2 className="text-lg lg:text-xl font-bold text-gray-900">Register New Model</h2>
                    <p className="text-sm text-gray-500">Submit a new model to the registry</p>
                  </div>
                  <button
                    onClick={() => setShowRegisterModel(false)}
                    className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
                  >
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <form
                  className="p-6 lg:p-8 space-y-6"
                  onSubmit={async (e) => {
                    e.preventDefault()
                    setLoading(true)
                    setError(null)
                    try {
                      await apiClient.registerModel({
                        name: registerModelData.name,
                        modelType: registerModelData.modelType,
                        description: registerModelData.description,
                        tags: registerModelData.tags,
                      })
                      setShowRegisterModel(false)
                      setRegisterModelData({
                        name: '',
                        modelType: 'xgboost',
                        description: '',
                        tags: [],
                      })
                      fetchModels() // Refresh the model list
                    } catch (err) {
                      setError('Failed to register model')
                      console.error('Failed to register model:', err)
                    } finally {
                      setLoading(false)
                    }
                  }}
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="label">Model Name</label>
                      <input
                        type="text"
                        value={registerModelData.name}
                        onChange={(e) =>
                          setRegisterModelData({
                            ...registerModelData,
                            name: e.target.value,
                          })
                        }
                        className="input"
                        required
                        placeholder="e.g. Upishield TextCNN"
                      />
                    </div>
                    <div>
                      <label className="label">Model Type</label>
                      <select
                        value={registerModelData.modelType}
                        onChange={(e) =>
                          setRegisterModelData({
                            ...registerModelData,
                            modelType: e.target.value as
                              | 'xgboost'
                              | 'lightgbm'
                              | 'transformer'
                              | 'gnn'
                              | 'ensemble',
                          })
                        }
                        className="input"
                      >
                        <option value="xgboost">XGBoost</option>
                        <option value="lightgbm">LightGBM</option>
                        <option value="transformer">Transformer</option>
                        <option value="gnn">GNN</option>
                        <option value="ensemble">Ensemble</option>
                      </select>
                    </div>
                    <div>
                      <label className="label">Description</label>
                      <textarea
                        value={registerModelData.description}
                        onChange={(e) =>
                          setRegisterModelData({
                            ...registerModelData,
                            description: e.target.value,
                          })
                        }
                        className="input textarea"
                        rows={3}
                        placeholder="Brief description of the model's architecture and purpose"
                        required
                      />
                    </div>
                    <div>
                      <label className="label">Tags (comma-separated)</label>
                      <textarea
                        value={registerModelData.tags.join(', ')}
                        onChange={(e) =>
                          setRegisterModelData({
                            ...registerModelData,
                            tags: e.target.value
                              .split(',')
                              .map((t: string) => t.trim())
                              .filter((t: string) => t.length > 0),
                          })
                        }
                        className="input textarea"
                        rows={3}
                        placeholder="e.g. text-cnn, upi-fraud, production-ready"
                      />
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button
                      type="submit"
                      className={cn(
                        'btn-primary flex-1',
                        loading && 'opacity-50 cursor-not-allowed'
                      )}
                      disabled={loading}
                    >
                      <Loader2 className="h-4 w-4 mr-2" />
                      {loading ? 'Registering...' : 'Register Model'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowRegisterModel(false)}
                      className="btn-secondary"
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Filters Panel */}
          {showFilters && (
            <div className="card mb-6 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4">
                <div>
                  <label className="label">Model Type</label>
                  <select
                    value={filters.type}
                    onChange={(e) => setFilters({ ...filters, type: e.target.value })}
                    className="input"
                  >
                    {modelTypes.map(type => (
                      <option key={type} value={type}>{type === 'all' ? 'All Types' : type.charAt(0).toUpperCase() + type.slice(1)}</option>
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
                    {statuses.map(status => (
                      <option key={status} value={status}>{status === 'all' ? 'All Statuses' : status.charAt(0).toUpperCase() + status.slice(1)}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Search</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search models, tags..."
                      value={filters.search}
                      onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                      className="input pl-10"
                    />
                  </div>
                </div>
                <div className="flex items-end">
                  <button
                    onClick={() => setFilters({ type: 'all', status: 'all', search: '' })}
                    className="btn-ghost text-sm"
                  >
                    Clear Filters
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Models Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
            {loading ? (
              <div className="col-span-full text-center py-12">
                <Loader2 className="h-12 w-12 mx-auto text-primary-600 animate-spin mb-2" />
                <p className="text-gray-500">Loading models...</p>
              </div>
            ) : filteredModels.map((model) => (
              <div
                key={model.id}
                className={cn('card card-hover', selectedModel?.id === model.id && 'ring-2 ring-primary-500')}
                onClick={() => { setSelectedModel(model); setActiveTab('overview'); setSelectedVersion(null) }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-3 bg-primary-100 rounded-xl">
                      {getTypeIcon(model.type)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{model.name}</h3>
                      <p className="text-xs text-gray-500 capitalize">{model.type}</p>
                    </div>
                  </div>
                  {model.versions.find(v => v.version === model.currentVersion) && getStatusBadge(model.versions.find(v => v.version === model.currentVersion)!.status)}
                </div>

                <p className="text-sm text-gray-600 mb-3 line-clamp-2">{model.description}</p>

                <div className="flex flex-wrap gap-1 mb-3">
                  {model.tags.slice(0, 4).map((tag, idx) => (
                    <span key={idx} className="badge badge-info text-xs">{tag}</span>
                  ))}
                  {model.tags.length > 4 && (
                    <span className="badge badge-info text-xs">+{model.tags.length - 4}</span>
                  )}
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                  <div className="text-xs text-gray-500">
                    <span>v{model.currentVersion}</span>
                    <span className="mx-1">•</span>
                    <span>Updated {new Date(model.updatedAt).toLocaleDateString()}</span>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setSelectedModel(model); setActiveTab('overview'); setSelectedVersion(null) }}
                    className="text-primary-600 hover:underline text-sm"
                  >
                    View Details
                  </button>
                </div>
              </div>
            ))}
            {!loading && filteredModels.length === 0 && (
              <div className="col-span-full text-center py-12">
                <Search className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                <p className="text-gray-500">No models found</p>
                <p className="text-sm text-gray-400 mt-1">Try adjusting your filters</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Detail Panel */}
      {selectedModel && (
        <div className="fixed inset-0 z-50 flex items-end justify-end p-4 lg:p-0">
          <div className="bg-white rounded-t-xl lg:rounded-xl shadow-xl w-full lg:w-[700px] max-h-[90vh] lg:max-h-[calc(100vh-4rem)] overflow-hidden animate-slide-in-right flex flex-col">
            {/* Header */}
            <div className="p-4 lg:p-6 border-b border-gray-100 flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <div className="p-2 bg-primary-100 rounded-lg">
                    {getTypeIcon(selectedModel.type)}
                  </div>
                  <div>
                    <h2 className="text-lg lg:text-xl font-bold text-gray-900">{selectedModel.name}</h2>
                    <p className="text-sm text-gray-500 capitalize">{selectedModel.type}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {selectedModel.tags.map((tag, idx) => (
                    <span key={idx} className="badge badge-info text-xs">{tag}</span>
                  ))}
                  {selectedModel.versions.find(v => v.version === selectedModel.currentVersion) && getStatusBadge(selectedModel.versions.find(v => v.version === selectedModel.currentVersion)!.status)}
                </div>
              </div>
              <button
                onClick={() => { setSelectedModel(null); setSelectedVersion(null) }}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors lg:hidden"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-100 px-4 lg:px-6">
              <nav className="flex gap-1" aria-label="Model details tabs">
                {[
                  { id: 'overview', label: 'Overview', icon: <Eye className="h-4 w-4" /> },
                  { id: 'versions', label: 'Versions', icon: <History className="h-4 w-4" /> },
                  { id: 'metrics', label: 'Metrics', icon: <BarChart2 className="h-4 w-4" /> },
                  { id: 'deployment', label: 'Deployment', icon: <Cloud className="h-4 w-4" /> },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => { setActiveTab(tab.id as any); setSelectedVersion(null) }}
                    className={cn(
                      'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                      activeTab === tab.id
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    )}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-6">
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
                    <p className="text-gray-700">{selectedModel.description}</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="card p-4">
                      <p className="text-xs text-gray-500 mb-1">Owner</p>
                      <p className="font-medium text-gray-900">{selectedModel.owner}</p>
                    </div>
                    <div className="card p-4">
                      <p className="text-xs text-gray-500 mb-1">Created</p>
                      <p className="font-medium text-gray-900">{new Date(selectedModel.createdAt).toLocaleDateString()}</p>
                    </div>
                    <div className="card p-4">
                      <p className="text-xs text-gray-500 mb-1">Current Version</p>
                      <p className="font-medium text-gray-900">{selectedModel.currentVersion}</p>
                    </div>
                  </div>

                  {/* Current Production Version Metrics */}
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3">Current Production Metrics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {(() => {
                        const prodVersion = selectedModel.versions.find(v => v.status === 'production')
                        if (!prodVersion) return <p className="text-gray-500">No production version</p>
                        return [
                          { label: 'Accuracy', value: `${(prodVersion.accuracy * 100).toFixed(1)}%`, icon: <CheckCircle className="h-5 w-5 text-success-600" /> },
                          { label: 'Precision', value: `${(prodVersion.precision * 100).toFixed(1)}%`, icon: <Target className="h-5 w-5 text-primary-600" /> },
                          { label: 'Recall', value: `${(prodVersion.recall * 100).toFixed(1)}%`, icon: <Search className="h-5 w-5 text-blue-600" /> },
                          { label: 'F1 Score', value: `${(prodVersion.f1 * 100).toFixed(1)}%`, icon: <BarChart2 className="h-5 w-5 text-purple-600" /> },
                          { label: 'Latency', value: `${prodVersion.latencyMs}ms`, icon: <Zap className="h-5 w-5 text-warning-600" /> },
                          { label: 'Training Data', value: `${(prodVersion.metadata.trainingDataSize / 1e6).toFixed(1)}M`, icon: <Database className="h-5 w-5 text-gray-600" /> },
                        ].map((metric, idx) => (
                          <div key={idx} className="card p-4 text-center">
                            <div className="flex items-center justify-center mb-2">{metric.icon}</div>
                            <p className="text-2xl font-bold text-gray-900">{metric.value}</p>
                            <p className="text-xs text-gray-500">{metric.label}</p>
                          </div>
                        ))
                      })()}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'versions' && (
                <div className="space-y-4">
                  {selectedModel.versions.map((version) => (
                    <div
                      key={version.id}
                      className={cn('card p-4 flex items-center justify-between gap-4', selectedVersion?.id === version.id && 'ring-2 ring-primary-500')}
                      onClick={() => setSelectedVersion(version)}
                    >
                      <div className="flex items-center gap-4 flex-1 min-w-0">
                        <div className={cn('p-2 rounded-lg', version.status === 'production' && 'bg-success-100', version.status === 'staging' && 'bg-warning-100', version.status === 'archived' && 'bg-gray-100', version.status === 'training' && 'bg-primary-100', version.status === 'failed' && 'bg-danger-100')}>
                          {getTypeIcon(version.modelType)}
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold text-gray-900 truncate">{version.version}</h4>
                            {getStatusBadge(version.status)}
                          </div>
                          <p className="text-sm text-gray-500 truncate">
                            Trained: {new Date(version.trainingDate).toLocaleDateString()}
                            {version.deployedDate && ` • Deployed: ${new Date(version.deployedDate).toLocaleDateString()}`}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-600 hidden md:flex">
                        <span>Acc: {(version.accuracy * 100).toFixed(1)}%</span>
                        <span>F1: {(version.f1 * 100).toFixed(1)}%</span>
                        <span>Latency: {version.latencyMs}ms</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button className="btn-ghost btn-sm p-2" title="View Details">
                          <Eye className="h-4 w-4" />
                        </button>
                        {version.status === 'staging' && (
                          <button
                            className="btn-primary btn-sm"
                            title="Promote to Production"
                            onClick={(e) => { e.stopPropagation(); handlePromoteVersion(selectedModel.id.replace('model_', ''), version.version) }}
                          >
                            <Cloud className="h-4 w-4" />
                          </button>
                        )}
                        {version.status === 'production' && (
                          <button className="btn-secondary btn-sm" title="Rollback">
                            <RotateCcw className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}

                  {/* Version Detail Panel */}
                  {selectedVersion && (
                    <div className="card p-4 border-l-4 border-primary-500 animate-fade-in">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h4 className="font-semibold text-gray-900">{selectedVersion.version}</h4>
                          <p className="text-sm text-gray-500">{selectedVersion.modelType} • {getStatusBadge(selectedVersion.status)}</p>
                        </div>
                        <button onClick={() => setSelectedVersion(null)} className="text-gray-500 hover:text-gray-700">
                          <XCircle className="h-5 w-5" />
                        </button>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className="text-xs text-gray-500">Accuracy</p>
                          <p className="font-bold text-gray-900">{(selectedVersion.accuracy * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Precision</p>
                          <p className="font-bold text-gray-900">{(selectedVersion.precision * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Recall</p>
                          <p className="font-bold text-gray-900">{(selectedVersion.recall * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">F1 Score</p>
                          <p className="font-bold text-gray-900">{(selectedVersion.f1 * 100).toFixed(1)}%</p>
                        </div>
                        <div className="md:col-span-2">
                          <p className="text-xs text-gray-500">Latency</p>
                          <p className="font-bold text-gray-900">{selectedVersion.latencyMs}ms</p>
                        </div>
                        <div className="md:col-span-2">
                          <p className="text-xs text-gray-500">Training Data</p>
                          <p className="font-bold text-gray-900">{(selectedVersion.metadata.trainingDataSize / 1e6).toFixed(1)}M samples</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h5 className="font-medium text-gray-900 mb-2">Hyperparameters</h5>
                          <pre className="bg-gray-50 p-3 rounded text-xs overflow-x-auto max-h-40"><code>{JSON.stringify(selectedVersion.metadata.hyperparameters, null, 2)}</code></pre>
                        </div>
                        <div>
                          <h5 className="font-medium text-gray-900 mb-2">Artifacts</h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex items-center gap-2 text-gray-700">
                              <Database className="h-4 w-4 text-gray-400" />
                              <span>ONNX: {selectedVersion.artifacts.onnxPath}</span>
                            </div>
                            <div className="flex items-center gap-2 text-gray-700">
                              <Settings className="h-4 w-4 text-gray-400" />
                              <span>Config: {selectedVersion.artifacts.configPath}</span>
                            </div>
                            <div className="flex items-center gap-2 text-gray-700">
                              <BarChart2 className="h-4 w-4 text-gray-400" />
                              <span>Metrics: {selectedVersion.artifacts.metricsPath}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-3 mt-4 pt-4 border-t border-gray-100">
                        <button
                          className="btn-primary btn-sm"
                          onClick={() => handleDownloadArtifacts(selectedVersion)}
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Download Artifacts
                        </button>
                        {selectedVersion.status === 'staging' && (
                          <button
                            className="btn-success btn-sm"
                            onClick={(e) => { e.stopPropagation(); handlePromoteVersion(selectedModel.id.replace('model_', ''), selectedVersion.version) }}
                          >
                            <Cloud className="h-4 w-4 mr-1" />
                            Promote to Production
                          </button>
                        )}
                        {selectedVersion.status === 'production' && (
                          <button className="btn-warning btn-sm">
                            <RotateCcw className="h-4 w-4 mr-1" />
                            Rollback
                          </button>
                        )}
                        <button className="btn-danger btn-sm">
                          <Trash2 className="h-4 w-4 mr-1" />
                          Archive
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'metrics' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3">Performance Comparison Across Versions</h3>
                    <FraudTypeBar
                      data={selectedModel.versions.map(v => ({
                        type: v.version,
                        count: Math.round(v.accuracy * 100),
                      }))}
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="card">
                      <h4 className="font-semibold text-gray-900 mb-3">Accuracy Trend</h4>
                      <TrendChart
                        data={selectedModel.versions.map(v => ({
                          timestamp: v.version,
                          transactions: v.accuracy * 100,
                          fraud_detected: 0,
                          avg_latency_ms: 0,
                        }))}
                      />
                    </div>
                    <div className="card">
                      <h4 className="font-semibold text-gray-900 mb-3">Latency Trend</h4>
                      <TrendChart
                        data={selectedModel.versions.map(v => ({
                          timestamp: v.version,
                          transactions: v.latencyMs,
                          fraud_detected: 0,
                          avg_latency_ms: 0,
                        }))}
                      />
                    </div>
                  </div>

                  <div className="card">
                    <h4 className="font-semibold text-gray-900 mb-3">Detailed Metrics Table</h4>
                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Version</th>
                            <th>Status</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                            <th>Latency (ms)</th>
                            <th>Training Date</th>
                            <th>Training Data</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedModel.versions.map((v, idx) => (
                            <tr key={idx} className={v.status === 'production' ? 'bg-success-50 font-medium' : ''}>
                              <td className="font-mono">{v.version}</td>
                              <td>{getStatusBadge(v.status)}</td>
                              <td>{(v.accuracy * 100).toFixed(1)}%</td>
                              <td>{(v.precision * 100).toFixed(1)}%</td>
                              <td>{(v.recall * 100).toFixed(1)}%</td>
                              <td>{(v.f1 * 100).toFixed(1)}%</td>
                              <td className={cn('font-medium', v.latencyMs > 20 ? 'text-warning-600' : 'text-success-600')}>{v.latencyMs}ms</td>
                              <td>{new Date(v.trainingDate).toLocaleDateString()}</td>
                              <td>{(v.metadata.trainingDataSize / 1e6).toFixed(1)}M</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'deployment' && (
                <div className="space-y-6">
                  <div className="card">
                    <h3 className="font-semibold text-gray-900 mb-4">Deployment Configuration</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <h4 className="font-medium text-gray-900">Production Endpoint</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Status</span>
                            <span className="badge badge-success">Active</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Endpoint</span>
                            <code className="text-xs font-mono">https://api.scamshield.ai/v1/models/{selectedModel.id}/predict</code>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Protocol</span>
                            <span>gRPC + REST</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Authentication</span>
                            <span>mTLS + API Key</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Rate Limit</span>
                            <span>10,000 req/s</span>
                          </div>
                        </div>
                      </div>
                      <div className="space-y-4">
                        <h4 className="font-medium text-gray-900">Infrastructure</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Compute</span>
                            <span>Kubernetes (GKE/EKS)</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Replicas</span>
                            <span>3-20 (HPA)</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Resources</span>
                            <span>2 CPU / 4GiB per pod</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">GPU</span>
                            <span>{selectedModel.type === 'transformer' ? 'NVIDIA T4 (1)' : 'None'}</span>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-gray-500">Quantization</span>
                            <span>INT8 (ONNX Runtime)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="card">
                    <h3 className="font-semibold text-gray-900 mb-4">Deployment History</h3>
                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Version</th>
                            <th>Environment</th>
                            <th>Deployed At</th>
                            <th>Deployed By</th>
                            <th>Status</th>
                            <th>Rollback</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedModel.versions
                            .filter(v => v.deployedDate)
                            .sort((a, b) => new Date(b.deployedDate!).getTime() - new Date(a.deployedDate!).getTime())
                            .map((v, idx) => (
                              <tr key={idx}>
                                <td className="font-mono">{v.version}</td>
                                <td><span className="badge badge-primary">Production</span></td>
                                <td>{new Date(v.deployedDate!).toLocaleString()}</td>
                                <td>ml-pipeline</td>
                                <td><span className="badge badge-success">Success</span></td>
                                <td>
                                  <button className="btn-ghost btn-sm text-danger-600 hover:text-danger-700">
                                    <RotateCcw className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="card border-warning-200 bg-warning-50">
                    <h3 className="font-semibold text-warning-800 mb-3 flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5" />
                      Deployment Actions
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900">Promote Staging to Production</p>
                          <p className="text-sm text-gray-500">Deploy the latest staging version to production with blue-green deployment</p>
                        </div>
                        <button
                          className="btn-primary btn-sm"
                          onClick={() => handlePromoteVersion(selectedModel.id.replace('model_', ''), selectedModel.versions.find(v => v.status === 'staging')?.version || '')}
                        >
                          <Cloud className="h-4 w-4 mr-1" />
                          Promote
                        </button>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900">Rollback Production</p>
                          <p className="text-sm text-gray-500">Rollback to the previous production version</p>
                        </div>
                        <button className="btn-warning btn-sm">
                          <RotateCcw className="h-4 w-4 mr-1" />
                          Rollback
                        </button>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-white rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900">Canary Deployment</p>
                          <p className="text-sm text-gray-500">Deploy new version to 5% of traffic for validation</p>
                        </div>
                        <button className="btn-secondary btn-sm">
                          <Zap className="h-4 w-4 mr-1" />
                          Start Canary
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {selectedModel && (
        <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => { setSelectedModel(null); setSelectedVersion(null) }} />
      )}
    </div>
  )
}