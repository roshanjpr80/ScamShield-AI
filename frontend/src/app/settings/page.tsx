'use client'

import { useState, useEffect, useCallback } from 'react'
import { User, Shield, Key, Bell, Palette, Globe, Save, Loader2, AlertCircle, Eye, EyeOff, Sun, Moon, Monitor, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'
import { apiClient } from '@/lib/api'

interface UserProfile {
  id: string
  name: string
  email: string
  role: string
  avatar?: string
  createdAt: string
  lastLogin?: string
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  notifications: {
    email: boolean
    push: boolean
    alerts: boolean
    reports: boolean
  }
  language: string
  timezone: string
  apiKeys: Array<{
    id: string
    name: string
    key: string
    createdAt: string
    lastUsed?: string
    permissions: string[]
  }>
}

interface SettingsState {
  profile: UserProfile | null
  preferences: UserPreferences | null
  loading: boolean
  saving: boolean
  error: string | null
  activeTab: 'profile' | 'preferences' | 'security' | 'api' | 'notifications'
}

const themes = [
  { value: 'light', label: 'Light', icon: <Sun className="h-4 w-4" /> },
  { value: 'dark', label: 'Dark', icon: <Moon className="h-4 w-4" /> },
  { value: 'system', label: 'System', icon: <Monitor className="h-4 w-4" /> },
]

const languages = [
  { value: 'en', label: 'English' },
  { value: 'hi', label: 'Hindi' },
  { value: 'ta', label: 'Tamil' },
  { value: 'te', label: 'Telugu' },
  { value: 'bn', label: 'Bengali' },
]

const timezones = [
  { value: 'Asia/Kolkata', label: 'IST (UTC+5:30)' },
  { value: 'UTC', label: 'UTC' },
  { value: 'America/New_York', label: 'EST (UTC-5)' },
  { value: 'America/Los_Angeles', label: 'PST (UTC-8)' },
  { value: 'Europe/London', label: 'GMT (UTC+0)' },
]

export default function SettingsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [state, setState] = useState<SettingsState>({
    profile: null,
    preferences: null,
    loading: true,
    saving: false,
    error: null,
    activeTab: 'profile',
  })

  const fetchSettings = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    try {
      const [profileRes, prefsRes] = await Promise.all([
        Promise.resolve({
          id: 'user_001',
          name: 'Admin User',
          email: 'admin@scamshield.ai',
          role: 'Administrator',
          createdAt: '2024-01-15T10:00:00Z',
          lastLogin: new Date().toISOString(),
        }),
        Promise.resolve({
          theme: 'system',
          notifications: {
            email: true,
            push: true,
            alerts: true,
            reports: false,
          },
          language: 'en',
          timezone: 'Asia/Kolkata',
          apiKeys: [
            {
              id: 'key_001',
              name: 'Production API Key',
              key: 'sk_live_****************',
              createdAt: '2024-01-10T10:00:00Z',
              lastUsed: '2024-01-15T09:30:00Z',
              permissions: ['read', 'write', 'score', 'models'],
            },
            {
              id: 'key_002',
              name: 'Development API Key',
              key: 'sk_test_****************',
              createdAt: '2024-01-12T14:00:00Z',
              lastUsed: '2024-01-14T16:45:00Z',
              permissions: ['read', 'score'],
            },
          ],
        }),
      ])

      setState(prev => ({
        ...prev,
        profile: profileRes,
        preferences: prefsRes,
        loading: false,
      }))
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: 'Failed to load settings',
        loading: false,
      }))
    }
  }, [])

  useEffect(() => {
    fetchSettings()
  }, [fetchSettings])

  const handleSave = async (section: string) => {
    setState(prev => ({ ...prev, saving: true, error: null }))
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))

      if (section === 'preferences' && state.preferences) {
        localStorage.setItem('userPreferences', JSON.stringify(state.preferences))
      }

      setState(prev => ({ ...prev, saving: false }))
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: `Failed to save ${section}`,
        saving: false,
      }))
    }
  }

  const handlePreferenceChange = <K extends keyof UserPreferences>(key: K, value: UserPreferences[K]) => {
    setState(prev => ({
      ...prev,
      preferences: prev.preferences ? { ...prev.preferences, [key]: value } : null,
    }))
  }

  const handleNestedPreferenceChange = (parent: keyof UserPreferences, key: string, value: boolean) => {
    setState(prev => ({
      ...prev,
      preferences: prev.preferences ? {
        ...prev.preferences,
        [parent]: { ...prev.preferences[parent] as object, [key]: value } as any
      } : null,
    }))
  }

  const createApiKey = async () => {
    const name = prompt('Enter API key name:')
    if (!name) return

    setState(prev => ({ ...prev, saving: true, error: null }))
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))

      const newKey = {
        id: `key_${Date.now()}`,
        name,
        key: `sk_live_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`,
        createdAt: new Date().toISOString(),
        permissions: ['read', 'score'],
      }

      setState(prev => ({
        ...prev,
        preferences: prev.preferences ? {
          ...prev.preferences,
          apiKeys: [...prev.preferences.apiKeys, newKey]
        } : null,
        saving: false,
      }))
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: 'Failed to create API key',
        saving: false,
      }))
    }
  }

  const revokeApiKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) return

    setState(prev => ({ ...prev, saving: true }))
    try {
      await new Promise(resolve => setTimeout(resolve, 500))
      setState(prev => ({
        ...prev,
        preferences: prev.preferences ? {
          ...prev.preferences,
          apiKeys: prev.preferences.apiKeys.filter(k => k.id !== keyId)
        } : null,
        saving: false,
      }))
    } catch (err) {
      setState(prev => ({ ...prev, error: 'Failed to revoke API key', saving: false }))
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  if (state.loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Sidebar onCollapseChange={setSidebarCollapsed} collapsed={sidebarCollapsed} />
        <Header sidebarCollapsed={sidebarCollapsed} />
        <main className={cn('pt-16 transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-64')}>
          <div className="p-6 flex items-center justify-center h-[calc(100vh-4rem)]">
            <Loader2 className="h-12 w-12 text-primary-600 animate-spin" />
          </div>
        </main>
      </div>
    )
  }

  const tabs = [
    { id: 'profile', label: 'Profile', icon: <User className="h-4 w-4" /> },
    { id: 'preferences', label: 'Preferences', icon: <Palette className="h-4 w-4" /> },
    { id: 'security', label: 'Security', icon: <Shield className="h-4 w-4" /> },
    { id: 'api', label: 'API Keys', icon: <Key className="h-4 w-4" /> },
    { id: 'notifications', label: 'Notifications', icon: <Bell className="h-4 w-4" /> },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar onCollapseChange={setSidebarCollapsed} collapsed={sidebarCollapsed} />
      <Header sidebarCollapsed={sidebarCollapsed} />

      <main className={cn('pt-16 transition-all duration-300', sidebarCollapsed ? 'ml-16' : 'ml-64')}>
        <div className="p-6">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
            <p className="text-gray-500 mt-1">Manage your account, preferences, and security</p>
          </div>

          {state.error && (
            <div className="mb-6 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-danger-600" />
              <span className="text-danger-700">{state.error}</span>
            </div>
          )}

          <div className="mb-6 border-b border-gray-200">
            <nav className="flex gap-1" aria-label="Settings tabs">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setState(prev => ({ ...prev, activeTab: tab.id as any }))}
                  className={cn(
                    'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                    state.activeTab === tab.id
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

          <div className="max-w-3xl">
            {state.activeTab === 'profile' && state.profile && (
              <div className="space-y-6 animate-fade-in">
                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Profile Information</h2>
                    <div className="flex items-center gap-6 mb-6">
                      <div className="relative">
                        <div className="h-24 w-24 rounded-full bg-primary-100 flex items-center justify-center">
                          <User className="h-12 w-12 text-primary-600" />
                        </div>
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold text-gray-900">{state.profile.name}</h3>
                        <p className="text-gray-500">{state.profile.email}</p>
                        <span className="badge badge-primary mt-2 inline-block">{state.profile.role}</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-6 border-t border-gray-100">
                      <div>
                        <p className="text-xs text-gray-500">Member Since</p>
                        <p className="font-medium text-gray-900">{new Date(state.profile.createdAt).toLocaleDateString()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500">Last Login</p>
                        <p className="font-medium text-gray-900">{state.profile.lastLogin ? new Date(state.profile.lastLogin).toLocaleString() : 'Never'}</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Edit Profile</h2>
                    <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); handleSave('profile') }}>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="label">Full Name</label>
                          <input
                            type="text"
                            defaultValue={state.profile.name}
                            className="input"
                            required
                          />
                        </div>
                        <div>
                          <label className="label">Email</label>
                          <input
                            type="email"
                            defaultValue={state.profile.email}
                            className="input"
                            required
                          />
                        </div>
                      </div>
                      <div>
                        <label className="label">Role</label>
                        <select className="input" defaultValue={state.profile.role} disabled>
                          <option value="Administrator">Administrator</option>
                          <option value="Analyst">Analyst</option>
                          <option value="Viewer">Viewer</option>
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Role changes require administrator approval</p>
                      </div>
                      <button
                        type="submit"
                        className={cn('btn-primary', state.saving && 'opacity-50')}
                        disabled={state.saving}
                      >
                        {state.saving ? (
                          <>
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            Saving...
                          </>
                        ) : (
                          <>
                            <Save className="h-4 w-4 mr-2" />
                            Save Changes
                          </>
                        )}
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            )}

            {state.activeTab === 'preferences' && state.preferences && (
              <div className="space-y-6 animate-fade-in">
                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Appearance</h2>
                    <div>
                      <label className="label">Theme</label>
                      <div className="grid grid-cols-3 gap-3">
                        {themes.map((theme) => (
                          <button
                            key={theme.value}
                            type="button"
                            onClick={() => handlePreferenceChange('theme', theme.value as any)}
                            className={cn(
                              'p-4 rounded-lg border-2 transition-all',
                              state.preferences?.theme === theme.value
                                ? 'border-primary-500 bg-primary-50'
                                : 'border-gray-200 hover:border-gray-300'
                            )}
                          >
                            <div className="flex justify-center mb-2">{theme.icon}</div>
                            <p className="text-sm font-medium text-gray-900 text-center">{theme.label}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Localization</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="label">Language</label>
                        <select
                          value={state.preferences.language}
                          onChange={(e) => handlePreferenceChange('language', e.target.value)}
                          className="input"
                        >
                          {languages.map((lang) => (
                            <option key={lang.value} value={lang.value}>{lang.label}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="label">Timezone</label>
                        <select
                          value={state.preferences.timezone}
                          onChange={(e) => handlePreferenceChange('timezone', e.target.value)}
                          className="input"
                        >
                          {timezones.map((tz) => (
                            <option key={tz.value} value={tz.value}>{tz.label}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => handleSave('preferences')}
                    className={cn('btn-primary', state.saving && 'opacity-50')}
                    disabled={state.saving}
                  >
                    {state.saving ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4 mr-2" />
                        Save Preferences
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {state.activeTab === 'security' && (
              <div className="space-y-6 animate-fade-in">
                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Change Password</h2>
                    <form className="space-y-4 max-w-md">
                      <div>
                        <label className="label">Current Password</label>
                        <div className="relative">
                          <input type="password" className="input pr-10" placeholder="********" />
                          <button type="button" className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                            <Eye className="h-5 w-5" />
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="label">New Password</label>
                        <div className="relative">
                          <input type="password" className="input pr-10" placeholder="********" />
                          <button type="button" className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                            <EyeOff className="h-5 w-5" />
                          </button>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">Must be at least 8 characters with uppercase, lowercase, number, and special character</p>
                      </div>
                      <div>
                        <label className="label">Confirm New Password</label>
                        <input type="password" className="input" placeholder="********" />
                      </div>
                      <button type="submit" className="btn-primary">
                        <Save className="h-4 w-4 mr-2" />
                        Update Password
                      </button>
                    </form>
                  </div>
                </div>

                <div className="card border-warning-200 bg-warning-50">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-warning-800 mb-4 flex items-center gap-2">
                      <AlertCircle className="h-5 w-5" />
                      Danger Zone
                    </h2>
                    <p className="text-warning-700 mb-4">These actions are irreversible. Please proceed with caution.</p>
                    <div className="flex items-center justify-between p-4 bg-white rounded-lg">
                      <div>
                        <p className="font-medium text-gray-900">Delete Account</p>
                        <p className="text-sm text-gray-500">Permanently delete your account and all associated data</p>
                      </div>
                      <button className="btn-danger">Delete Account</button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {state.activeTab === 'api' && state.preferences && (
              <div className="space-y-6 animate-fade-in">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900">API Keys</h2>
                  <button onClick={createApiKey} className="btn-primary">
                    <Key className="h-4 w-4 mr-2" />
                    Create API Key
                  </button>
                </div>

                <div className="card">
                  <div className="overflow-x-auto">
                    <table className="table w-full">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Key</th>
                          <th>Permissions</th>
                          <th>Created</th>
                          <th>Last Used</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {state.preferences.apiKeys.map((key) => (
                          <tr key={key.id}>
                            <td className="font-medium text-gray-900">{key.name}</td>
                            <td>
                              <div className="flex items-center gap-2">
                                <code className="text-xs font-mono text-gray-600">{key.key}</code>
                                <button
                                  onClick={() => copyToClipboard(key.key)}
                                  className="p-1 text-gray-400 hover:text-gray-600"
                                  title="Copy key"
                                >
                                  <Copy className="h-4 w-4" />
                                </button>
                              </div>
                            </td>
                            <td>
                              <div className="flex flex-wrap gap-1">
                                {key.permissions.map((perm) => (
                                  <span key={perm} className="badge badge-info text-xs">{perm}</span>
                                ))}
                              </div>
                            </td>
                            <td className="text-sm text-gray-500">{new Date(key.createdAt).toLocaleDateString()}</td>
                            <td className="text-sm text-gray-500">{key.lastUsed ? new Date(key.lastUsed).toLocaleString() : 'Never'}</td>
                            <td>
                              <button
                                onClick={() => revokeApiKey(key.id)}
                                className="text-danger-600 hover:text-danger-700 text-sm font-medium"
                              >
                                Revoke
                              </button>
                            </td>
                          </tr>
                        ))}
                        {state.preferences.apiKeys.length === 0 && (
                          <tr>
                            <td colSpan={6} className="text-center py-8 text-gray-500">
                              No API keys created yet. Click "Create API Key" to get started.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="card border-info-200 bg-info-50">
                  <div className="p-4">
                    <h3 className="font-medium text-info-800 mb-2 flex items-center gap-2">
                      <Key className="h-4 w-4" />
                      API Key Usage
                    </h3>
                    <ul className="text-sm text-info-700 space-y-1 list-disc list-inside">
                      <li>Include API key in Authorization header: <code className="bg-white px-1 rounded">Authorization: Bearer &lt;your-api-key&gt;</code></li>
                      <li>Production keys start with <code className="bg-white px-1 rounded">sk_live_</code>, test keys with <code className="bg-white px-1 rounded">sk_test_</code></li>
                      <li>Keep your keys secure - never commit them to version control</li>
                      <li>Revoke compromised keys immediately</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {state.activeTab === 'notifications' && state.preferences && (
              <div className="space-y-6 animate-fade-in">
                <div className="card">
                  <div className="p-6">
                    <h2 className="text-lg font-semibold text-gray-900 mb-6">Notification Preferences</h2>
                    <div className="space-y-4">
                      {[
                        { key: 'email', label: 'Email Notifications', description: 'Receive notifications via email' },
                        { key: 'push', label: 'Push Notifications', description: 'Receive browser push notifications' },
                        { key: 'alerts', label: 'Alert Notifications', description: 'Get notified when fraud alerts are triggered' },
                        { key: 'reports', label: 'Report Notifications', description: 'Receive scheduled reports and summaries' },
                      ].map((item) => (
                        <div key={item.key} className="flex items-center justify-between">
                          <div>
                            <p className="font-medium text-gray-900">{item.label}</p>
                            <p className="text-sm text-gray-500">{item.description}</p>
                          </div>
                          <button
                            onClick={() => handleNestedPreferenceChange('notifications', item.key, !state.preferences?.notifications[item.key as keyof typeof state.preferences.notifications])}
                            className={cn(
                              'relative w-11 h-6 rounded-full transition-colors',
                              state.preferences?.notifications[item.key as keyof typeof state.preferences.notifications]
                                ? 'bg-primary-600'
                                : 'bg-gray-300'
                            )}
                            role="switch"
                            aria-checked={state.preferences?.notifications[item.key as keyof typeof state.preferences.notifications]}
                          >
                            <span className={cn(
                              'absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform',
                              state.preferences?.notifications[item.key as keyof typeof state.preferences.notifications]
                                ? 'translate-x-5'
                                : 'translate-x-0.5'
                            )} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => handleSave('preferences')}
                    className={cn('btn-primary', state.saving && 'opacity-50')}
                    disabled={state.saving}
                  >
                    {state.saving ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4 mr-2" />
                        Save Notification Preferences
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}