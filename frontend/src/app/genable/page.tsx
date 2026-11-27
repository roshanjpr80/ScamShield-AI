'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2, Shield, AlertTriangle, TrendingUp, BarChart2, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api'
import { Sidebar } from '@/components/Sidebar'
import { Header } from '@/components/Header'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface QuickAction {
  label: string
  prompt: string
  icon: React.ReactNode
}

export default function GenablePage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const quickActions: QuickAction[] = [
    { label: 'Analyze Transaction', prompt: 'Analyze this UPI transaction for fraud: ', icon: <Shield className="h-4 w-4" /> },
    { label: 'UPI Scam Info', prompt: 'What is a UPI scam and how can I avoid it? ', icon: <AlertTriangle className="h-4 w-4" /> },
    { label: 'SMS Phishing', prompt: 'Is this SMS text a phishing attempt? ', icon: <AlertTriangle className="h-4 w-4" /> },
    { label: 'Job Scam', prompt: 'Is this job offer legitimate? ', icon: <Shield className="h-4 w-4" /> },
    { label: 'Fraud Prevention', prompt: 'How can I protect myself from UPI scams? ', icon: <Shield className="h-4 w-4" /> },
    { label: 'Model Insights', prompt: 'Explain how the fraud detection model makes decisions. ', icon: <BarChart2 className="h-4 w-4" /> },
  ]

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleQuickAction = (prompt: string) => {
    const messageId = Date.now().toString()
    setMessages(prev => [...prev, {
      id: messageId,
      role: 'user',
      content: prompt,
      timestamp: new Date(),
    }])
    handleSendMessage(prompt)
  }

  const handleSendMessage = async (customPrompt?: string) => {
    const content = customPrompt || inputValue
    if (!content.trim()) return

    setInputValue('')
    setIsLoading(true)

    const messageId = Date.now().toString()
    const userMessage: Message = {
      id: messageId,
      role: 'user',
      content,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])

    try {
      const sessionMessages = [...messages, userMessage].map(m => ({
        role: m.role,
        content: m.content,
      }))

      // Use streaming API
      const stream = await apiClient.chatStream(sessionMessages)

      let assistantContent = ''
      let sources: any[] = []

      stream.onToken((token, isFinal) => {
        setMessages((prev: Message[]) => {
          const lastMsg = prev[prev.length - 1]
          if (lastMsg?.role === 'assistant' && lastMsg.id === messageId + '_stream') {
            return prev.slice(0, -1).concat({
              ...lastMsg,
              content: (lastMsg as any).content + token,
            } as Message)
          } else {
            return [...prev, {
              id: messageId + '_stream',
              role: 'assistant' as const,
              content: assistantContent + token,
              timestamp: new Date(),
            } as Message].slice(0, -1).concat([{
              ...prev[prev.length - 1],
              content: (prev[prev.length - 1] as any).content + token,
            } as Message])
          }
        })
      })

      stream.onSources((data) => {
        sources = data
      })

      stream.onComplete(() => {
        setIsLoading(false)
      })

      stream.onError((err) => {
        console.error('Stream error:', err)
        setIsLoading(false)
        setMessages(prev => prev.filter(m => m.id !== messageId + '_stream'))
      })

    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString() + '_error',
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        }
      ])
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (isLoading || !inputValue.trim()) return
    handleSendMessage()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Ctrl' + 'Enter' || e.key === 'Cmd' + 'Enter') {
      handleSubmit(e as any)
    }
  }

  if (messages.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Sidebar />
        <div className="ml-64">
          <Header sidebarCollapsed={false} />

          <main className="p-6 lg:p-8">
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-12">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-4">
                  <Sparkles className="h-8 w-8 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-gray-900 mb-4">
                  Welcome to Genable
                </h1>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                  Your AI assistant for fraud detection, scam education, and security analysis. Ask me anything about UPI fraud, SMS phishing, job scams, or any other security question.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
                {quickActions.map((action) => (
                  <button
                    key={action.label}
                    onClick={() => handleQuickAction(action.prompt)}
                    disabled={isLoading}
                    className={cn(
                      "p-4 border rounded-lg text-left transition-all hover:shadow-md",
                      isLoading && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      {action.icon}
                      <span className="font-medium text-gray-900">{action.label}</span>
                    </div>
                    <p className="text-sm text-gray-500">{action.prompt.substring(0, 50)}...</p>
                  </button>
                ))}
              </div>

              <div className="card">
                <div className="p-6">
                  <h3 className="font-semibold text-gray-900 mb-4">Start a conversation</h3>
                  <p className="text-gray-600 mb-4">
                    Type your question below. I can help with:
                  </p>
                  <ul className="list-disc list-inside text-gray-600 space-y-2">
                    <li>UPI fraud detection and prevention</li>
                    <li>SMS phishing identification</li>
                    <li>Job scam detection</li>
                    <li>Transaction analysis</li>
                    <li>Fraud model explanations</li>
                    <li>Compliance and security guidelines</li>
                  </ul>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <div className="ml-64">
        <Header sidebarCollapsed={false} />

        <main className="p-6 lg:p-8">
          <div className="max-w-4xl mx-auto">
            <div className="space-y-4 mb-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-4",
                    message.role === 'user' ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[80%] lg:max-w-[60%] p-4 rounded-lg",
                      message.role === 'user'
                        ? "bg-blue-600 text-white"
                        : "bg-white border border-gray-200"
                    )}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    <div className={cn(
                      "text-xs mt-2",
                      message.role === 'user' ? "text-blue-100" : "text-gray-500"
                    )}>
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div ref={messagesEndRef} />

            <div className="card">
              <div className="p-4">
                <form onSubmit={handleSubmit} className="flex gap-2">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about fraud detection, scams, or security..."
                    disabled={isLoading}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !inputValue.trim()}
                    className={cn(
                      "px-4 py-2 rounded-lg transition-all",
                      isLoading
                        ? "bg-gray-300 cursor-not-allowed"
                        : "bg-blue-600 hover:bg-blue-700 text-white"
                    )}
                  >
                    {isLoading ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Send className="h-5 w-5" />
                    )}
                  </button>
                </form>
                <div className="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Ctrl+Enter to send</span>
                  <span>•</span>
                  <span>Shift+Enter for new line</span>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}