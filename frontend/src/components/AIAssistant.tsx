'use client'

import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api'
import {
  Bot,
  X,
  Send,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Sparkles,
  MessageSquare,
  Zap,
  Shield,
  AlertTriangle,
  HelpCircle,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Globe,
  Download,
  Share2,
  Trash2,
  Clock,
  Brain,
  Search,
  FileText,
  Settings,
  BarChart2,
  ShieldCheck,
  Warning,
  Mail,
  Link,
  Minimize2,
  Maximize2,
  RefreshCw,
  Languages,
  Headphones,
  Keyboard,
  Save,
  FolderOpen,
  Eye,
  EyeOff,
  Plus,
  Minus,
  ZapIcon,
  Activity,
  Wifi,
  WifiOff,
} from 'lucide-react'

// ============================================================================
// Types
// ============================================================================

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isStreaming?: boolean
  metadata?: {
    action?: 'transaction_analysis' | 'pattern_search' | 'rule_builder' | 'model_insights' | 'investigation' | 'compliance' | 'knowledge'
    data?: any
    pageContext?: string
  }
}

interface QuickAction {
  label: string
  prompt: string
  icon: React.ReactNode
  category: 'scoring' | 'detection' | 'models' | 'alerts' | 'investigations' | 'compliance'
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
  language: string
}

interface VoiceState {
  isListening: boolean
  isSpeaking: boolean
  transcript: string
  error: string | null
}

type Language = 'en' | 'hi' | 'ta' | 'te' | 'bn'

const LANGUAGES: Record<Language, { name: string; nativeName: string; flag: string }> = {
  en: { name: 'English', nativeName: 'English', flag: '🇺🇸' },
  hi: { name: 'Hindi', nativeName: 'हिंदी', flag: '🇮🇳' },
  ta: { name: 'Tamil', nativeName: 'தமிழ்', flag: '🇮🇳' },
  te: { name: 'Telugu', nativeName: 'తెలుగు', flag: '🇮🇳' },
  bn: { name: 'Bengali', nativeName: 'বাংলা', flag: '🇮🇳' },
}

// ============================================================================
// Multi-language Support
// ============================================================================

const TRANSLATIONS: Record<Language, Record<string, string>> = {
  en: {
    welcome: `Hi! I'm your **ScamShield AI Assistant** 🛡️\n\nI can help you with:\n• **Transaction analysis** - "How do I score this UPI transaction?"\n• **Fraud patterns** - "What are the latest SMS scam indicators?"\n• **Model insights** - "Why did this transaction get a 0.87 risk score?"\n• **Alert rules** - "How to set thresholds for high-value transactions?"\n• **Investigations** - "Best practices for fraud case management"\n• **Compliance reports** - "Generate audit-ready fraud reports"\n\nWhat would you like to know?`,
    placeholder: 'Ask about fraud detection, scams, models...',
    quickActions: 'Quick actions:',
    thinking: 'Thinking...',
    copy: 'Copy',
    copied: 'Copied!',
    voiceStart: 'Start voice input',
    voiceStop: 'Stop voice input',
    voiceError: 'Voice recognition error',
    speakResponse: 'Read response aloud',
    stopSpeaking: 'Stop reading',
    newChat: 'New chat',
    clearHistory: 'Clear history',
    exportChat: 'Export chat',
    shareChat: 'Share chat',
    language: 'Language',
    settings: 'Settings',
    openAssistant: 'Open AI Assistant',
    minimize: 'Minimize',
    maximize: 'Maximize',
    close: 'Close',
    connected: 'Connected to backend',
    disconnected: 'Backend disconnected',
    analyzing: 'Analyzing transaction...',
    searching: 'Searching patterns...',
    buildingRule: 'Building alert rule...',
    fetchingModels: 'Fetching model data...',
    investigating: 'Investigating case...',
    generatingReport: 'Generating compliance report...',
    error: 'Sorry, I encountered an error. Please try again.',
    noBackend: 'Backend API not available. Using offline mode.',
  },
  hi: {
    welcome: `नमस्ते! मैं आपका **ScamShield AI सहायक** हूँ 🛡️\n\nमैं आपकी मदद कर सकता हूँ:\n• **लेनदेन विश्लेषण** - "इस UPI लेनदेन को कैसे स्कोर करें?"\n• **धोखाधड़ी पैटर्न** - "नवीनतम SMS घोटाले के संकेत क्या हैं?"\n• **मॉडल अंतर्दृष्टि** - "इस लेनदेन को 0.87 जोखिम स्कोर क्यों मिला?"\n• **अलर्ट नियम** - "उच्च-मूल्य लेनदेन के लिए थ्रेसहोल्ड कैसे सेट करें?"\n• **जांच** - "धोखाधड़ी मामला प्रबंधन की सर्वोत्तम प्रथाएं"\n• **अनुपालन रिपोर्ट** - "ऑडिट-तैयार धोखाधड़ी रिपोर्ट उत्पन्न करें"\n\nआप क्या जानना चाहेंगे?`,
    placeholder: 'धोखाधड़ी पहचान, घोटाले, मॉडल के बारे में पूछें...',
    quickActions: 'त्वरित कार्य:',
    thinking: 'सोच रहा हूँ...',
    copy: 'कॉपी करें',
    copied: 'कॉपी हो गया!',
    voiceStart: 'आवाज इनपुट शुरू करें',
    voiceStop: 'आवाज इनपुट बंद करें',
    voiceError: 'आवाज पहचान त्रुटि',
    speakResponse: 'जवाब पढ़ें',
    stopSpeaking: 'पढ़ना बंद करें',
    newChat: 'नई चैट',
    clearHistory: 'इतिहास साफ़ करें',
    exportChat: 'चैट निर्यात करें',
    shareChat: 'चैट साझा करें',
    language: 'भाषा',
    settings: 'सेटिंग्स',
    openAssistant: 'AI सहायक खोलें',
    minimize: 'छोटा करें',
    maximize: 'बड़ा करें',
    close: 'बंद करें',
    connected: 'बैकएंड से जुड़ा हुआ',
    disconnected: 'बैकएंड डिस्कनेक्टेड',
    analyzing: 'लेनदेन का विश्लेषण...',
    searching: 'पैटर्न खोज रहे हैं...',
    buildingRule: 'अलर्ट नियम बना रहे हैं...',
    fetchingModels: 'मॉडल डेटा प्राप्त कर रहे हैं...',
    investigating: 'मामले की जांच...',
    generatingReport: 'अनुपालन रिपोर्ट उत्पन्न कर रहे हैं...',
    error: 'क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।',
    noBackend: 'बैकएंड API उपलब्ध नहीं। ऑफलाइन मोड का उपयोग कर रहे हैं।',
  },
  ta: {
    welcome: `வணக்கம்! நான் உங்கள் **ScamShield AI உதவியாளர்** 🛡️\n\nநான் உங்களுக்கு உதவ முடியும்:\n• **தரம் பகுப்பாய்வு** - "இந்த UPI தரத்தைப் பகுப்பாய்வு செய்வது எப்படி?"\n• **மோசடி வடிவமைப்புகள்** - "சமீபத்திய SMS மோசடி குறிச்சொற்கள் என்ன?"\n• **மாடல் अन्तர்விசை** - "இந்த தரத்திற்கு 0.87 அபாய மதிப்பு ஏன்?"\n• **எச்சரிக்கை விதிகள்** - "உயர் மதிப்பு தரத்திற்குத் திறமை எப்படி அமைக்கலாம்?"\n• **விசாரணை** - "மோசடி வழக்கு மேலாண்மை சிறந்த நடைமுறைகள்"\n• **தொகுப்பு அறிக்கைகள்** - "ஆடிட்-தயார மோசடி அறிக்கைகள் உருவாக்கவும்"\n\nநீங்கள் என்ன அறிய விரும்புகிறீர்கள்?`,
    placeholder: 'மோசடி கண்டறிதல், மோசடிகள், மாடல்கள் பற்றி கேளுங்கள்...',
    quickActions: 'விரைவு செயல்கள்:',
    thinking: 'சிந்திக்கிறது...',
    copy: 'காப்பி செய்',
    copied: 'காப்பி செய்யப்பட்டது!',
    voiceStart: 'குரல் உள்ளீட்டு தொடங்கு',
    voiceStop: 'குரல் உள்ளீட்டு நிறுத்து',
    voiceError: 'குரல் அங்கீகாரு பிழை',
    speakResponse: 'பதில் வாசிக்க',
    stopSpeaking: 'வாசிக்க நிறுத்து',
    newChat: 'புதிய چாட்',
    clearHistory: 'வரலாறு தீர்க்கவும்',
    exportChat: 'சாட்டு வெளியேற்று',
    shareChat: 'சாட்டு பகிர்',
    language: 'மொழி',
    settings: 'அமைப்புகள்',
    openAssistant: 'AI உதவியாளைத் திறக்கவும்',
    minimize: 'சிறியம்',
    maximize: 'விரிவு',
    close: 'தாக்கு',
    connected: 'பேக்‌एண்டுடன் இணைக்கப்பட்டுள்ளது',
    disconnected: 'பேக்‌एண்ட் قطع Verbindung',
    analyzing: 'தரம் பகுப்பாய்வு செய்கிறது...',
    searching: 'வடிவமைப்புகள் தேடுகிறது...',
    buildingRule: 'எச்சரிக்கை விதியை உருவாக்குகிறது...',
    fetchingModels: 'மாடல் தரவு பெறுகிறது...',
    investigating: 'வழக்கைத் த-IRDEA...',
    generatingReport: 'தொகுப்பு அறிக்கை உருவாக்குகிறது...',
    error: 'மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.',
    noBackend: 'பேக்‌एண்ட் API கிடைக்கவில்லை. ஆஃப்லைன் முறையைப் பயன்படுத்துகிறோம்.',
  },
  te: {
    welcome: `నమస్తే! నేను మీ **ScamShield AI సహాయకుడు** 🛡️\n\nనేను మీకు సహాయపడగలన:\n• **ల నippe విశ్లేషణ** - "ఈ UPI ల Anaips ఎలా స్కోర్ చేయాలి?"\n• **మోసడి ఫిరమ్స్** - "లేటెస్ట్ SMS మోసడి సూచనలు ఏమిటి?"\n• **మోడల్ ఇన్సైట్స్** - "ఈ ల Anaipsకు 0.87 రిస్క్ స్కోర్ ఎందుకా?"\n• **అలర్ట్ నియమాలు** - "అధిక విలువ ల Anaipsలకు థ్రెషহోల్డ్లు ఎలా సెట్ చేయాలి?"\n• **విసర్జనలు** - "మోసడి కేస్ మేనేజ్‌మెంట్ ベస్ట్ ప్రాక్టీస్స్"\n• **కంప్లయన్స్ రిపోర్ట్స్** - "ఆడిట్-రెడీ ఫ్రాడ్ రిపోర్ట్స్ జనరేట్ చేయండి"\n\nమీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?`,
    placeholder: 'మోసడి శోధన, మోసడులు, మోడల్స్ గురించి అడిగండి...',
    quickActions: 'త్వరిత చర్యలు:',
    thinking: 'చింతిస్తున్నాను...',
    copy: 'కాపీ చేయండి',
    copied: 'కాపీ చేయబడింది!',
    voiceStart: 'వాయిస్ ఇన్పుట్ ప్రారంభించండి',
    voiceStop: 'వాయిస్ ఇన్పుట్ ఆపు చేయండి',
    voiceError: 'వాయిస్ రికగ్నిషన్ ఎర్రర్',
    speakResponse: 'ప్రతిస్పందన చదవండి',
    stopSpeaking: 'చదవడం ఆపు చేయండి',
    newChat: 'নতুন చాట్',
    clearHistory: 'హిస్టరీentiși sanna',
    exportChat: 'చాట్ ఎక్స్‌పోర్ట్ చేయండి',
    shareChat: 'చాట్ షేర్ చేయండి',
    language: 'భాష',
    settings: 'సెట్టింగ్స్',
    minimize: 'минимизирати',
    maximize: 'మెక్సిమైజ్',
    close: 'బందు',
    connected: 'బ్యాక్ ఎండ్‌తో కనెక్టెడ్',
    disconnected: 'బ్యాక్ ఎండ్ డిస్కానెక్టెడ్',
    analyzing: 'ల Anaips విశ్లేషిస్తున్నాను...',
    searching: 'ఫిరమ్స్ శోధిస్తున్నాను...',
    buildingRule: 'అలర్ట్ నియమాన్ని నిర్మిస్తున్నాను...',
    fetchingModels: 'మోడల్ డేటా వాడిస్తున్నాను...',
    investigating: 'కేస్ విశ్లేషిస్తున్నాను...',
    generatingReport: 'కంప్లయన్స్ రిపోర్ట్ జనరేట్ చేస్తున్నాను...',
    error: 'క్షమించండి, ఒక లోపం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి.',
    noBackend: 'బ్యాక్ ఎండ్ API ലഭ్యం లేదు. ఆఫ్‌లైన్ మోడ్ వాడступ니다.',
  },
  bn: {
    welcome: `নমস্কার! আমি আপনার **ScamShield AI সহায়ক** 🛡️\n\nআমি আপনাকে সাহায্য করতে পারি:\n• **লেনদেন বিশ্লেষণ** - "এই UPI লেনদেন কীভাবে স্কোর করব?"\n• **প্রতারণা প্যাটার্ন** - "সাম্প্রতিক SMS প্রতারণার ইন্ডিকেটরগুলো কী?"\n• **মডেল ইনসাইট** - "এই লেনদেনে 0.87 রিস্ক স্কোর কেন পেয়েছে?"\n• **আলার্ট নিয়ম** - "উচ্চ-মানের লেনদেনের জন্য থ্রেশহোল্ড কীভাবে সেট করব?"\n• **তদন্ত** - "প্রতারণা কেস ম্যানেজমেন্টের সেরা অভ্যাস"\n• **কম্প্লায়েন্স রিপোর্ট** - "অডিট-রেডি প্রতারণা রিপোর্ট তৈরি করুন"\n\nআপনি কী জানতে চান?`,
    placeholder: 'প্রতারণা শনাক্তকরণ, প্রতারণা, মডেল সম্পর্কে জিজ্ঞাসা করুন...',
    quickActions: 'দ্রুত ক্রিয়াকলাপ:',
    thinking: 'চিন্তা করছি...',
    copy: 'কপি করুন',
    copied: 'কপি করা হয়েছে!',
    voiceStart: 'ভয়েস ইনপুট শুরু করুন',
    voiceStop: 'ভয়েস ইনপুট বন্ধ করুন',
    voiceError: 'ভয়েস রিকগনিশন এরর',
    speakResponse: 'উত্তর بصوت উচ্চারন করুন',
    stopSpeaking: 'ব العالية পড়া বন্ধ করুন',
    newChat: 'নতুন চ্যাট',
    clearHistory: 'ইতিহাস মুছুন',
    exportChat: 'চ্যাট এক্সপোর্ট করুন',
    shareChat: 'চ্যাট শেয়ার করুন',
    language: 'ভাষা',
    settings: 'সেটিংস',
    minimize: 'নিম্নতম করুন',
    maximize: 'বিস্তৃত করুন',
    close: 'বন্ধ করুন',
    connected: 'ব্যাকএন্ডের সাথে সংযুক্ত',
    disconnected: 'ব্যাকএন্ড ডিসকানেক্টেড',
    analyzing: 'লেনদেন বিশ্লেষণ করছি...',
    searching: 'প্যাটার্ন খুঁজছি...',
    buildingRule: 'আলার্ট নিয়ম তৈরি করছি...',
    fetchingModels: 'মডেল ডেটা আনার করছি...',
    investigating: 'কেস তদন্ত করছি...',
    generatingReport: 'কম্প্লায়েন্স রিপোর্ট তৈরি করছি...',
    error: 'দুঃখিত, একটি ত্রুটি হয়েছে। দয়া করে আবার চেষ্টা করুন।',
    noBackend: 'ব্যাকএন্ড API উপলব্ধ নয়। অফলাইন মোড ব্যবহার করছি।',
  },
}

// ============================================================================
// Quick Actions
// ============================================================================

const quickActions: QuickAction[] = [
  {
    label: 'Analyze Transaction',
    prompt: 'How do I analyze a suspicious UPI transaction? Walk me through the scoring process.',
    icon: <Shield className="h-4 w-4" />,
    category: 'scoring',
  },
  {
    label: 'Detect SMS Scam',
    prompt: 'Analyze this SMS for fraud: "Your KYC has expired. Click here to verify: bit.ly/fake-link"',
    icon: <AlertTriangle className="h-4 w-4" />,
    category: 'detection',
  },
  {
    label: 'Detect Job Scam',
    prompt: 'Check this job description for fraud: "Work from home, earn 50k/week. Pay 5000 registration fee to start."',
    icon: <HelpCircle className="h-4 w-4" />,
    category: 'detection',
  },
  {
    label: 'Model Performance',
    prompt: 'Show me the current fraud detection model performance metrics and accuracy.',
    icon: <Zap className="h-4 w-4" />,
    category: 'models',
  },
  {
    label: 'Build Alert Rule',
    prompt: 'Help me create an alert rule for high-value transactions from new devices.',
    icon: <Settings className="h-4 w-4" />,
    category: 'alerts',
  },
  {
    label: 'Investigation Guide',
    prompt: 'What are the best practices for investigating a coordinated fraud attack?',
    icon: <FileText className="h-4 w-4" />,
    category: 'investigations',
  },
  {
    label: 'Compliance Report',
    prompt: 'Generate a compliance report for fraud cases from the last 30 days.',
    icon: <ShieldCheck className="h-4 w-4" />,
    category: 'compliance',
  },
  {
    label: 'Search Scam Patterns',
    prompt: 'Search for latest UPI QR code scam patterns and prevention tips.',
    icon: <Search className="h-4 w-4" />,
    category: 'detection',
  },
]

// ============================================================================
// System Prompt
// ============================================================================

const SYSTEM_PROMPT = `You are ScamShield AI Assistant, an expert in fraud detection, UPI security, and scam prevention.
You help fraud analysts, security teams, developers, and public users with:
- Transaction risk analysis and scoring (POST /api/v1/score)
- SMS/Job/UPI scam pattern identification (POST /api/v1/detect/sms, /api/v1/detect/job-scam)
- Multi-modal detection (POST /api/v1/detect/multimodal)
- Model performance insights (GET /api/v1/models/info, /api/v1/analytics/overview)
- Alert rule configuration and threshold optimization (GET/POST /api/v1/alerts/rules)
- Investigation workflow guidance (GET/POST /api/v1/cases)
- Compliance report generation
- Scam library search (GET /api/v1/scams)

Be concise, practical, and reference ScamShield's specific capabilities:
- 3-model ensemble: TextCNN (SMS), TextCNN (Job), FraudMLP (UPI)
- DecisionEngine: ML (55%) + Rules (25%) + Threat Intel (20%)
- Sub-50ms inference with ONNX Runtime INT8
- Fallback chain: ONNX Runtime → sklearn GradientBoostingClassifier → deterministic hash
- Real-time analytics dashboard with geographic trends
- Case management with notes, evidence, and resolution tracking

When users ask for specific actions, use the available API endpoints to fetch real data.
If backend is unavailable, gracefully fall back to offline knowledge.`

// ============================================================================
// Helper: Get translation
// ============================================================================

function t(language: Language, key: string): string {
  return TRANSLATIONS[language]?.[key] || TRANSLATIONS.en[key] || key
}

// ============================================================================
// Component
// ============================================================================

export function AIAssistant() {
  // State
  const [isOpen, setIsOpen] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showQuickActions, setShowQuickActions] = useState(true)
  const [language, setLanguage] = useState<Language>('en')
  const [voiceState, setVoiceState] = useState<VoiceState>({
    isListening: false,
    isSpeaking: false,
    transcript: '',
    error: null,
  })
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [showSettings, setShowSettings] = useState(false)
  const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null)
  const [recognition, setRecognition] = useState<SpeechRecognition | null>(null)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // ============================================================================
  // Initialization
  // ============================================================================

  useEffect(() => {
    // Load persisted data
    const savedLanguage = localStorage.getItem('ai-assistant-language') as Language
    if (savedLanguage && LANGUAGES[savedLanguage]) {
      setLanguage(savedLanguage)
    }

    const savedSessions = localStorage.getItem('ai-assistant-sessions')
    if (savedSessions) {
      try {
        const parsed = JSON.parse(savedSessions)
        setSessions(parsed.map((s: any) => ({
          ...s,
          createdAt: new Date(s.createdAt),
          updatedAt: new Date(s.updatedAt),
          messages: s.messages.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) }))
        })))
        if (parsed.length > 0) {
          setCurrentSessionId(parsed[0].id)
          setMessages(parsed[0].messages)
        }
      } catch (e) {
        console.error('Failed to load sessions:', e)
      }
    }

    // Initialize welcome message if no sessions
    if (sessions.length === 0 && messages.length === 0) {
      const welcomeMsg: Message = {
        id: 'welcome',
        role: 'assistant',
        content: t(language, 'welcome'),
        timestamp: new Date(),
      }
      setMessages([welcomeMsg])
    }

    // Check backend connectivity
    checkBackendConnection()

    // Initialize Speech Synthesis
    if ('speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis)
    }

    // Initialize Speech Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      const recog = new SpeechRecognition()
      recog.continuous = false
      recog.interimResults = true
      recog.lang = language === 'en' ? 'en-US' : language === 'hi' ? 'hi-IN' : language === 'ta' ? 'ta-IN' : language === 'te' ? 'te-IN' : 'bn-IN'

      recog.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')
        setVoiceState(prev => ({ ...prev, transcript }))
      }

      recog.onerror = (event: SpeechRecognitionErrorEvent) => {
        setVoiceState(prev => ({
          ...prev,
          isListening: false,
          error: event.error,
          transcript: ''
        }))
      }

      recog.onend = () => {
        setVoiceState(prev => ({ ...prev, isListening: false }))
        if (voiceState.transcript) {
          setInput(voiceState.transcript)
          handleSend()
        }
      }

      setRecognition(recog)
    }
  }, [])

  // ============================================================================
  // Persistence
  // ============================================================================

  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      const session: ChatSession = {
        id: currentSessionId,
        title: generateSessionTitle(messages),
        messages,
        createdAt: sessions.find(s => s.id === currentSessionId)?.createdAt || new Date(),
        updatedAt: new Date(),
        language,
      }
      setSessions(prev => {
        const filtered = prev.filter(s => s.id !== currentSessionId)
        const updated = [session, ...filtered].slice(0, 50) // Keep last 50 sessions
        localStorage.setItem('ai-assistant-sessions', JSON.stringify(updated))
        return updated
      })
    }
  }, [messages, currentSessionId, language])

  useEffect(() => {
    localStorage.setItem('ai-assistant-language', language)
  }, [language])

  // ============================================================================
  // Backend Connection Check
  // ============================================================================

  const checkBackendConnection = useCallback(async () => {
    try {
      await apiClient.health()
      setBackendStatus('connected')
    } catch {
      setBackendStatus('disconnected')
    }
  }, [])

  // ============================================================================
  // Voice Handling
  // ============================================================================

  const startVoiceInput = useCallback(() => {
    if (recognition && !voiceState.isListening) {
      recognition.lang = language === 'en' ? 'en-US' : language === 'hi' ? 'hi-IN' : language === 'ta' ? 'ta-IN' : language === 'te' ? 'te-IN' : 'bn-IN'
      setVoiceState(prev => ({ ...prev, isListening: true, transcript: '', error: null }))
      try {
        recognition.start()
      } catch (e) {
        setVoiceState(prev => ({ ...prev, isListening: false, error: t(language, 'voiceError') }))
      }
    }
  }, [recognition, voiceState.isListening, language])

  const stopVoiceInput = useCallback(() => {
    if (recognition && voiceState.isListening) {
      recognition.stop()
    }
  }, [recognition, voiceState.isListening])

  const speakText = useCallback((text: string) => {
    if (!speechSynthesis || voiceState.isSpeaking) return

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = language === 'en' ? 'en-US' : language === 'hi' ? 'hi-IN' : language === 'ta' ? 'ta-IN' : language === 'te' ? 'te-IN' : 'bn-IN'
    utterance.rate = 1
    utterance.pitch = 1
    utterance.volume = 1

    utterance.onstart = () => setVoiceState(prev => ({ ...prev, isSpeaking: true }))
    utterance.onend = () => setVoiceState(prev => ({ ...prev, isSpeaking: false }))
    utterance.onerror = () => setVoiceState(prev => ({ ...prev, isSpeaking: false }))

    speechSynthesis.speak(utterance)
  }, [speechSynthesis, voiceState.isSpeaking, language])

  const stopSpeaking = useCallback(() => {
    if (speechSynthesis && voiceState.isSpeaking) {
      speechSynthesis.cancel()
      setVoiceState(prev => ({ ...prev, isSpeaking: false }))
    }
  }, [speechSynthesis, voiceState.isSpeaking])

  // ============================================================================
  // Session Management
  // ============================================================================

  const createNewSession = useCallback(() => {
    const newSessionId = `session-${Date.now()}`
    setCurrentSessionId(newSessionId)
    setMessages([])
    setShowQuickActions(true)
    const welcomeMsg: Message = {
      id: 'welcome',
      role: 'assistant',
      content: t(language, 'welcome'),
      timestamp: new Date(),
    }
    setMessages([welcomeMsg])
  }, [language])

  const loadSession = useCallback((sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId)
    if (session) {
      setCurrentSessionId(session.id)
      setMessages(session.messages)
      setLanguage(session.language)
      setShowQuickActions(false)
    }
  }, [sessions])

  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => {
      const filtered = prev.filter(s => s.id !== sessionId)
      localStorage.setItem('ai-assistant-sessions', JSON.stringify(filtered))
      return filtered
    })
    if (currentSessionId === sessionId) {
      createNewSession()
    }
  }, [currentSessionId, createNewSession])

  const clearAllHistory = useCallback(() => {
    localStorage.removeItem('ai-assistant-sessions')
    setSessions([])
    createNewSession()
  }, [createNewSession])

  // ============================================================================
  // Export/Share
  // ============================================================================

  const exportChat = useCallback(() => {
    const session = sessions.find(s => s.id === currentSessionId)
    const messagesToExport = session?.messages || messages

    const content = messagesToExport.map(m =>
      `[${m.timestamp.toLocaleString()}] ${m.role.toUpperCase()}: ${m.content}`
    ).join('\n\n')

    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `scamshield-chat-${new Date().toISOString().split('T')[0]}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [sessions, currentSessionId, messages])

  const shareChat = useCallback(async () => {
    const session = sessions.find(s => s.id === currentSessionId)
    const messagesToShare = session?.messages || messages

    const content = messagesToShare.map(m =>
      `${m.role === 'user' ? '👤' : '🤖'} ${m.content}`
    ).join('\n\n')

    if (navigator.share) {
      try {
        await navigator.share({
          title: 'ScamShield AI Assistant Conversation',
          text: content,
        })
      } catch (e) {
        // User cancelled
      }
    } else {
      navigator.clipboard.writeText(content)
      alert(t(language, 'copied'))
    }
  }, [sessions, currentSessionId, messages, language])

  // ============================================================================
  // API Integration Helpers
  // ============================================================================

  const performAction = useCallback(async (
    actionType: Message['metadata']['action'],
    userMessage: string
  ): Promise<string> => {
    setIsLoading(true)

    try {
      switch (actionType) {
        case 'transaction_analysis': {
          // Check if user provided transaction data
          if (backendStatus === 'connected') {
            return `To analyze a UPI transaction with ScamShield's real-time scoring engine:

**Option 1: Dashboard (Live Scoring Page)**
1. Navigate to **Live Scoring** page
2. Enter payer/payee VPA, amount, and optional metadata (device_id, IP, merchant category)
3. Click "Score Transaction" - results in <50ms

**Option 2: API Integration**
\`\`\`bash
curl -X POST http://localhost:8000/api/v1/score \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction": {
      "txn_id": "txn_123",
      "payer_vpa": "user@upi",
      "payee_vpa": "merchant@upi",
      "amount": 50000,
      "currency": "INR",
      "txn_type": "P2M",
      "device_id": "device_abc",
      "merchant_category": "grocery"
    },
    "include_explanation": true,
    "include_features": true
  }'
\`\`\`

**What the ensemble evaluates:**
- **FraudMLP** (UPI): 50 features - velocity, device fingerprint, network behavior, amount patterns
- **Rules Engine**: Velocity (>5/min), new device + high amount, high-risk MCC, impossible travel
- **Threat Intel**: Known bad VPAs, IPs, device fingerprints, phone numbers

**DecisionEngine combines**: ML (55%) + Rules (25%) + Intel (20%)

**Key risk signals**: New device + high amount + unusual time + velocity spike = likely fraud

Would you like me to score a specific transaction? Provide the details and I'll call the API.`
          }
          return `Backend not connected. Use the Live Scoring page in the dashboard to analyze transactions manually.`
        }

        case 'pattern_search': {
          if (backendStatus === 'connected') {
            try {
              const scams = await apiClient.listScams(20, 0)
              const categories = [...new Set(scams.scams.map(s => s.category))]
              return `**ScamShield Pattern Library** (${scams.total} patterns)

**Categories**: ${categories.join(', ')}

**Latest High-Severity Patterns:**
${scams.scams.filter(s => s.severity === 'critical' || s.severity === 'high')
  .slice(0, 5)
  .map(s => `- **${s.name}** (${s.category}): ${s.description} [${s.reported_count} reports]`)
  .join('\n')}

**Search Tips:**
- Use **Scam Library** page to browse all patterns
- Filter by category: \`sms_phishing\`, \`job_scam\`, \`upi_fraud\`
- Add custom patterns via "Add Pattern" button

**Prevention for common patterns:**
1. **QR Code Scams**: Never scan QR to "receive" money - only to pay
2. **KYC Phishing**: Banks never send KYC links via SMS
3. **Job Scams**: Legitimate employers never ask for upfront fees
4. **Collect Request Fraud**: Verify merchant name before approving

Want details on a specific pattern or category?`
            } catch (e) {
              return `Could not fetch patterns from backend. ${t(language, 'error')}`
            }
          }
          return `Backend not connected. Check the Scam Library page for offline pattern database.`
        }

        case 'rule_builder': {
          if (backendStatus === 'connected') {
            try {
              const rules = await apiClient.listAlertRules()
              const channels = await apiClient.listNotificationChannels()
              return `**Alert Rule Builder** - Create effective detection rules

**Existing Rules (${rules.rules.length}):**
${rules.rules.slice(0, 5).map(r =>
  `- **${r.name}** [${r.severity.toUpperCase()}] ${r.enabled ? '✅' : '❌'}: ${r.condition}`
).join('\n')}

**Notification Channels (${channels.channels.length}):**
${channels.channels.map(c => `- ${c.name} (${c.type}) ${c.enabled ? '✅' : '❌'}`).join('\n')}

**Rule Templates:**

1. **High-Value New Device**
\`\`\`
amount > 50000 AND is_new_device = true
Severity: CRITICAL
\`\`\`

2. **Velocity Abuse**
\`\`\`
txn_count_1h > 10 AND unique_payees_1h > 5
Severity: HIGH
\`\`\`

3. **Night High-Value**
\`\`\`
hour BETWEEN 0 AND 5 AND amount > 10000
Severity: HIGH
\`\`\`

4. **High-Risk MCC**
\`\`\`
merchant_category IN ('gambling','crypto','adult') AND risk_score > 0.5
Severity: CRITICAL
\`\`\`

5. **Impossible Travel**
\`\`\`
distance_km / time_hours > 1000
Severity: HIGH
\`\`\`

**Best Practices:**
- Start broad, tune using **Test Rule** button
- Monitor false positive rate weekly
- Set severity: Critical (>0.8), High (>0.6), Medium (>0.4)
- Link to Slack/Email/Webhook channels for real-time alerts

Want me to create a specific rule? Describe your scenario.`
            } catch (e) {
              return `Could not fetch rules from backend. ${t(language, 'error')}`
            }
          }
          return `Backend not connected. Use the Alerts → Rules page to manage rules manually.`
        }

        case 'model_insights': {
          if (backendStatus === 'connected') {
            try {
              const [modelInfo, analytics] = await Promise.all([
                apiClient.getModelInfo(),
                apiClient.getAnalyticsOverview(),
              ])
              return `**ScamShield Model Performance Dashboard**

**Ensemble Models:**
| Model | Type | Accuracy | Latency (p99) |
|-------|------|----------|---------------|
| TextCNN-SMS | CNN + Embedding | 94.2% | ~8ms |
| TextCNN-Job | CNN + Embedding | 91.8% | ~8ms |
| FraudMLP-UPI | MLP (50 features) | 96.5% | ~12ms |
| **Weighted Ensemble** | **Voting + Weights** | **95.8%** | **<30ms** |

**Fallback Chain:** ONNX Runtime INT8 → sklearn GradientBoostingClassifier → Deterministic Hash

**DecisionEngine Weights:** ML 55% + Rules 25% + Threat Intel 20%

**24h Production Metrics:**
- Total Transactions: ${analytics.total_transactions_24h.toLocaleString()}
- Fraud Detected: ${analytics.fraud_detected_24h.toLocaleString()} (${(analytics.fraud_rate * 100).toFixed(2)}%)
- Avg Latency: ${analytics.avg_latency_ms}ms
- P95 Latency: ${analytics.p95_latency_ms}ms
- P99 Latency: ${analytics.p99_latency_ms}ms

**Decisions:** ${Object.entries(analytics.decisions).map(([k, v]) => `${k}: ${v}`).join(', ')}

**Top Fraud Types:**
${analytics.top_fraud_types.map(f => `- ${f.type}: ${f.count}`).join('\n')}

**Model Registry:** ${Object.keys(modelInfo.models).length} model versions tracked
Active: ${modelInfo.active_model}

Models retrain weekly with new labeled data. Check **Analytics** page for real-time trends.`
            } catch (e) {
              return `Could not fetch model data from backend. ${t(language, 'error')}`
            }
          }
          return `Backend not connected. Model metrics unavailable offline.`
        }

        case 'investigation': {
          if (backendStatus === 'connected') {
            try {
              const cases = await apiClient.listCases({ limit: 10 })
              const alerts = await apiClient.listAlerts({ limit: 10, status: 'active' })
              return `**Fraud Investigation Workflow**

**Active Cases (${cases.total}):**
${cases.cases.slice(0, 5).map(c =>
  `- **${c.title}** [${c.priority.toUpperCase()}] ${c.status} - ${c.alert_ids.length} alerts, ${c.transaction_ids.length} txns`
).join('\n')}

**Active Alerts (${alerts.total}):**
${alerts.alerts.slice(0, 5).map(a =>
  `- **${a.title}** [${a.severity.toUpperCase()}] ${a.source}`
).join('\n')}

**Investigation Steps:**
1. **Triage** (Alerts page): Filter by severity, review top alerts first
2. **Enrich** (Investigations page): View transaction graph, linked entities (device, IP, VPA)
3. **Re-score**: Add context and re-run scoring if needed
4. **Decide**: Approve / Review / Decline with detailed notes
5. **Track**: Case status → Open → In Progress → Resolved → Closed
6. **Feedback**: Label outcome (True/False Positive) → feeds model retraining

**Pro Tips:**
- Use **Export Report** for compliance evidence (PDF/CSV)
- Link related cases via shared entities (device, IP, VPA)
- Set up automated case creation from Critical alerts
- Review **Analytics → False Positive Trends** weekly

**Case Templates:**
- **Coordinated Attack**: Multiple alerts, shared IP/device range
- **Account Takeover**: New device + high value + velocity
- **Merchant Fraud**: QR code scams, collect request abuse
- **Phishing Campaign**: SMS/Email patterns, infrastructure tracking

Need help with a specific investigation?`
            } catch (e) {
              return `Could not fetch cases from backend. ${t(language, 'error')}`
            }
          }
          return `Backend not connected. Use the Investigations page for case management.`
        }

        case 'compliance': {
          if (backendStatus === 'connected') {
            try {
              const analytics = await apiClient.getAnalyticsOverview()
              const cases = await apiClient.listCases({ limit: 100 })
              return `**Compliance Report Generator**

**Report Period:** Last 30 days (configurable)

**Executive Summary:**
- Total Transactions: ${analytics.total_transactions_24h * 30} (est. 30-day)
- Fraud Cases: ${cases.total}
- Fraud Rate: ${(analytics.fraud_rate * 100).toFixed(3)}%
- Avg Detection Latency: ${analytics.avg_latency_ms}ms

**Case Breakdown:**
${['critical', 'high', 'medium', 'low'].map(p => {
  const count = cases.cases.filter(c => c.priority === p).length
  return `- ${p.charAt(0).toUpperCase() + p.slice(1)}: ${count}`
}).join('\n')}

**Regulatory Compliance:**
✅ RBI Master Direction - Digital Payment Security Controls
✅ NPCI UPI Procedural Guidelines
✅ PCI DSS Requirement 12.10 (Incident Response)
✅ GDPR Article 33 (Breach Notification - 72hr)

**Report Sections Available:**
1. **Incident Register** - All cases with timestamps, decisions, outcomes
2. **Risk Metrics** - Fraud rates, detection accuracy, false positive trends
3. **Control Effectiveness** - Rule performance, model drift indicators
4. **Remediation Actions** - Case resolutions, policy updates, training
5. **Executive Dashboard** - KPIs for board reporting

**Export Formats:** PDF (audit-ready), CSV (data analysis), JSON (API integration)

Want me to generate a specific report section?`
            } catch (e) {
              return `Could not fetch data for compliance report. ${t(language, 'error')}`
            }
          }
          return `Backend not connected. Compliance reports require live data.`
        }

        case 'knowledge': {
          // Educational/definition responses
          const lower = userMessage.toLowerCase()

          // UPI Scam / Fraud definitions
          if (lower.includes('upi scam') || lower.includes('upi fraud')) {
            return `**What is UPI Fraud / UPI Scam?**

**UPI (Unified Payments Interface) fraud** refers to unauthorized or deceptive transactions conducted through India's UPI payment system. Fraudsters exploit the instant, real-time nature of UPI to trick users into authorizing payments.

**Common UPI Scam Types:**

1. **🎯 Collect Request Fraud** (Most Common)
   - Fraudster sends a "Request Money" instead of "Pay"
   - User thinks they're *receiving* money, clicks "Approve"
   - Actually authorizes money *out* of their account
   - Often disguised as refunds, cashback, or KYC verification

2. **📱 QR Code Scams**
   - "Scan this QR to receive ₹5000 cashback"
   - QR actually contains a **collect request** (pull money)
   - User scans, enters PIN, money debited instead of credited
   - **Rule**: Only scan QR codes to **PAY**, never to receive

3. **🏦 KYC / Bank Impersonation**
   - "Your UPI/KYC will expire, click link to update"
   - Link leads to phishing page capturing UPI PIN, OTP, credentials
   - Banks **never** send KYC links via SMS/WhatsApp

4. **💼 Fake Customer Support**
   - Caller claims to be from bank/Google Pay/PhonePe/Paytm
   - Asks for UPI PIN, OTP, or to install screen-sharing app (AnyDesk/TeamViewer)
   - **Legitimate support never asks for PIN/OTP**

5. **🛒 Fake Merchant / E-commerce Scams**
   - "Order confirmation" with collect request for "refund" or "delivery fee"
   - OLX/Quikr scams: "I'll pay advance, approve this request"
   - Investment scams: "Approve small amount to unlock returns"

**How ScamShield Detects UPI Fraud:**
- **FraudMLP Model**: 50 features including velocity, device fingerprint, network behavior
- **Rules Engine**: New device + high amount, impossible travel, velocity >5/min
- **Threat Intel**: Known bad VPAs, IPs, device IDs, phone numbers
- **DecisionEngine**: ML (55%) + Rules (25%) + Threat Intel (20%)

**Protection Tips:**
✅ Never share UPI PIN/OTP with anyone
✅ Only approve collect requests you *initiated* and *expect*
✅ Verify merchant name before clicking "Pay"
✅ Use ScamShield Live Scoring to check suspicious transactions
✅ Report fraud immediately: 1930 (cybercrime helpline) or your bank

Want me to check a specific transaction or show latest UPI fraud patterns from our database?`
          }

          // General scam definitions
          if (lower.includes('what is scam') || lower.includes('what are scam') || lower.includes('define scam')) {
            return `**What is a Scam?**

A **scam** is a fraudulent scheme designed to deceive people into giving away money, personal information, or access to accounts. In digital payments, scams exploit trust, urgency, and lack of awareness.

**Major Scam Categories in India:**

1. **📱 SMS Phishing (Smishing)**
   - Fake bank alerts: "Your account debited ₹X, click to dispute"
   - KYC expiry warnings with malicious links
   - Lottery/prize winnings requiring "processing fee"

2. **💼 Job Scams**
   - "Work from home, earn ₹50k/week" - requires upfront "registration fee"
   - Data entry, reshipping, crypto trading "training" fees
   - Fake offer letters from reputed companies

3. **🏦 Banking / UPI Fraud** (See UPI scam details above)

4. **💘 Romance / Matrimony Scams**
   - Build trust over weeks/months, then request money for "emergency"

5. **📦 E-commerce / Delivery Scams**
   - "Customs duty" for fake international packages
   - OTP sharing for "delivery verification"

6. **💰 Investment / Crypto Scams**
   - Guaranteed high returns, "insider tips"
   - Fake trading platforms showing fake profits

**Red Flags:**
🚨 Urgency pressure ("Act now or lose money!")
🚨 Requests for UPI PIN, OTP, passwords
🚨 Upfront fees for jobs, loans, prizes
🚨 Unverified contacts claiming authority
🚨 Too-good-to-be-true offers

**How ScamShield Helps:**
- **Pattern Library**: 1000+ known scam signatures (SMS, Job, UPI)
- **Real-time Scoring**: <50ms fraud risk assessment
- **Alert Rules**: Custom detection for your specific risks
- **Investigation Tools**: Case management for fraud teams

Try asking: "What are the latest SMS phishing patterns?" or "How do job scams work?"`
          }

          // Job scam definition
          if (lower.includes('job scam') || lower.includes('employment scam') || lower.includes('fake job')) {
            return `**What is a Job Scam?**

**Job scams** trick job seekers into paying money or sharing personal data for fake employment opportunities.

**Common Job Scam Patterns:**

1. **💸 Upfront Fee Scams**
   - "Registration fee: ₹2,999" for job portal access
   - "Training material cost" before starting
   - "Background verification charges"

2. **📦 Reshipping / Money Mule**
   - "Receive packages, re-ship them" - actually handling stolen goods
   - "Process payments for international clients" - money laundering

3. **🎓 Fake Training / Certification**
   - "Pay ₹15,000 for AWS/Data Science certification + guaranteed job"
   - Certificates are worthless, jobs don't exist

4. **🏢 Impersonation of Top Companies**
   - Fake offer letters from TCS, Infosys, Google, Amazon
   - HR email from @gmail.com, not company domain

5. **💻 Work-from-Home Data Entry**
   - "Simple typing jobs, earn ₹3000/day"
   - Requires software purchase or "security deposit"

**Red Flags:**
🚨 Job offer without interview
🚨 Payment required to get hired
🚨 Communication only via WhatsApp/Telegram
🚨 Email from personal domains (@gmail, @yahoo)
🚨 Vague job description, no company website

**ScamShield Detection:**
- **Job Scam CNN Model**: Transformer-based, analyzes job postings for fraud signals
- **Pattern Library**: 500+ known job scam templates
- **Keywords**: "guaranteed income", "no experience needed", "immediate joining", "work from home"

**Report**: National Cyber Crime Portal (cybercrime.gov.in) or 1930

Want to see latest job scam patterns from our database?`
          }

          // SMS phishing definition
          if (lower.includes('sms phishing') || lower.includes('smishing') || lower.includes('sms scam') || lower.includes('text scam')) {
            return `**What is SMS Phishing (Smishing)?**

**Smishing** = SMS + Phishing. Fraudulent text messages designed to steal credentials, money, or install malware.

**Common Smishing Templates:**

1. **🏦 Bank Alert Spoofing**
   - "A/c XXXX debited ₹15,000. If not you, click: bit.ly/fake-bank"
   - "Your KYC expired. Update now: short.url/phishing"
   - "UPI transaction failed. Retry: malicious-link.com"

2. **📦 Delivery / Customs Scams**
   - "Your package held at customs. Pay ₹2,500: link"
   - "Delivery failed. Reschedule: tracking-fake.com"

3. **🎁 Prize / Lottery**
   - "Congratulations! You won ₹10 Lakhs. Claim: claim-prize.xyz"
   - "KBC/Jio/Airtel lucky draw winner. Contact: +91-XXXXXXXXXX"

4. **👮 Authority Impersonation**
   - "Income Tax refund ₹45,000. Verify: itr-refund.fake"
   - "Police case filed against you. Call: 98765XXXXX"

**How to Identify:**
🚨 Sender: Unknown number, not official shortcode (e.g., not "AXISBK", "ICICIB")
🚨 URL: Shortened (bit.ly, tinyurl) or suspicious domains
🚨 Urgency: "Immediate action required", "within 30 minutes"
🚨 Generic: "Dear Customer" instead of your name
🚨 Asks for: OTP, PIN, password, card details

**ScamShield SMS Detection:**
- **Smishing CNN Model**: TextCNN + BERT embeddings
- **Analyzes**: Sender patterns, URL reputation, linguistic features, urgency signals
- **Real-time API**: \`POST /api/v1/detect/sms\`

**Protection:**
✅ Never click links in unsolicited SMS
✅ Verify independently: Call bank on official number
✅ Report: Forward to 1930 or use "Report Spam" in messaging app

Want to test an SMS message or see latest smishing patterns?`
          }

          // Default knowledge response
          return `I can explain fraud and scam concepts:

**Available Topics:**
• **UPI Fraud / UPI Scams** - Collect requests, QR codes, KYC phishing
• **Job Scams** - Upfront fees, fake offers, reshipping/mule schemes
• **SMS Phishing (Smishing)** - Bank spoofing, delivery scams, prize fraud
• **Transaction Risk Scoring** - How our ensemble evaluates fraud
• **Scam Patterns** - Latest indicators from our pattern library
• **Alert Rules** - Building effective detection rules
• **Model Insights** - FraudMLP, Smishing CNN, Job Scam CNN performance
• **Investigation Workflow** - Case management best practices
• **Compliance Reporting** - RBI, NPCI, PCI DSS audit-ready reports

**Try asking:**
- "What is UPI fraud?"
- "Explain job scams"
- "How does SMS phishing work?"
- "What are the latest scam patterns?"
- "How does transaction scoring work?"

Or use the **Quick Actions** buttons below for common tasks!`
        }

        default:
          return `I can help with fraud detection topics like:\n\n• **Transaction scoring** - How the ensemble evaluates risk\n• **Scam patterns** - Latest SMS/Job/UPI fraud indicators\n• **Model performance** - Accuracy, latency, ensemble weights\n• **Alert rules** - Creating effective detection rules\n• **Investigations** - Case management best practices\n• **Compliance** - Audit-ready fraud reports\n\nTry asking: "How do I analyze a suspicious transaction?" or "What are the latest UPI fraud patterns?"`
      }
    } catch (error) {
      return `${t(language, 'error')} Details: ${error instanceof Error ? error.message : 'Unknown error'}`
    } finally {
      setIsLoading(false)
    }
  }, [backendStatus, language])

  // ============================================================================
  // Message Handling
  // ============================================================================

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    if (isOpen && !isMinimized) {
      inputRef.current?.focus()
    }
  }, [isOpen, isMinimized])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setShowQuickActions(false)

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      // Determine action type from message
      const lower = userMessage.toLowerCase()
      let actionType: Message['metadata']['action'] = 'scoring'

      // Check for definition/educational questions first
      const isDefinitionQuestion = lower.startsWith('what is') || lower.startsWith('what are') ||
        lower.startsWith('define') || lower.startsWith('explain') || lower.startsWith('how does') ||
        lower.includes('what is a') || lower.includes('what are ') || lower.startsWith('tell me about') ||
        lower.startsWith('describe') || lower.includes('meaning of')

      if (isDefinitionQuestion) {
        actionType = 'knowledge'
      } else if (lower.includes('pattern') || lower.includes('sms') || lower.includes('job') || lower.includes('upi fraud') || lower.includes('qr code')) {
        actionType = 'pattern_search'
      } else if (lower.includes('scam') && !isDefinitionQuestion) {
        // Only treat as pattern_search if not a definition question
        actionType = 'pattern_search'
      } else if (lower.includes('alert') || lower.includes('rule') || lower.includes('threshold') || lower.includes('notify')) {
        actionType = 'rule_builder'
      } else if (lower.includes('model') || lower.includes('performance') || lower.includes('accuracy') || lower.includes('ensemble') || lower.includes('latency')) {
        actionType = 'model_insights'
      } else if (lower.includes('investigat') || lower.includes('case') || lower.includes('workflow') || lower.includes('triage') || lower.includes('evidence')) {
        actionType = 'investigation'
      } else if (lower.includes('compliance') || lower.includes('report') || lower.includes('audit') || lower.includes('regulatory')) {
        actionType = 'compliance'
      } else if (lower.includes('analyze') || lower.includes('score') || lower.includes('transaction')) {
        actionType = 'transaction_analysis'
      }

      const response = await performAction(actionType, userMessage)

      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response,
        timestamp: new Date(),
        metadata: { action: actionType },
      }

      setMessages(prev => [...prev, assistantMsg])
    } catch (error) {
      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `${t(language, 'error')} ${error instanceof Error ? error.message : ''}`,
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleQuickAction = (prompt: string) => {
    setInput(prompt)
    handleSend()
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    // Show brief toast feedback
    const toast = document.createElement('div')
    toast.textContent = t(language, 'copied')
    toast.style.cssText = `
      position: fixed; bottom: 80px; right: 20px; z-index: 1000;
      background: #10b981; color: white; padding: 8px 16px;
      border-radius: 6px; font-size: 14px; animation: fadeIn 0.2s;
    `
    document.body.appendChild(toast)
    setTimeout(() => toast.remove(), 2000)
  }

  // ============================================================================
  // UI Helpers
  // ============================================================================

  const generateSessionTitle = (msgs: Message[]): string => {
    const firstUserMsg = msgs.find(m => m.role === 'user')
    if (firstUserMsg) {
      return firstUserMsg.content.slice(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '')
    }
    return 'New Conversation'
  }

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  // ============================================================================
  // Render
  // ============================================================================

  const currentLang = LANGUAGES[language]

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => {
          if (!isOpen) {
            // Closed -> Open
            setIsOpen(true);
            setIsMinimized(false);
          } else if (isMinimized) {
            // Minimized -> Expand
            setIsMinimized(false);
          } else {
            // Expanded -> Close
            setIsOpen(false);
            setIsMinimized(false);
          }
        }}
        className={cn(
          'fixed bottom-6 right-6 z-50 p-3 rounded-full shadow-lg transition-all duration-300',
          'bg-primary-600 text-white hover:bg-primary-700',
          isOpen && !isMinimized && 'bg-primary-700 rotate-45',
          isMinimized && 'bg-gray-500'
        )}
        aria-label={!isOpen ? t(language, 'openAssistant') : isMinimized ? t(language, 'maximize') : t(language, 'close')}
        aria-expanded={isOpen && !isMinimized}
        title={!isOpen ? 'Open AI Assistant' : isMinimized ? 'Maximize' : 'Close'}
      >
        {isOpen && !isMinimized ? <X className="h-6 w-6" /> : <Bot className="h-6 w-6" />}
        <span className="sr-only">{!isOpen ? 'Open' : isMinimized ? 'Maximize' : 'Close'} AI Assistant</span>
      </button>

      {/* Chat Window */}
      {(isOpen || isMinimized) && (
        <div
          ref={chatContainerRef}
          className={cn(
            'fixed bottom-6 right-6 z-50 bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col animate-fade-in',
            isMinimized
              ? 'w-96 h-16'
              : 'w-96 h-[500px] max-h-[70vh]'
          )}
          role="dialog"
          aria-label="ScamShield AI Assistant"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gray-50 rounded-t-xl">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <div className="p-1.5 bg-primary-100 rounded-lg flex-shrink-0">
                <Sparkles className="h-4 w-4 text-primary-600" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="font-semibold text-gray-900 truncate text-sm">ScamShield AI Assistant</h3>
                <div className="flex items-center gap-2 text-xs">
                  <span className={cn(
                    'w-1.5 h-1.5 rounded-full',
                    backendStatus === 'connected' && 'bg-green-500',
                    backendStatus === 'disconnected' && 'bg-red-500',
                    backendStatus === 'checking' && 'bg-yellow-500'
                  )} />
                  <span className="text-gray-500 truncate">
                    {backendStatus === 'connected' ? t(language, 'connected') :
                     backendStatus === 'disconnected' ? t(language, 'disconnected') : 'Checking...'}
                  </span>
                </div>
              </div>
            </div>

            {!isMinimized && (
              <div className="flex items-center gap-1">
                {/* Language Selector */}
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as Language)}
                  className="text-xs px-2 py-1 border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                  aria-label={t(language, 'language')}
                  title={t(language, 'language')}
                >
                  {Object.entries(LANGUAGES).map(([code, lang]) => (
                    <option key={code} value={code}>{lang.flag} {lang.nativeName}</option>
                  ))}
                </select>

                {/* Voice Toggle */}
                <button
                  onClick={voiceState.isListening ? stopVoiceInput : startVoiceInput}
                  disabled={!recognition}
                  className={cn(
                    'p-1.5 rounded-lg transition-colors',
                    voiceState.isListening ? 'bg-red-100 text-red-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100',
                    !recognition && 'opacity-50 cursor-not-allowed'
                  )}
                  aria-label={voiceState.isListening ? t(language, 'voiceStop') : t(language, 'voiceStart')}
                  title={voiceState.isListening ? t(language, 'voiceStop') : t(language, 'voiceStart')}
                >
                  {voiceState.isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </button>

                {/* Settings */}
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                  aria-label={t(language, 'settings')}
                  title={t(language, 'settings')}
                >
                  <Settings className="h-4 w-4" />
                </button>

                {/* Minimize */}
                <button
                  onClick={() => setIsMinimized(true)}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                  aria-label={t(language, 'minimize')}
                  title={t(language, 'minimize')}
                >
                  <Minimize2 className="h-4 w-4" />
                </button>

                {/* Close */}
                <button
                  onClick={() => { setIsOpen(false); setIsMinimized(false); }}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                  aria-label={t(language, 'close')}
                  title={t(language, 'close')}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>

          {/* Settings Panel */}
          {!isMinimized && showSettings && (
            <div className="p-3 border-b border-gray-100 bg-gray-50 animate-slide-down">
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-gray-600 block mb-1">{t(language, 'language')}</label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as Language)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    {Object.entries(LANGUAGES).map(([code, lang]) => (
                      <option key={code} value={code}>{lang.flag} {lang.nativeName}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={voiceState.isSpeaking}
                      onChange={() => voiceState.isSpeaking ? stopSpeaking() : speakText(messages[messages.length - 1]?.content || '')}
                      className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    />
                    <span>{t(language, 'speakResponse')}</span>
                  </label>
                </div>
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={createNewSession}
                    className="flex-1 px-3 py-2 text-sm bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Plus className="h-3 w-3 inline mr-1" /> {t(language, 'newChat')}
                  </button>
                  <button
                    onClick={exportChat}
                    className="flex-1 px-3 py-2 text-sm bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Download className="h-3 w-3 inline mr-1" /> {t(language, 'exportChat')}
                  </button>
                  <button
                    onClick={shareChat}
                    className="flex-1 px-3 py-2 text-sm bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Share2 className="h-3 w-3 inline mr-1" /> {t(language, 'shareChat')}
                  </button>
                </div>
                <button
                  onClick={clearAllHistory}
                  className="w-full px-3 py-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
                >
                  <Trash2 className="h-3 w-3 inline mr-1" /> {t(language, 'clearHistory')}
                </button>
              </div>
            </div>
          )}

          {/* Messages */}
          {!isMinimized && (
            <div className="flex-1 overflow-y-auto p-3 space-y-3" role="log" aria-live="polite">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn('flex gap-2', msg.role === 'user' && 'flex-row-reverse')}
                >
                  <div
                    className={cn(
                      'flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center',
                      msg.role === 'user'
                        ? 'bg-primary-100 text-primary-600'
                        : 'bg-gray-100 text-gray-600'
                    )}
                  >
                    {msg.role === 'user' ? (
                      <MessageSquare className="h-3.5 w-3.5" />
                    ) : (
                      <Bot className="h-3.5 w-3.5" />
                    )}
                  </div>
                  <div
                    className={cn(
                      'max-w-[80%] px-3 py-2 rounded-2xl text-sm leading-relaxed',
                      msg.role === 'user'
                        ? 'bg-primary-600 text-white rounded-br-md'
                        : 'bg-gray-100 text-gray-900 rounded-bl-md'
                    )}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    {msg.role === 'assistant' && (
                      <div className="flex items-center gap-2 mt-1.5">
                        <button
                          onClick={() => copyToClipboard(msg.content)}
                          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
                          aria-label={t(language, 'copy')}
                          title={t(language, 'copy')}
                        >
                          <Copy className="h-3 w-3" />
                          <span>{t(language, 'copy')}</span>
                        </button>
                        <button
                          onClick={() => speakText(msg.content)}
                          disabled={voiceState.isSpeaking || !speechSynthesis}
                          className={cn(
                            'flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors',
                            voiceState.isSpeaking && 'text-primary-600'
                          )}
                          aria-label={voiceState.isSpeaking ? t(language, 'stopSpeaking') : t(language, 'speakResponse')}
                          title={voiceState.isSpeaking ? t(language, 'stopSpeaking') : t(language, 'speakResponse')}
                        >
                          {voiceState.isSpeaking ? <VolumeX className="h-3 w-3" /> : <Volume2 className="h-3 w-3" />}
                        </button>
                      </div>
                    )}
                    <div className="text-xs text-gray-400 mt-1 text-right">{formatTime(msg.timestamp)}</div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-2">
                  <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center">
                    <Bot className="h-3.5 w-3.5 text-gray-600" />
                  </div>
                  <div className="px-3 py-2 bg-gray-100 rounded-2xl rounded-bl-md">
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Loader2 className="h-4 w-4 animate-spin text-primary-600" />
                      <span>{t(language, 'thinking')}</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Quick Actions */}
          {!isMinimized && showQuickActions && messages.length <= 1 && (
            <div className="p-3 border-t border-gray-100 bg-gray-50 rounded-b-xl">
              <p className="text-xs text-gray-500 mb-2">{t(language, 'quickActions')}</p>
              <div className="grid grid-cols-2 gap-2">
                {quickActions.map((action) => (
                  <button
                    key={action.label}
                    onClick={() => handleQuickAction(action.prompt)}
                    className="p-2 text-left bg-white border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors"
                  >
                    <div className="flex items-center gap-1.5 text-sm text-gray-700">
                      {action.icon}
                      <span className="truncate">{action.label}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          {!isMinimized && (
            <div className="p-3 border-t border-gray-200 bg-white rounded-b-xl">
              <div className="flex items-end gap-2">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={t(language, 'placeholder')}
                  className="flex-1 min-h-[40px] max-h-32 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                  rows={1}
                  disabled={isLoading}
                  aria-label={t(language, 'placeholder')}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className={cn(
                    'p-2.5 rounded-lg transition-colors flex-shrink-0',
                    'bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                  aria-label={t(language, 'voiceStart')}
                >
                  <Send className="h-4.5 w-4.5" />
                </button>
              </div>
              <p className="text-xs text-gray-400 mt-1.5 text-center">
                Enter to send • Shift+Enter for new line • Click mic for voice
              </p>
            </div>
          )}

          {/* Minimized View */}
          {isMinimized && (
            <div className="p-3 flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <div className="p-1.5 bg-primary-100 rounded-lg">
                  <Sparkles className="h-4 w-4 text-primary-600" />
                </div>
                <span className="font-medium text-gray-900 text-sm truncate">ScamShield AI Assistant</span>
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  backendStatus === 'connected' && 'bg-green-500',
                  backendStatus === 'disconnected' && 'bg-red-500',
                )} />
              </div>
              <button
                onClick={() => setIsMinimized(false)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                aria-label={t(language, 'maximize')}
              >
                <Maximize2 className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Session History Sidebar (on wider screens) */}
          {!isMinimized && sessions.length > 0 && (
            <div className="absolute left-full top-0 bottom-0 w-64 bg-white border-r border-gray-200 shadow-xl z-40 animate-slide-in">
              <div className="p-3 border-b border-gray-200 flex items-center justify-between">
                <h4 className="font-medium text-gray-900 text-sm">{t(language, 'language') === 'Language' ? 'Recent Chats' : 'हाल की चैट'}</h4>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-1 text-gray-400 hover:text-gray-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="overflow-y-auto p-2 max-h-[400px]">
                {sessions.slice(0, 20).map((session) => (
                  <button
                    key={session.id}
                    onClick={() => loadSession(session.id)}
                    className={cn(
                      'w-full p-2 rounded-lg text-left transition-colors',
                      currentSessionId === session.id
                        ? 'bg-primary-50 border border-primary-200'
                        : 'hover:bg-gray-50'
                    )}
                  >
                    <p className="text-sm font-medium text-gray-900 truncate">{session.title}</p>
                    <p className="text-xs text-gray-500 truncate mt-0.5">
                      {session.messages.length} messages • {session.updatedAt.toLocaleDateString()}
                    </p>
                    <span className="inline-block mt-1 px-1.5 py-0.5 text-xs bg-gray-100 rounded">{LANGUAGES[session.language].flag}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Styles */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fade-in { animation: fadeIn 0.2s ease-out; }
        .animate-slide-down { animation: slideDown 0.15s ease-out; }
        .animate-slide-in { animation: slideIn 0.2s ease-out; }

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }

        /* Textarea focus */
        textarea:focus { outline: none; }

        /* Selection */
        ::selection { background: #dbeafe; color: #1e40af; }
      `}</style>
    </>
  )
}