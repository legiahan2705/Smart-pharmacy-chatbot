"use client";

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Send, Bot, User, Home, Menu, X, Globe } from "lucide-react"

interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
}

export default function ChatbotPage() {
  const [language, setLanguage] = useState<'en' | 'vi'>('vi')
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: language === 'vi' 
        ? 'Xin chào! Tôi là trợ lý y tế thông minh của Long Châu Pharmacy. Tôi có thể giúp bạn về thông tin thuốc, triệu chứng, hoặc các câu hỏi về sức khỏe. Bạn cần hỗ trợ gì hôm nay?'
        : 'Hello! I am the Smart Medicine Assistant from Long Chau Pharmacy. I can help you with information about medicines, symptoms, or health questions. What can I help you with today?',
      timestamp: new Date()
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const content = {
    en: {
      sidebar: {
        title: "Long Chau Pharmacy",
        newChat: "+ New conversation",
        history: "CONVERSATION HISTORY",
        home: "Home"
      },
      header: {
        title: "Smart Medicine Assistant",
        status: "● Online"
      },
      quickQuestions: {
        label: "Suggested questions:",
        items: [
          "How to use paracetamol?",
          "Common flu symptoms",
          "How to prevent cold",
          "Fever medication for children"
        ]
      },
      conversationHistory: [
        { title: "Ask about pain reliever", date: "Today, 10:30" },
        { title: "High fever symptoms", date: "Yesterday, 14:20" },
        { title: "Multivitamin", date: "2 days ago" }
      ],
      input: {
        placeholder: "Type your question...",
        disclaimer: "This is an AI assistant and does not replace professional medical advice"
      },
      responses: {
        paracetamol: "Paracetamol is a common pain reliever and fever reducer. Adult dosage: 500-1000mg every 4-6 hours, not exceeding 4000mg/day. Should be taken after meals to reduce stomach irritation. Note: Not for use in severe liver disease patients. Do you need more advice about this medicine?",
        flu: "Common flu symptoms: fever, cough, sore throat, stuffy nose, headache, fatigue. Treatment:\n\n1. Get adequate rest\n2. Drink plenty of fluids\n3. Use fever reducers if temperature exceeds 38.5°C\n4. Gargle with salt water\n\nIf symptoms persist for more than 7 days or high fever above 39°C, consult a doctor. Do you have any specific symptoms?",
        children: "For children, medication use requires special care:\n\n- Always calculate dosage by weight\n- Do not use adult medications for children without consultation\n- Use syrup or chewable tablets for young children\n\nCan you provide the child's age and weight for more specific advice?",
        vitamin: "Multivitamins help supplement essential nutrients for the body. Should take:\n\n- After breakfast\n- With plain water\n- Consistently for 1-3 months\n\nNote: Do not exceed recommended dosage. Who are you looking for vitamins for: adults, children, or elderly?",
        default: "Thank you for sharing. I have noted your question. For more accurate advice, you can provide additional information such as: age, specific symptoms, or medication type of interest. Or you can try the suggested questions below!"
      }
    },
    vi: {
      sidebar: {
        title: "Nhà Thuốc Long Châu",
        newChat: "+ Cuộc trò chuyện mới",
        history: "LỊCH SỬ TRÒ CHUYỆN",
        home: "Trang chủ"
      },
      header: {
        title: "Trợ Lý Y Tế Thông Minh",
        status: "● Đang hoạt động"
      },
      quickQuestions: {
        label: "Câu hỏi gợi ý:",
        items: [
          "Thuốc paracetamol dùng như thế nào?",
          "Triệu chứng cảm cúm thông thường",
          "Cách phòng ngừa cảm lạnh",
          "Thuốc hạ sốt cho trẻ em"
        ]
      },
      conversationHistory: [
        { title: "Hỏi về thuốc giảm đau", date: "Hôm nay, 10:30" },
        { title: "Triệu chứng sốt cao", date: "Hôm qua, 14:20" },
        { title: "Vitamin tổng hợp", date: "2 ngày trước" }
      ],
      input: {
        placeholder: "Nhập câu hỏi của bạn...",
        disclaimer: "Đây là trợ lý AI và không thay thế cho tư vấn y tế chuyên nghiệp"
      },
      responses: {
        paracetamol: "Paracetamol là thuốc giảm đau, hạ sốt phổ biến. Liều dùng người lớn: 500-1000mg mỗi 4-6 giờ, không quá 4000mg/ngày. Nên uống sau ăn để giảm kích ứng dạ dày. Lưu ý: Không dùng cho người bệnh gan nặng. Bạn có cần tư vấn thêm về thuốc này không?",
        flu: "Triệu chứng cảm cúm thường gặp: sốt, ho, đau họng, nghẹt mũi, đau đầu, mệt mỏi. Cách xử lý:\n\n1. Nghỉ ngơi đầy đủ\n2. Uống nhiều nước\n3. Dùng thuốc hạ sốt nếu sốt trên 38.5°C\n4. Súc họng nước muối\n\nNếu triệu chứng kéo dài trên 7 ngày hoặc sốt cao trên 39°C, nên đến bác sĩ khám. Bạn có triệu chứng nào đặc biệt không?",
        children: "Đối với trẻ em, việc dùng thuốc cần đặc biệt cẩn thận:\n\n- Luôn tính liều theo cân nặng\n- Không tự ý dùng thuốc người lớn cho trẻ\n- Dùng thuốc dạng siro hoặc viên nhai cho trẻ nhỏ\n\nBạn có thể cho biết độ tuổi và cân nặng của bé để tôi tư vấn cụ thể hơn không?",
        vitamin: "Vitamin tổng hợp giúp bổ sung các chất cần thiết cho cơ thể. Nên uống:\n\n- Sau bữa ăn sáng\n- Uống với nước lọc\n- Dùng đều đặn 1-3 tháng\n\nLưu ý: Không nên dùng quá liều khuyến cáo. Bạn đang tìm vitamin cho ai: người lớn, trẻ em hay người cao tuổi?",
        default: "Cảm ơn bạn đã chia sẻ. Tôi đã ghi nhận câu hỏi của bạn. Để tư vấn chính xác hơn, bạn có thể cung cấp thêm thông tin như: độ tuổi, triệu chứng cụ thể, hoặc loại thuốc bạn quan tâm. Hoặc bạn có thể thử các câu hỏi gợi ý bên dưới!"
      }
    }
  }

  const t = content[language]

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Update initial bot message when language changes
    setMessages([{
      id: '1',
      type: 'bot',
      content: language === 'vi' 
        ? 'Xin chào! Tôi là trợ lý y tế thông minh của Long Châu Pharmacy. Tôi có thể giúp bạn về thông tin thuốc, triệu chứng, hoặc các câu hỏi về sức khỏe. Bạn cần hỗ trợ gì hôm nay?'
        : 'Hello! I am the Smart Medicine Assistant from Long Chau Pharmacy. I can help you with information about medicines, symptoms, or health questions. What can I help you with today?',
      timestamp: new Date()
    }])
  }, [language])

  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'vi' : 'en')
  }

  const handleSend = async () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsTyping(true)

    // Simulate bot response
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: generateBotResponse(inputValue),
        timestamp: new Date()
      }
      setMessages(prev => [...prev, botResponse])
      setIsTyping(false)
    }, 1500)
  }

  const generateBotResponse = (userInput: string): string => {
    const input = userInput.toLowerCase()
    const responses = content[language].responses
    
    if (input.includes('paracetamol') || input.includes('giảm đau') || input.includes('pain')) {
      return responses.paracetamol
    } else if (input.includes('cảm cúm') || input.includes('cảm lạnh') || input.includes('flu') || input.includes('cold')) {
      return responses.flu
    } else if (input.includes('trẻ em') || input.includes('con') || input.includes('children') || input.includes('kid')) {
      return responses.children
    } else if (input.includes('vitamin')) {
      return responses.vitamin
    } else {
      return responses.default
    }
  }

  const handleQuickQuestion = (question: string) => {
    setInputValue(question)
    inputRef.current?.focus()
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={`${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-80 bg-white border-r border-gray-200 transition-transform duration-300 ease-in-out flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-6">
             <div className="flex items-center gap-3">
            <img
              src="/logo_LongChau.png"
              alt="Long Chau Logo"
              className="h-15 w-auto"
            />
            <h1 className="font-bold text-3xl  text-[#072D94]">{t.sidebar.title}</h1>
          </div>

            <button 
              onClick={() => setIsSidebarOpen(false)}
              className="lg:hidden text-gray-500 hover:text-gray-700"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <Button className="w-full bg-[#001A61] hover:bg-[#072D94] text-white">
            {t.sidebar.newChat}
          </Button>
        </div>

        {/* Conversation History */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-sm font-semibold text-gray-500 mb-3 px-2">{t.sidebar.history}</h3>
          <div className="space-y-2">
            {t.conversationHistory.map((conv, index) => (
              <div 
                key={index}
                className="p-3 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <p className="text-sm font-medium text-gray-800 truncate">{conv.title}</p>
                <p className="text-xs text-gray-500 mt-1">{conv.date}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-200">
          <a href="/" className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition-colors">
            <Home className="w-5 h-5 text-[#001A61]" />
            <span className="text-sm font-medium text-gray-700">{t.sidebar.home}</span>
          </a>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Chat Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden text-gray-600 hover:text-gray-800"
            >
              <Menu className="w-6 h-6" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-[#001A61] flex items-center justify-center">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg text-gray-800">{t.header.title}</h1>
                <p className="text-sm text-green-600">{t.header.status}</p>
              </div>
            </div>
          </div>
          
          {/* Language Toggle Button */}
          <button
            onClick={toggleLanguage}
            className="flex items-center gap-2 px-4 py-2 bg-[#072D94] text-white rounded-lg hover:bg-[#001A61] transition-colors font-medium"
          >
            <Globe className="w-5 h-5" />
            {language === 'en' ? 'VI' : 'EN'}
          </button>
        </header>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((message) => (
            <div 
              key={message.id}
              className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.type === 'user' ? 'bg-[#072D94]' : 'bg-[#001A61]'
              }`}>
                {message.type === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>
              <div className={`flex flex-col ${message.type === 'user' ? 'items-end' : 'items-start'} max-w-2xl`}>
                <div className={`rounded-2xl px-4 py-3 ${
                  message.type === 'user' 
                    ? 'bg-[#072D94] text-white' 
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}>
                  <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                </div>
                <span className="text-xs text-gray-500 mt-1 px-2">
                  {message.timestamp.toLocaleTimeString(language === 'vi' ? 'vi-VN' : 'en-US', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-[#001A61] flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        {messages.length === 1 && (
          <div className="px-6 pb-4">
            <p className="text-sm text-gray-600 mb-3">{t.quickQuestions.label}</p>
            <div className="flex flex-wrap gap-2">
              {t.quickQuestions.items.map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickQuestion(question)}
                  className="px-4 py-2 bg-white border border-gray-300 rounded-full text-sm text-gray-700 hover:bg-gray-50 hover:border-[#072D94] transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={t.input.placeholder}
                  className="w-full px-4 py-3 pr-12 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#072D94] focus:border-transparent resize-none"
                />
              </div>
              <Button
                onClick={handleSend}
                disabled={!inputValue.trim() || isTyping}
                className="bg-[#001A61] hover:bg-[#072D94] text-white px-6 py-3 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              {t.input.disclaimer}
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}