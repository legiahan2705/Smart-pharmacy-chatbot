"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Send, Bot, User, Home, Menu, X, Globe } from "lucide-react";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

async function askAgent(userInput: string) {
  try {
    const response = await fetch("http://localhost:8080/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: userInput,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // data.answer chính là câu trả lời của AI
    console.log(data.answer);
    return data.answer;
  } catch (error) {
    console.error("Lỗi khi gọi API chat:", error);
    return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau.";
  }
}

// Cách sử dụng:
// askAgent("Pharmaton có tác dụng phụ gì?");

export default function ChatbotPage() {
  const [language, setLanguage] = useState<"en" | "vi">("vi");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "bot",
      content:
        language === "vi"
          ? "Xin chào! Tôi là trợ lý y tế thông minh của Long Châu Pharmacy. Tôi có thể giúp bạn về thông tin thuốc, triệu chứng, hoặc các câu hỏi về sức khỏe. Bạn cần hỗ trợ gì hôm nay?"
          : "Hello! I am the Smart Medicine Assistant from Long Chau Pharmacy. I can help you with information about medicines, symptoms, or health questions. What can I help you with today?",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const content = {
    en: {
      sidebar: {
        title: "Long Chau Pharmacy",
        newChat: "+ New conversation",
        history: "CONVERSATION HISTORY",
        home: "Home",
      },
      header: {
        title: "Smart Medicine Assistant",
        status: "● Online",
      },
      quickQuestions: {
        label: "Suggested questions:",
        items: [
          "How to use paracetamol?",
          "Common flu symptoms",
          "How to prevent cold",
          "Fever medication for children",
        ],
      },
      conversationHistory: [
        { title: "Ask about pain reliever", date: "Today, 10:30" },
        { title: "High fever symptoms", date: "Yesterday, 14:20" },
        { title: "Multivitamin", date: "2 days ago" },
      ],
      input: {
        placeholder: "Type your question...",
        disclaimer:
          "This is an AI assistant and does not replace professional medical advice",
      },
    },
    vi: {
      sidebar: {
        title: "Nhà Thuốc Long Châu",
        newChat: "+ Cuộc trò chuyện mới",
        history: "LỊCH SỬ TRÒ CHUYỆN",
        home: "Trang chủ",
      },
      header: {
        title: "Trợ Lý Y Tế Thông Minh",
        status: "● Đang hoạt động",
      },
      quickQuestions: {
        label: "Câu hỏi gợi ý:",
        items: [
          "Thuốc paracetamol dùng như thế nào?",
          "Triệu chứng cảm cúm thông thường",
          "Cách phòng ngừa cảm lạnh",
          "Thuốc hạ sốt cho trẻ em",
        ],
      },
      conversationHistory: [
        { title: "Hỏi về thuốc giảm đau", date: "Hôm nay, 10:30" },
        { title: "Triệu chứng sốt cao", date: "Hôm qua, 14:20" },
        { title: "Vitamin tổng hợp", date: "2 ngày trước" },
      ],
      input: {
        placeholder: "Nhập câu hỏi của bạn...",
        disclaimer:
          "Đây là trợ lý AI và không thay thế cho tư vấn y tế chuyên nghiệp",
      },
    },
  };

  const t = content[language];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Update initial bot message when language changes
    setMessages([
      {
        id: "1",
        type: "bot",
        content:
          language === "vi"
            ? "Xin chào! Tôi là trợ lý y tế thông minh của Long Châu Pharmacy. Tôi có thể giúp bạn về thông tin thuốc, triệu chứng, hoặc các câu hỏi về sức khỏe. Bạn cần hỗ trợ gì hôm nay?"
            : "Hello! I am the Smart Medicine Assistant from Long Chau Pharmacy. I can help you with information about medicines, symptoms, or health questions. What can I help you with today?",
        timestamp: new Date(),
      },
    ]);
  }, [language]);

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "en" ? "vi" : "en"));
  };

  // HÀM MỚI (ĐÃ KẾT NỐI VỚI AGENT THẬT)
  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    // Thêm tin nhắn của người dùng vào danh sách
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true); // Bật "đang gõ..."

    // Sử dụng try...catch...finally để đảm bảo luôn tắt "đang gõ"
    try {
      // 1. GỌI AGENT THẬT
      // (Chúng ta dùng userMessage.content thay vì inputValue vì inputValue đã bị xóa)
      const botAnswer = await askAgent(userMessage.content);

      // 2. Tạo tin nhắn bot với câu trả lời thật
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: botAnswer, // <-- Sử dụng câu trả lời thật từ API
        timestamp: new Date(),
      };
      // 3. Thêm tin nhắn của bot vào danh sách
      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      // Xử lý nếu hàm askAgent thất bại (mặc dù nó đã có catch riêng)
      console.error("Lỗi nghiêm trọng trong handleSend:", error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: "Xin lỗi, đã có lỗi kết nối. Vui lòng thử lại.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      // 4. Luôn luôn tắt "đang gõ..." sau khi hoàn tất
      setIsTyping(false);
    }
  };

  const handleQuickQuestion = (question: string) => {
    setInputValue(question);
    inputRef.current?.focus();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside
        className={`${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-80 bg-white border-r border-gray-200 transition-transform duration-300 ease-in-out flex flex-col`}
      >
        {/* Sidebar Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <img
                src="/logo_LongChau.png"
                alt="Long Chau Logo"
                className="h-15 w-auto"
              />
              <h1 className="font-bold text-3xl  text-[#072D94]">
                {t.sidebar.title}
              </h1>
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
          <h3 className="text-sm font-semibold text-gray-500 mb-3 px-2">
            {t.sidebar.history}
          </h3>
          <div className="space-y-2">
            {t.conversationHistory.map((conv, index) => (
              <div
                key={index}
                className="p-3 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <p className="text-sm font-medium text-gray-800 truncate">
                  {conv.title}
                </p>
                <p className="text-xs text-gray-500 mt-1">{conv.date}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-200">
          <a
            href="/"
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Home className="w-5 h-5 text-[#001A61]" />
            <span className="text-sm font-medium text-gray-700">
              {t.sidebar.home}
            </span>
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
                <h1 className="font-bold text-lg text-gray-800">
                  {t.header.title}
                </h1>
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
            {language === "en" ? "VI" : "EN"}
          </button>
        </header>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === "user" ? "flex-row-reverse" : "flex-row"
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === "user" ? "bg-[#072D94]" : "bg-[#001A61]"
                }`}
              >
                {message.type === "user" ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>
              <div
                className={`flex flex-col ${
                  message.type === "user" ? "items-end" : "items-start"
                } max-w-2xl`}
              >
                <div
                  className={`rounded-2xl px-4 py-3 ${
                    message.type === "user"
                      ? "bg-[#072D94] text-white"
                      : "bg-white border border-gray-200 text-gray-800"
                  }`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">
                    {message.content}
                  </p>
                </div>
                <span className="text-xs text-gray-500 mt-1 px-2">
                  {message.timestamp.toLocaleTimeString(
                    language === "vi" ? "vi-VN" : "en-US",
                    { hour: "2-digit", minute: "2-digit" }
                  )}
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
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        {messages.length === 1 && (
          <div className="px-6 pb-4">
            <p className="text-sm text-gray-600 mb-3">
              {t.quickQuestions.label}
            </p>
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
  );
}
