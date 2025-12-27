"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Send,
  Bot,
  User,
  Home,
  Menu,
  X,
  Globe,
  Trash2,
  Mic,
  Square,
  Loader2,
} from "lucide-react";
import { v4 as uuidv4 } from "uuid"; // Cần cài: npm install uuid @types/uuid
import ReactMarkdown from "react-markdown";

// --- TYPES ---
interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

interface ChatSession {
  id: string;
  title: string;
  updatedAt: Date;
}

// --- API CALL ---
async function askAgent(userInput: string, threadId: string) {
  try {
    // Sử dụng biến môi trường cho URL
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

    const response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: userInput,
        thread_id: threadId, // Gửi thread_id thật xuống BE
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.answer;
  } catch (error) {
    console.error("Lỗi khi gọi API chat:", error);
    return "Xin lỗi, tôi đang gặp sự cố kết nối với server. Vui lòng kiểm tra lại backend.";
  }
}

export default function ChatbotPage() {
  // --- STATE ---
  const [language, setLanguage] = useState<"en" | "vi">("vi");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // State quản lý Chat thật
  const [sessions, setSessions] = useState<ChatSession[]>([]); // Danh sách lịch sử bên trái
  const [activeThreadId, setActiveThreadId] = useState<string>(""); // ID cuộc trò chuyện đang mở
  const [messages, setMessages] = useState<Message[]>([]); // Tin nhắn của cuộc trò chuyện hiện tại

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // --- STATE CHO VOICE CHAT ---
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // --- TEXT CONTENT (Đa ngôn ngữ) ---
  const content = {
    en: {
      sidebar: {
        title: "Long Chau Pharmacy",
        newChat: "+ New conversation",
        history: "CONVERSATION HISTORY",
        home: "Home",
        empty: "No history yet",
      },
      header: { title: "Smart Medicine Assistant", status: "● Online" },
      quickQuestions: {
        label: "Suggested questions:",
        items: [
          "How to use paracetamol?",
          "Common flu symptoms",
          "How to prevent cold",
          "Fever medication for children",
        ],
      },
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
        empty: "Chưa có lịch sử",
      },
      header: { title: "Trợ Lý Y Tế Thông Minh", status: "Đang hoạt động" },
      quickQuestions: {
        label: "Câu hỏi gợi ý:",
        items: [
          "Thuốc paracetamol dùng như thế nào?",
          "Thuốc hạ sốt cho trẻ em",
          
        ],
      },
      input: {
        placeholder: "Nhập câu hỏi của bạn...",
        disclaimer:
          "Đây là trợ lý AI và không thay thế cho tư vấn y tế chuyên nghiệp",
      },
    },
  };
  const t = content[language];

  // --- EFFECTS ---

  // 1. Khởi tạo: Load lịch sử từ LocalStorage khi vào trang
  useEffect(() => {
    const savedSessions = localStorage.getItem("chat_sessions");
    if (savedSessions) {
      const parsedSessions = JSON.parse(savedSessions);
      // Convert string date back to Date object
      const formattedSessions = parsedSessions.map((s: any) => ({
        ...s,
        updatedAt: new Date(s.updatedAt),
      }));
      setSessions(formattedSessions);

      // Nếu có lịch sử, load bài gần nhất. Nếu không, tạo mới.
      if (formattedSessions.length > 0) {
        loadSession(formattedSessions[0].id);
      } else {
        createNewSession();
      }
    } else {
      createNewSession();
    }
  }, []);

  // 2. Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // --- LOGIC CHỨC NĂNG ---

  // Tạo phiên chat mới
  const createNewSession = () => {
    const newId = uuidv4();
    const newSession: ChatSession = {
      id: newId,
      title: language === "vi" ? "Cuộc trò chuyện mới" : "New Conversation",
      updatedAt: new Date(),
    };

    // Cập nhật state sessions
    const updatedSessions = [newSession, ...sessions];
    setSessions(updatedSessions);
    setActiveThreadId(newId);

    // Reset tin nhắn về mặc định
    setMessages([getWelcomeMessage(language)]);

    // Lưu session mới vào LocalStorage
    localStorage.setItem("chat_sessions", JSON.stringify(updatedSessions));

    // Đóng sidebar trên mobile sau khi chọn
    if (window.innerWidth < 1024) setIsSidebarOpen(false);
  };

  // Load nội dung của một phiên chat cũ
  const loadSession = (threadId: string) => {
    setActiveThreadId(threadId);
    const savedMessages = localStorage.getItem(`chat_messages_${threadId}`);

    if (savedMessages) {
      const parsed = JSON.parse(savedMessages).map((m: any) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      }));
      setMessages(parsed);
    } else {
      setMessages([getWelcomeMessage(language)]);
    }

    if (window.innerWidth < 1024) setIsSidebarOpen(false);
  };

  // Xóa một phiên chat
  const deleteSession = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation(); // Ngăn sự kiện click lan ra cha
    const updatedSessions = sessions.filter((s) => s.id !== sessionId);
    setSessions(updatedSessions);
    localStorage.setItem("chat_sessions", JSON.stringify(updatedSessions));
    localStorage.removeItem(`chat_messages_${sessionId}`);

    // Nếu xóa session đang active, chuyển sang cái khác hoặc tạo mới
    if (activeThreadId === sessionId) {
      if (updatedSessions.length > 0) {
        loadSession(updatedSessions[0].id);
      } else {
        createNewSession();
      }
    }
  };

  // Hàm lấy câu chào (helper)
  const getWelcomeMessage = (lang: "en" | "vi"): Message => ({
    id: "welcome",
    type: "bot",
    content:
      lang === "vi"
        ? "Xin chào! Tôi là trợ lý y tế thông minh của Long Châu Pharmacy. Tôi có thể giúp bạn về thông tin thuốc, triệu chứng, hoặc các câu hỏi về sức khỏe. Bạn cần hỗ trợ gì hôm nay?"
        : "Hello! I am the Smart Medicine Assistant from Long Chau Pharmacy. I can help you with information about medicines, symptoms, or health questions. What can I help you with today?",
    timestamp: new Date(),
  });

  // Gửi tin nhắn
  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const currentThreadId = activeThreadId; // Capture ID hiện tại đề phòng user đổi tab nhanh

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    // 1. Cập nhật UI ngay lập tức
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue("");
    setIsTyping(true);

    // 2. Lưu tin nhắn vào LocalStorage cho thread hiện tại
    localStorage.setItem(
      `chat_messages_${currentThreadId}`,
      JSON.stringify(newMessages)
    );

    // 3. Cập nhật Tiêu đề Session (nếu đây là tin nhắn đầu tiên của user)
    // Kiểm tra xem session hiện tại có đang là tên mặc định không
    const currentSessionIndex = sessions.findIndex(
      (s) => s.id === currentThreadId
    );
    if (
      currentSessionIndex !== -1 &&
      messages.filter((m) => m.type === "user").length === 0
    ) {
      // Logic đơn giản: Lấy 30 ký tự đầu làm tiêu đề
      const updatedSessions = [...sessions];
      updatedSessions[currentSessionIndex].title =
        userMessage.content.substring(0, 30) +
        (userMessage.content.length > 30 ? "..." : "");
      updatedSessions[currentSessionIndex].updatedAt = new Date();
      // Đánh dấu là đã đổi tên (trick nhỏ, hoặc chỉ cần check length messages)
      setSessions(updatedSessions);
      localStorage.setItem("chat_sessions", JSON.stringify(updatedSessions));
    }

    try {
      // 4. Gọi API
      const botAnswer = await askAgent(userMessage.content, currentThreadId);

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: botAnswer,
        timestamp: new Date(),
      };

      // 5. Cập nhật tin nhắn Bot
      const finalMessages = [...newMessages, botResponse];
      setMessages(finalMessages);
      localStorage.setItem(
        `chat_messages_${currentThreadId}`,
        JSON.stringify(finalMessages)
      );
    } catch (error) {
      console.error(error);
    } finally {
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

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "en" ? "vi" : "en"));
    // Cập nhật lại câu chào trong message hiện tại nếu nó là tin nhắn duy nhất
    if (messages.length === 1 && messages[0].id === "welcome") {
      setMessages([getWelcomeMessage(language === "en" ? "vi" : "en")]);
    }
  };

  // --- LOGIC VOICE CHAT ---

  // 1. Bắt đầu thu âm
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = handleStopRecording; // Gắn hàm xử lý khi dừng
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Lỗi Micro:", error);
      alert("Không thể truy cập Micro. Vui lòng kiểm tra quyền!");
    }
  };

  // 2. Dừng thu âm (User bấm nút dừng)
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
    }
  };

  // 3. Xử lý file âm thanh -> Gửi lên Server -> Nhận Text & Audio
  const handleStopRecording = async () => {
    setIsProcessingVoice(true); // Hiển thị loading xoay vòng
    const currentThreadId = activeThreadId;

    try {
      // Gom file âm thanh
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/mp3" });
      const audioFile = new File([audioBlob], "voice_input.mp3", {
        type: "audio/mp3",
      });

      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("thread_id", currentThreadId);

      // Gọi API Full Flow
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/chat-voice-flow`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Lỗi API Voice");

      const data = await response.json();

      // Nếu không nghe được gì
      if (!data.user_text) return;

      // --- CẬP NHẬT UI & LOCAL STORAGE (Giống logic handleSend) ---

      // A. Tạo tin nhắn User (từ text nhận diện được)
      const userMessage: Message = {
        id: Date.now().toString(),
        type: "user",
        content: data.user_text,
        timestamp: new Date(),
      };

      // B. Tạo tin nhắn Bot
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: data.bot_answer,
        timestamp: new Date(),
      };

      // C. Cập nhật State
      const newMessages = [...messages, userMessage, botMessage];
      setMessages(newMessages);

      // D. Lưu LocalStorage
      localStorage.setItem(
        `chat_messages_${currentThreadId}`,
        JSON.stringify(newMessages)
      );

      // E. Cập nhật tiêu đề Session nếu là tin nhắn đầu
      if (
        messages.length === 0 ||
        (messages.length === 1 && messages[0].id === "welcome")
      ) {
        const updatedSessions = sessions.map((s) =>
          s.id === currentThreadId
            ? {
                ...s,
                title: data.user_text.substring(0, 30) + "...",
                updatedAt: new Date(),
              }
            : s
        );
        setSessions(updatedSessions);
        localStorage.setItem("chat_sessions", JSON.stringify(updatedSessions));
      }

      // F. PHÁT ÂM THANH TRẢ LỜI
      if (data.audio_base64) {
        const audio = new Audio(`data:audio/mp3;base64,${data.audio_base64}`);
        audio.play().catch((e) => console.error("Lỗi phát audio:", e));
      }
    } catch (error) {
      console.error(error);
      alert("Có lỗi khi xử lý giọng nói.");
    } finally {
      setIsProcessingVoice(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside
        className={`${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-80 bg-white border-r border-gray-200 transition-transform duration-300 ease-in-out flex flex-col shadow-lg lg:shadow-none`}
      >
        {/* Sidebar Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <img
                src="/logo_LongChau.png"
                alt="Logo"
                className="h-10 w-auto"
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
              {/* Fallback text nếu ảnh lỗi */}
              <h1 className="font-bold text-xl text-[#072D94] leading-tight">
                Long Châu
                <br />
                AI Chatbot
              </h1>
            </div>
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="lg:hidden text-gray-500"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <Button
            onClick={createNewSession}
            className="w-full bg-[#001A61] hover:bg-[#072D94] text-white flex gap-2"
          >
            {t.sidebar.newChat}
          </Button>
        </div>

        {/* Conversation History List */}
        <div className="flex-1 overflow-y-auto p-3">
          <h3 className="text-xs font-semibold text-gray-500 mb-3 px-2 uppercase tracking-wider">
            {t.sidebar.history}
          </h3>
          <div className="space-y-1">
            {sessions.length === 0 && (
              <p className="text-sm text-gray-400 text-center italic mt-4">
                {t.sidebar.empty}
              </p>
            )}
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => loadSession(session.id)}
                className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                  activeThreadId === session.id
                    ? "bg-blue-50 border-l-4 border-[#072D94] shadow-sm"
                    : "hover:bg-gray-100 border-l-4 border-transparent"
                }`}
              >
                <div className="flex-1 min-w-0 pr-2">
                  <p
                    className={`text-sm font-medium truncate ${
                      activeThreadId === session.id
                        ? "text-[#072D94]"
                        : "text-gray-700"
                    }`}
                  >
                    {session.title}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(session.updatedAt).toLocaleDateString(
                      language === "vi" ? "vi-VN" : "en-US",
                      {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      }
                    )}
                  </p>
                </div>
                {/* Nút xóa session */}
                <button
                  onClick={(e) => deleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-100 hover:text-red-600 rounded-md transition-all"
                  title="Xóa cuộc trò chuyện"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-200">
          <a
            href="/"
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition-colors text-gray-700"
          >
            <Home className="w-5 h-5 text-[#001A61]" />
            <span className="text-sm font-medium">{t.sidebar.home}</span>
          </a>
        </div>
      </aside>

      {/* Overlay mobile */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full w-full">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm z-10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden text-gray-600"
            >
              <Menu className="w-6 h-6" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-[#001A61] flex items-center justify-center shadow-md">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg text-gray-800 leading-tight">
                  {t.header.title}
                </h1>
                <p className="text-xs text-green-600 font-medium flex items-center gap-1">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                  {t.header.status}
                </p>
              </div>
            </div>
          </div>
          <button
            onClick={toggleLanguage}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-[#072D94] rounded-full hover:bg-gray-200 transition-colors text-sm font-bold border border-gray-200"
          >
            <Globe className="w-4 h-4" />
            {language === "en" ? "VN" : "EN"}
          </button>
        </header>

        {/* Messages List */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 bg-gray-50/50">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === "user" ? "flex-row-reverse" : "flex-row"
              } animate-in fade-in slide-in-from-bottom-2 duration-300`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm ${
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
                } max-w-[85%] md:max-w-2xl`}
              >
                <div
                  className={`rounded-2xl px-5 py-3.5 shadow-sm ${
                    message.type === "user"
                      ? "bg-[#072D94] text-white rounded-br-none"
                      : "bg-white border border-gray-200 text-gray-800 rounded-bl-none"
                  }`}
                >
                  <div className="prose prose-sm max-w-none leading-relaxed text-[15px]">
                    {message.type === "bot" ? (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    )}
                  </div>
                </div>
                <span className="text-[10px] text-gray-400 mt-1.5 px-1">
                  {new Date(message.timestamp).toLocaleTimeString(
                    language === "vi" ? "vi-VN" : "en-US",
                    { hour: "2-digit", minute: "2-digit" }
                  )}
                </span>
              </div>
            </div>
          ))}

          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex gap-3 animate-pulse">
              <div className="w-8 h-8 rounded-full bg-[#001A61] flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 rounded-bl-none shadow-sm">
                <div className="flex gap-1.5">
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

        {/* Quick Suggestions */}
        {messages.length === 1 && !isTyping && (
          <div className="px-6 pb-2">
            <p className="text-xs text-gray-500 mb-3 uppercase tracking-wide font-semibold ml-1">
              {t.quickQuestions.label}
            </p>
            <div className="flex flex-wrap gap-2">
              {t.quickQuestions.items.map((question, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickQuestion(question)}
                  className="px-4 py-2 bg-white border border-gray-200 rounded-full text-sm text-gray-600 hover:bg-[#072D94] hover:text-white hover:border-[#072D94] transition-all shadow-sm"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4 md:p-5">
          <div className="max-w-4xl mx-auto relative flex items-center gap-2">
            {/* INPUT TEXT */}
            <div className="relative flex-1">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  isRecording ? "Đang nghe bạn nói..." : t.input.placeholder
                }
                disabled={isRecording || isProcessingVoice} // Khóa khi đang thu âm
                className={`w-full pl-5 pr-12 py-3.5 rounded-full border focus:outline-none focus:ring-2 transition-all shadow-sm ${
                  isRecording
                    ? "border-red-500 bg-red-50 text-red-600 placeholder-red-400 focus:ring-red-200"
                    : "border-gray-300 focus:ring-[#072D94]/20 focus:border-[#072D94] text-gray-700"
                }`}
              />

              {/* Nút Gửi (Text) - Nằm bên trong Input */}
              <Button
                onClick={handleSend}
                disabled={!inputValue.trim() || isTyping || isRecording}
                className="absolute right-2 top-1.5 bottom-1.5 bg-[#001A61] hover:bg-[#072D94] text-white rounded-full w-10 h-10 p-0 flex items-center justify-center disabled:opacity-50 transition-all shadow-md"
              >
                <Send className="w-4 h-4 ml-0.5" />
              </Button>
            </div>

            {/* NÚT MICROPHONE (VOICE) - Nằm bên cạnh Input */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isTyping || isProcessingVoice}
              className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center shadow-md transition-all duration-300 ${
                isProcessingVoice
                  ? "bg-gray-200 cursor-wait"
                  : isRecording
                  ? "bg-red-500 hover:bg-red-600 animate-pulse ring-4 ring-red-200"
                  : "bg-white border border-gray-200 hover:bg-gray-100 text-gray-600 hover:text-[#072D94]"
              }`}
              title="Nói chuyện với AI"
            >
              {isProcessingVoice ? (
                <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
              ) : isRecording ? (
                <Square className="w-5 h-5 text-white fill-current" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="text-[10px] text-gray-400 mt-2 text-center select-none">
            {t.input.disclaimer}
          </p>
        </div>
      </main>
    </div>
  );
}
