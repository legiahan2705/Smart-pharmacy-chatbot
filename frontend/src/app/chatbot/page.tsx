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
  Volume2,
  Pill,
  AlertTriangle,
  Info,
  CheckCircle2,
  ClipboardList,
} from "lucide-react";
import Link from "next/link";
import { v4 as uuidv4 } from "uuid";
import ReactMarkdown from "react-markdown";

// --- API CALL ---
async function askAgent(userInput: string, threadId: string) {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: userInput, thread_id: threadId }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    return data.answer;
  } catch (error) {
    return "Xin lỗi, tôi đang gặp sự cố kết nối với server.";
  }
}

export default function ChatbotPage() {
  const [mounted, setMounted] = useState(false);
  const [language, setLanguage] = useState<"en" | "vi">("vi");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [typingStatus, setTypingStatus] = useState("");
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>("");
  const [messages, setMessages] = useState<any[]>([]);

  // Nút dừng
  const abortControllerRef = useRef<boolean>(false);

  // Voice & Loading States
  const [isRecording, setIsRecording] = useState(false);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(
    null,
  );
  const [loadingAudioId, setLoadingAudioId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const botAudioRef = useRef<HTMLAudioElement | null>(null);

  const content = {
    en: {
      sidebar: {
        title: "Long Chau AI",
        newChat: "+ New Chat",
        history: "HISTORY",
        backHome: "Back to Home",
      },
      header: { title: "Smart Medicine Assistant", status: "Online" },
      typingStatuses: ["Thinking...", "Searching database..."],
      input: { placeholder: "Ask me anything...", recording: "Listening..." },
    },
    vi: {
      sidebar: {
        title: "Long Châu AI",
        newChat: "+ Hội thoại mới",
        history: "LỊCH SỬ TRÒ CHUYỆN",
        backHome: "Về trang chủ",
      },
      header: { title: "Trợ Lý Y Tế Thông Minh", status: "Đang hoạt động" },
      typingStatuses: ["Đang suy nghĩ...", "Đang tra cứu dữ liệu..."],
      input: {
        placeholder: "Nhập câu hỏi của bạn...",
        recording: "Đang nghe bạn nói...",
      },
    },
  };
  const t = content[language];

  useEffect(() => {
    setMounted(true);
  }, []);

  const ActiveVolumeIcon = () => (
    <div className="relative flex items-center justify-center w-4 h-4 mr-1">
      <Volume2 size={14} className="relative z-10" />
      <span className="absolute inset-0 bg-white/40 rounded-full animate-ping"></span>
      <div className="absolute flex gap-[1px] items-end justify-center -right-2 h-3">
        <span
          className="w-[1.5px] bg-white rounded-full animate-bounce h-2"
          style={{ animationDuration: "0.5s" }}
        ></span>
        <span
          className="w-[1.5px] bg-white rounded-full animate-bounce h-3"
          style={{ animationDuration: "0.8s" }}
        ></span>
        <span
          className="w-[1.5px] bg-white rounded-full animate-bounce h-2"
          style={{ animationDuration: "0.6s" }}
        ></span>
      </div>
    </div>
  );

  const beautifyText = (text: string) => {
    return text
      .replace(/(?:^|\n)(\d+\. )/g, "\n\n$1")
      .replace(/Công dụng:|Tác dụng:|Chỉ định:/gi, "\n\n**💊 Công dụng:**")
      .replace(/Chống chỉ định:/gi, "\n\n**🚫 Chống chỉ định:**")
      .replace(/Thận trọng:|Cảnh báo:/gi, "\n\n**⚠️ Thận trọng:**")
      .replace(/Lưu ý quan trọng:|Lưu ý:/gi, "\n\n**💡 Lưu ý quan trọng:**")
      .replace(/Liều dùng:|Cách dùng:/gi, "\n\n**📋 Liều dùng & Cách dùng:**")
      .replace(/Tương tác thuốc:|Tương tác:/gi, "\n\n**🔄 Tương tác thuốc:**")
      .replace(/Tác dụng phụ:|Phản ứng phụ:/gi, "\n\n**⚡ Tác dụng phụ:**");
  };

  const splitText = (text: string) => {
    return text
      .split(/([.!?])\s+/)
      .reduce((acc: string[], cur, i) => {
        if (i % 2 === 0) acc.push(cur);
        else acc[acc.length - 1] += cur;
        return acc;
      }, [])
      .filter((s) => s.trim().length > 0);
  };

  const startRecording = () => {
    stopBotAudio();
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Trình duyệt không hỗ trợ Mic.");
    const recognition = new SpeechRecognition();
    recognition.lang = language === "vi" ? "vi-VN" : "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setInputValue(transcript);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(() => {
        stopRecording();
        handleSendInternal(transcript);
      }, 5000);
    };
    recognition.start();
    recognitionRef.current = recognition;
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (recognitionRef.current) recognitionRef.current.stop();
    setIsRecording(false);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
  };

  const handleStopGeneration = () => {
    abortControllerRef.current = true;
    setIsTyping(false);
  };

  const handleSendInternal = async (text: string) => {
    if (!text.trim()) return;
    abortControllerRef.current = false; // Reset nút dừng
    const currentThreadId = activeThreadId;
    const userMsg = {
      id: Date.now().toString(),
      type: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsTyping(true);
    setTypingStatus(
      t.typingStatuses[Math.floor(Math.random() * t.typingStatuses.length)],
    );

    try {
      const botAnswer = await askAgent(text, currentThreadId);

      // Kiểm tra xem người dùng có nhấn nút dừng khi đang gọi API không
      if (abortControllerRef.current) return;

      const botMsgId = (Date.now() + 1).toString();
      const botMsg = {
        id: botMsgId,
        type: "bot",
        content: "",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMsg]);

      let displayed = "";
      const words = botAnswer.split(" ");
      for (let word of words) {
        // Kiểm tra nút dừng trong khi đang gõ
        if (abortControllerRef.current) break;

        displayed += word + " ";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsgId ? { ...m, content: displayed } : m,
          ),
        );
        await new Promise((r) => setTimeout(r, 20));
      }

      if (!abortControllerRef.current) {
        const savedMsgs = JSON.parse(
          localStorage.getItem(`chat_messages_${currentThreadId}`) || "[]",
        );
        localStorage.setItem(
          `chat_messages_${currentThreadId}`,
          JSON.stringify([
            ...savedMsgs,
            userMsg,
            { ...botMsg, content: botAnswer },
          ]),
        );
      }
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = () => handleSendInternal(inputValue);

  const playTextToSpeech = async (text: string, messageId: string) => {
    if (speakingMessageId === messageId) {
      stopBotAudio();
      return;
    }
    stopBotAudio();
    const chunks = splitText(text);
    if (chunks.length === 0) return;
    setLoadingAudioId(messageId);
    let index = 0;
    const playNext = async () => {
      if (index >= chunks.length) {
        setIsBotSpeaking(false);
        setSpeakingMessageId(null);
        return;
      }
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${apiUrl}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: chunks[index] }),
        });
        const audioBlob = await res.blob();
        const audio = new Audio(URL.createObjectURL(audioBlob));
        botAudioRef.current = audio;
        audio.onplay = () => {
          setLoadingAudioId(null);
          setIsBotSpeaking(true);
          setSpeakingMessageId(messageId);
        };
        audio.onended = () => {
          index++;
          playNext();
        };
        await audio.play();
      } catch {
        setLoadingAudioId(null);
        setIsBotSpeaking(false);
      }
    };
    playNext();
  };

  const stopBotAudio = () => {
    if (botAudioRef.current) {
      botAudioRef.current.pause();
      botAudioRef.current = null;
    }
    setIsBotSpeaking(false);
    setSpeakingMessageId(null);
    setLoadingAudioId(null);
  };

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("chat_sessions");
      if (saved) {
        const parsed = JSON.parse(saved).map((s: any) => ({
          ...s,
          updatedAt: new Date(s.updatedAt),
        }));
        setSessions(parsed);
        if (parsed.length > 0) loadSession(parsed[0].id);
        else createNewSession();
      } else createNewSession();
    }
  }, []);

  const createNewSession = () => {
    const id = uuidv4();
    const sess = { id, title: "Hội thoại mới", updatedAt: new Date() };

    // SỬA Ở ĐÂY: Lưu danh sách session vào State VÀ cả LocalStorage
    setSessions((prev) => {
      const updatedSessions = [sess, ...prev];
      localStorage.setItem("chat_sessions", JSON.stringify(updatedSessions));
      return updatedSessions;
    });

    setActiveThreadId(id);
    setMessages([
      {
        id: "welcome",
        type: "bot",
        timestamp: new Date(),
        content: "Xin chào! Tôi là trợ lý y tế thông minh.",
      },
    ]);
  };

  const loadSession = (id: string) => {
    setActiveThreadId(id);
    const saved = localStorage.getItem(`chat_messages_${id}`);
    if (saved)
      setMessages(
        JSON.parse(saved).map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp),
        })),
      );
    else
      setMessages([
        {
          id: "welcome",
          type: "bot",
          timestamp: new Date(),
          content: "Xin chào! Tôi là trợ lý y tế thông minh.",
        },
      ]);
  };

  return (
    <div
      className="flex h-screen bg-[#F8F9FB] text-slate-900 overflow-hidden font-sans"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      {/* Sidebar */}
      <aside
        className={`${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200 flex flex-col transition-transform duration-300 shadow-sm`}
      >
        <div className="p-5 border-b border-slate-100">
          <Link
            href="/"
            className="flex items-center gap-3 mb-6 hover:opacity-80 transition-opacity"
          >
            <img src="/logo_LongChau.png" alt="Logo" className="h-18 w-auto" />
            <div>
              <h1 className="font-bold text-2xl text-[#072D94] leading-tight">
                Long Chau Pharmacy
              </h1>
              <p className="text-[12px] font-semibold text-[#072D94] ">
                Smart Medicine Assistant
              </p>
            </div>
          </Link>
          <Button
            onClick={createNewSession}
            className="w-full bg-[#072D94] text-white rounded-xl shadow-md py-6 font-semibold"
          >
            + Hội thoại mới
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <p className="px-3 py-2 text-[11px] font-bold text-slate-400 uppercase tracking-widest">
            LỊCH SỬ
          </p>
          {mounted &&
            sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => loadSession(s.id)}
                className={`p-3 rounded-xl cursor-pointer transition-all ${
                  activeThreadId === s.id
                    ? "bg-blue-50 text-[#072D94] font-medium"
                    : "hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <ClipboardList size={16} className="opacity-50" />
                  <p className="text-sm truncate font-medium">{s.title}</p>
                </div>
              </div>
            ))}
        </div>
        <div className="p-4 border-t border-slate-100">
          <Link href="/">
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 text-[#072D94] hover:bg-blue-50 rounded-xl py-6"
            >
              <Home size={20} />{" "}
              <span className="font-semibold text-sm">Về trang chủ</span>
            </Button>
          </Link>
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col h-full bg-[#F8F9FB] overflow-hidden">
        <header className="flex-none bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 py-4 flex justify-between items-center z-10 shadow-sm">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden p-2 hover:bg-slate-100 rounded-lg text-[#072D94]"
            >
              <Menu size={20} />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#072D94] flex items-center justify-center shadow-sm shrink-0">
                <Bot size={22} className="text-white" />
              </div>
              <div>
                <h1 className="font-bold text-[#072D94] leading-tight">
                  Trợ Lý Y Tế Thông Minh
                </h1>
                <p className="text-[11px] text-green-500 font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>{" "}
                  Đang hoạt động
                </p>
              </div>
            </div>
          </div>
          <Button
            onClick={() => setLanguage((l) => (l === "vi" ? "en" : "vi"))}
            variant="outline"
            className="rounded-full px-4 text-xs font-bold border-slate-200 hover:bg-slate-50"
          >
            {language.toUpperCase()}
          </Button>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-10 space-y-8 pb-32 scroll-smooth">
          {mounted &&
            messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-4 ${
                  m.type === "user" ? "flex-row-reverse" : "flex-row"
                } animate-in fade-in slide-in-from-bottom-2 duration-500`}
              >
                <div
                  className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 shadow-sm ${
                    m.type === "user"
                      ? "bg-[#072D94]"
                      : "bg-white border border-slate-200"
                  }`}
                >
                  {m.type === "user" ? (
                    <User size={18} className="text-white" />
                  ) : (
                    <Bot size={18} className="text-[#072D94]" />
                  )}
                </div>
                <div
                  className={`flex flex-col w-full ${
                    m.type === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <div
                    className={`p-5 md:p-6 rounded-3xl shadow-sm leading-relaxed max-w-[95%] md:max-w-[85%] ${
                      m.type === "user"
                        ? "bg-[#072D94] text-white rounded-tr-none font-medium"
                        : "bg-white border border-slate-200 rounded-tl-none text-slate-800"
                    }`}
                  >
                    <div
                      className={`prose prose-sm max-w-none prose-slate ${
                        m.type === "user" ? "prose-invert" : ""
                      } text-[16px] text-justify prose-strong:font-bold`}
                    >
                      <ReactMarkdown>
                        {m.type === "bot" ? beautifyText(m.content) : m.content}
                      </ReactMarkdown>
                    </div>
                    {m.type === "bot" && m.content && (
                      <div className="mt-6 pt-4 border-t border-slate-100 flex items-center gap-2">
                        <button
                          onClick={() => playTextToSpeech(m.content, m.id)}
                          disabled={
                            loadingAudioId !== null && loadingAudioId !== m.id
                          }
                          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-[12px] font-bold transition-all shadow-sm ${
                            speakingMessageId === m.id ||
                            loadingAudioId === m.id
                              ? "bg-blue-600 text-white"
                              : "bg-slate-50 text-slate-500 hover:bg-slate-100"
                          }`}
                        >
                          {loadingAudioId === m.id ? (
                            <Loader2 size={14} className="animate-spin" />
                          ) : speakingMessageId === m.id ? (
                            <ActiveVolumeIcon />
                          ) : (
                            <Volume2 size={14} />
                          )}
                          {speakingMessageId === m.id
                            ? "DỪNG ĐỌC"
                            : "NGHE NỘI DUNG"}
                        </button>
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-slate-400 mt-2 px-1 font-medium">
                    {mounted
                      ? new Date(m.timestamp).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : "--:--"}
                  </p>
                </div>
              </div>
            ))}
          {isTyping && mounted && (
            <div className="flex gap-4 animate-in fade-in duration-300">
              <div className="w-9 h-9 rounded-xl bg-white border border-slate-200 flex items-center justify-center shadow-sm shrink-0">
                <Bot size={18} className="text-[#072D94] animate-bounce" />
              </div>
              <div className="bg-white border border-slate-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex flex-col gap-2 min-w-[150px]">
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-[#072D94] rounded-full animate-bounce"></span>
                  <span className="w-1.5 h-1.5 bg-[#072D94] rounded-full animate-bounce [animation-delay:0.2s]"></span>
                </div>
                <span className="text-xs font-bold text-[#072D94] animate-pulse italic">
                  {typingStatus}
                </span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-4" />
        </div>

        {/* Input area */}
        <div className="flex-none p-4 md:p-6 bg-gradient-to-t from-white via-white/95 to-transparent border-t border-slate-200 z-20">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-center">
              <div className="flex-1 flex gap-3 bg-white p-2 rounded-2xl border border-slate-200 focus-within:border-[#072D94] focus-within:ring-2 focus-within:ring-blue-50 transition-all shadow-sm">
                <input
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && !isTyping && handleSend()
                  }
                  placeholder={
                    isRecording ? t.input.recording : t.input.placeholder
                  }
                  className={`flex-1 px-4 py-3 bg-transparent outline-none text-sm placeholder:text-slate-400 font-medium ${
                    isRecording ? "text-blue-600 animate-pulse" : ""
                  }`}
                />

                {/* NÚT GỬI HOẶC DỪNG */}
                {isTyping ? (
                  <Button
                    onClick={handleStopGeneration}
                    className="bg-[#072D94] hover:bg-[#001A61] rounded-xl px-5 transition-all shadow-md"
                  >
                    <Square size={18} fill="white" />
                  </Button>
                ) : (
                  <Button
                    onClick={handleSend}
                    disabled={!inputValue.trim() || isRecording}
                    className="bg-[#072D94] hover:bg-[#001A61] rounded-xl px-5 transition-all disabled:bg-slate-300 shadow-md"
                  >
                    <Send size={18} className="text-white" />
                  </Button>
                )}
              </div>

              {/* Nút Mic */}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isTyping}
                className={`p-4 rounded-full transition-all shadow-lg ${
                  isRecording
                    ? "bg-red-500 text-white animate-pulse"
                    : isTyping
                      ? "bg-slate-100 text-slate-300 cursor-not-allowed"
                      : "bg-white text-[#072D94] border border-slate-200 hover:bg-slate-50"
                }`}
              >
                {isRecording ? (
                  <Square size={24} fill="white" />
                ) : (
                  <Mic size={24} />
                )}
              </button>
            </div>
            <p className="text-[10px] text-center text-slate-400 mt-3 font-medium opacity-70">
              AI Assistant - Không thay thế lời khuyên y khoa chuyên nghiệp
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
