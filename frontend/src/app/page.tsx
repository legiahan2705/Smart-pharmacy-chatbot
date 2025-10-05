"use client";
import { useState } from "react";
import PharmacyCarousel from "@/components/pharmacy_carousel";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Globe } from "lucide-react";

export default function Home() {
  const [language, setLanguage] = useState<"en" | "vi">("en");

  const content = {
    en: {
      header: {
        title: "Long Chau Pharmacy",
        home: "Home",
        signin: "Sign in",
      },
      main: {
        title: "Smart Medicine Assistant",
        description:
          "Get answers to all your questions about medicines, symptoms, and health — available 24/7",
        startChat: "Start chatting now",
        learnMore: "Learn more",
      },
    },
    vi: {
      header: {
        title: "Nhà Thuốc Long Châu",
        home: "Trang chủ",
        signin: "Đăng nhập",
      },
      main: {
        title: "Trợ Lý Y Tế Thông Minh",
        description:
          "Nhận câu trả lời cho mọi câu hỏi về thuốc, triệu chứng và sức khỏe — luôn sẵn sàng 24/7",
        startChat: "Bắt đầu trò chuyện",
        learnMore: "Tìm hiểu thêm",
      },
    },
  };

  const t = content[language];

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "en" ? "vi" : "en"));
  };

  return (
    <div className="relative min-h-screen flex flex-col">
      {/* --- Header --- */}
      <header className="bg-[#DEECFF] text-[#072D94] px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img
              src="/logo_LongChau.png"
              alt="Long Chau Logo"
              className="h-15 w-auto"
            />
            <h1 className="font-bold text-3xl">{t.header.title}</h1>
          </div>

          <nav className="flex gap-8 items-center">
            <a
              href="/"
              className="hover:text-blue-200 transition-colors font-medium text-[20px]"
            >
              {t.header.home}
            </a>
            <a
              href="/signin"
              className="hover:text-blue-200 transition-colors font-medium text-[20px]"
            >
              {t.header.signin}
            </a>
            <button
              onClick={toggleLanguage}
              className="flex items-center gap-2 px-4 py-2 bg-[#072D94] text-white rounded-lg hover:bg-[#001A61] transition-colors font-medium"
            >
              <Globe className="w-5 h-5" />
              {language === "en" ? "VI" : "EN"}
            </button>
          </nav>
        </div>
      </header>

      {/* --- Carousel Section (Full Width) --- */}
      <section className="w-full p-0">
        <PharmacyCarousel />
      </section>

      {/* --- Main Content Section --- */}
      <main className="flex-1 bg-gradient-to-b from-[#DEECFF] via-[#5f7cca] to-[#001A61] text-white">
        <div className="max-w-4xl mx-auto px-6 py-16 text-center">
          <h2 className="text-5xl font-bold mb-6 leading-tight">
            {t.main.title}
          </h2>
          <p className="text-xl mb-10 text-blue-100 leading-relaxed max-w-4xl mx-auto">
            {t.main.description}
          </p>
          <div className="flex gap-4 justify-center items-center">
            <Link href="/chatbot">
              <Button className="bg-white border-2 border-white text-[#001A61] hover:bg-blue-50 px-8 py-6 text-lg font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all">
                {t.main.startChat}
              </Button>
            </Link>

            <Link href="/learn_more">
              <button className="text-white hover:bg-white hover:text-[#001A61] px-8 py-3 text-lg font-semibold rounded-lg transition-all border-2 border-white">
                {t.main.learnMore}
              </button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
