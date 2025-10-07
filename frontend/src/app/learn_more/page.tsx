"use client";
import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  MessageSquare,
  Clock,
  Shield,
  Heart,
  Pill,
  Activity,
  Globe,
} from "lucide-react";

const LearnMore = () => {
  const [language, setLanguage] = useState<"en" | "vi">("en");

  const content = {
    en: {
      hero: {
        title: "Smart Medicine Assistant",
        subtitle: "Your 24/7 Healthcare Companion",
        description:
          "Powered by advanced AI technology, our Smart Medicine Assistant provides instant, reliable healthcare information to help you make informed decisions about your health.",
      },
      features: {
        title: "How We Help You",
        items: [
          {
            title: "Instant Answers",
            description:
              "Get immediate responses to your health questions without waiting in line or scheduling appointments.",
          },
          {
            title: "24/7 Available",
            description:
              "Access healthcare information anytime, anywhere. Our assistant never sleeps so you get help when you need it.",
          },
          {
            title: "Medicine Info",
            description:
              "Learn about medications, their uses, side effects, and interactions in clear, easy-to-understand language.",
          },
          {
            title: "Symptom Checker",
            description:
              "Describe your symptoms and receive guidance on potential causes and when to seek professional care.",
          },
          {
            title: "Safe & Secure",
            description:
              "Your privacy is our priority. All conversations are confidential and stored securely.",
          },
          {
            title: "Health Tips",
            description:
              "Receive personalized health advice and wellness tips to help you maintain a healthier lifestyle.",
          },
        ],
      },
      howItWorks: {
        title: "How It Works",
        steps: [
          {
            title: "Ask Your Question",
            description:
              "Type your health-related question or describe your symptoms in natural language.",
          },
          {
            title: "Get Instant Response",
            description:
              "Our AI analyzes your question and provides accurate, helpful information within seconds.",
          },
          {
            title: "Take Action",
            description:
              "Use the information to make informed decisions or know when to consult a healthcare professional.",
          },
        ],
      },
      cta: {
        title: "Ready to Get Started?",
        description:
          "Join thousands of users who trust our Smart Medicine Assistant for their healthcare needs.",
        startButton: "Start chatting now",
        backButton: "Back to Home",
      },
    },
    vi: {
      hero: {
        title: "Trợ Lý Y Tế Thông Minh",
        subtitle: "Người Đồng Hành Sức Khỏe 24/7",
        description:
          "Được hỗ trợ bởi công nghệ AI tiên tiến, Trợ lý Y tế Thông minh của chúng tôi cung cấp thông tin y tế đáng tin cậy ngay lập tức để giúp bạn đưa ra quyết định sáng suốt về sức khỏe.",
      },
      features: {
        title: "Chúng Tôi Hỗ Trợ Bạn Như Thế Nào",
        items: [
          {
            title: "Trả Lời Tức Thì",
            description:
              "Nhận phản hồi ngay lập tức cho các câu hỏi về sức khỏe mà không cần chờ đợi hay đặt lịch hẹn.",
          },
          {
            title: "Luôn Sẵn Sàng 24/7",
            description:
              "Truy cập thông tin y tế mọi lúc, mọi nơi. Trợ lý của chúng tôi không bao giờ ngủ để bạn luôn nhận được sự giúp đỡ khi cần.",
          },
          {
            title: "Thông Tin Thuốc",
            description:
              "Tìm hiểu về thuốc, công dụng, tác dụng phụ và tương tác bằng ngôn ngữ rõ ràng, dễ hiểu.",
          },
          {
            title: "Kiểm Tra Triệu Chứng",
            description:
              "Mô tả triệu chứng của bạn và nhận hướng dẫn về nguyên nhân tiềm ẩn và khi nào cần tìm chuyên gia.",
          },
          {
            title: "An Toàn & Bảo Mật",
            description:
              "Quyền riêng tư của bạn là ưu tiên hàng đầu. Mọi cuộc trò chuyện đều được bảo mật và lưu trữ an toàn.",
          },
          {
            title: "Lời Khuyên Sức Khỏe",
            description:
              "Nhận tư vấn sức khỏe cá nhân hóa và mẹo chăm sóc sức khỏe để duy trì lối sống lành mạnh hơn.",
          },
        ],
      },
      howItWorks: {
        title: "Cách Hoạt Động",
        steps: [
          {
            title: "Đặt Câu Hỏi",
            description:
              "Nhập câu hỏi liên quan đến sức khỏe hoặc mô tả triệu chứng của bạn bằng ngôn ngữ tự nhiên.",
          },
          {
            title: "Nhận Phản Hồi Ngay",
            description:
              "AI của chúng tôi phân tích câu hỏi và cung cấp thông tin chính xác, hữu ích trong vài giây.",
          },
          {
            title: "Hành Động",
            description:
              "Sử dụng thông tin để đưa ra quyết định sáng suốt hoặc biết khi nào cần tư vấn chuyên gia y tế.",
          },
        ],
      },
      cta: {
        title: "Sẵn Sàng Bắt Đầu?",
        description:
          "Tham gia cùng hàng nghìn người dùng tin tưởng Trợ lý Y tế Thông minh cho nhu cầu chăm sóc sức khỏe.",
        startButton: "Bắt đầu trò chuyện",
        backButton: "Về Trang Chủ",
      },
    },
  };

  const t = content[language];

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "en" ? "vi" : "en"));
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header with Language Toggle */}
      <header className="bg-[#DEECFF] text-[#072D94] px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img
              src="/logo_LongChau.png"
              alt="Long Chau Logo"
              className="h-15 w-auto"
            />
            <h1 className="font-bold text-3xl">
              {language === "en" ? "Long Chau Pharmacy" : "Nhà Thuốc Long Châu"}
            </h1>
          </div>

          <nav className="flex gap-8 items-center">
            <a
              href="/"
              className="hover:text-[#072D94] hover:bg-white active:scale-90 transition-transform duration-150 font-medium text-[20px] px-2  rounded-lg"
            >
              {language === "en" ? "Home" : "Trang chủ"}
            </a>
            <a
              href="/signin"
              className="hover:text-[#072D94] hover:bg-white active:scale-90 transition-transform duration-150 font-medium text-[20px] px-2  rounded-lg"
            >
              {language === "en" ? "Sign in" : "Đăng nhập"}
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

      {/* Hero Section */}
      <section className="bg-gradient-to-b from-[#DEECFF] to-[#5f7cca] text-[#001A61] py-20">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            {t.hero.title}
          </h1>
          <p className="text-2xl md:text-3xl mb-8 text-white font-semibold">
            {t.hero.subtitle}
          </p>
          <p className="text-lg max-w-3xl mx-auto leading-relaxed text-neutral-100">
            {t.hero.description}
          </p>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-background py-20">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-16 text-[#001A61]">
            {t.features.title}
          </h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {t.features.items.map((feature, index) => {
              const icons = [
                MessageSquare,
                Clock,
                Pill,
                Activity,
                Shield,
                Heart,
              ];
              const Icon = icons[index];

              return (
                <div
                  key={index}
                  className="bg-[#DEECFF] p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow"
                >
                  <div className="bg-[#072D94] w-16 h-16 rounded-full flex items-center justify-center mb-6">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-[#001A61]">
                    {feature.title}
                  </h3>
                  <p className="text-[#072D94] leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-gradient-to-b from-[#5f7cca] to-[#001A61] py-20 text-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-16">
            {t.howItWorks.title}
          </h2>

          <div className="grid md:grid-cols-3 gap-12">
            {t.howItWorks.steps.map((step, index) => (
              <div key={index} className="text-center">
                <div className="bg-white text-[#001A61] w-20 h-20 rounded-full flex items-center justify-center text-3xl font-bold mx-auto mb-6">
                  {index + 1}
                </div>
                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                <p className="text-blue-100 leading-relaxed">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-[#DEECFF] py-20">
        <div className="max-w-5xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-6 text-[#001A61]">
            {t.cta.title}
          </h2>
          <p className="text-xl mb-10 text-[#072D94] leading-relaxed">
            {t.cta.description}
          </p>
          <div className="flex gap-4 justify-center items-center flex-wrap">
            <Link href="/chatbot">
              <Button className="bg-[#001A61] border-2 border-[#001A61] text-white hover:bg-[#072D94] px-8 py-6 text-lg font-semibold rounded-lg shadow-lg hover:shadow-xl active:scale-90 transition-transform duration-150">
                {t.cta.startButton}
              </Button>
            </Link>
            <Link
              href="/"
              className="text-[#001A61] hover:bg-[#001A61] hover:text-white px-8 py-3 text-lg font-semibold rounded-lg border-2 border-[#001A61] active:scale-90 transition-transform duration-150"
            >
              {t.cta.backButton}
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LearnMore;
