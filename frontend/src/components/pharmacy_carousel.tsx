"use client";

import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Lottie from "lottie-react";

interface Slide {
  id: number;
  titleEn: string;
  titleVi: string;
  color: string;
  lottieFileName: string; // Tên file JSON trong thư mục public
}

// Định nghĩa kiểu dữ liệu cho object chứa tất cả Lottie data
interface LottieData {
  [key: string]: any;
}

const PharmacyCarousel = ({ language }: { language: "en" | "vi" }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  // TẠO STATE MỚI để lưu dữ liệu của TẤT CẢ các animation
  const [lottieData, setLottieData] = useState<LottieData>({});
  // State để theo dõi xem dữ liệu đã tải xong chưa
  const [isLoading, setIsLoading] = useState(true);

  const slides: Slide[] = [
    // LƯU Ý: Đảm bảo tên file JSON khớp chính xác với tên trong thư mục public
    {
      id: 1,
      titleEn: "24/7 Symptom Checker",
      titleVi: "Kiểm tra Triệu chứng 24/7",
      color: "bg-[#DEECFF]",
      lottieFileName: "Data_Scanning.json" 
    },
    {
      id: 2,
      titleEn: "Medical Technology Applications",
      titleVi: "Ứng dụng Công nghệ Y tế",
      color: "bg-[#DEECFF]",
      lottieFileName: "Healthcare_Heart_icon_animation.json" 
    },
    {
      id: 3,
      titleEn: "Health Technology",
      titleVi: "Công nghệ sức khỏe",
      color: "bg-[#DEECFF]",
      lottieFileName: "AI_application_in_healthcare.json" 
    },
    {
      id: 4,
      titleEn: "AI-powered Drug Advice Support",
      titleVi: "Hỗ trợ Tư vấn Thuốc bằng AI",
      color: "bg-[#DEECFF]",
      lottieFileName: "Chatbot.json" 
    },
  ];

  useEffect(() => {
    setIsLoading(true);
    const fetchLottieData = async () => {
      const data: LottieData = {};
      const fetchPromises = slides.map(async (slide) => {
        try {
          // Fetch từ đường dẫn public
          const res = await fetch(`/${slide.lottieFileName}`);
          if (!res.ok) throw new Error(`Failed to fetch ${slide.lottieFileName}`);
          const json = await res.json();
          data[slide.lottieFileName] = json;
        } catch (error) {
          console.error(`Error loading Lottie for ${slide.lottieFileName}:`, error);
          // Để tránh lỗi crash, nếu file lỗi, ta lưu null
          data[slide.lottieFileName] = null;
        }
      });

      await Promise.all(fetchPromises);
      setLottieData(data);
      setIsLoading(false);
    };

    fetchLottieData();

    // Thiết lập tự động chuyển slide
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % slides.length);
    }, 3000);
    return () => clearInterval(interval);

  }, []); // Chỉ chạy một lần khi component mount

  const goToSlide = (index: number) => setCurrentIndex(index);
  const goToPrevious = () =>
    setCurrentIndex((prev) => (prev - 1 + slides.length) % slides.length);
  const goToNext = () =>
    setCurrentIndex((prev) => (prev + 1) % slides.length);

  return (
    <div className="w-full mx-auto">
      <div className="relative overflow-hidden">
        {/* Carousel Container */}
        <div className="relative h-96">
          <div
            className="flex transition-transform duration-700 ease-in-out h-full"
            style={{ transform: `translateX(-${currentIndex * 100}%)` }}
          >
            {slides.map((slide) => {
              const animationData = lottieData[slide.lottieFileName];
              return (
                <div
                  key={slide.id}
                  className={`min-w-full h-full flex items-center justify-center relative ${slide.color}`}
                >
                  <div className="relative z-10 text-center p-8">

                    {/* Hiển thị animation Lottie nếu đã tải xong và hợp lệ */}
                    {!isLoading && animationData ? (
                      <div className="w-63 h-64 mx-auto mb-6"> 
                        <Lottie
                          animationData={animationData}
                          loop={true}
                          autoplay={true}
                        />
                      </div>
                    ) : (
                      // Hiển thị placeholder hoặc spinner nếu đang tải
                      <div className="w-48 h-48 mx-auto mb-6 flex items-center justify-center text-4xl text-gray-500">
                        {isLoading ? 'Đang tải...' : 'Lỗi tải animation'}
                      </div>
                    )}

                    <h3 className="text-2xl font-bold text-[#072D94]">
                      {language === "vi" ? slide.titleVi : slide.titleEn}
                    </h3>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Navigation Arrows */}
        <button
          onClick={goToPrevious}
          className="absolute left-4 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white p-3 rounded-full shadow-lg transition-all hover:scale-110 z-20"
        >
          <ChevronLeft className="w-6 h-6 text-[#072D94]" />
        </button>
        <button
          onClick={goToNext}
          className="absolute right-4 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white p-3 rounded-full shadow-lg transition-all hover:scale-110 z-20"
        >
          <ChevronRight className="w-6 h-6 text-[#072D94]" />
        </button>

        {/* Dots Indicator */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex gap-3 z-20">
          {slides.map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`transition-all rounded-full ${currentIndex === index
                ? "bg-[#072D94] w-8 h-3"
                : "bg-[#cad5f3]  hover:bg-white/80 w-3 h-3"
                }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PharmacyCarousel;
