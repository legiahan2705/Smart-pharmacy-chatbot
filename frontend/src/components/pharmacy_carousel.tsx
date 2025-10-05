"use client";

import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Slide {
  id: number;
  title: string;
  color: string;
  icon: string;
}

const PharmacyCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const slides: Slide[] = [
     { id: 1, title: "Medicine Consultation", color: "bg-[#DEECFF]", icon: "💊" },
    { id: 2, title: "Professional Pharmacists", color: "bg-[#DEECFF]", icon: "⚕️" },
    { id: 3, title: "Health Technology", color: "bg-[#DEECFF]", icon: "🔬" },
    { id: 4, title: "E-Prescriptions", color: "bg-[#DEECFF]", icon: "📋" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % slides.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [slides.length]);

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
            {slides.map((slide) => (
              <div
                key={slide.id}
                className={`min-w-full h-full flex items-center justify-center relative ${slide.color}`}
              >
                <div className="relative z-10 text-center p-8">
                  <div className="text-8xl mb-6 animate-bounce">{slide.icon}</div>
                  <h3 className="text-3xl font-bold text-[#072D94]">
                    {slide.title}
                  </h3>
                </div>
              </div>
            ))}
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
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-3 z-20">
          {slides.map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`transition-all rounded-full ${
                currentIndex === index
                  ? "bg-[#072D94] w-8 h-3"
                  : "bg-white/60 hover:bg-white/80 w-3 h-3"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PharmacyCarousel;
