import React from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";
import { Pagination, Navigation } from "swiper/modules";

const HowItWorks = () => {
  return (
    <div className="slider-show">
      <Swiper
        spaceBetween={30}
        slidesPerView={1}
        pagination={{ clickable: true }}
        loop={true} 
        modules={[Pagination, Navigation]}
      >
        <SwiperSlide>
          <div className="slide slide-1">
            <img
              src="assets/Slide-0.png"
              alt="Step 1"
              className="slide-image"
              draggable="false"
              onContextMenu={(e) => e.preventDefault()}
            />
          </div>
        </SwiperSlide>
        <SwiperSlide>
          <div className="slide slide-2">
            <img
              src="assets/Slide-1.png"
              alt="Step 2"
              className="slide-image"
              draggable="false"
              onContextMenu={(e) => e.preventDefault()}
            />
          </div>
        </SwiperSlide>
        <SwiperSlide>
          <div className="slide slide-3">
            <img
              src="assets/Slide-2.png"
              alt="Step 3"
              className="slide-image"
              draggable="false"
              onContextMenu={(e) => e.preventDefault()}
            />
          </div>
        </SwiperSlide>
        <SwiperSlide>
          <div className="slide slide-4">
            <img
              src="assets/Slide-3.png"
              alt="Step 4"
              className="slide-image"
              draggable="false"
              onContextMenu={(e) => e.preventDefault()}
            />
          </div>
        </SwiperSlide>
      </Swiper>
    </div>
  );
};

export default HowItWorks;
