// src/pages/Home.jsx

import React from 'react';
import HeroSection from '../components/HeroSection';
import Categories from '../components/Categories';
import Courses from '../components/Courses';
import FAQs from '../components/FAQs';
import Testimonials from '../components/Testimonials';

const Home = () => {
  return (
    <>
      <HeroSection />
      <Categories />
      <Courses />
      <FAQs />
      <Testimonials />
    </>
  );
};

export default Home;