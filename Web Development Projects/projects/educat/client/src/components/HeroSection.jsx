// src/components/HeroSection.jsx
import React from 'react';
import heroImage from '../assets/hero-image.jpg';

const HeroSection = () => {
  return (
    <header className="hero-section">
      <img src={heroImage} alt="Education Platform" className="hero-image" />
      <h1>Welcome to Our Education Platform</h1>
      <p>Explore the best courses to enhance your skills</p>
      <button className="btn btn-primary">Get Started</button>
    </header>
  );
};

export default HeroSection;