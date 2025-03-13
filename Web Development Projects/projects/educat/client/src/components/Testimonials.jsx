// src/components/Testimonials.jsx
import React from 'react';

const testimonials = [
  { name: 'Alice Johnson', feedback: 'This platform has transformed my career!' },
  { name: 'Michael Lee', feedback: 'Great courses and amazing instructors!' },
  { name: 'Sophia Martinez', feedback: 'Highly recommended for learning new skills.' }
];

const Testimonials = () => {
  return (
    <section className="testimonials">
      <h2>What Our Students Say</h2>
      <div className="testimonial-list">
        {testimonials.map((testimonial, index) => (
          <div key={index} className="testimonial-item">
            <p>"{testimonial.feedback}"</p>
            <h4>- {testimonial.name}</h4>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Testimonials;
