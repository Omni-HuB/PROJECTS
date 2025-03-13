// src/components/FAQs.jsx
import React from 'react';

const faqs = [
  { question: 'How do I enroll in a course?', answer: 'Simply click the enroll button and follow the instructions.' },
  { question: 'Are the courses free?', answer: 'Some courses are free, while others require a payment.' },
  { question: 'Can I get a certificate?', answer: 'Yes, certificates are available for most courses.' }
];

const FAQs = () => {
  return (
    <section className="faqs">
      <h2>Frequently Asked Questions</h2>
      <div className="faq-list">
        {faqs.map((faq, index) => (
          <div key={index} className="faq-item">
            <h3>{faq.question}</h3>
            <p>{faq.answer}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default FAQs;
