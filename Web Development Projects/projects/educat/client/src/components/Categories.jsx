// src/components/Categories.jsx
import React from 'react';

const Categories = () => {
  const categories = ['Web Development', 'Data Science', 'Graphic Design', 'Business', 'Marketing'];

  return (
    <section className="categories">
      <h2>Categories</h2>
      <div className="category-list">
        {categories.map((category, index) => (
          <div key={index} className="category-item">{category}</div>
        ))}
      </div>
    </section>
  );
};

export default Categories;