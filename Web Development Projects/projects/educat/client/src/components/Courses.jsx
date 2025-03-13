// src/components/Courses.jsx
import React from 'react';

const courses = [
  { title: 'React for Beginners', instructor: 'John Doe' },
  { title: 'Advanced JavaScript', instructor: 'Jane Smith' },
  { title: 'UI/UX Design Basics', instructor: 'Emily Brown' }
];

const Courses = () => {
  return (
    <section className="courses">
      <h2>Popular Courses</h2>
      <div className="course-list">
        {courses.map((course, index) => (
          <div key={index} className="course-item">
            <h3>{course.title}</h3>
            <p>Instructor: {course.instructor}</p>
            <button className="btn btn-secondary">Enroll Now</button>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Courses;