import { Link } from "react-router-dom";
import "../assets/css/style.css"; 

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/">Home</Link>
      <Link to="/about">About</Link>
      <Link to="/courses">Courses</Link>
      <Link to="/contact">Contact</Link>
    </nav>
  );
};

export default Navbar;
