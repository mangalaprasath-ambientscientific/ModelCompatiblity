/**
 * @file PageTransition.jsx
 * @description Component that handles page transitions with fade effect
 * @author [Gourav Pradhan]
 * @copyright (c) 2025 Ambient Scientific Inc.
 * @version 1.0.0
 */

import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';

/**
 * PageTransition component that adds a fade-out animation when navigating between pages
 * 
 * @component
 * @param {Object} props - Component props
 * @param {string} props.to - Target route path to navigate to
 * @param {React.ReactNode} props.children - Child elements to render inside the transition container
 * @returns {JSX.Element} The rendered transition container with click handler
 */
const PageTransition = ({ to, children }) => {
  /**
   * React Router's navigate function for programmatic navigation
   * @type {Function}
   */
  const navigate = useNavigate(); 
  
  /**
   * State to control fade animation
   * @type {[boolean, Function]} Fade state and setter
   */
  const [fade, setFade] = useState(false);

  /**
   * Handles click events on the transition container
   * Triggers fade-out animation and then navigates to target route
   */
  const handleClick = () => {
    setFade(true); // Start fade-out animation
    
    // Navigate to target route after animation starts
    setTimeout(() => {
      navigate(to); 
    }, 100); // Delay matches CSS transition duration
  };

  /**
   * Reset fade state when target route changes
   * Ensures component is ready for next transition
   */
  useEffect(() => {
    setFade(false);
  }, [to]);

  return (
    <div 
      onClick={handleClick} 
      className={`circle-btn ${fade ? 'fade-out' : ''}`}
    >
      {children}
    </div>
  );
};

export default PageTransition;