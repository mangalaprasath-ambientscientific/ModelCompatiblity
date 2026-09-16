import React, { useState } from 'react';
import '../styles/Button.css';

const Button = ({ text, onClick, logo, addClass, disabled }) => {
  // State to track button interactions
  const [clicked, setClicked] = useState(false); // Tracks click animation state
  const [hovered, setHovered] = useState(false); // Tracks hover animation state

  const handleClick = () => {
    if (disabled) return; // Prevent click if disabled
    setClicked(true); // Activate click animation
    onClick && onClick(); // Execute callback if provided

    setTimeout(() => {
      setClicked(false);
    }, 1000); // Animation duration in milliseconds
  };

  const handleMouseEnter = () => {
    if (!disabled) setHovered(true);
  };

  const handleMouseLeave = () => {
    setHovered(false);
  };

  return (
    <button
      className={`${addClass ? addClass : 'button'} ${clicked ? 'clicked' : ''} ${hovered && !clicked ? 'hovered' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      disabled={disabled}
    >
      {logo && <span className="button-logo">{logo}</span>}
      {text}
    </button>
  );
};

export default Button;
