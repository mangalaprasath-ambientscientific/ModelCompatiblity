/**
 * @file Body.jsx
 * @description Landing page Body component with animation effects
 * @author [Gourav Pradhan]
 * @copyright (c) 2025 Ambient Scientific Inc.
 * @version 1.0.0
 */

import React, { useEffect } from "react";
import { gsap } from "gsap";
import "../styles/body.css";

/**
 * Body component for the EdgeSphere landing page
 * Displays the main heading and description with GSAP animations
 * 
 * @component
 * @returns {JSX.Element} The rendered Body component
 */
const Body = () => {
  /**
   * Set up GSAP animations on component mount
   * Creates entrance animations for heading, subheadings, and paragraph text
   */
  useEffect(() => {
    // Create GSAP context for clean animation management
    const ctx = gsap.context(() => {
      // Animate main heading from top with fade-in effect
      gsap.from(".body-container-body h1", {
        opacity: 0,
        y: -50,
        duration: 1,
        ease: "power2.out",
      });

      // Animate subheadings from bottom with staggered timing
      gsap.from(".body-container-body h2", {
        opacity: 0,
        y: 50,
        duration: 1,
        stagger: 0.3, // Stagger each h2 animation by 0.3 seconds
        ease: "power2.out",
      });

      // Animate paragraph text with slight delay
      gsap.from(".body-container-body p", {
        opacity: 0,
        y: 20,
        duration: 1.2,
        delay: 0.5, // Start after heading animations
        ease: "power2.out",
      });
    });

    // Clean up animations when component unmounts
    return () => ctx.revert();
  }, []);

  return (
    <div className="body-container-body">
      <h1>
        <span className="edge">Edge</span>
        <span className="sphere">Sphere</span>
      </h1>
      <p>
        - A holistic toolchain for data collection, <br /> feature engineering, AI modeling, and deployment at the edge.
      </p>
    </div>
  );
};

export default Body;