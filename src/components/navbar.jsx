import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import "../styles/navbar.css";

const Navbar = () => {
  const navigate = useNavigate();
  const [activeModule, setActiveModule] = useState("");
  const [showConfirm, setShowConfirm] = useState(false);
  const [pendingNav, setPendingNav] = useState(null);

  const location = useLocation();
  const [isBannerVisible, setIsBannerVisible] = useState(false);
  // const [currentPage, setCurrentPage] = useState(0);
  const [slideDirection, setSlideDirection] = useState(null);
  const menuButtonRef = useRef(null);
  const bannerRef = useRef(null);
  const [startIndex, setStartIndex] = useState(0);
  const [isMobileView, setIsMobileView] = useState(window.innerWidth <= 768);

  const handleFlasherClick = async () => {
    try {
      window.dispatchEvent(new Event("flasher-reset"));
      console.log("Frontend reset event fired");
 
      // stop SSE if still open
      if (window.evtSource) {
        window.evtSource.close();
        window.evtSource = null;
      }
 
      // terminate backend (stops monitor, frees J-Link) — same as popup "yes"
      const resp = await fetch("http://localhost:9262/terminate", { method: "POST" });
      if (resp.ok) {
        console.log("Backend terminated on entering Flasher");
      } else {
        console.warn("Terminate failed on entering Flasher");
      }
    } catch (err) {
      console.error("Error handling Flasher click:", err);
    }
  };
  const handleRefreshClick = async () => {
    try {
      const resp = await fetch("http://localhost:9262/terminate", {
        method: "POST",
      });
 
      if (resp.ok) {
        console.log("Backend terminated successfully");
        window.location.reload();   // refresh UI after cleanup
      } else {
        console.warn("Terminate failed");
      }
    } catch (err) {
      console.error("Terminate error:", err);
    }
  };
 
  const menuItems = [
    { path: "/howitworks", label: "How it works" },
    { path: "/audiodatacollection", label: "Data Collection" },
    { path: "/preprocessing", label: "Pre-Processing" },
    { path: "/modelcompatibility", label: "Model Compatibility" },
    // { path: "/modelembbededintegration", label: "Model Embedded Integration" },
    // { path: "/runningdebugging", label: "Debugger" },
    { path: "/Flasher", label: "Firmware Deployer" },
    { path: "/resultlog", label: "Result Log" },
  ];
 
  const terminateDebugger = async () => {
    try {
      const resp = await fetch("http://localhost:9262/terminate", {
        method: "POST",
      });
 
      if (!resp.ok) {
        console.warn("Terminate failed");
      } else {
        console.log("Debugger terminated successfully");
      }
    } catch (err) {
      console.error("Terminate error:", err);
    }
  };
 
  // Number of items to show per page
  const itemsPerPage = 6;
 
  const handleLogoClick = (e) => {
    const isOnDebugger = location.pathname === "/runningdebugging";
    const isDebuggerRunning =
  window.__DEBUGGER_RUNNING__ === true &&
  window.__DEBUGGER_SUCCESS__ !== true;
 
    const isOnFlasher = location.pathname === "/Flasher";
    const isFlasherRunning = window.__FLASHER_RUNNING__ === true;
 
    if ((isOnDebugger && isDebuggerRunning) || (isOnFlasher && isFlasherRunning)) {
      e.preventDefault();
      setPendingNav({ path: "/LandingPage" });
 
      setActiveModule(isOnDebugger ? "Debugger" : "Flasher");
      setShowConfirm(true);
    }
  };
  // Calculate total number of pages
  const totalPages = Math.ceil(menuItems.length / itemsPerPage);
 
  const toggleMenu = () => {
    setIsBannerVisible(!isBannerVisible);
  };
 
  const closeBanner = () => {
    setIsBannerVisible(false);
  };
 
  const handleProtectedNav = async (e, item) => {
    const isOnDebugger = location.pathname === "/runningdebugging";
    const isGoingToDebugger = item.path === "/runningdebugging";
 
const isDebuggerRunning =
  window.__DEBUGGER_RUNNING__ === true &&
  window.__DEBUGGER_SUCCESS__ !== true;
    const isOnFlasher = location.pathname === "/Flasher";
    const isGoingAwayFromFlasher = item.path !== "/Flasher";
    const isFlasherRunning = window.__FLASHER_RUNNING__ === true;
 
    if (isOnDebugger && isGoingToDebugger) {
      e.preventDefault();
      closeBanner();
      return;
    }
 
    if (isOnFlasher && isGoingAwayFromFlasher && isFlasherRunning) {
      e.preventDefault();
      setPendingNav(item);
      setActiveModule("Flasher");
      setShowConfirm(true);
      return;
    }
 
    if (isOnDebugger && !isGoingToDebugger && isDebuggerRunning) {
      e.preventDefault();
      setPendingNav(item);
      setActiveModule("Debugger");
      setShowConfirm(true);
      return;
    }
 
    if (!isOnDebugger && isGoingToDebugger) {
      e.preventDefault();
      closeBanner();
 
      await terminateDebugger();
      navigate("/runningdebugging");
      return;
    }
 
    if (item.path === "/Flasher") {
      handleFlasherClick();
    }
  };
 
  const changePage = (direction) => {
    if (direction === 'next' && startIndex + itemsPerPage < menuItems.length) {
      setSlideDirection('right');
      setStartIndex(startIndex + itemsPerPage);
    } else if (direction === 'prev' && startIndex - itemsPerPage >= 0) {
      setSlideDirection('left');
      setStartIndex(startIndex - itemsPerPage);
    }
  };
 
  useEffect(() => {
    if (slideDirection) {
      const timer = setTimeout(() => setSlideDirection(null), 300); // Match CSS duration
      return () => clearTimeout(timer);
    }
  }, [slideDirection]);
 
 
  useEffect(() => {
    const handleResize = () => {
      setIsMobileView(window.innerWidth <= 768);
    };
 
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
 
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        menuButtonRef.current && !menuButtonRef.current.contains(event.target) &&
        bannerRef.current && !bannerRef.current.contains(event.target)
      ) {
        setIsBannerVisible(false);
      }
    };
 
    // Add event listener when component mounts
    document.addEventListener("mousedown", handleClickOutside);
 
    // Remove event listener when component unmounts
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);
 
 
  return (
    <nav className="navbar">
      {/* Company logo with link to landing page */}
      <div className="navbar-logo">
        <Link to="/LandingPage"
          onClick={handleLogoClick}
          draggable="false"
          onContextMenu={(e) => e.preventDefault()}>
          <img
            src="assets/amb_logo 1.png"
            alt="Logo"
            className="logo"
            draggable="false"
            onContextMenu={(e) => e.preventDefault()}
          />
        </Link>
      </div>
       
      {/* Navigation menu button */}
      <ul className="navbar-links">

        <li className="menu-item">
          <button onClick={toggleMenu} ref={menuButtonRef} className="menu-button">
            <span className="hamburger-icon">
              <span></span>
              <span></span>
              <span></span>
            </span>
            <span className="menu-text">Menu</span>
          </button>
        </li>
      </ul>
 
      <div className={`menu-banner ${isBannerVisible ? "show" : ""}`} ref={bannerRef}>
        {isMobileView ? (
          <div className="menu-items-wrapper scrollable-wrapper">
            <div className="menu-items-container no-slide">
              {menuItems.map((item, index) => (
                <Link
                  key={index}
                  to={item.path}
                  className="compartment"
                  onClick={(e) => {
                    closeBanner();
                    handleProtectedNav(e, item);
                  }}
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                >
                  {item.label}
                </Link>
              ))}
 
            </div>
          </div>
        ) : (
          <>
            {totalPages > 1 && (
              <button
                className="nav-arrow left-arrow"
                onClick={() => changePage("prev")}
                disabled={startIndex === 0}
              >
                <ChevronLeft size={24} />
              </button>
            )}
 
            <div className="menu-items-wrapper">
              <div
                className={`menu-items-container ${slideDirection ? `slide-${slideDirection}` : ""}`}
                key={startIndex}
              >
                {menuItems.slice(startIndex, startIndex + itemsPerPage).map((item, index) => (
                  <Link
                    key={index}
                    to={item.path}
                    className="compartment"
                    onClick={(e) => {
                      closeBanner();
                      handleProtectedNav(e, item);
                    }}
 
                    draggable="false"
                    onContextMenu={(e) => e.preventDefault()}
                  >
                    {item.label}
                  </Link>
                ))}
 
              </div>
            </div>
 
            {totalPages > 1 && (
              <button
                className="nav-arrow right-arrow"
                onClick={() => changePage("next")}
                disabled={startIndex + itemsPerPage >= menuItems.length}
              >
                <ChevronRight size={24} />
              </button>
            )}
          </>
        )}
      </div>
      {showConfirm && (
        <div className="confirm-overlay">
          <div className="confirm-box">
            <h3>{activeModule} is running...</h3>
            <p>Are you sure you want to leave this page?</p>
 
            <div className="confirm-actions">
 
              <button
                className="confirm-cancel"
                onClick={() => {
                  setShowConfirm(false);
                  setPendingNav(null);
                }}
              >
   
                No
              </button>
         
              <button
                className="confirm-ok"
                onClick={async () => {
                  setShowConfirm(false);
 
                  await terminateDebugger();
 
                  if (activeModule === "Flasher") {
                    // stop SSE + cleanup
                    if (window.evtSource) {
                      window.evtSource.close();
                      window.evtSource = null;
                    }
                  }
   
                  if (pendingNav?.path === "/Flasher") {
                    handleFlasherClick();
                  }
   
                  navigate(pendingNav.path);
                }}
              >
                yes
              </button>
            </div>
          </div>
        </div>
      )}

    </nav>
  );
};
 
export default Navbar;
 