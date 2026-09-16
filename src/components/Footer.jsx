/**
 * @file Footer.jsx
 * @description Application footer component displaying copyright, logo, and connection status
 * @author [Gourav Pradhan]
 * @copyright (c) 2025 Ambient Scientific Inc.
 * @version 1.1.2
 */

import React from "react";
import useConnectionStatus from "../components/useConnectionStatus";
import "../styles/footer.css";

/**
 * Footer component that displays company information and connection status
 *
 * Shows copyright information, company logo, and visual indicators for
 * device connection statuses (JTAG and UART). Clicking either indicator asks
 * the backend to re-probe the hardware immediately.
 *
 * @component
 * @returns {JSX.Element} The rendered Footer component
 */
const Footer = () => {
  /**
   * Live hardware status. `ready` is latched by the hook - it turns true on the
   * first answer from the backend and never goes back to false - and is used
   * for the tooltip wording only, not for the colour.
   */
  const {
    deviceName,
    connections,
    statusMessage,
    refreshConnection,
    ready,
    stale,
  } = useConnectionStatus();

  /**
   * TWO STATES ONLY: connected (green) or disconnected (red).
   *
   * There is deliberately no neutral/"checking" colour any more. Anything that
   * is not a confirmed connection reads as red, so the indicator can never sit
   * in a third colour the user has to interpret. `stale` is not mixed in
   * either - dimming a connected box because one probe cycle ran long is what
   * used to produce the dark-green indicator on healthy hardware.
   */
  const boxState = (isConnected) => (isConnected ? "connected" : "disconnected");

  return (
    <footer className="footer">
      <div className="footer-content">
        {/* Copyright information with dynamically updated year */}
        <p>
          © Copyright Ambient Scientific Inc. {new Date().getFullYear()}, All
          rights Reserved.
        </p>

        {/* Company logo with link to website */}
        <div className="footer-logo">
          <a
            href="https://www.ambientscientific.ai/"
            target="_blank"
            rel="noopener noreferrer"
            draggable="false"
            onContextMenu={(e) => e.preventDefault()}
          >
            <img
              src="assets/footer_logo.png"
              alt="Ambient Scientific Logo"
              className="footer-logo-img"
              draggable="false"
              onContextMenu={(e) => e.preventDefault()}
            />
          </a>
        </div>

        {/* Connection status indicators section */}
        <div className="footer-right">
          <div className="footer-color-boxes">
            {/* Status message and device name displays */}
            <div className="footer-text">
              <p>{statusMessage}</p>
            </div>

            {/* Interactive connection status indicators */}
            <div className="footer-buttons">
              {/* JTAG connection status indicator - click to re-probe */}
              <div
                className={`footer-color-box jtag ${boxState(connections.jtag)}`}
                onClick={refreshConnection}
                title={
                  !ready
                    ? "JTag not detected (waiting for the first hardware check) — click to refresh"
                    : `JTag ${connections.jtag ? "connected" : "not detected"}${
                        stale ? " (last update is old)" : ""
                      } — click to refresh`
                }
              ></div>

              {/* UART indicator - click to re-probe. Grey, not red, when absent:
                  nothing requires it any more, so it reports presence only. */}
              <div
                className={`footer-color-box uart ${
                  connections.uart ? "connected" : "optional"
                }`}
                onClick={refreshConnection}
                title={
                  !ready
                    ? "UART not detected (waiting for the first hardware check) — click to refresh"
                    : `UART ${
                        connections.uart
                          ? "connected"
                          : "not detected (optional — data and trace come over the J-Link)"
                      }${stale ? " (last update is old)" : ""} — click to refresh`
                }
              ></div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;