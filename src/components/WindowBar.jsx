import React, { useState, useEffect } from "react";
import "../styles/windowbar.css";

const WindowBar = () => {
    const [apiReady, setApiReady] = useState(false);
    useEffect(() => {
        function waitForPyWebView() {
            if (window.pywebview) {
                setApiReady(true);
            } else {
                setTimeout(waitForPyWebView, 100);
            }
        }
        waitForPyWebView();
    }, []);
    const minimize = () => {
        if (apiReady) window.pywebview.api.minimize();
    };

    const maximize = () => {
        if (apiReady) window.pywebview.api.maximize();
    };

    const close = () => {
        if (apiReady) window.pywebview.api.close();
    };
    const refresh = async () => {
        if (!apiReady) {
            window.location.reload();
            return;
        }

        try {

            await window.pywebview.api.refresh();
            window.location.reload();

        } catch (err) {
            console.error("Refresh error", err);
            window.location.reload();
        }
    };
    return (

        <div className="window-bar">

            {/* LEFT */}
            <div className="left-controls">
                <button className="refresh-btn" onClick={refresh}>⟳</button>
            </div>

            {/* CENTER */}
            <div className="title">EdgeSphere V-3.0.0</div>

            {/* RIGHT */}
            <div className="window-buttons">
                <button onClick={minimize}>—</button>
                <button onClick={maximize}>☐</button>
                <button className="close-btn" onClick={close}>✕</button>
            </div>

        </div>

    );
};

export default WindowBar;