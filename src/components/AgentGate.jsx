// =====================================================
// EdgeSphere - agent gate
// frontend/src/components/AgentGate.jsx
// =====================================================
// Wraps any page that needs local hardware. Renders its children only when
// the agent is present and able to flash; otherwise it explains WHICH
// problem occurred and offers the matching fix.
//
// Usage in App.js (HashRouter):
//     <Route path="/Flasher" element={
//       <AgentGate feature="Flashing"><Flasher /></AgentGate>
//     } />
//
// Pages that do NOT touch hardware (Landing, Pre-Processing, Model
// Compatibility, Result Log) are NOT wrapped. They must work with no install
// at all - that is the whole point of hosting the UI.
//
// -----------------------------------------------------------------------
// CHANGE LOG
// -----------------------------------------------------------------------
// FIX 1  The permission prompt is triggered by a CLICK, not by the probe on
//        mount. On a hosted https page the first request to 127.0.0.1 makes
//        Chrome ask "...wants to look for and connect to any device on your
//        local network". Firing that during page load, with no explanation
//        on screen, is how users end up blocking it - and a blocked
//        permission cannot be undone by your code.
//
// FIX 2  A BLOCKED panel exists. A denied permission used to show the
//        "install the agent" screen, sending users to reinstall software
//        they already had.
//
// FIX 3  Polling continues while the gate is open, not only after a download
//        click. The page unlocks by itself when the agent starts, which is
//        what the download screen promises.
// =====================================================

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  AgentState,
  detectOS,
  installerFor,
  isSecurePage,
  OS_LABELS,
  probeAgent,
  readLocalNetworkPermission,
  waitForAgent,
} from "../agent";
import { autoStartAgent } from "../utils/agentLauncher";
import "../styles/agent-gate.css";

export default function AgentGate({ children, feature = "This page" }) {
  const [result, setResult] = useState({ state: AgentState.CHECKING });
  const [started, setStarted] = useState(false);
  const cancelRef = useRef(null);

  const os = detectOS();
  const installer = installerFor(os);

  // ---- decide whether we may probe silently --------------------------
  // 'prompt' on a secure page means the browser will show its dialog on our
  // first request. Wait for a click so the user can see why it is being
  // asked. 'granted', 'denied' and 'unsupported' all need no gesture:
  // granted just works, denied fails fast into the BLOCKED panel, and
  // unsupported means this browser does not enforce the permission at all.
  useEffect(() => {
    let alive = true;
    readLocalNetworkPermission().then((state) => {
      if (!alive) return;
      const needsGesture = state === "prompt" && isSecurePage();
      if (!needsGesture) setStarted(true);
    });
    return () => { alive = false; };
  }, []);

  // ---- probe, then keep polling until the agent is usable ------------
  const start = useCallback(() => {
    if (cancelRef.current) cancelRef.current();
    setResult({ state: AgentState.CHECKING });
    // index.js also calls this at page load, but it bails out there when the
    // local-network permission is still undecided rather than firing the
    // browser prompt with nothing on screen to explain it. Reaching this line
    // means we are allowed to talk to loopback, so the edgesphere:// launch
    // can go ahead - an installed-but-not-running agent starts itself and the
    // poll below picks it up. It is latched to once per page load internally.
    autoStartAgent();
    cancelRef.current = waitForAgent(setResult, { interval: 2000 });
  }, []);

  useEffect(() => {
    if (!started) return undefined;
    start();
    return () => {
      if (cancelRef.current) cancelRef.current();
      cancelRef.current = null;
    };
  }, [started, start]);

  // A user who plugs in hardware or starts the agent in another window and
  // then comes back should not wait for the next tick.
  useEffect(() => {
    if (!started) return undefined;
    const onWake = () => {
      if (document.visibilityState === "visible") {
        probeAgent().then(setResult);
      }
    };
    window.addEventListener("focus", onWake);
    document.addEventListener("visibilitychange", onWake);
    return () => {
      window.removeEventListener("focus", onWake);
      document.removeEventListener("visibilitychange", onWake);
    };
  }, [started]);

  const retry = () => {
    setStarted(true);
    start();
  };

  // ---------- the page works ----------
  if (result.state === AgentState.READY) {
    return <>{children}</>;
  }

  // ---------- waiting for the user to allow the browser prompt ----------
  if (!started) {
    return (
      <div className="agent-gate">
        <div className="agent-card">
          <h2>Connect to your hardware</h2>
          <p>
            {feature} controls a J-Link probe plugged into this computer.
            Your browser will ask for permission to reach it - choose{" "}
            <strong>Allow</strong>.
          </p>
          <button className="agent-btn agent-btn-primary" onClick={retry}>
            Allow and connect
          </button>
          <p className="agent-note">
            The connection stays on this machine. Nothing about your hardware
            leaves your computer.
          </p>
        </div>
      </div>
    );
  }

  // ---------- first probe in flight ----------
  if (result.state === AgentState.CHECKING) {
    return (
      <div className="agent-gate">
        <div className="agent-card">
          <div className="agent-spinner" />
          <p>Looking for the EdgeSphere Agent…</p>
        </div>
      </div>
    );
  }

  // ---------- the browser is refusing us ----------
  if (result.state === AgentState.BLOCKED) {
    return (
      <div className="agent-gate">
        <div className="agent-card">
          <h2>Your browser is blocking access to the agent</h2>
          <p>
            {feature} needs to reach the EdgeSphere Agent on this computer, and
            permission for that is currently set to Block. Reinstalling the
            agent will not change this - it is a browser setting.
          </p>
          <ol className="agent-steps">
            <li>Click the icon at the left of the address bar.</li>
            <li>
              Find <strong>Local network access</strong> and set it to{" "}
              <strong>Allow</strong>.
            </li>
            <li>Reload this page.</li>
          </ol>
          <button className="agent-btn" onClick={retry}>Check again</button>
          <p className="agent-hint">
            On a managed work computer this can be controlled by IT. The policy
            they need is <code>LocalNetworkAccessAllowedForUrls</code>, listing
            this website's address.
          </p>
        </div>
      </div>
    );
  }

  // ---------- agent running, but cannot flash ----------
  if (result.state === AgentState.NOT_READY) {
    const missing = (result.info && result.info.missing_tools) || [];
    return (
      <div className="agent-gate">
        <div className="agent-card">
          <h2>Agent found, but it can't flash yet</h2>
          <p>
            EdgeSphere Agent {result.info && result.info.agent_version} is
            running on {result.info && result.info.platform}, but these
            components are missing:
          </p>
          <ul className="agent-missing">
            {missing.map((m) => <li key={m}><code>{m}</code></li>)}
          </ul>
          <p className="agent-hint">
            This usually means the installer did not finish. Reinstalling
            restores the bundled J-Link runtime and toolchain.
          </p>
          <DownloadLink installer={installer} os={os} label="Reinstall for" />
          <button className="agent-link" onClick={retry}>Retry</button>
        </div>
      </div>
    );
  }

  // ---------- agent too old ----------
  if (result.state === AgentState.OUTDATED) {
    return (
      <div className="agent-gate">
        <div className="agent-card">
          <h2>Your agent needs updating</h2>
          <p>
            This page needs a newer EdgeSphere Agent. You have{" "}
            <strong>{result.info && result.info.agent_version}</strong>.
          </p>
          <DownloadLink installer={installer} os={os} label="Update for" />
          <p className="agent-hint">
            Your reports and recordings are kept — updating replaces only the
            application.
          </p>
        </div>
      </div>
    );
  }

  // ---------- no agent ----------
  const originRejected = result.error === "origin_not_allowed";

  return (
    <div className="agent-gate">
      <div className="agent-card">
        {originRejected ? (
          <>
            <h2>The agent is running but does not trust this site</h2>
            <p>
              An agent answered on this computer and refused the request. This
              website's address is not in the agent's allow-list, which means
              the installed agent is older than this deployment.
            </p>
            <DownloadLink installer={installer} os={os} label="Reinstall for" />
            <button className="agent-link" onClick={retry}>Check again</button>
          </>
        ) : (
          <>
            <h2>Edgesphere Agent Not Detected</h2>
            <p>
              {feature} talks to a J-Link probe over USB. Browsers can't reach
              USB devices directly, so EdgeSphere uses a small local agent on
              your machine.
            </p>

            <DownloadLink
              installer={installer}
              os={os}
              label="Install for"
              primary
            />
            {installer.configured && (
              <p className="agent-note">{installer.note}</p>
            )}

            <details className="agent-other">
              <summary>Other platforms</summary>
              <ul>
                {["windows", "mac", "linux"]
                  .filter((k) => k !== os)
                  .map((k) => {
                    const inst = installerFor(k);
                    return (
                      <li key={k}>
                        {inst.configured ? (
                          <a href={inst.url} download>{inst.label}</a>
                        ) : (
                          <span className="agent-unavailable">
                            {inst.label} — {inst.note}
                          </span>
                        )}
                      </li>
                    );
                  })}
              </ul>
            </details>

            <p className="agent-waiting">
              <span className="agent-spinner agent-spinner-sm" />
              Waiting for the agent… this page will unlock automatically.
            </p>
            <button className="agent-link" onClick={retry}>
              Already installed? Check again
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/** One download button, or an honest message when the build has no URL. */
function DownloadLink({ installer, os, label, primary = false }) {
  if (!installer.configured) {
    return (
      <p className="agent-warn">
        This build has no download link configured. Set{" "}
        <code>REACT_APP_AGENT_DOWNLOAD_BASE</code> in{" "}
        <code>frontend/.env.production</code> and deploy again.
      </p>
    );
  }
  return (
    <a
      className={`agent-btn${primary ? " agent-btn-primary" : ""}`}
      href={installer.url}
      download
    >
      {label} {OS_LABELS[os]}
    </a>
  );
}

/** Small status indicator for the navbar. Independent of the gate. */
export function AgentStatusPill() {
  const [result, setResult] = useState({ state: AgentState.CHECKING });

  useEffect(() => {
    const cancel = waitForAgent(setResult, { interval: 5000 });
    return cancel;
  }, []);

  const LABELS = {
    checking: "Checking",
    ready: "Connected",
    not_ready: "Limited",
    outdated: "Update needed",
    blocked: "Blocked",
    missing: "Not detected",
  };

  return (
    <span className={`agent-pill agent-pill-${result.state}`}>
      <span className="agent-pill-dot" aria-hidden="true" />
      Agent: {LABELS[result.state]}
      {result.info && result.info.agent_version
        ? ` · ${result.info.agent_version}`
        : ""}
    </span>
  );
}