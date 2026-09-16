// =====================================================
// EdgeSphere - local agent detection + launch (frontend only)
// frontend/src/utils/agentLauncher.js
// =====================================================
// Why this file exists
// -------------------
// Hosted as a website, the page has no way to run a process. There is no
// browser API that takes "C:\Program Files\EdgeSphere\EdgeSphereAgent.exe"
// and starts it, and navigating to file:///...exe is blocked outright.
//
// The one mechanism a web page is allowed to use is a registered custom URL
// protocol: the page navigates to `edgesphere://start`, the browser hands the
// URL to Windows, and Windows launches whatever the registry says handles it.
// That registry entry is written once by the installer:
//
//   HKCU\Software\Classes\edgesphere              @ = "URL:EdgeSphere Agent Protocol"
//                                                 "URL Protocol" = ""
//   HKCU\Software\Classes\edgesphere\shell\open\command
//       @ = "\"C:\\Program Files\\EdgeSphere\\EdgeSphereAgent.exe\""
//
// So: detection is a plain fetch to the agent's /health, and launching is a
// protocol hand-off. autoStartAgent() runs this once at page load and is
// entirely silent - it renders nothing and never blocks the UI.
// =====================================================

import { AGENT_API, IS_LOCAL_UI } from "../config";
import { isSecurePage, readLocalNetworkPermission } from "../agent";

/** Where the installer puts the agent. The registry command above points at
 *  this; JS itself can never execute a path. */
export const AGENT_EXE_PATH =
  "C:\\Program Files\\EdgeSphere\\EdgeSphereAgent.exe";

/** What the page hands to the OS. Windows resolves it through
 *  HKCU\Software\Classes\edgesphere\shell\open\command -> AGENT_EXE_PATH. */
export const AGENT_PROTOCOL_URL = "edgesphere://start";

const HEALTH_URL = `${AGENT_API}/health`;

/** A dead port refuses instantly; this only bounds the pathological case. */
const PROBE_TIMEOUT_MS = 1500;

/** The agent unpacks its bundled toolchain on a cold start, so the first
 *  /health can take a while. Too short here means a false "not running". */
export const AGENT_START_TIMEOUT_MS = 30000;

const POLL_INTERVAL_MS = 800;

/**
 * True when something is answering /health on the agent port.
 * Any failure - refused connection, abort, CORS - counts as "not running",
 * which is the useful answer: the frontend cannot talk to it either way.
 */
export async function isAgentRunning(timeoutMs = PROBE_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    // GET with no custom header stays a simple request, so no preflight.
    // Cache-buster keeps a stale 200 from masking an agent that just died.
    const resp = await fetch(`${HEALTH_URL}?t=${Date.now()}`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
    return resp.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ask the OS to start the agent. Returns the route taken, not whether it
 * worked - nothing here can know that. Poll with waitForAgent() after.
 *
 * The hand-off goes through a hidden iframe rather than a top-level
 * navigation. That matters at page load, where there is no user gesture: a
 * browser that refuses the launch just logs a console warning, and an
 * unregistered-protocol error stays inside the iframe instead of navigating
 * the app away to an error page.
 */
export function requestAgentLaunch() {
  if (typeof window === "undefined" || !document.body) return "unavailable";

  // Desktop build: the main process can spawn directly, no protocol needed.
  if (window.electronAPI && typeof window.electronAPI.startAgent === "function") {
    try {
      window.electronAPI.startAgent();
      return "electron";
    } catch {
      /* fall through to the protocol hand-off */
    }
  }

  try {
    const frame = document.createElement("iframe");
    frame.style.display = "none";
    frame.src = AGENT_PROTOCOL_URL;
    document.body.appendChild(frame);
    // The hand-off is synchronous; the element has no reason to stick around.
    setTimeout(() => frame.remove(), 2000);
    return "protocol";
  } catch {
    return "failed";
  }
}

/**
 * Poll /health until the agent answers or the deadline passes.
 * @param {number} timeoutMs
 */
export async function waitForAgent(timeoutMs = AGENT_START_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await isAgentRunning(1200)) return true;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  // One last look - the agent may have come up during the final sleep.
  return isAgentRunning(PROBE_TIMEOUT_MS);
}

// The protocol hand-off is not idempotent from the user's point of view - two
// hand-offs can mean two "Open EdgeSphere Agent?" prompts - so it happens once
// per page load, whatever calls in.
let started = false;

/**
 * Starts the agent if it is not already up. Called at startup from index.js
 * and again by AgentGate when it is cleared to reach loopback (see below) -
 * the work itself still happens at most once per page load.
 *
 * Completely silent: no dialog, no spinner, nothing rendered. The result is
 * published on `window.__AGENT_RUNNING__` for anything that wants to check.
 *
 * @returns {Promise<boolean>} whether the agent is up when this settles
 */
export async function autoStartAgent() {
  if (started) return Boolean(window.__AGENT_RUNNING__);

  // Do not be the first thing to touch 127.0.0.1.
  //
  // On a hosted https page, Chrome 142+ answers the first request to loopback
  // with "<site> wants to look for and connect to any device on your local
  // network". Asking that from here means it appears during page load with
  // nothing on screen to explain it, and a user who clicks Block cannot be
  // un-blocked by any code we write. AgentGate asks the same question behind
  // an "Allow and connect" button, with the reason visible, so when the
  // permission is still undecided the first contact is left to the gate - the
  // user reaches it by opening a hardware page, which is the only time the
  // agent is needed anyway. AgentGate calls this function back once it has
  // that gesture, so the protocol launch below still happens - just later.
  // Note the `started` latch is deliberately NOT set on this path, which is
  // what leaves that second call able to run.
  //
  // Every other case proceeds as before: 'granted' just works, 'denied' fails
  // fast, 'unsupported' means this browser does not enforce the permission,
  // and the agent-served offline build is same-origin so none of this applies.
  if (!IS_LOCAL_UI && isSecurePage()) {
    const permission = await readLocalNetworkPermission();
    if (permission === "prompt") {
      window.__AGENT_RUNNING__ = false;
      return false;
    }
  }

  started = true;

  if (await isAgentRunning()) {
    window.__AGENT_RUNNING__ = true;
    return true;
  }

  window.__AGENT_RUNNING__ = false;
  requestAgentLaunch();

  const up = await waitForAgent();
  window.__AGENT_RUNNING__ = up;

  if (!up) {
    // Nothing user-facing by design. If this shows up, either the agent is
    // not installed or the edgesphere:// registry key is missing.
    console.warn(
      `EdgeSphere Agent did not come up on ${AGENT_API}. Expected at ${AGENT_EXE_PATH}`
    );
  }
  return up;
}
