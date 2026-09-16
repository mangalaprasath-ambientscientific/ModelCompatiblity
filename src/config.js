// =====================================================
// EdgeSphere - frontend configuration
// frontend/src/config.js
// =====================================================
// FINAL version. Replaces the copy you attached. Three fixes:
//
//   FIX 1  AGENT_API is now a single literal string.
//          `http://${AGENT_HOST}:${AGENT_PORT}` is split apart by the
//          minifier, so the built bundle no longer contained the text
//          "127.0.0.1:9262" and build-all.bat's verification grep failed on
//          a build that was actually correct.
//
//   FIX 2  DOWNLOAD_BASE no longer defaults to "/downloads".
//          On a hosted page that resolves to https://yoursite/downloads/...,
//          which does not exist, so the browser saves the 404 HTML page as
//          a .exe. Now it defaults to empty and the UI says the build is
//          misconfigured instead of shipping a broken download.
//
//   FIX 3  The client header is added only to state-changing requests.
//          Adding it to GET /health turned a simple request into a
//          preflighted one, doubling the round trips of a poll that runs
//          every 2 seconds on every customer machine.
//
// This file is the ONLY place that decides who the frontend talks to. There
// are three different servers in this product and confusing them is the
// single most common cause of "agent not detected":
//
//   AGENT_API      http://127.0.0.1:9262   local hardware
//   CLOUD_API      https://api.…           pure computation
//   window.origin  the Pages deployment    static files only
// =====================================================

import axios from "axios";

// ---------------------------------------------------------------------------
//  AGENT - always the local machine, in every environment, forever
// ---------------------------------------------------------------------------

export const AGENT_HOST = "127.0.0.1";
export const AGENT_PORT = 9262;

/** FIX 1: one literal, so it survives minification and can be grepped. */
const AGENT_DEFAULT = "http://127.0.0.1:9262";

/** Override only for unusual setups, e.g. a remote lab rig. */
export const AGENT_API = (
  process.env.REACT_APP_AGENT_URL || AGENT_DEFAULT
).replace(/\/+$/, "");

/** Socket.IO needs an absolute URL. Without this, io() connects to the
 *  page's own origin, so live hardware events silently never arrive when the
 *  app is hosted. */
export const SOCKET_URL = AGENT_API;

export const SOCKET_OPTIONS = {
  transports: ["websocket", "polling"],
  withCredentials: false,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
};

/** True when this bundle is being served BY the agent (offline mode). */
export const IS_LOCAL_UI =
  typeof window !== "undefined" &&
  window.location.origin.replace(/\/+$/, "") === AGENT_API;

// ---------------------------------------------------------------------------
//  CLOUD - your server
// ---------------------------------------------------------------------------

/** Set REACT_APP_CLOUD_API at build time. Empty string means "not deployed
 *  yet": the UI should disable those tabs rather than call a URL that does
 *  not exist. */
export const CLOUD_API = (process.env.REACT_APP_CLOUD_API || "")
  .replace(/\/+$/, "");

export const CLOUD_READY = CLOUD_API.length > 0;

// ---------------------------------------------------------------------------
//  BACK COMPAT
//  Existing code imports { API } or the default export and expects the
//  agent. Keeping both names means no call site has to change today.
// ---------------------------------------------------------------------------

export const API = AGENT_API;
export default AGENT_API;

// ---------------------------------------------------------------------------
//  VERSIONS
// ---------------------------------------------------------------------------

/** This website build. Sent to the agent so an old agent can say "update me". */
export const WEBSITE_VERSION = process.env.REACT_APP_WEBSITE_VERSION || "4.0.0";

/** Oldest agent this build of the website can work with. Raise it when you
 *  start calling an endpoint older agents do not have. */
export const REQUIRED_AGENT_VERSION = "4.0.0";

/** Oldest agent API shape this build understands. */
export const REQUIRED_API_VERSION = 1;

/** Semver-ish compare. Returns -1, 0 or 1. */
export function compareVersions(a, b) {
  const pa = String(a || "0").split(".").map((n) => parseInt(n, 10) || 0);
  const pb = String(b || "0").split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i += 1) {
    const x = pa[i] || 0;
    const y = pb[i] || 0;
    if (x !== y) return x < y ? -1 : 1;
  }
  return 0;
}

/** True when the agent is older than this website build requires. */
export function agentIsOutdated(agentVersion) {
  return compareVersions(agentVersion, REQUIRED_AGENT_VERSION) < 0;
}

// ---------------------------------------------------------------------------
//  CLIENT HEADER
//  security.py rejects POST/PUT/PATCH/DELETE to hardware routes without it.
//  That is deliberate: requiring a custom header forces a CORS preflight,
//  which the agent's origin allow-list then polices. A plain HTML form POST
//  from a hostile page cannot send a custom header, so it cannot reach a
//  hardware endpoint.
// ---------------------------------------------------------------------------

export const CLIENT_HEADER = "X-EdgeSphere-Client";
export const CLIENT_HEADER_VALUE = "1";

/** Methods that need the header. FIX 3: GET and HEAD stay "simple" requests
 *  so /health polling remains one round trip. Must match
 *  STATE_CHANGING_METHODS in security.py. */
const HEADER_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

export function isAgentUrl(url) {
  if (!url) return false;
  if (String(url).startsWith(AGENT_API)) return true;
  // Relative URL: only an agent call if the page itself is served by the agent.
  if (!/^https?:\/\//i.test(url)) {
    return IS_LOCAL_UI;
  }
  return false;
}

let _installed = false;

export function installAgentHeaders() {
  if (_installed) return;
  _installed = true;

  // --- axios ---
  axios.interceptors.request.use((cfg) => {
    const full = cfg.baseURL
      ? `${cfg.baseURL.replace(/\/+$/, "")}${cfg.url || ""}`
      : cfg.url;
    const method = String(cfg.method || "get").toUpperCase();
    if (isAgentUrl(full) && HEADER_METHODS.has(method)) {
      cfg.headers = cfg.headers || {};
      cfg.headers[CLIENT_HEADER] = CLIENT_HEADER_VALUE;
    }
    return cfg;
  });

  // --- fetch ---
  // Some call sites use fetch directly. Wrapping it once here means one
  // change instead of hunting every occurrence.
  //
  // Note what this CANNOT cover: EventSource (the /flashing-status stream)
  // and navigator.sendBeacon (/terminate on pagehide) cannot carry custom
  // headers at all. Both are handled on the server side - /flashing-status
  // is a GET, and /terminate is in CLIENT_HEADER_EXEMPT_PREFIXES.
  if (typeof window !== "undefined" && typeof window.fetch === "function") {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
      const url = typeof input === "string" ? input : (input && input.url);
      const method = String(
        init.method || (typeof input !== "string" && input && input.method) || "GET"
      ).toUpperCase();

      if (isAgentUrl(url) && HEADER_METHODS.has(method)) {
        const headers = new Headers(
          init.headers ||
            (typeof input !== "string" ? input && input.headers : undefined)
        );
        if (!headers.has(CLIENT_HEADER)) {
          headers.set(CLIENT_HEADER, CLIENT_HEADER_VALUE);
        }
        return nativeFetch(input, { ...init, headers });
      }
      return nativeFetch(input, init);
    };
  }
}

installAgentHeaders();

// ---------------------------------------------------------------------------
//  DOWNLOADS
//  The installer is ~500 MB. Cloudflare Pages refuses any single file over
//  25 MB, so it is served from R2, not from this site.
//
//  frontend/.env.production:
//      REACT_APP_AGENT_DOWNLOAD_BASE=https://pub-xxxxxxxx.r2.dev
// ---------------------------------------------------------------------------

/** FIX 2: empty by default. No silent fallback to a path that 404s. */
export const DOWNLOAD_BASE = (
  process.env.REACT_APP_AGENT_DOWNLOAD_BASE || ""
).replace(/\/+$/, "");

export const DOWNLOADS_CONFIGURED = DOWNLOAD_BASE.length > 0;

export const INSTALLERS = {
  windows: {
    label: "Windows 10/11 (64-bit)",
    file:
      process.env.REACT_APP_AGENT_INSTALLER ||
      "EdgeSphere_Agent_Setup_4.0.0.exe",
    note: "Includes J-Link and the ARM toolchain. Around 500 MB.",
    available: true,
  },
  mac: {
    label: "macOS",
    file: "EdgeSphere-Agent.dmg",
    note: "Not yet released.",
    available: false,
  },
  linux: {
    label: "Linux (Debian/Ubuntu)",
    file: "edgesphere-agent.deb",
    note: "Not yet released.",
    available: false,
  },
};

export function installerFor(os) {
  const entry = INSTALLERS[os] || INSTALLERS.windows;
  return {
    ...entry,
    url: DOWNLOADS_CONFIGURED ? `${DOWNLOAD_BASE}/${entry.file}` : "",
    configured: DOWNLOADS_CONFIGURED && entry.available,
  };
}

/** Best guess at the visitor's OS, for picking a default download. */
export function detectOS() {
  if (typeof navigator === "undefined") return "windows";
  const p = (
    (navigator.userAgentData && navigator.userAgentData.platform) ||
    navigator.platform ||
    ""
  ).toLowerCase();
  if (p.includes("win")) return "windows";
  if (p.includes("mac")) return "mac";
  if (p.includes("linux")) return "linux";
  return "windows";
}

export const OS_LABELS = {
  windows: "Windows",
  mac: "macOS",
  linux: "Linux",
  unknown: "your system",
};
