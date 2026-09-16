// =====================================================
// EdgeSphere - local agent detection
// frontend/src/agent.js
// =====================================================
// The hosted UI runs from a CDN. Pages that only compute (Pre-Processing,
// Model Compatibility, docs) work with no install at all. Pages that touch
// the J-Link need the local agent, because a server cannot see a USB device
// plugged into the user's machine.
//
// This module answers one question, repeatedly and cheaply:
//     is the agent running on this machine right now, and can it flash?
//
// Four states matter and must NOT be collapsed into "offline":
//   MISSING    nothing answered           -> offer the installer
//   BLOCKED    the browser refused us     -> explain the permission
//   OUTDATED   answered, api too old      -> offer an update
//   NOT_READY  answered, toolchain broken -> show what is missing
// A UI that shows "install the agent" to someone who already has it running
// sends them in a circle.
//
// -----------------------------------------------------------------------
// CHANGE LOG
// -----------------------------------------------------------------------
// FIX 1  The installer table, detectOS and OS_LABELS were duplicated here
//        AND in config.js with DIFFERENT filenames, so whichever module a
//        component imported decided whether the download worked. They live
//        in config.js only now, re-exported from here so existing imports
//        keep working.
//
// FIX 2  BLOCKED is reachable. It used to be defined but never returned, so
//        a user who denied the browser permission got the "install the
//        agent" screen for software they already had - and reinstalling does
//        not fix a permission.
//
// FIX 3  targetAddressSpace REMOVED from the probe. See the long note in
//        probeAgent(). This is what made /health never arrive at the agent
//        while every other endpoint worked.
// =====================================================

import {
  AGENT_API,
  REQUIRED_API_VERSION,
  WEBSITE_VERSION,
  detectOS,
  installerFor,
  INSTALLERS,
  OS_LABELS,
} from "./config";

export const AGENT_URL = AGENT_API;
export { REQUIRED_API_VERSION, detectOS, installerFor, INSTALLERS, OS_LABELS };

export const AgentState = {
  CHECKING: "checking",
  READY: "ready",
  NOT_READY: "not_ready",
  OUTDATED: "outdated",
  MISSING: "missing",
  BLOCKED: "blocked",
};

// --------------------------------------------------------------
// LOCAL NETWORK ACCESS
//
// Since Chrome 142 (and Edge/Brave/Opera on the same engine), a request from
// a public https page to 127.0.0.1 requires a USER PERMISSION:
//
//   "edgesphere-probe.pages.dev wants to look for and connect to any device
//    on your local network."
//
// This replaced the older Private Network Access scheme, which worked through
// server headers. Nothing the agent sends can grant it any more.
//
// Two consequences this module has to handle:
//   * A denied permission fails fetch() exactly like a missing agent. The
//     permission query below is the only way to tell them apart.
//   * The prompt should be triggered by a click, not by a background poll on
//     page load, or users block a dialog they have no context for. AgentGate
//     does that.
//
// Requests to a 127.0.0.0/8 literal are also exempt from mixed-content
// blocking, which is the reason an https page may call http://127.0.0.1 at
// all. Do not move the agent to a hostname - it loses that exemption.
// --------------------------------------------------------------

/** 'granted' | 'denied' | 'prompt' | 'unsupported' */
export async function readLocalNetworkPermission() {
  try {
    if (!navigator.permissions || !navigator.permissions.query) {
      return "unsupported";
    }
    const status = await navigator.permissions.query({
      name: "local-network-access",
    });
    return status.state;
  } catch {
    // The browser does not know this permission name, so it does not enforce
    // the restriction either.
    return "unsupported";
  }
}

export function isSecurePage() {
  return typeof window !== "undefined"
    && window.location.protocol === "https:";
}

// --------------------------------------------------------------
// PROBE
// --------------------------------------------------------------
/**
 * Ask the agent who it is.
 *
 * Deliberately a "simple" request - GET, no custom headers - so the browser
 * sends no CORS preflight and a poll costs one round trip. The website
 * version travels as a query parameter for the same reason.
 */
export async function probeAgent({ timeout = 2500 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    // ------------------------------------------------------------------
    // NO targetAddressSpace ANNOTATION HERE. This is deliberate, and the
    // comment is long because the failure it causes is invisible.
    //
    // An earlier version set `options.targetAddressSpace = "local"` on
    // secure pages. But 127.0.0.1 is in the LOOPBACK address space, which
    // Chrome treats as distinct from LOCAL (private LAN ranges such as
    // 192.168.x.x). Annotating a loopback request as "local" is a MISMATCH,
    // so Chrome rejects it inside the browser - the request never goes out,
    // never reaches the agent, and never appears in the agent's log.
    //
    // The symptom is brutal to diagnose: /health silently fails while every
    // other call to the SAME origin (/check-connections, /hw-refresh,
    // /terminate) succeeds, because those are made by components that do not
    // set the option. It looks like the agent is selectively ignoring one
    // endpoint.
    //
    // No annotation is needed anyway:
    //   * Loopback literals are already exempt from mixed-content blocking.
    //   * The Local Network Access permission is granted PER REQUESTING
    //     ORIGIN, so once the user has allowed this site, the whole origin is
    //     allowed - there is nothing per-request left to declare.
    // ------------------------------------------------------------------
    const options = {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    };

    const res = await fetch(
      `${AGENT_URL}/health?wv=${encodeURIComponent(WEBSITE_VERSION)}`,
      options
    );

    if (!res.ok) {
      // 403 here means the agent is running but does not trust this origin:
      // this site is not in DEFAULT_ALLOWED_ORIGINS in security.py, which
      // means the installed agent predates this deployment.
      return {
        state: AgentState.MISSING,
        info: null,
        error: res.status === 403
          ? "origin_not_allowed"
          : `HTTP ${res.status}`,
      };
    }

    const info = await res.json();

    if ((info.api_version || 0) < REQUIRED_API_VERSION) {
      return { state: AgentState.OUTDATED, info, error: null };
    }
    if (!info.ready_to_flash) {
      return { state: AgentState.NOT_READY, info, error: null };
    }
    return { state: AgentState.READY, info, error: null };

  } catch (err) {
    const aborted = err && err.name === "AbortError";

    // FIX 2: distinguish "the browser said no" from "nothing is there".
    const permission = await readLocalNetworkPermission();
    if (permission === "denied") {
      return {
        state: AgentState.BLOCKED,
        info: null,
        error: "permission_denied",
      };
    }

    return {
      state: AgentState.MISSING,
      info: null,
      error: aborted ? "timeout" : ((err && err.message) || "unreachable"),
      permission,
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Poll until the agent appears, then stop.
 *
 * This is what makes "install and it just works" true: the user runs the
 * installer in another window and the page unlocks itself. Asking them to
 * refresh is where this flow usually feels broken.
 *
 * Returns a cancel function.
 */
export function waitForAgent(onChange, { interval = 2000 } = {}) {
  let cancelled = false;
  let timer = null;

  const tick = async () => {
    if (cancelled) return;
    const result = await probeAgent();
    if (cancelled) return;
    onChange(result);
    if (result.state === AgentState.READY) return;      // done polling
    if (result.state === AgentState.BLOCKED) return;    // polling cannot fix
    timer = setTimeout(tick, interval);
  };

  tick();

  return () => {
    cancelled = true;
    if (timer) clearTimeout(timer);
  };
}