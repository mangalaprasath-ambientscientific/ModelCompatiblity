import { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const API = "http://127.0.0.1:9262";

const POLL_MS = 2500;      // REST fallback - cheap, backend reads a cache
const WATCHDOG_MS = 7000;  // no update for this long -> force a refresh

const INITIAL = {
  jtag: 0,
  uart: null,
  chip: null,
  chip_name: null,
  chip_error: null,
  detecting: true,
  paused: false,
  ready: false,
  stale: true,
  device_name: "",
};

// The J-Link is the only link the app actually needs: audio/video samples are
// read out of target memory over the probe, and trace comes off the J-Link log
// channel. UART is a fallback, so its absence is reported, never faulted.
const buildMessage = (snap) => {
  if (snap.jtag && snap.uart) {
    return `JTag and UART connected`;
  }
  if (snap.jtag) return `JTag connected (UART not required)`;
  if (snap.uart) return `Only UART connected`;
  return "No device connected";
};

const useConnectionStatus = () => {
  const [snapshot, setSnapshot] = useState(INITIAL);

  const socketRef = useRef(null);
  const mountedRef = useRef(true);
  const lastUpdateRef = useRef(0);
  const inFlightRef = useRef(false);

  const apply = useCallback((data) => {
    if (!mountedRef.current || !data) return;
    lastUpdateRef.current = Date.now();
    setSnapshot((prev) => ({
      ...prev,
      ...data,
      // `ready` is LATCHED. It only ever means "the backend has answered us at
      // least once". A late/early payload that still carries ready:false (for
      // example a /check-connections served before the monitor's first cycle)
      // used to flip the footer back to the neutral grey "checking" colour on a
      // working setup - that is the random grey the tester reported.
      ready: prev.ready || Boolean(data.ready),
      // Once we have live data, never re-assert staleness from an older
      // payload; only an explicit stale:true from a fresh answer counts.
      stale: data.stale === undefined ? prev.stale : Boolean(data.stale),
    }));
  }, []);

  /**
   * Read the backend snapshot. `force` asks the backend to probe right now
   * instead of serving its last cycle - used on mount, on focus and when the
   * watchdog fires, so a J-Link plugged in a moment ago shows up immediately.
   */
  const fetchStatus = useCallback(async (force = false) => {
    if (inFlightRef.current) return null;
    inFlightRef.current = true;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), force ? 6000 : 4000);

    try {
      const res = await fetch(`${API}${force ? "/hw-refresh" : "/check-connections"}`, {
        method: force ? "POST" : "GET",
        signal: controller.signal,
        cache: "no-store",
      });
      const data = await res.json();
      apply(data);
      return data;
    } catch {
      // Do not paint an error state: a single missed poll is not a
      // disconnection. The next tick will correct it.
      return null;
    } finally {
      clearTimeout(timer);
      inFlightRef.current = false;
    }
  }, [apply]);

  // ---------------- socket + polling + watchdog ----------------
  useEffect(() => {
    mountedRef.current = true;

    const socket = io(API, {
      // Polling first, then upgrade - see the note in Flasher.jsx. A direct
      // websocket connection 500s on the werkzeug dev server; the upgrade from
      // an established polling session does not.
      transports: ["polling", "websocket"],
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionDelayMax: 3000,
      timeout: 8000,
    });
    socketRef.current = socket;

    socket.on("connection_status", apply);
    socket.on("connect", () => {
      socket.emit("request_hw_refresh");
      fetchStatus(false);
    });

    // First paint should not wait for a socket round trip.
    fetchStatus(true);

    const poll = setInterval(() => fetchStatus(false), POLL_MS);

    const watchdog = setInterval(() => {
      if (Date.now() - lastUpdateRef.current > WATCHDOG_MS) {
        // Socket has gone quiet (Electron refresh, backend restart, sleep).
        fetchStatus(true);
        if (socketRef.current && !socketRef.current.connected) {
          socketRef.current.connect();
        }
      }
    }, 2500);

    const onWake = () => {
      if (document.visibilityState !== "hidden") fetchStatus(true);
    };
    window.addEventListener("focus", onWake);
    window.addEventListener("online", onWake);
    document.addEventListener("visibilitychange", onWake);

    return () => {
      mountedRef.current = false;
      clearInterval(poll);
      clearInterval(watchdog);
      window.removeEventListener("focus", onWake);
      window.removeEventListener("online", onWake);
      document.removeEventListener("visibilitychange", onWake);
      socket.off("connection_status", apply);
      socket.disconnect();
      socketRef.current = null;
    };
  }, [apply, fetchStatus]);
 
  const refreshConnection = useCallback(() => {
    if (socketRef.current) socketRef.current.emit("request_hw_refresh");
    return fetchStatus(true);
  }, [fetchStatus]);

  const connections = {
    jtag: Boolean(snapshot.jtag),
    uart: Boolean(snapshot.uart),
  };

  const statusMessage = buildMessage(snapshot);
  // Green on the J-Link alone. Requiring UART here is what made a perfectly
  // healthy probe-only rig read as disconnected.
  const connectionColor =
    !snapshot.ready ? "gray" : snapshot.jtag ? "green" : "red";

  return {
    // shape the old Footer expects
    deviceName: snapshot.device_name || "",
    connections,
    statusMessage,
    connectionColor,
    refreshConnection,
    checkConnectionsManually: refreshConnection, // legacy alias
    // extras
    chip: snapshot.chip,
    chipName: snapshot.chip_name,
    chipError: snapshot.chip_error,
    chipMessage: snapshot.chip_message || null,
    detecting: snapshot.detecting,
    ready: snapshot.ready,
    // Staleness is informational only now (tooltip / debugging). The footer no
    // longer tints a connected box with it, which is what produced the
    // "dark green" boxes on healthy hardware.
    stale: snapshot.ready ? Boolean(snapshot.stale) : false,
    paused: snapshot.paused,
    snapshot,
  };
};

export default useConnectionStatus;