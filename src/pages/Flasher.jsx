import React, { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import "../styles/Flasher.css";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper";
import { io } from "socket.io-client";
import CustomDropdown from "../components/CustomDropdown";
import { API } from "../config";

const CHIP_ERROR_MESSAGES = {
  NO_JLINK: "J-Link not connected. Please check and try again",
  JLINK_CONNECT_FAIL: "Cannot connect to J-Link. Please reconnect and try again.",
  JLINK_BUSY: "J-Link is busy. Please wait a moment.",
  PIN15_RESET: "Chip not detected. Please check the connection between J-Link and target.",
  JTAG_CHAIN_FAIL: "JTAG chain failure detected.",
  TARGET_CONNECT_FAIL: "Target not detected.Please check and try again",
  OUT_OF_SYNC: "Error occurred due to out of sync. Please check the J-Link cable and target connection, then try again.",
  NO_FIRMWARE: "No firmware / chip not detected",
};

// Short forms for the narrow "Detected chip" pill.
const CHIP_ERROR_LABELS = {
  NO_JLINK: "No chip",
  JLINK_CONNECT_FAIL: "J-Link error",
  JLINK_BUSY: "Busy",
  PIN15_RESET: "No chip",
  JTAG_CHAIN_FAIL: "JTAG error",
  TARGET_CONNECT_FAIL: "No target",
  OUT_OF_SYNC: "Out of sync",
  NO_FIRMWARE: "No chip",
};

// A target fault that persists is re-announced this often, so the notice does
// not disappear for the rest of the session after a single toast.
const FAULT_TOAST_REPEAT_MS = 8000;
// Not a target fault: somebody else legitimately owns the J-Link right now.
const SILENT_FAULTS = new Set(["JLINK_BUSY"]);

const faultText = (snap) =>
  snap?.chip_message ||
  CHIP_ERROR_MESSAGES[snap?.chip_error] ||
  "Chip not detected. Please check the connection and try again.";

const DEC_ADDR_RE = /^[0-9]{1,10}$/;
const validateBinLayout = (bins) => {
  const imgs = [];

  for (let i = 0; i < bins.length; i++) {
    const label = bins.length === 1 ? "BIN" : `BIN ${i + 1}`;

    const s = (bins[i].address || "").trim();

    if (!DEC_ADDR_RE.test(s))
      return `${label}: "${bins[i].address}" is not a valid address (enter a number).`;

    const addr = parseInt(s, 10);
    const end = addr + bins[i].file.size - 1;
    imgs.push({ label, addr, end });
  }

  imgs.sort((a, b) => a.addr - b.addr);

  for (let i = 1; i < imgs.length; i++) {
    if (imgs[i].addr <= imgs[i - 1].end) {
      return `${imgs[i].label} overlaps ${imgs[i - 1].label}. Please check the address.`;
    }
  }

  return null;
};

// =====================================================
// FLASH STAGES  (drives the left-hand device graphic)
// -----------------------------------------------------
// One entry per real device event, in the order the flash reports them. Each
// stage owns exactly ONE signature animation on the board graphic — see the
// `st-*` rules in section 14 of Flasher.css — plus the caption printed under
// the board, so every status update the user sees on the right has a matching
// physical change on the left.
//
//   key         reported by                        board shows
//   prep        flash start (5%)                   probe wakes, amber standby
//   checked     /upload-and-convert (15%)          probe face verify tick
//   built       /make-action validate (35%)         ribbon energises, lane by lane
//   linked      /start-debug-session (45%)          JTAG handshake on the header
//   verified    code 100 (48%)                     current reaches the chip
//   ready       code 110 (52%)                     memory array powers up
//   ident       code 810 (56%)                     ID read sweep across the die
//   checkedmem  code 210 (58%)                     erase wash clears the die
//   writing     code 310 (65%)                     programming: fill + write head
//   firstboot   code 410 (72%)                     bootloader sector latches
//   header      code 510 (95%)                     header sector latches
//   application code 610 (99%)                     application sector latches
//   sealed      code 710 (100%)                    link released, target boots
//   fault       any failure                        red hold on the failing part
// =====================================================
const FLASH_STAGES = [
  "idle",
  "prep",
  "checked",
  "built",
  "linked",
  "verified",
  "ready",
  "ident",
  "checkedmem",
  "writing",
  "firstboot",
  "header",
  "application",
  "sealed",
];

// Caption under the board. Present tense while the device is working on it,
// past tense once the step is a reported result — it reads as a status line,
// not a label.
const STAGE_CAPTIONS = {
  idle: "Standby",
  prep: "Preparing firmware",
  checked: "Firmware verified",
  built: "Build complete",
  linked: "JTAG link established",
  verified: "Target communication verified",
  ready: "Flash memory initialised",
  ident: "Flash ID verified",
  checkedmem: "Write / read check passed",
  writing: "Programming flash",
  firstboot: "Firstboot image written",
  header: "Application header written",
  application: "Application image written",
  sealed: "Flash complete",
  fault: "Operation halted",
};

// Human-readable size for the picked firmware file.
const formatSize = (bytes) => {
  if (typeof bytes !== "number") return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

// =====================================================
// MAIN COMPONENT: Flasher
// =====================================================
function Flasher() {

  // =====================================================
  // REFS
  // =====================================================
  const binTableRef = useRef(null);
  const [binScroll, setBinScroll] = useState({ top: true, bottom: true });

  const logRef = useRef(null);
  const TimerRef = useRef(null);
  const idleTimerRef = useRef(null);
  const fileInputRef = useRef(null);
  const terminalRef = useRef(null);
  const socketRef = useRef(null);
  const stage56TimerRef = useRef(null);
  const stage52TimerRef = useRef(null);
  const stage48TimerRef = useRef(null);
  // Watchdog for the pre-flash hold: progress idles up to 52% while we wait for
  // the device to report the first flash status code. If the device is stuck,
  // the backend emits nothing, so this timer is what turns the hang into an error.
  const flashStartTimerRef = useRef(null);
  // Holds the reset until the progress screen has finished sliding out.
  const leaveTimerRef = useRef(null);
  // Mirrors `success` for the watchdog callbacks: they close over the state
  // value from the render they were created in, so reading `success` inside
  // them would always see the stale false.
  const successRef = useRef(false);

  // Latest hardware snapshot from the backend. Kept in a ref as well as state
  // so the Continue handler always reads the current value, never a closure
  // captured at render time.
  const hwRef = useRef({ jtag: 0, uart: null, chip: null, chip_name: null, ready: false });
  // True whenever a flash is in flight: the detected chip must stay pinned to
  // whatever we started with, so live snapshots must not overwrite it.
  const busyRef = useRef(false);
  // Last fault we announced + when, so a fault that is still present keeps
  // re-announcing instead of latching silent after the first toast.
  const faultToastRef = useRef({ code: null, at: 0 });
  const prevChipRef = useRef(null);
  // Mirrors `chipDetected` for applyHwSnapshot, which is a [] useCallback and
  // so can never read the state itself.
  const chipKnownRef = useRef(false);

  // Snapshot ORDERING. Four things read hardware status - the 3 s poll, the
  // socket push, the focus/visibility probe and the Continue press - and none
  // of them were ordered against each other. /check-connections returns a
  // cached snapshot in milliseconds while /hw-refresh forces a real probe, so
  // a poll fired before a probe routinely answers after it, and whichever
  // reply landed last won. That is how the UI ended up showing "Detecting..."
  // for a chip a newer snapshot had already identified.
  //
  // Every read takes a ticket; a reply is applied only if no newer snapshot
  // has been applied while it was in flight.
  const hwReqSeqRef = useRef(0);
  const hwAppliedSeqRef = useRef(0);

  // =====================================================
  // STATE
  // =====================================================
  const [currentPage, setCurrentPage] = useState(1);
  const [isDragging, setIsDragging] = useState(false);      // 1 = type/board/chip, 2 = interface/upload
  const [flasherType, setFlasherType] = useState("single");   // "single" | "multi"
  const [checkingNext, setCheckingNext] = useState(false);
  const [selectedBoard, setSelectedBoard] = useState("CRT");
  const [selectedInterface, setSelectedInterface] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);     // single-bin file
  const [flashStarted, setFlashStarted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [success, setSuccess] = useState(false);
  const [errorStopped, setErrorStopped] = useState(false);
  const [logs, setLogs] = useState([]);
  const [selectedChip, setSelectedChip] = useState(null);
  const [chipDetected, setChipDetected] = useState(false);
  const [dragIndex, setDragIndex] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [failedStep, setFailedStep] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isDetectingChip, setIsDetectingChip] = useState(true);
  const [terminalOutput, setTerminalOutput] = useState([]);
  const [currentStep, setCurrentStep] = useState("idle");
  // Fine-grained flash stage — one entry per REAL device event, set only where
  // the device actually reports something. `progress` cannot drive this: the
  // idle creepers walk the number between events (45->52, 52->65), so a
  // percentage threshold would fire stages the device never reported.
  // This is what the left-hand device graphic animates off. See FLASH_STAGES.
  const [stage, setStage] = useState("idle");
  const [jlinkPresent, setJlinkPresent] = useState(true);
  // Current hardware fault, for display only.
  const [hwFault, setHwFault] = useState({ code: null, message: "" });

  // True for the length of the Finish / Try again exit animation: the flash
  // state is still set, but the config screen has already been asked to come
  // back. See LEAVE_MS and handleLeaveProgress.
  const [leavingProgress, setLeavingProgress] = useState(false);

  // multi-bin files (index 0 = FirstBoot)
  const [binFiles, setBinFiles] = useState([{ file: null, address: "" }]);

  const showInterface = selectedBoard === "CRT" && selectedChip === "GPX 10 Pro";
  const steps = ["build", "connect", "flashing", "done"];

  // multi-bin limits / sizes
  const maxBins =
    selectedChip === "GPX 10" || selectedChip === "GPX 10 Pro" ? 5 : 0;

  const getTotalBinSize = (files) =>
    files.reduce((total, item) => total + (item.file ? item.file.size : 0), 0);

  const getMaxAllowedSize = () => {
    if (selectedChip === "GPX 10") return 246 * 1024;          // 246 KB
    if (selectedChip === "GPX 10 Pro") return 1.9 * 1024 * 1024; // 1.9 MB
    return 0;
  };
  // Not a target fault: somebody else legitimately owns the J-Link right now.
  // Faults serious enough to announce on their own, without waiting for a button
  // press, and to keep re-announcing while they persist.
  const LOUD_FAULTS = new Set(["PIN15_RESET", "OUT_OF_SYNC"]);
  const applyHwSnapshot = useCallback((snap) => {
    if (!snap || typeof snap !== "object") return;

    hwRef.current = snap;
    setJlinkPresent(Boolean(snap.jtag));

    // Unplugging re-arms the fault notice, so replugging a still-faulty target
    // reports it again from scratch.
    if (!snap.jtag) faultToastRef.current = { code: null, at: 0 };

    // Mid-flash: display only, never touch the chip selection.
    if (busyRef.current) return;

    if (snap.chip_name) {
      faultToastRef.current = { code: null, at: 0 };
      setSelectedChip((prev) => (prev === snap.chip_name ? prev : snap.chip_name));
      setChipDetected(true);
      chipKnownRef.current = true;
      setIsDetectingChip(false);
      setHwFault({ code: null, message: "" });
      return;
    }

    // A fault the backend has already diagnosed is not "detecting". The old
    // `snap.detecting` check alone returned early on the retry snapshots the
    // monitor emits while the chip stays unknown, which is what hid the fault
    // handling below on most ticks.
    if (!snap.ready || (snap.detecting && !snap.chip_error)) {
      // A re-probe of a target we have ALREADY identified is not news. Flipping
      // the readout back to "Detecting..." on those ticks is what made a chip
      // that was confirmed seconds ago look forgotten - and what made a
      // perfectly legitimate Continue look like it had advanced mid-detection.
      // Hold the known chip on screen until the probe actually resolves, to
      // this same chip or to a fault (the fault path below clears it).
      //
      // Only while the J-Link is still attached: with the probe unplugged, a
      // stale chip name on screen would be a lie, so that falls through.
      if (snap.ready && snap.jtag && chipKnownRef.current) return;
      setIsDetectingChip(true);
      return;
    }

    // No chip. Keep `selectedChip` as-is so a momentary drop cannot wipe the
    // files the user already picked; `chipDetected` is what gates flashing.
    setChipDetected(false);
    chipKnownRef.current = false;
    setIsDetectingChip(false);
    setHwFault({
      code: snap.chip_error || null,
      message: faultText(snap),
    });

    // Only LOUD_FAULTS announce themselves. PIN15_RESET is a physical wiring
    // problem the user has to act on, so it keeps coming back every
    // FAULT_TOAST_REPEAT_MS while it persists. Every other fault is display
    // only here and is named by handleContinue on the button press instead.
    const code = snap.chip_error;
    if (snap.jtag && code && LOUD_FAULTS.has(code)) {
      const now = Date.now();
      const seen = faultToastRef.current;
      if (seen.code !== code || now - seen.at >= FAULT_TOAST_REPEAT_MS) {
        faultToastRef.current = { code, at: now };
        toast.error(faultText(snap), { toastId: `HW_${code}` });
      }
    }
  }, []);

  /** Read the backend snapshot. `force` makes the backend probe immediately. */
  const fetchHwSnapshot = useCallback(async (force = false, timeout = 6000) => {
    const seq = ++hwReqSeqRef.current;
    try {
      const res = force
        ? await axios.post(`${API}/hw-refresh`, {}, { timeout })
        : await axios.get(`${API}/check-connections`, { timeout });

      // Superseded while in flight. The caller still gets what it asked for -
      // it may be the Continue press, which needs an answer - but this reply
      // is older than what is already on screen, so it must not be applied.
      if (seq < hwAppliedSeqRef.current) return res.data;

      hwAppliedSeqRef.current = seq;
      applyHwSnapshot(res.data);
      return res.data;
    } catch {
      return null;
    }
  }, [applyHwSnapshot]);

  /** A snapshot the backend pushed at us (socket, or an endpoint that answers
   *  with one). It is newer than anything still in flight over REST by
   *  definition, so it claims the current ticket and those replies are
   *  dropped when they land. */
  const applyPushedSnapshot = useCallback((snap) => {
    hwAppliedSeqRef.current = hwReqSeqRef.current;
    applyHwSnapshot(snap);
  }, [applyHwSnapshot]);

  // =====================================================
  // PROGRESS HANDLERS
  // =====================================================
  const startIdleProgress = (cap) => {
    clearInterval(idleTimerRef.current);
    idleTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= cap) {
          clearInterval(idleTimerRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, 3000);
  };

  const startStagedIdleProgress = (start, cap) => {
    clearInterval(idleTimerRef.current);
    setProgress(start);
    idleTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= cap) {
          clearInterval(idleTimerRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, 3000);
  };

  const stopIdleProgress = () => {
    clearInterval(idleTimerRef.current);
  };

  const smoothSetProgress = (target) => {
    stopIdleProgress();
    clearInterval(TimerRef.current);
    TimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= target) {
          clearInterval(TimerRef.current);
          return target;
        }
        return prev + (target - prev > 10 ? 3 : 1);
      });
    }, 120);
  };

  // =====================================================
  // RESET
  // =====================================================
  // Must outlast the .fl-screen cross-fade so the progress panel is invisible
  // before its data is wiped: 0.3s transition + the 0.06s incoming delay that
  // .fl-work.is-leaving adds, rounded up.
  const LEAVE_MS = 400;

  const resetFlasher = () => {
    clearTimeout(leaveTimerRef.current);
    setLeavingProgress(false);
    clearInterval(TimerRef.current);
    stopIdleProgress();
    clearTimeout(stage56TimerRef.current);
    clearTimeout(stage48TimerRef.current);
    clearTimeout(stage52TimerRef.current);
    clearTimeout(flashStartTimerRef.current);
    setFailedStep(null);
    setSuccess(false);
    successRef.current = false;
    setErrorStopped(false);
    setFlashStarted(false);
    setProgress(0);
    setLogs([]);
    setCurrentStep("idle");
    setStage("idle");
    setErrorMessage("");
    setSelectedFile(null);
    setSelectedInterface(null);
    setSelectedChip(null);
    setChipDetected(false);
    chipKnownRef.current = false;
    setIsDetectingChip(true);
    setCurrentPage(1);
    setBinFiles([{ file: null, address: "" }]);
    setHwFault({ code: null, message: "" });
    prevChipRef.current = null;
    faultToastRef.current = { code: null, at: 0 };
    busyRef.current = false;
    window.__FLASHER_RUNNING__ = false;
    socketRef.current?.emit("leave_upgrade_page");
    socketRef.current?.emit("request_hw_refresh");
    setTerminalOutput([]);

    // Ask the backend to re-probe now instead of waiting for the next tick,
    // so returning to page 1 after a flash shows live status immediately.
    fetchHwSnapshot(true).catch(() => { });
  };

  // Finish / Try again. Calling resetFlasher straight from the button made the
  // progress panel disappear in a single frame - it is conditionally rendered,
  // so clearing success / errorStopped unmounts it before its exit transition
  // can run, and the config screen faded in over nothing.
  //
  // So the exit is split in two. Now: flip the screens, but keep every piece of
  // flash state, which is what leaves the panel mounted and its content frozen
  // while it slides out. LEAVE_MS later: the actual reset, by which point the
  // panel is invisible and nothing about wiping it can be seen.
  const handleLeaveProgress = () => {
    if (leavingProgress) return;          // double click during the animation
    setLeavingProgress(true);
    // Rewind the wizard NOW rather than in the reset, so it happens while the
    // config screen is still fully transparent. Left until later, the user
    // watched the panel arrive on step 2 and then slide itself back to step 1.
    // .fl-work.is-leaving suppresses the page transition to keep it a snap.
    setCurrentPage(1);
    clearTimeout(leaveTimerRef.current);
    leaveTimerRef.current = setTimeout(resetFlasher, LEAVE_MS);
  };

  // =====================================================
  // EFFECTS
  // =====================================================
  // Expose flasher running state to Navbar
  useEffect(() => {
    window.__FLASHER_RUNNING__ = flashStarted;
    return () => { window.__FLASHER_RUNNING__ = false; };
  }, [flashStarted]);

  // Keep the "a flash owns the chip" flag in sync for applyHwSnapshot.
  useEffect(() => {
    busyRef.current = isStarting || flashStarted || success || errorStopped;
  }, [isStarting, flashStarted, success, errorStopped]);

  // When the chip actually CHANGES to a different part, reset the multi-bin
  // layout. A transient loss (chip goes null) must not wipe the user's files,
  // which is what the old `[selectedChip]` effect did on every hiccup.
  useEffect(() => {
    if (!selectedChip) return;
    const prev = prevChipRef.current;
    if (prev === selectedChip) return;
    prevChipRef.current = selectedChip;

    // First identification is not a CHANGE. Nothing can need invalidating -
    // no files can have been chosen while no chip was known - and rewinding
    // here undid the page the Continue press had just moved to whenever that
    // press was itself what identified the chip: the wizard flicked to step 2
    // and was pulled straight back to step 1. Only a swap to a DIFFERENT part
    // invalidates the addresses below.
    if (!prev) return;

    setBinFiles([{ file: null, address: "" }]);
    setCurrentPage(1);
  }, [selectedChip]);

  // Mount reset. The cleanup drops the pending exit timer: leaving the page
  // mid-animation would otherwise fire resetFlasher into an unmounted tree.
  useEffect(() => {
    resetFlasher();
    return () => clearTimeout(leaveTimerRef.current);
  }, []);

  // On (re)load, force-stop any flash session left running by a previous page
  // or a mid-flash refresh. A native/Electron refresh (Ctrl+R) never hits
  // /terminate, so without this the backend stays BUSY and holds the J-Link.
  // The response carries a fresh hardware snapshot, so status comes back at
  // once after a refresh instead of staying red.
  useEffect(() => {
    axios.post(`${API}/abort-flash`)
      .then((res) => { if (res?.data?.hw) applyPushedSnapshot(res.data.hw); })
      .catch(() => { })
      .finally(() => { fetchHwSnapshot(true).catch(() => { }); });
  }, [applyPushedSnapshot, fetchHwSnapshot]);
  // Browser teardown: Electron used to call /terminate before a refresh or
  // close, and a browser does not. A normal fetch is cancelled during unload,
  // so sendBeacon is the only reliable way to release the J-Link when the user
  // hits Ctrl+R or closes the tab mid-flash.
  useEffect(() => {
    const releaseHardware = () => {
      try {
        navigator.sendBeacon(`${API}/abort-flash`, new Blob([], { type: "text/plain" }));
      } catch { /* teardown is best-effort */ }
    };
    window.addEventListener("pagehide", releaseHardware);
    return () => window.removeEventListener("pagehide", releaseHardware);
  }, []);
  // recompute fade state on scroll / row count change
  const updateBinScroll = () => {
    const el = binTableRef.current;
    if (!el) return;
    const canScroll = el.scrollHeight > el.clientHeight + 1;
    setBinScroll({
      top: !canScroll || el.scrollTop <= 2,
      bottom: !canScroll || el.scrollTop + el.clientHeight >= el.scrollHeight - 2,
    });
  };

  useEffect(() => {
    const el = binTableRef.current;
    if (el) el.scrollTop = el.scrollHeight;   // keep newest row in view
    updateBinScroll();
  }, [binFiles.length]);

  // External reset event
  useEffect(() => {
    const handler = () => resetFlasher();
    window.addEventListener("flasher-reset", handler);
    return () => window.removeEventListener("flasher-reset", handler);
  }, []);

  // Socket setup — also the transport for live hardware status
  useEffect(() => {
    const socket = io(API, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionDelayMax: 3000,
    });
    socketRef.current = socket;

    socket.on("uart_output", (data) => {
      if (!data?.line) return;
      // COM output only. The backend pushes the drained J-Link flash trace
      // over this same event tagged source: "jlink" (emit_console_lines in
      // app.py), which is what was filling this panel with the flash log.
      // Only the serial reader tags its lines "uart".
      if (data.source !== "uart") return;
      setTerminalOutput(prev => [...prev, data.line]);
    });

    socket.on("connection_status", applyPushedSnapshot);

    socket.on("connect", () => {
      socket.emit("request_hw_refresh");
    });

    return () => {
      socket.off("connection_status", applyPushedSnapshot);
      socket.emit("leave_upgrade_page");
      socket.disconnect();
      socketRef.current = null;
    };
  }, [applyPushedSnapshot]);

  // REST fallback for hardware status. Cheap (the backend serves a cached
  // snapshot), and it self-heals if the socket ever goes quiet.
  useEffect(() => {
    let cancelled = false;

    const tick = () => { if (!cancelled) fetchHwSnapshot(false); };
    const interval = setInterval(tick, 3000);

    const onWake = () => {
      if (!cancelled && document.visibilityState !== "hidden") fetchHwSnapshot(true);
    };
    window.addEventListener("focus", onWake);
    document.addEventListener("visibilitychange", onWake);

    return () => {
      cancelled = true;
      clearInterval(interval);
      window.removeEventListener("focus", onWake);
      document.removeEventListener("visibilitychange", onWake);
    };
  }, [fetchHwSnapshot]);

  // Terminal auto scroll
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalOutput]);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (window.evtSource) {
        window.evtSource.close();
        window.evtSource = null;
      }
    };
  }, []);

  // Log auto scroll
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  // =====================================================
  // FILE UPLOAD HANDLER (single-bin)
  // =====================================================
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file || !file.name.toLowerCase().endsWith(".bin")) {
      toast.error("Only .bin files are allowed!");
      e.target.value = null;
      return;
    }
    setSelectedFile(file);
    setSuccess(false);
    setErrorStopped(false);
    setFlashStarted(false);
    setProgress(0);
    setStage("idle");
    setLogs([]);
    e.target.value = null;
  };

  // =====================================================
  // MULTI-BIN HANDLERS
  // =====================================================
  const handleMultiBinChange = (index, file, event) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".bin")) {
      toast.error("Only .bin files are allowed!");
      event.target.value = null;
      return;
    }

    const updated = [...binFiles];
    const previousFile = updated[index].file;   // keep old in case of size overflow
    updated[index].file = file;

    const totalSize = getTotalBinSize(updated);
    const maxAllowed = getMaxAllowedSize();

    if (totalSize > maxAllowed) {
      updated[index].file = previousFile;        // restore
      const limitText = selectedChip === "GPX 10" ? "246 KB" : "1.9 MB";
      toast.error(`Total BIN size exceeded ${limitText}`);
      event.target.value = null;
      return;
    }

    setBinFiles(updated);
    setSuccess(false);
    setErrorStopped(false);
    setFlashStarted(false);
    setProgress(0);
    setStage("idle");
    setLogs([]);
    event.target.value = null;
  };

  const handleAddressChange = (index, value) => {
    const clean = value
      .replace(/[^0-9]/g, "")
      .slice(0, 10);

    const updated = [...binFiles];
    updated[index].address = clean;
    setBinFiles(updated);
  };

  const addAnotherBin = () => {
    if (binFiles.length >= maxBins) return;
    const defaultAddress = binFiles.length === 1 ? "" : "";
    setBinFiles([...binFiles, { file: null, address: defaultAddress }]);
  };

  const removeBin = (index) => {
    const updated = [...binFiles];
    updated.splice(index, 1);
    setBinFiles(updated.length ? updated : [{ file: null, address: "" }]);
  };

  // =====================================================
  // ERROR HANDLER
  // =====================================================
  const handleFlashError = (msg, step = "flashing") => {
    // A stage watchdog can fire after 710 if its clearing code was lost;
    // never paint an error over a completed flash.
    if (successRef.current) return;
    clearInterval(TimerRef.current);
    stopIdleProgress();
    clearTimeout(stage56TimerRef.current);
    clearTimeout(stage52TimerRef.current);
    clearTimeout(stage48TimerRef.current);
    clearTimeout(flashStartTimerRef.current);
    setFailedStep(step);
    setStage("fault");
    toast.error(msg);
    setErrorMessage(msg);
    setLogs(prev => [...prev, { message: msg, success: false }]);
    setErrorStopped(true);
    setProgress(0);
    setFlashStarted(false);
    if (window.evtSource) {
      window.evtSource.close();
      window.evtSource = null;
    }
    // Hand the hardware back to the status monitor right away.
    fetchHwSnapshot(true).catch(() => { });
  };

  // =====================================================
  // MULTI FLASH ENTRY — validate, then run the normal flash flow
  // =====================================================
  const startMultiFlashing = async () => {
    // Every bin needs a file + address (no FirstBoot anymore)
    for (let i = 0; i < binFiles.length; i++) {
      const binLabel = binFiles.length === 1 ? "BIN" : `BIN ${i + 1}`;

      if (!binFiles[i]?.file) {
        toast.error(`Please select ${binLabel} file`);
        return;
      }
      if (!binFiles[i]?.address) {
        toast.error(`Please enter ${binLabel} address`);
        return;
      }
    }
    const layoutError = validateBinLayout(binFiles);

    if (layoutError) {
      toast.error(layoutError);
      return;
    }
    // validation passed -> run the shared flashing flow
    startFlashing();
  };

  // =====================================================
  // FLASHING  (shared by single + multi)
  // =====================================================
  const startFlashing = async () => {
    let detectedChip = null;

    // active firmware file = single file OR firstboot of the multi list
    const activeFile = flasherType === "multi" ? binFiles[0]?.file : selectedFile;

    if (!activeFile) {
      toast.warning("Please select a .bin file before flashing!");
      return;
    }

    try {
      setIsStarting(true);
      busyRef.current = true;

      // ===== STOP ANY PREVIOUS FLASH SESSION =====
      // Enforce "only one flashing at a time": if a flash from an earlier
      // (possibly refreshed) attempt is still running on the backend, end it
      // now — on whatever stage it's at — so this new flash can take over.
      // Best-effort: proceed even if the abort call fails.
      try {
        await axios.post(`${API}/abort-flash`, {}, { timeout: 15000 });
      } catch {
        /* ignore — abort is best-effort */
      }

      // ===== JLINK CHECK + CHIP IDENTIFICATION =====
      const pre = await axios.post(`${API}/pre-flash-check`, {}, { timeout: 30000 });
      if (!pre.data.success) {
        toast.error(pre.data.message);
        setIsStarting(false);
        busyRef.current = false;
        return;
      }

      detectedChip =
        pre.data.chip_name || (pre.data.chip === 1 ? "GPX 10 Pro" : "GPX 10");
      setSelectedChip(detectedChip);
      prevChipRef.current = detectedChip;
      setChipDetected(true);
      chipKnownRef.current = true;
      setIsDetectingChip(false);

      // ===== INTERFACE VALIDATION =====
      if (selectedBoard === "CRT" && detectedChip === "GPX 10 Pro" && !selectedInterface) {
        toast.warning("Please select interface before flashing");
        setIsStarting(false);
        busyRef.current = false;
        return;
      }

      setFlashStarted(true);
      setCurrentStep("build");
      setStage("prep");
      setProgress(5);
      setLogs([]);
      setErrorStopped(false);
      setSuccess(false);
      successRef.current = false;
      startIdleProgress(10);
      smoothSetProgress(10);
      setIsStarting(false);

    } catch (err) {
      toast.error(
        err?.code === "ECONNABORTED"
          ? "Hardware communication error. Please try again."
          : "Could not reach server"
      );
      setIsStarting(false);
      busyRef.current = false;
      return;
    }

    // ===== STEP 1: BIN UPLOAD =====
    try {
      const chipMap = { "GPX 10": 0, "GPX 10 Pro": 1 };
      const boardMap = { "CRT": 1, "Sparsh": 0 };
      const interfaceMap = { "SPI0": 3, "QPI": 2, "SPI1": 0 };

      // number of bins sent (single = 1, multi = count of selected files)
      const noBinFile =
        flasherType === "multi" ? binFiles.filter(b => b.file).length : 1;

      // Files to flash: single = one file, multi = every selected bin.
      // Backend expects binFile_1 .. binFile_N and emits write_application_1..N.
      const filesToSend =
        flasherType === "multi"
          ? binFiles.filter(b => b.file).map(b => b.file)
          : [selectedFile];

      const formData = new FormData();
      filesToSend.forEach((file, i) => {
        formData.append(`binFile_${i + 1}`, file);
      });
      formData.append("user_selected_chip", chipMap[detectedChip] ?? 0);
      formData.append("user_selected_board", boardMap[selectedBoard] ?? 0);
      formData.append("user_selected_interface", interfaceMap[selectedInterface] ?? 0);
      formData.append("multi_bin_page", flasherType === "multi" ? 1 : 0);
      formData.append("no_bin_file", noBinFile);

      // multi-bin addresses (addr_2..addr_5)
      if (flasherType === "multi") {
        const rows = binFiles.filter((b) => b.file);

        rows.forEach((b, i) => {
          formData.append(`addr_${i + 1}`, b.address.trim());
        });
      }

      const response = await axios.post(`${API}/upload-and-convert`, formData);

      setLogs(prev => [
        ...prev,
        { message: `Firmware file checked and verified (${response.data.file_size_kb} KB)`, success: true }
      ]);
      setCurrentStep("build");
      setStage("checked");
      smoothSetProgress(15);

    } catch (err) {
      handleFlashError(err?.response?.data?.message || "BIN upload failed", "build");
      return;
    }

    // ===== STEP 2: BUILD =====
    try {
      const mappedChip = { "GPX 10": 0, "GPX 10 Pro": 1 }[detectedChip] ?? 0;
      await axios.post(`${API}/make-action`, { action: "clean", user_selected_chip: mappedChip });
      await axios.post(`${API}/make-action`, { action: "build", user_selected_chip: mappedChip });
      await axios.post(`${API}/make-action`, { action: "validate", user_selected_chip: mappedChip });

      setLogs(prev => [...prev, { message: "Flash firmware build completed successfully.", success: true }]);
      setCurrentStep("connect");
      setStage("built");
      smoothSetProgress(35);

    } catch (error) {
      const errorType = error?.response?.data?.error_type;
      const errorMsg = error?.response?.data?.message;
      const overflowBytes = error?.response?.data?.overflow_bytes;

      if (errorType === "ram_overflow") {
        handleFlashError(`RAM overflow detected. RAM limit exceeded by ${overflowBytes} bytes`, "build");
      } else {
        handleFlashError(errorMsg || "Firmware build failed. Please try again.", "build");
      }
      return;
    }

    // ===== STEP 3: GDB + JLINK =====
    try {
      await axios.post(`${API}/start-gdb-server`);
      await axios.post(`${API}/start-debug-session`);
      setLogs(prev => [...prev, { message: "Device connection established (JTAG interface).", success: true }]);
      setCurrentStep("flashing");
      setStage("linked");
      smoothSetProgress(45);
      startStagedIdleProgress(45, 52);

    } catch {
      handleFlashError("J-Link communication lost. Please check the device connection and try again.", "connect");
      return;
    }

    // ===== STEP 4: FLASH MONITOR =====
    if (window.evtSource) window.evtSource.close();

    const evtSource = new EventSource(`${API}/flashing-status`);
    window.evtSource = evtSource;

    // Pre-flash watchdog: from the moment the monitor opens, the device must
    // report its first real flash status code within 70s. Until then progress
    // just idles up to 52% and holds. If nothing arrives (device stuck / backend
    // silently looping), fail with a hardware error instead of hanging forever.
    clearTimeout(flashStartTimerRef.current);
    flashStartTimerRef.current = setTimeout(() => {
      handleFlashError(
        "Hardware connection interrupted. Please check and try again.",
        "flashing"
      );
      if (window.evtSource) {
        window.evtSource.close();
        window.evtSource = null;
      }
    }, 70000);

    evtSource.onmessage = function (e) {
      let data;
      try { data = JSON.parse(e.data); } catch { return; }
      const { code } = data;

      const FAIL_CODES = {
        201: "Flash memory read/write verification failed. Please retry the operation.",
        401: "Unable to write the Firstboot firmware. Please try again.",
        501: "Failed to write the Application header. Please try again.",
        601: "Application write operation failed. Please try again.",
        801: "Flash memory identification failed. Please verify the hardware and try again.",
        901: "Incorrect flash interface selected. Please select the correct interface and try again.",
        TIMEOUT: "Firmware flashing timed out. Please reconnect the device and try again.",
        HW_LOST: "Hardware connection interrupted. Please reconnect and try again.",
        CONNECT_FAIL: "Could not connect to the device. Please reconnect and try again.",
        BUSY: "A flash session is already running. Please wait or refresh.",
      };

      // PIN15 + OUT_OF_SYNC are connect-step failures; the rest are flashing-step
      if (code === "PIN15_RESET") {
        handleFlashError("Pin 15 error detected. Cannot connect to target.", "connect");
        evtSource.close();
        window.evtSource = null;
        return;
      }
      if (code === "OUT_OF_SYNC") {
        handleFlashError(
          "Error occurred due to out of sync. Please check the J-Link cable and target connection, then try again.",
          "connect"
        );
        evtSource.close();
        window.evtSource = null;
        return;
      }
      if (code in FAIL_CODES) {
        const step = code === "CONNECT_FAIL" ? "connect" : "flashing";
        handleFlashError(FAIL_CODES[code], step);
        evtSource.close();
        window.evtSource = null;
        return;
      }

      const updateLogOnce = (text) => {
        setLogs(prev => {
          if (prev[prev.length - 1]?.message === text) return prev;
          return [...prev, { message: text, success: true }];
        });
      };

      // The device reported real progress -> it's alive, so cancel the pre-flash
      // watchdog. (INFO/ACK heartbeat codes are NOT in this list, so a stream of
      // heartbeats with no real progress still times out.)
      const PROGRESS_CODES = [100, 110, 810, 210, 310, 410, 510, 610, 710];
      if (PROGRESS_CODES.includes(code)) {
        clearTimeout(flashStartTimerRef.current);
      }

      switch (code) {
        case 100: {
          const ifaceLabel = selectedInterface || "SPI0";
          updateLogOnce(`Target Device communication verified (${ifaceLabel}).`);
          setStage("verified");
          smoothSetProgress(48);
          clearTimeout(stage48TimerRef.current);

          stage48TimerRef.current = setTimeout(() => {
            console.log("48 watchdog fired");
            handleFlashError(
              "Hardware connection interrupted. Please check and try again.",
              "flashing"
            );
            if (window.evtSource) {
              window.evtSource.close();
              window.evtSource = null;
            }
          }, 70000);
          break;
        }
        case 110:
          clearTimeout(stage48TimerRef.current);
          updateLogOnce("Flash memory initialized.");
          setStage("ready");
          smoothSetProgress(52);
          startStagedIdleProgress(52, 65);
          clearTimeout(stage52TimerRef.current);
          stage52TimerRef.current = setTimeout(() => {
            handleFlashError(
              "Hardware connection interrupted. Please check and try again.",
              "flashing"
            );
            if (window.evtSource) {
              window.evtSource.close();
              window.evtSource = null;
            }
          }, 70000);
          break;
        case 810:
          clearTimeout(stage52TimerRef.current);
          updateLogOnce("Flash memory ID verified successfully.");
          setStage("ident");
          smoothSetProgress(56);
          clearTimeout(stage56TimerRef.current);
          stage56TimerRef.current = setTimeout(() => {
            handleFlashError(
              "Hardware connection interrupted. Please check and try again.",
              "flashing"
            );
            if (window.evtSource) {
              window.evtSource.close();
              window.evtSource = null;
            }
          }, 70000);
          break;
        case 210:
          clearTimeout(stage56TimerRef.current);
          // 810 is the only flag the device writes without an ACK wait, so the
          // backend can miss it entirely — clear its watchdog here as well.
          clearTimeout(stage52TimerRef.current);
          updateLogOnce("Flash Memory Write/Read Verification Passed.");
          setStage("checkedmem");
          smoothSetProgress(58);
          break;
        case 310:
          updateLogOnce("Firmware Flashing process initiated - Please wait");
          setStage("writing");
          smoothSetProgress(65);
          startStagedIdleProgress(65, 82);
          break;
        case 410:
          updateLogOnce("Firstboot Image Flashed Successfully.");
          setStage("firstboot");
          smoothSetProgress(82);
          break;
        case 510:
          updateLogOnce("Application Header Flashed Successfully.");
          setStage("header");
          smoothSetProgress(95);
          break;
        case 610:
          updateLogOnce("Application Image Flashed Successfully.");
          setStage("application");
          smoothSetProgress(99);
          break;
        case 710:
          setCurrentStep("done");
          clearInterval(TimerRef.current);
          stopIdleProgress();
          clearTimeout(stage48TimerRef.current);
          clearTimeout(stage52TimerRef.current);
          clearTimeout(stage56TimerRef.current);
          clearTimeout(flashStartTimerRef.current);
          updateLogOnce("Flashing Completed Successfully.");
          setStage("sealed");
          setProgress(100);
          setSuccess(true);
          successRef.current = true;
          setFlashStarted(false);
          if (flasherType === "single") {
            setTerminalOutput([
              "Firmware flashed successfully please wait for COM output...",
              "--------------------------------------------------------",
            ]);
            socketRef.current?.emit("start_uart");
          }
          evtSource.close();
          window.evtSource = null;
          // The GDB server has released the J-Link — ask for a fresh probe so
          // the footer and chip status are correct the moment the user is back
          // on page 1, instead of sitting red for 10-20 s.
          fetchHwSnapshot(true).catch(() => { });
          break;
        default:
          break; // INFO, ACK codes (105/120/...), anything else: ignore
      }
    };
  };

  // =====================================================
  // CONTINUE (page 1 -> page 2)
  // -----------------------------------------------------
  // /hw-refresh reads a cached snapshot on the backend (it never opens the
  // J-Link), so this can no longer sit behind a 20 s hardware timeout — that
  // is what left the button spinning forever.
  // =====================================================
  const handleContinue = async () => {
    if (checkingNext) return;
    setCheckingNext(true);

    try {
      const fresh = await fetchHwSnapshot(true, 12000);

      // No `|| hwRef.current` fallback here any more. A forced probe that
      // failed says NOTHING about the target right now, while the last
      // snapshot can easily still name a chip that has since dropped -
      // advancing on it is what put the wizard on page 2 while the readout
      // still said "Detecting...". A failed probe is a failed probe.
      if (!fresh) {
        toast.error("Hardware status unavailable. Please try again.");
        return;
      }

      // Decide on the newest snapshot that actually reached the UI, not
      // necessarily the reply we are holding: if a live push overtook our
      // probe while it was in flight, the push is the truth and hwRef has it.
      const snap = hwRef.current || fresh;

      if (!snap.ready) {
        toast.error("Hardware status unavailable. Please try again.");
        return;
      }

      // This is a button press, so every fault gets named explicitly here —
      // this is where the errors the poller stayed quiet about surface.
      if (!snap.jtag) {
        toast.error(CHIP_ERROR_MESSAGES.NO_JLINK);
        return;
      }

      if (snap.chip_name) {
        setCurrentPage(2);
        return;
      }

      // Fault BEFORE `detecting`: a retry probe can flip `detecting` back on
      // while the fault is already diagnosed, and that must not mask it behind
      // "Please wait while the chip is detecting".
      if (snap.chip_error && !SILENT_FAULTS.has(snap.chip_error)) {
        toast.error(faultText(snap));
        return;
      }

      if (snap.detecting || snap.chip_error === "JLINK_BUSY") {
        toast.warning("Please wait while the chip is detecting");
        return;
      }

      toast.error(faultText(snap));
    } finally {
      // Unconditional: the spinner can never be left running.
      setCheckingNext(false);
    }
  };

  // =====================================================
  // DERIVED UI VALUES
  // =====================================================
  const chipStatusClass = isDetectingChip
    ? "is-detecting"
    : chipDetected
      ? "is-detected"
      : "is-none";
  const chipStatusLabel = isDetectingChip
    ? "Detecting"
    : chipDetected
      ? selectedChip
      : CHIP_ERROR_LABELS[hwFault.code] || (jlinkPresent ? "No chip" : "No J-Link");

  // Hovering the pill gives the full message from the table, so the fault is
  // still explainable between toasts.
  const chipStatusTitle = isDetectingChip
    ? "Detecting the connected chip"
    : chipDetected
      ? `Detected chip: ${selectedChip}`
      : hwFault.message || CHIP_ERROR_MESSAGES.NO_FIRMWARE;

  const stepLabels = [
    "Firmware Preparation",
    "Device Connection",
    "Flashing Firmware",
    "Flash Completed",
  ];

  const flashing = isStarting || flashStarted;

  // Two related but different questions, and the exit animation is the reason
  // they have to be asked separately:
  //   flashSession    - is the progress panel in the DOM? Stays true through
  //                     the exit, so the panel can animate out.
  //   onProgressScreen - which screen is on top? Goes false the instant
  //                     Finish / Try again is pressed, which is what starts
  //                     the cross-fade.
  const flashSession = flashStarted || success || errorStopped;
  const onProgressScreen = flashSession && !leavingProgress;

  // Fill for the stepper rail — purely visual, derived from the step already
  // being rendered below.
  const railIndex = failedStep
    ? steps.indexOf(failedStep)
    : currentStep === "done"
      ? steps.length - 1
      : steps.indexOf(currentStep);
  const railFill = `${(Math.max(0, railIndex) / (steps.length - 1)) * 100}%`;

  // =====================================================
  // DEVICE GRAPHIC STATE  (derived only — no new data)
  // =====================================================
  const devState = errorStopped
    ? "is-fail"
    : success
      ? "is-done"
      : flashStarted
        ? "is-flash"
        : !jlinkPresent
          ? "is-off"
          : chipDetected
            ? "is-ready"
            : isDetectingChip
              ? "is-scan"
              : "is-idle";

  // Where we are on the FLASH_STAGES timeline. "fault" is not on it, so it
  // lands at -1: the stage layer drops all the "in flight" motion and holds
  // red. The sector map is read off the PEAK instead, because an image that
  // was written stays written — a failure at 95% must still show the
  // bootloader sector it did commit, not an empty die.
  const stageIdx = FLASH_STAGES.indexOf(stage);
  const peakStageRef = useRef(0);
  useEffect(() => {
    peakStageRef.current = stage === "idle" ? 0 : Math.max(stageIdx, peakStageRef.current);
  }, [stage, stageIdx]);
  // Read straight from the stage on "idle" rather than the ref: a ref does
  // not re-render, so a reset has to be correct in the same paint.
  const mapIdx = stage === "idle" ? 0 : Math.max(stageIdx, peakStageRef.current);
  const reached = (name) => mapIdx >= FLASH_STAGES.indexOf(name);

  // Overall completion — the probe readout and the segment meter, i.e. the
  // two places that stand in for the percentage on the right.
  const barFill = success ? 1 : Math.max(0, Math.min(1, progress / 100));

  // The die is a MAP OF THE FLASH, so it stays dark through the first ~65%:
  // upload, build, connect and identify are host-side work and write nothing
  // to the chip.
  //
  // It then fills in two parts, and the split is what keeps it honest AND
  // smooth. It cannot simply track the percentage 1:1 from the start of the
  // write, because that would snap the die straight to ~65% full in one
  // step. So the write opens EMPTY at 66% and climbs with the percentage
  // until, at 72%, it has caught up and is 72% full. From there it is 1:1 —
  // 95% full at 95% — so the die and the number on the right agree for the
  // whole back half. Both branches give 0.72 at DIE_CATCHUP, so the handover
  // is invisible.
  const DIE_START = 66;    // write begins here, die empty
  const DIE_CATCHUP = 82;  // die has caught up with the percentage here
  const dieFill = success
    ? 1
    : !reached("writing") || progress < DIE_START
      ? 0
      : progress < DIE_CATCHUP
        ? ((progress - DIE_START) / (DIE_CATCHUP - DIE_START)) * (DIE_CATCHUP / 100)
        : Math.min(1, progress / 100);
  const dieY = 272 + 72 * (1 - dieFill);
  const dieH = 72 * dieFill;

  // Caption under the board. During a flash it is the stage; before one it
  // reports what the probe can actually see, so the graphic is never mute.
  const vizCaption =
    stage === "fault"
      ? STAGE_CAPTIONS.fault
      : stage !== "idle"
        ? STAGE_CAPTIONS[stage]
        : !jlinkPresent
          ? "Probe not connected"
          : isDetectingChip
            ? "Detecting target"
            : chipDetected
              ? `${selectedChip} ready to flash`
              : "No target detected";

  // Activity meter under the caption: 16 segments off the same real progress
  // value the percentage uses, so it can never disagree with it.
  const VIZ_SEGMENTS = 16;
  const litSegments = Math.round(barFill * VIZ_SEGMENTS);

  // Single-bin drag handlers — shared by the empty dropzone and the file card
  // so a replacement file can be dropped on either. Same behaviour as before.
  const onFileDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
    document.body.classList.add("dragging");
  };
  const onFileDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
    document.body.classList.remove("dragging");
  };
  const onFileDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    document.body.classList.remove("dragging");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (file.name.endsWith(".bin")) {
      setSelectedFile(file);
      setSuccess(false);
      setErrorStopped(false);
      setFlashStarted(false);
      setProgress(0);
      setStage("idle");
      setLogs([]);
    } else {
      toast.error("Only .bin files are allowed!");
    }
  };
  const openFilePicker = () => fileInputRef.current?.click();

  // =====================================================
  // RENDER
  // =====================================================
  return (
    <>
      <ToastWrapper />

      <div className="fl">

        <div className="fl-bg" aria-hidden="true">
          <span className="fl-bg-grid" />
          <span className="fl-bg-glow" />
        </div>

        <div className="fl-inner">

          {/* ===================== HEADER ===================== */}
          <header className={`fl-head ${flashStarted ? "is-live" : ""}`}>
            <div className="fl-head-id">
              <span className="fl-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" />
                  <rect x="10" y="10" width="4" height="4" rx="1" />
                  <path d="M12 2v4.5M12 17.5V22M2 12h4.5M17.5 12H22M8 2v4.5M16 2v4.5M8 17.5V22M16 17.5V22M2 8h4.5M2 16h4.5M17.5 8H22M17.5 16H22" />
                </svg>
              </span>
              <h1 className="fl-title">Firmware Upgrade</h1>
            </div>
          </header>

          {/* ===================== MAIN ===================== */}
          <main className="fl-main">

            {/* --------- LEFT: live device graphic ---------
                 Every stage in FLASH_STAGES owns one visible change here. The
                 layers below are inert on their own: `st-<stage>` on the root
                 svg is what switches each one on, so the graphic and the log
                 on the right can never drift apart.

                 `key={stage}` on .fl-dev-fx is deliberate — remounting the
                 group restarts its one-shot rings, which is what makes EVERY
                 status update register as motion even between neighbouring
                 stages that share a steady state. --------- */}
            <aside className="fl-viz" aria-hidden="true">
              <svg
                className={`fl-dev ${devState} st-${stage}`}
                viewBox="0 0 280 470"
                preserveAspectRatio="xMidYMid meet"
              >
                <defs>
                  <clipPath id="flDieClip">
                    <rect x="104" y="272" width="72" height="72" rx="6" />
                  </clipPath>
                  {/* Programmed region: denser at the bottom, where the write
                      started, so the fill reads as settled cells rather than
                      flat paint. */}
                  <linearGradient id="flFillGrad" x1="0" y1="1" x2="0" y2="0">
                    <stop offset="0" stopColor="#53d824" stopOpacity="0.5" />
                    <stop offset="1" stopColor="#53d824" stopOpacity="0.16" />
                  </linearGradient>
                  {/* Scan head: bright leading edge, trailing fade. */}
                  <linearGradient id="flScanGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stopColor="#8ef25f" stopOpacity="0" />
                    <stop offset="0.75" stopColor="#8ef25f" stopOpacity="0.42" />
                    <stop offset="1" stopColor="#eafce0" stopOpacity="0.95" />
                  </linearGradient>
                  <linearGradient id="flEraseGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stopColor="#eef2ee" stopOpacity="0.9" />
                    <stop offset="1" stopColor="#eef2ee" stopOpacity="0" />
                  </linearGradient>
                  {/* Board-wide light bar for the boot sweep at 100%. */}
                  <linearGradient id="flBootGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stopColor="#53d824" stopOpacity="0" />
                    <stop offset="0.5" stopColor="#8ef25f" stopOpacity="0.3" />
                    <stop offset="1" stopColor="#53d824" stopOpacity="0" />
                  </linearGradient>
                </defs>

                <circle className="fl-dev-halo" cx="140" cy="308" r="78" />
                <circle className="fl-dev-core" cx="140" cy="308" r="46" />

                <path className="fl-dev-cable" d="M140 0 V20" />
                <g className="fl-dev-probe">
                  <rect className="fl-dev-probe-body" x="80" y="20" width="120" height="54" rx="11" />
                  <rect className="fl-dev-probe-face" x="92" y="32" width="96" height="30" rx="7" />
                  <circle className="fl-dev-led" cx="105" cy="47" r="3.5" />
                  {/* Centred on the probe face (x 92..188, y 32..62). The
                      extra 1 on x compensates for letter-spacing: SVG adds
                      the tracking AFTER the last glyph too, so a
                      text-anchor: middle string sits half a space left of
                      true centre. */}
                  <text className="fl-dev-brand" x="141" y="51">J-LINK</text>
                </g>

                {/* Ribbon: the dim physical lanes the cable is made of. These
                    never move - they are the hardware. */}
                <g className="fl-dev-ribbon">
                  <path d="M116 74 V162" />
                  <path d="M128 74 V162" />
                  <path d="M140 74 V162" />
                  <path d="M152 74 V162" />
                  <path d="M164 74 V162" />
                </g>

                {/* Data on the ribbon, drawn as the SAME dashed stream the
                    traces inside the board use (.fl-dev-trace-fx): short
                    dashes running down each lane. One visual language for
                    the whole path, so the transfer reads as continuous from
                    the probe into the die instead of changing metaphor at
                    the board edge. Geometry is identical to .fl-dev-ribbon
                    above, so the dashes ride exactly on the physical
                    lanes. */}
                <g className="fl-dev-pkts">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <path key={`pk${i}`} d={`M${116 + i * 12} 74 V162`} style={{ "--i": i }} />
                  ))}
                </g>

                <g className="fl-dev-board">
                  <rect className="fl-dev-pcb" x="24" y="162" width="232" height="286" rx="14" />
                  <rect className="fl-dev-pcb-inner" x="34" y="172" width="212" height="266" rx="10" />

                  <circle className="fl-dev-hole" cx="46" cy="184" r="4.5" />
                  <circle className="fl-dev-hole" cx="234" cy="184" r="4.5" />
                  <circle className="fl-dev-hole" cx="46" cy="426" r="4.5" />
                  <circle className="fl-dev-hole" cx="234" cy="426" r="4.5" />

                  <rect className="fl-dev-header" x="106" y="152" width="68" height="26" rx="5" />
                  <g className="fl-dev-header-pins">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <circle key={`ht${i}`} cx={116 + i * 12} cy="161" r="1.9" style={{ "--i": i }} />
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <circle key={`hb${i}`} cx={116 + i * 12} cy="170" r="1.9" style={{ "--i": 4 - i }} />
                    ))}
                  </g>

                  <g className="fl-dev-trace">
                    <path d="M116 178 V210 H106 V262" />
                    <path d="M140 178 V262" />
                    <path d="M164 178 V210 H174 V262" />
                  </g>
                  {/* Same three traces, drawn again as the live current path. */}
                  <g className="fl-dev-trace-fx">
                    <path d="M116 178 V210 H106 V262" style={{ "--i": 0 }} />
                    <path d="M140 178 V262" style={{ "--i": 1 }} />
                    <path d="M164 178 V210 H174 V262" style={{ "--i": 2 }} />
                  </g>

                  <g className="fl-dev-smd">
                    <rect x="58" y="236" width="15" height="7" rx="1.5" />
                    <rect x="58" y="252" width="15" height="7" rx="1.5" />
                    <rect x="207" y="236" width="15" height="7" rx="1.5" />
                    <rect x="207" y="252" width="15" height="7" rx="1.5" />
                    <rect x="58" y="330" width="20" height="12" rx="2" />
                    <rect x="202" y="328" width="26" height="16" rx="3" />
                  </g>

                  {/* Board indicators: PWR latches on once the link is up, ACT
                      only blinks while the device is actually being written. */}
                  <rect className="fl-dev-pwrled" x="58" y="296" width="15" height="7" rx="2" />
                  <rect className="fl-dev-actled" x="207" y="296" width="15" height="7" rx="2" />

                  <g className="fl-dev-pins">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <rect key={`pt${i}`} x={102 + i * 17} y="252" width="5" height="10" rx="2" />
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <rect key={`pb${i}`} x={102 + i * 17} y="354" width="5" height="10" rx="2" />
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <rect key={`pl${i}`} x="84" y={270 + i * 17} width="10" height="5" rx="2" />
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <rect key={`pr${i}`} x="186" y={270 + i * 17} width="10" height="5" rx="2" />
                    ))}
                  </g>

                  <g className="fl-dev-chip">
                    <rect className="fl-dev-chip-glow" x="94" y="262" width="92" height="92" rx="10" />
                    <rect className="fl-dev-chip-body" x="94" y="262" width="92" height="92" rx="10" />
                    <rect className="fl-dev-die" x="104" y="272" width="72" height="72" rx="6" />

                    <g clipPath="url(#flDieClip)">
                      {/* Cell array — powered up at "ready", and the surface
                          everything after it is drawn on. */}
                      <g className="fl-dev-array">
                        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                          <path key={`ar${i}`} d={`M104 ${272 + i * 9} H176`} />
                        ))}
                        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                          <path key={`ac${i}`} d={`M${104 + i * 9} 272 V344`} />
                        ))}
                      </g>

                      {/* Programmed region + write head. */}
                      <rect className="fl-dev-fill" x="104" y={dieY} width="72" height={dieH} />
                      {dieFill > 0 && (
                        <rect className="fl-dev-fill-edge" x="104" y={dieY} width="72" height="2" />
                      )}

                      {/* Committed sectors, in real flash order: bootloader at
                          the bottom of the map, then the header, then the
                          application image. Each one latches as its code
                          arrives. */}
                      <rect
                        className={`fl-dev-sector s-boot ${reached("firstboot") ? "is-on" : ""}`}
                        x="104" y="326" width="72" height="18"
                      />
                      <rect
                        className={`fl-dev-sector s-hdr ${reached("header") ? "is-on" : ""}`}
                        x="104" y="310" width="72" height="16"
                      />
                      <rect
                        className={`fl-dev-sector s-app ${reached("application") ? "is-on" : ""}`}
                        x="104" y="272" width="72" height="38"
                      />

                      {/* One-shot passes: ID read, then the erase wash. */}
                      <rect className="fl-dev-scan" x="104" y="272" width="72" height="16" />
                      <rect className="fl-dev-erase" x="104" y="272" width="72" height="72" />
                    </g>

                    <circle className="fl-dev-pin1" cx="113" cy="281" r="2.6" />
                    <text className="fl-dev-mark" x="140" y="313">
                      {chipDetected && selectedChip ? selectedChip : ""}
                    </text>
                  </g>

                  {/* Target boot indicators — only meaningful once the flash
                      has completed and the device runs its own firmware. */}
                  <g className="fl-dev-boot">
                    {[0, 1, 2].map((i) => (
                      <circle key={`bt${i}`} cx={124 + i * 16} cy="382" r="3.4" style={{ "--i": i }} />
                    ))}
                  </g>

                  <text className="fl-dev-silk" x="140" y="404">{selectedBoard}</text>
                  <rect className="fl-dev-edge" x="104" y="420" width="72" height="12" rx="2" />

                  {/* Power-up light bar that runs down the board at 100%. */}
                  <rect className="fl-dev-bootsweep" x="24" y="162" width="232" height="70" />
                </g>

                {/* One-shot acknowledgement for the stage that just landed. */}
                <g className="fl-dev-fx" key={stage}>
                  <circle className="fl-dev-ring r1" cx="140" cy="308" r="50" />
                  <circle className="fl-dev-ring r2" cx="140" cy="308" r="50" />
                </g>
              </svg>

              {/* Readout under the board: what the device is doing, plus a
                  segment meter off the same progress value as the percentage.
                  Named parts, not decoration — it is the caption for the
                  animation above it. */}
              <div className={`fl-viz-readout st-${stage}`}>
        
                <span className="fl-viz-meter">
                  {Array.from({ length: VIZ_SEGMENTS }).map((_, i) => (
                    <span
                      key={i}
                      className={`fl-viz-seg ${i < litSegments ? "is-on" : ""}`}
                      style={{ "--i": i }}
                    />
                  ))}
                </span>
              </div>
            </aside>

            {/* --------- RIGHT: work column --------- */}
            <section
              className={`fl-work ${onProgressScreen ? "go-progress" : ""} ${leavingProgress ? "is-leaving" : ""}`}
            >

              {/* ============ CONFIGURATION ============ */}
              <div className="fl-screen fl-screen-config">

                {/* hidden single-bin input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".bin"
                  onChange={handleFileChange}
                  className="fl-file-input"
                />

                {/* ---------------- PAGE 1 ---------------- */}
                <div className={`fl-page ${currentPage === 1 ? "is-active" : "is-left"}`}>
                  {/* .fl-card holds the fields AND the CTA, so the page reads as
                      one bordered panel instead of two loose groups. */}
                  <div className="fl-card">

                    <div className="fl-stack">

                      {!onProgressScreen && (
                        <div className="fl-steps">
                          <span className="fl-steps-text">Step 1 of 2</span>
                        </div>
                      )}

                      <div className="fl-field" style={{ "--d": "0ms" }}>
                        <span className="fl-label">Flasher mode</span>

                        {/* Segmented control, not two cards with ticks. One
                            inset track holds a single thumb that slides to the
                            selected segment: --i is the selected index, --n the
                            segment count, and the CSS derives the thumb's width
                            and travel from those two numbers. The radios stay
                            underneath, so arrow-key navigation and the form
                            semantics are the browser's, not ours. */}
                        <div
                          className="fl-seg"
                          style={{ "--n": 2, "--i": flasherType === "multi" ? 1 : 0 }}
                          role="radiogroup"
                          aria-label="Flasher mode"
                        >
                          <span className="fl-seg-thumb" aria-hidden="true" />

                          <label className={`fl-seg-opt ${flasherType === "single" ? "is-active" : ""}`}>
                            <input
                              type="radio"
                              name="flasherType"
                              value="single"
                              checked={flasherType === "single"}
                              onChange={() => { setFlasherType("single"); setCurrentPage(1); }}
                            />
                            <span className="fl-seg-ic" aria-hidden="true">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                              </svg>
                            </span>
                            <span className="fl-seg-text">
                              <span className="fl-seg-title">Single-bin</span>
                            </span>
                          </label>

                          <label className={`fl-seg-opt ${flasherType === "multi" ? "is-active" : ""}`}>
                            <input
                              type="radio"
                              name="flasherType"
                              value="multi"
                              checked={flasherType === "multi"}
                              onChange={() => { setFlasherType("multi"); }}
                            />
                            <span className="fl-seg-ic" aria-hidden="true">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M15 3H9a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7z" />
                                <polyline points="15 3 15 7 19 7" />
                                <path d="M15 21H7a2 2 0 0 1-2-2V7" opacity="0.5" />
                              </svg>
                            </span>
                            <span className="fl-seg-text">
                              <span className="fl-seg-title">Multi-bin</span>
                              {/* maxBins is chip-dependent, so the count is read
                                  off it rather than hard-coded. */}

                            </span>
                          </label>
                        </div>
                      </div>

                      <div className="fl-grid2">
                        <div className="fl-field fl-field-board">
                          <span className="fl-label">Target board</span>
                          <CustomDropdown
                            value={selectedBoard}
                            onChange={setSelectedBoard}
                            options={[
                              { value: "CRT", label: "CRT" },
                              { value: "Sparsh", label: "Sparsh" },
                            ]}
                          />
                        </div>

                        <div className="fl-field" style={{ "--d": "120ms" }}>
                          <span className="fl-label">Detected chip</span>
                          <div className={`fl-readout ${chipStatusClass}`} title={chipStatusTitle}>
                            <span className="fl-dot" />
                            <span className="fl-readout-value">
                              {chipStatusLabel}
                              {isDetectingChip && <span className="fl-dots" />}
                            </span>
                          </div>
                        </div>
                      </div>

                    </div>

                    <div className="fl-cta-continue">
                      <button
                        className="fl-btn fl-btn-primary"
                        disabled={checkingNext}
                        onClick={handleContinue}
                      >
                        {checkingNext && <span className="fl-spinner" />}
                        <span className="fl-btn-text">{checkingNext ? "loading" : "Continue"}</span>
                        {!checkingNext && (
                          <svg className="fl-btn-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <line x1="5" y1="12" x2="19" y2="12" />
                            <polyline points="12 5 19 12 12 19" />
                          </svg>
                        )}
                      </button>
                    </div>

                  </div>
                </div>

                {/* ---------------- PAGE 2 ---------------- */}
                {/* The mode has to be readable from here because everything
                    below is shared by both modes: is-multi shrinks the Flash
                    interface control, is-single spaces the CTA. Page 1 carries
                    neither, so its Continue button is never affected. */}
                <div className={`fl-page ${currentPage === 2 ? "is-active" : "is-right"} ${flasherType === "multi" ? "is-multi" : "is-single"}`}>
                  <div className="fl-card">

                    <div className="fl-stack">

                      <div className="fl-backrow">
                        <button className="fl-back" onClick={() => setCurrentPage(1)}>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <line x1="19" y1="12" x2="5" y2="12" />
                            <polyline points="12 19 5 12 12 5" />
                          </svg>
                          Back
                        </button>

                        {/* Same counter as page 1. It shares the top row with Back
                            so it lands on the exact same line / right edge on both
                            pages instead of shifting the fields down. */}
                        {!onProgressScreen && (
                          <div className="fl-steps">
                            <span className="fl-steps-text">Step 2 of 2</span>
                          </div>
                        )}
                      </div>

                      {showInterface ? (
                        // GPX 10 Pro -> selectable dropdown
                        <div className="fl-field fl-field-iface" style={{ "--d": "0ms" }}>
                          <span className="fl-label">Flash interface</span>
                          <CustomDropdown
                            value={selectedInterface}
                            onChange={setSelectedInterface}
                            placeholder="Select interface"
                            options={[
                              { value: "SPI0", label: "SPI0" },
                              { value: "QPI", label: "QPI" },
                              { value: "SPI1", label: "SPI1" },
                            ]}
                          />
                        </div>
                      ) : (selectedChip === "GPX 10" || selectedBoard === "Sparsh") ? (
                        // GPX 10 (any board) OR any Sparsh board -> interface fixed to SPI0 (display only)
                        <div className="fl-field" style={{ "--d": "0ms" }}>
                          <span className="fl-label">Flash interface</span>
                          <div className="fl-readout is-detected is-fixed">
                            <span className="fl-dot" />
                            <span className="fl-readout-value">SPI0</span>
                            <span className="fl-tag">Default</span>
                          </div>
                        </div>
                      ) : null}

                      {/* ---------- SINGLE BIN ---------- */}
                      {flasherType === "single" && (
                        <div className="fl-field" style={{ "--d": "60ms" }}>
                          <span className="fl-label">Firmware file</span>

                          {!selectedFile ? (
                            <div
                              className={`fl-drop ${isDragging ? "is-drag" : ""}`}
                              onClick={openFilePicker}
                              onDragOver={onFileDragOver}
                              onDragLeave={onFileDragLeave}
                              onDrop={onFileDrop}
                            >
                              <span className="fl-drop-badge">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                  <polyline points="17 8 12 3 7 8" />
                                  <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                              </span>
                              <span className="fl-drop-title">
                                {isDragging
                                  ? "Drop the .bin file here"
                                  : "Browse or drag and drop a .bin file"}
                              </span>
                            </div>
                          ) : (
                            <div
                              className="fl-filecard"
                              onDragOver={onFileDragOver}
                              onDragLeave={onFileDragLeave}
                              onDrop={onFileDrop}
                            >
                              <span className="fl-filecard-badge">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                  <polyline points="14 2 14 8 20 8" />
                                </svg>
                              </span>
                              <span className="fl-filecard-meta">
                                <span className="fl-filecard-name">{selectedFile.name}</span>
                                <span className="fl-filecard-size">{formatSize(selectedFile.size)}</span>
                              </span>
                              <button className="fl-btn fl-btn-quiet" onClick={openFilePicker}>
                                Browse
                              </button>
                            </div>
                          )}
                        </div>
                      )}

                      {/* ---------- MULTI BIN ---------- */}
                      {flasherType === "multi" && (
                        <div className="fl-field" style={{ "--d": "60ms" }}>
                          <span className="fl-label">Firmware files</span>

                          <div
                            className={`fl-bin-wrap ${binScroll.top ? "" : "can-up"} ${binScroll.bottom ? "" : "can-down"}`}
                          >
                            <div
                              className={`fl-bin-list ${binFiles.length > 3 ? "is-scroll" : ""}`}
                              ref={binTableRef}
                              onScroll={updateBinScroll}
                            >
                              {binFiles.map((bin, index) => (
                                <div className="fl-bin-row" key={index}>
                                  <span className="fl-bin-tag">
                                    {binFiles.length === 1 ? "BIN :" : `BIN ${index + 1} :`}
                                  </span>

                                  <div
                                    className={`fl-bin-name ${dragIndex === index ? "is-drag" : ""} ${bin.file ? "is-filled" : ""}`}
                                    onDragOver={(e) => {
                                      e.preventDefault();
                                      setDragIndex(index);
                                      document.body.classList.add("dragging");
                                    }}
                                    onDragLeave={(e) => {
                                      e.preventDefault();
                                      setDragIndex(null);
                                      document.body.classList.remove("dragging");
                                    }}
                                    onDrop={(e) => {
                                      e.preventDefault();
                                      setDragIndex(null);
                                      document.body.classList.remove("dragging");

                                      const file = e.dataTransfer.files[0];
                                      if (!file) return;

                                      handleMultiBinChange(index, file, {
                                        target: { value: null }
                                      });
                                    }}
                                  >
                                    {bin.file
                                      ? bin.file.name
                                      : dragIndex === index
                                        ? "Drop .bin file here"
                                        : "Browse or drag & drop"}
                                  </div>

                                  <div className="fl-addr-wrap">
                                    <input
                                      type="text"
                                      className="fl-bin-addr"
                                      placeholder="Int Address"
                                      value={bin.address}
                                      onChange={(e) => handleAddressChange(index, e.target.value)}
                                    />
                                    <div className="fl-addr-tip">Enter a valid bin Compiled location</div>
                                  </div>

                                  <button
                                    className="fl-bin-browse"
                                    onClick={() => document.getElementById(`bin-input-${index}`)?.click()}
                                  >
                                    Browse
                                  </button>

                                  {binFiles.length >= 1 ? (
                                    <button
                                      className="fl-bin-remove"
                                      onClick={() => removeBin(index)}
                                      aria-label={`Remove ${binFiles.length === 1 ? "BIN" : `BIN ${index + 1}`}`}
                                    >
                                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                      </svg>
                                    </button>
                                  ) : (
                                    <span className="fl-bin-spacer" aria-hidden="true" />
                                  )}

                                  <input
                                    type="file"
                                    accept=".bin"
                                    id={`bin-input-${index}`}
                                    style={{ display: "none" }}
                                    onChange={(e) => handleMultiBinChange(index, e.target.files[0], e)}
                                  />
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="fl-add-row">
                            {binFiles.length < maxBins ? (
                              <button className="fl-add-btn" onClick={addAnotherBin}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                                  <line x1="12" y1="5" x2="12" y2="19" />
                                  <line x1="5" y1="12" x2="19" y2="12" />
                                </svg>
                                Add another BIN
                              </button>
                            ) : <span className="fl-add-spacer" />}

                            {(selectedChip === "GPX 10" || selectedChip === "GPX 10 Pro") && (
                              <span className="fl-limit-text">Up to 5 BIN files</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="fl-cta">
                      <button
                        className="fl-btn fl-btn-primary"
                        onClick={flasherType === "multi" ? startMultiFlashing : startFlashing}
                        disabled={flashing}
                      >
                        {flashing && <span className="fl-spinner" />}
                        <span className="fl-btn-text">{flashing ? "Upgrading" : "Upgrade firmware"}</span>
                      </button>
                    </div>

                  </div>
                </div>
              </div>

              {/* ============ PROGRESS ============ */}
              {/* flashSession, NOT onProgressScreen: the panel has to outlive
                  the screen swap by one transition, or there is nothing left
                  on screen to animate out. */}
              {flashSession && (
                <div className="fl-screen fl-screen-progress">
                  <div
                    className={`fl-prog ${flashStarted ? "is-live" : ""} ${success ? "is-done" : ""} ${errorStopped ? "is-fail" : ""}`}
                  >

                    {/* headline + percentage */}
                    <div className="fl-prog-top">
                      <div className="fl-prog-label">
                        Upgrading firmware on <span className="fl-prog-chip">{selectedChip}</span>
                        {!success && !errorStopped && <span className="fl-dots" />}
                      </div>
                      <div className="fl-prog-value">
                        {Math.round(progress)}<span className="fl-prog-unit">%</span>
                      </div>
                    </div>

                    {/* bar */}
                    <div className="fl-bar">
                      <div className="fl-bar-fill" style={{ width: `${Math.round(progress)}%` }}>
                        <span className="fl-bar-head" />
                      </div>
                    </div>

                    {/* stepper, or the completion banner once done */}
                    {!success && (
                      <div className="fl-stepper">
                        <div className="fl-stepper-track">
                          <div className="fl-stepper-fill" style={{ width: railFill }} />
                        </div>
                        {stepLabels.map((label, i) => {
                          const stepKey = steps[i];
                          const isFailed = failedStep === stepKey;
                          const isDone = (steps.indexOf(currentStep) > i || (currentStep === "done" && i === steps.length - 1)) && !isFailed;
                          const isActive = steps.indexOf(currentStep) === i && currentStep !== "done" && !isFailed;
                          return (
                            <div
                              key={i}
                              className={`fl-step ${isDone ? "is-done" : ""} ${isActive ? "is-active" : ""} ${isFailed ? "is-failed" : ""}`}
                            >
                              <span className="fl-step-node">
                                {isFailed ? (
                                  <svg className="fl-step-glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
                                    <line x1="7" y1="7" x2="17" y2="17" />
                                    <line x1="17" y1="7" x2="7" y2="17" />
                                  </svg>
                                ) : isDone ? (
                                  <svg className="fl-step-glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="20 6 9 17 4 12" />
                                  </svg>
                                ) : (
                                  <span className="fl-step-num">{i + 1}</span>
                                )}
                              </span>
                              <span className="fl-step-name">{label}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {success && flasherType === "single" && (
                      <div className="fl-banner">
                        <span className="fl-banner-ic">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        Flashing Completed Successfully.
                      </div>
                    )}

                    {/* console */}
                    <div className={`fl-console ${flashStarted ? "is-live" : ""}`}>

                      {/* SUCCESS — UART TERMINAL */}
                      {success && (
                        flasherType === "single" ? (
                          <div className="fl-term">
                            <div className="fl-term-bar">
                              <span className="fl-term-title">
                                <span className="fl-dot" />
                                COM / UART OUTPUT
                              </span>
                              <button
                                type="button"
                                className="fl-term-clear"
                                onClick={() => setTerminalOutput([])}
                              >
                                Clear
                              </button>
                            </div>

                            <div className="fl-term-body" ref={terminalRef}>
                              {terminalOutput.length === 0 ? (
                                <div className="fl-term-empty">
                                  Waiting for UART output...
                                </div>
                              ) : (
                                terminalOutput.map((line, index) => (
                                  <div key={index}>{line}</div>
                                ))
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className="fl-result is-ok">
                            <span className="fl-result-badge">
                              <span className="fl-result-ripple" />
                              <svg viewBox="0 0 52 52" fill="none">
                                <circle className="fl-result-ring" cx="26" cy="26" r="23" />
                                <path className="fl-result-glyph" d="M15.5 26.5 L22.5 33.5 L36.5 19.5" />
                              </svg>
                            </span>
                            <p className="fl-result-msg">Firmware flashed successfully.</p>
                          </div>
                        )
                      )}

                      {/* ERROR */}
                      {errorStopped && (
                        <div className="fl-result is-fail">
                          <span className="fl-result-badge">
                            <span className="fl-result-ripple" />
                            <svg viewBox="0 0 52 52" fill="none">
                              <circle className="fl-result-ring" cx="26" cy="26" r="23" />
                              <path className="fl-result-glyph" d="M18 18 L34 34 M34 18 L18 34" />
                            </svg>
                          </span>
                          <p className="fl-result-msg">{errorMessage || "Flashing failed"}</p>
                        </div>
                      )}

                      {/* IN-PROGRESS LOGS — same .fl-term shell the UART view
                          uses, so the two consoles read as one thing: a title
                          bar, then a result line per completed step.

                          Each line is a RESULT, not terminal output: a tick
                          (or a cross when it failed) and the message, rising
                          into place on the same fl-pop-in the cards and the
                          banner use. */}
                      {!success && !errorStopped && (
                        <div className="fl-term">
                          <div className="fl-term-bar">
                            <span className="fl-term-title">
                              <span className="fl-dot" />
                              FLASH CONSOLE
                            </span>
                          </div>

                          <div className="fl-log" ref={logRef}>
                            {logs.map((log, index) => (
                              <div
                                key={index}
                                className={`fl-log-line ${log.success ? "is-ok" : "is-fail"}`}
                              >
                                {/* Badge glyph: a tinted disc, a ring, then the
                                    mark. All three paint with currentColor, so
                                    the ok/fail colour is set once in CSS on
                                    .fl-log-mark and the whole badge follows —
                                    and at this size the disc is what makes the
                                    mark legible, where a bare stroke check
                                    just reads as a smudge. */}
                                <span className="fl-log-mark" aria-hidden="true">
                                  {log.success ? (
                                    <svg viewBox="0 0 24 24">
                                      <circle cx="12" cy="12" r="10" fill="currentColor" fillOpacity="0.16" />
                                      <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.5" />
                                      <polyline
                                        points="7.6 12.5 10.6 15.4 16.4 9"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2.6"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      />
                                    </svg>
                                  ) : (
                                    <svg viewBox="0 0 24 24">
                                      <circle cx="12" cy="12" r="10" fill="currentColor" fillOpacity="0.16" />
                                      <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.5" />
                                      <path
                                        d="M8.8 8.8 L15.2 15.2 M15.2 8.8 L8.8 15.2"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2.6"
                                        strokeLinecap="round"
                                      />
                                    </svg>
                                  )}
                                </span>
                                <span className="fl-log-text">
                                  {log.message}
                                  {log.message === "Firmware Flashing process initiated - Please wait" && (
                                    <span className="fl-dots" />
                                  )}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* ACTION BUTTONS */}
                    {(success || errorStopped) && (
                      <div className="fl-cta">
                        {success && (
                          <button className="fl-btn fl-btn-primary" onClick={handleLeaveProgress}>
                            <span className="fl-btn-text">Finish</span>
                          </button>
                        )}
                        {errorStopped && (
                          <button className="fl-btn fl-btn-danger" onClick={handleLeaveProgress}>
                            <svg className="fl-btn-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <polyline points="1 4 1 10 7 10" />
                              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                            </svg>
                            <span className="fl-btn-text">Try again</span>
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

            </section>
          </main>
        </div>
      </div>
    </>
  );
}

// =====================================================
// EXPORT
// =====================================================
export default Flasher;
