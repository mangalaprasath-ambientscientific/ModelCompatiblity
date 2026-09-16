import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FaCog, FaExclamationTriangle } from 'react-icons/fa';

/* ============================================================================
 * NetronViewer
 *
 * Renders the model with the real Netron (vendored under public/netron - see
 * public/netron/VENDOR.md).  Netron is a static web app, so it is loaded in an
 * iframe from our own origin; being same-origin lets us hand it the uploaded
 * File through its own file-open path.
 *
 * Why not `?url=<blob url>` (Netron does support ?url=&identifier=):
 * browser.js `_fetch()` appends a `?cb=<timestamp>` cache-buster to every URL
 * that is not a `data:` URL, and a blob: URL with a query string does not
 * resolve.  Passing the File object avoids the round trip entirely - Netron
 * reads it with a FileReader, exactly as it does for drag & drop.
 * ==========================================================================*/

/* public/ is served at the app root; HashRouter keeps the document URL at
   index.html, so resolving against baseURI is correct in dev and in a build. */
const NETRON_URL = new URL('netron/index.html', document.baseURI).href;

const READY_TIMEOUT = 30000;
const POLL_MS = 60;

/* Netron sets <body class="welcome spinner"> in its markup and drops the
   spinner (view.show('welcome')) once the host has finished starting. */
function netronReady(win) {
  const doc = win && win.document;
  if (!doc || !doc.body || !win.__view__) return false;
  const cl = doc.body.classList;
  return cl.contains('welcome') && !cl.contains('spinner') && !!doc.getElementById('open-file-dialog');
}

/* Hand the File to Netron the way its own UI does: through the hidden file
   input, falling back to a drop on <body>.  Both paths end in Host._open(). */
function handOver(win, file) {
  const doc = win.document;

  let dt = null;
  try {
    dt = new win.DataTransfer();
    dt.items.add(file);
  } catch (e) {
    dt = null;
  }

  const input = doc.getElementById('open-file-dialog');
  if (dt && input) {
    try {
      input.files = dt.files;
    } catch (e) {
      /* assigning FileList is not supported - fall through to the drop path */
    }
    if (input.files && input.files.length === 1) {
      input.dispatchEvent(new win.Event('change', { bubbles: true }));
      return true;
    }
  }

  if (dt && win.DragEvent) {
    try {
      doc.body.dispatchEvent(new win.DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt }));
      return true;
    } catch (e) {
      /* fall through */
    }
  }

  /* last resort: Netron's host is reachable because we are same-origin */
  const host = win.__view__ && win.__view__._host;
  if (host && typeof host._open === 'function') {
    host._open(file, [file]);
    return true;
  }
  return false;
}

const NetronViewer = ({ file }) => {
  const frameRef = useRef(null);
  const timerRef = useRef(null);
  const seqRef = useRef(0);
  const [state, setState] = useState('idle'); // idle | loading | ready | error
  const [message, setMessage] = useState('');

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  /* Runs on every iframe load: wait for Netron to finish booting, then give it
     the file. A fresh document per file keeps Netron's own state clean. */
  const onLoad = useCallback(() => {
    const seq = seqRef.current;
    const started = Date.now();

    const tick = () => {
      if (seq !== seqRef.current) return;
      const frame = frameRef.current;
      if (!frame) return;

      let win = null;
      try {
        win = frame.contentWindow;
      } catch (e) {
        setState('error');
        setMessage('The graph viewer could not be reached (cross-origin).');
        return;
      }

      if (netronReady(win)) {
        if (!file) {
          setState('ready');
          return;
        }
        const ok = handOver(win, file);
        setState(ok ? 'ready' : 'error');
        if (!ok) setMessage('Netron did not accept the model file.');
        return;
      }

      if (Date.now() - started > READY_TIMEOUT) {
        setState('error');
        setMessage('The graph viewer did not finish loading.');
        return;
      }
      timerRef.current = setTimeout(tick, POLL_MS);
    };

    clearTimer();
    tick();
  }, [file]);

  /* New file -> new Netron document. */
  useEffect(() => {
    seqRef.current += 1;
    clearTimer();
    setState('loading');
    setMessage('');
    const frame = frameRef.current;
    if (frame) {
      /* a fresh document per file; the unused query param only guarantees the
         navigation happens (Netron reads `url`/`identifier`/`gist`, not `r`) */
      frame.src = `${NETRON_URL}?r=${seqRef.current}`;
    }
    return clearTimer;
  }, [file]);

  useEffect(() => clearTimer, []);

  return (
    <div className="netron-preview-container">
      {/* not sandboxed on purpose: Netron is our own asset and must stay
          same-origin so the File can be handed to it */}
      <iframe ref={frameRef} title="Netron" className="iframe-container" onLoad={onLoad} />
      {state === 'loading' && (
        <div className="netron-status">
          <FaCog className="fa-spin" />
          <span>Loading model graph…</span>
        </div>
      )}
      {state === 'error' && (
        <div className="netron-status netron-status-error">
          <FaExclamationTriangle />
          <span>{message}</span>
        </div>
      )}
    </div>
  );
};

export default NetronViewer;
