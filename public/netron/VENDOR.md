# Vendored Netron

This folder is the **real Netron web app**, self-hosted so the model graph is rendered
by Netron itself — no Python (`pip install netron`), no local netron server on :8181,
and no call out to netron.app.

| | |
|---|---|
| Upstream | https://github.com/lutzroeder/netron |
| Version | **v9.2.2** (`source/` directory of the release tarball) |
| Licence | MIT — see `LICENSE` in the upstream repo |

`frontend/src/components/NetronViewer.jsx` loads `netron/index.html` in an iframe and
hands the uploaded `.h5` to Netron's own file-open path, so what you see is Netron's
renderer, Netron's layout and Netron's sidebar.

## Changes made to the upstream files

**One patch**, in `browser.js`, marked with `EDGESPHERE PATCH`:

* `consent()` returns immediately. Upstream it fetches `https://ipinfo.io/json` to decide
  whether GDPR consent is required and then shows a blocking *"This app uses cookies…"*
  dialog. Neither belongs in an embedded viewer inside a desktop tool, and the network
  call would hang for 2s when offline.

Nothing else is modified. Telemetry needs no patch: it is gated on
`this._environment.packaged`, which is derived from the `<meta name="version">` in
`index.html` and is `0.0.0` in the source distribution, so it never starts.

## Files omitted

* `*.py` (`server.py`, `onnx.py`, `pytorch.py`, `__init__.py`) — the Python package.
* `desktop.mjs` — the Electron host.
* Four metadata files no `.h5` model can reach, dropped to keep the build from growing by
  another 16.5 MB: `mlir-metadata.json`, `onnx-metadata.json`, `tf-metadata.json`,
  `caffe2-metadata.json`.

  `onnx`/`pytorch`/`tflite` metadata is *prefetched* by `view.js` at start-up, but the
  prefetch is wrapped in `.catch(() => {})`, so the 404s are harmless. They would only
  matter if someone opened one of those formats through Netron's own **Open Model**
  button — this page only accepts `.h5`.

All 125 `*.js` modules are kept, because Netron resolves format readers lazily by name and
several of them are tried for a `.h5` file (`keras`, `pytorch`, `tflite`, `sklearn`,
`pickle`, `hickle`).

## Upgrading

1. Download the `source/` folder of the new release tag.
2. Copy `*.js`, `*.json`, `index.html`, `grapher.css`, `icon.png`, `favicon.ico` here.
3. Delete the four metadata files listed above.
4. Re-apply the `consent()` patch.
5. Check that `NetronViewer.jsx` still finds `#open-file-dialog` and that the ready state
   is still signalled by `document.body.class` going from `welcome spinner` to `welcome`.
