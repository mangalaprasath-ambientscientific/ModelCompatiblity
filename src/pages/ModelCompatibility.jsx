import React, { useRef, useState, useEffect, useCallback } from 'react';
import { FaUpload, FaRocket, FaCheck } from 'react-icons/fa';
// import { toast } from 'react-toastify';
import * as hdf5 from 'jsfive';
/* single stylesheet for the whole tab: layout, supported operators, validation */
import '../styles/modelCompatibility.css';
import SupportedOperators from '../components/SupportedOperators';
import ToastWrapper from '../components/ToastWrapper';
import NetronViewer from '../components/NetronViewer';
import ModelValidation from '../components/ModelValidation';

/* ============================================================================
 * 1. HDF5 / KERAS READER   (replaces  netron_view.py  +  h5py)
 * ==========================================================================*/

function readH5(arrayBuffer, filename) {
  const unwrap = (v) => (Array.isArray(v) ? v[0] : v);

  /* Model_Compatibility.py opens the file and reads two root attributes inside
     a single bare `except:` -

         try:
             with h5py.File(model_path, "r") as f:
                 model_backend = dict(f.attrs)['backend']
                 keras_version = dict(f.attrs)['keras_version']
         except:
             failure_reason = "Model is not a valid Tensorflow/Keras Model
                               Format, Please Verify Model."

     so a file h5py cannot open, and a .h5 whose root is missing either
     attribute, both land on the format message.  Nothing else does. */
  let f;
  let attrs;
  try {
    f = new hdf5.File(arrayBuffer, filename);
    attrs = f.attrs || {};
  } catch (e) {
    throw new Error('NOT_KERAS');
  }
  const backend = unwrap(attrs.backend);
  const kerasVersion = unwrap(attrs.keras_version);
  if (backend === undefined || kerasVersion === undefined) throw new Error('NOT_KERAS');

  /* `keras.models.load_model()` builds the graph out of the `model_config`
     attribute.  A missing or unparseable one is a FAILED LOAD, not an invalid
     format, so it goes down the `if not load:` path with config = null - which
     is what puts the "Older or Newer version of Python or Libraries" reason on
     screen instead of the generic format error.

     The stored weight arrays are read for shapes ONLY, to reproduce the stage-4
     assignment failures load_model() would have raised.  The heap/memory checks
     (wts_check / get_max_inter_output / heap_check) stay commented out at the
     tail of analyse(), exactly as they are in the python. */
  const rawConfig = unwrap(attrs.model_config);
  let cfg = null;
  let cfgRaw = null;
  if (rawConfig) {
    try {
      cfg = typeof rawConfig === 'string' ? JSON.parse(rawConfig) : rawConfig;
    } catch (e) {
      cfg = null;
    }
    /* a second, literal-preserving parse.  JSON.parse turns 0.0 into 0, and
       python would have printed 0.0 - the deserialisation message quotes the
       config verbatim, so the difference is visible to the user. */
    if (typeof rawConfig === 'string') cfgRaw = parseRawJson(rawConfig);
    /* python's json.loads() accepts Infinity / -Infinity / NaN and keras stores
       them unread, so such a model LOADS and reports normally.  JSON.parse()
       throws, so fall back to the literal-preserving parse rather than
       condemning the file. */
    if (cfg === null && cfgRaw !== null) cfg = fromRaw(cfgRaw);
  }

  return {
    config: cfg,
    backend: backend === undefined ? null : backend,
    keras_version: kerasVersion === undefined ? null : kerasVersion,
    config_raw: cfgRaw,
    weights: readWeights(f),
  };
}

/* The stored arrays, keyed by layer name:  { dense_1: [{ name, shape }, ...] }.
   load_model() assigns these into the variables the config just built, so a
   config that was edited by hand no longer fits them - see weightError().

   keras does NOT walk the file.  load_weights_from_hdf5_group() reads the
   `layer_names` attribute on /model_weights and, for each of those groups, the
   `weight_names` attribute, and it ignores every group or dataset those two
   lists do not mention.  Walking the tree instead made a leftover group, a
   `top_level_model_weights` entry or a stray dataset look like a layer-count or
   weight-count mismatch that keras never sees.

   The per-layer arrays are kept in weight_names ORDER, because batch_set_value()
   zips them against the layer's symbolic weights POSITIONALLY - it never matches
   them up by name. */
function readWeights(f) {
  let root = null;
  try {
    root = f.get('model_weights');
  } catch (e) {
    root = null;
  }
  /* load_model_from_hdf5() indexes f["model_weights"] unconditionally, so a file
     without that group is a hard load failure, not a model with no weights.
     null says "the group is not there"; {} says "it is there and declares
     nothing", which is a different error further down. */
  if (!root || !root.keys) return null;
  const per = {};
  /* the first weight_names entry that does not resolve inside its layer group */
  let missing = null;

  const asList = (v) => {
    if (v === null || v === undefined) return [];
    return (Array.isArray(v) ? v : [v]).map((x) => String(x).replace(/\u0000/g, '').trim()).filter(Boolean);
  };
  const leaf = (p) => String(p).split('/').pop();
  const record = (top, wname, node) => {
    if (!node || node.shape === undefined || node.dtype === undefined) return;
    (per[top] || (per[top] = [])).push({
      name: String(wname).replace(/:\d+$/, ''),
      shape: Array.prototype.slice.call(node.shape || []),
    });
  };

  /* load_weights_from_hdf5_group() reads the `layer_names` attribute and NOTHING
     else: a group the attribute does not mention is invisible to keras, and a
     file with no attribute at all presents ZERO saved layers, which the layer
     count guard below then reports.  Walking root.keys as a fallback made the
     port load a file the backend refuses. */
  const declared = asList((root.attrs || {}).layer_names);

  for (const layerName of declared) {
    let g;
    try {
      g = root.get(layerName);
    } catch (e) {
      continue; /* an unreadable group just means no weight check for that layer */
    }
    if (!g || !g.keys) continue;

    const wnames = asList((g.attrs || {}).weight_names);
    /* keras' filtered_layer_names drops a group whose weight_names is empty */
    if (!wnames.length) continue;
    for (const wn of wnames) {
      let d = null;
      try {
        d = g.get(wn);
      } catch (e) {
        d = null;
      }
      /* load_weights_from_hdf5_group() does, with g = /model_weights/<layer>:

             weight_values = [np.asarray(g[weight_name])
                              for weight_name in weight_names]

         so each name is resolved INSIDE the layer group.  keras writes the
         full variable name there, and the datasets nested to match, giving
         /model_weights/dense/dense/kernel:0.  A file whose weight_names say
         'dense/kernel' while the dataset sits at /model_weights/dense/kernel
         makes h5py look for /model_weights/dense/dense/kernel and raise

             'Unable to synchronously open object (component not found)'

         Retrying on the LEAF name papered over exactly that, so a file the
         backend cannot open reported as clean. */
      if (!d) {
        missing = missing || wn;
        continue;
      }
      record(layerName, leaf(wn), d);
    }
  }
  if (missing) {
    per.__missing__ = missing;
  }
  return per;
}

/* ============================================================================
 * 2. COMPATIBILITY ENGINE   (1:1 port of  Model_Compatibility.py)
 *    Returns the exact same { layer_info, layer_details, tips } contract that
 *    POST /validate-model used to return, so nothing downstream changes.
 * ==========================================================================*/

/* ---------------------------------------------------------------------------
 * LIBRARY / VERSION FALLBACK   (head of check_model_compatibility)
 *
 * Model_Compatibility.py reads `backend` / `keras_version` from the h5 root, but
 * it only consults them when keras itself FAILS to load the model:
 *
 *     try:    model = keras.models.load_model(model_path)
 *     except: if model_backend != supp_mb or keras_version != supp_kv:
 *                 "The Model was created on an Older or Newer version of
 *                  Python or Libraries, Please use Ambient recommended versions."
 *             else:
 *                 f"Unable to Load Model : {reason}"
 *
 * The version numbers are NOT a precondition.  keras 2.15 happily loads a model
 * written by 2.13.1, so the python reached the layer report and never printed
 * this message; only a model it could not load at all was rejected, and then
 * the version numbers merely explained why.
 * ------------------------------------------------------------------------ */
const SUPPORTED_BACKEND = 'tensorflow';
const SUPPORTED_KERAS = '2.15.0';
const OLD_LIBS_REASON =
  'The Model was created on an Older or Newer version of Python or Libraries, Please use Ambient recommended versions.';

/* the two root attributes every keras `.h5` writes, as trimmed strings */
function buildStack(parsed) {
  const str = (v) => (v === null || v === undefined ? '' : String(v).replace(/\u0000/g, '').trim());
  return { backend: str(parsed.backend), kerasVersion: str(parsed.keras_version) };
}

function isRecommendedStack(parsed) {
  const { backend, kerasVersion } = buildStack(parsed);
  return backend.toLowerCase() === SUPPORTED_BACKEND && kerasVersion === SUPPORTED_KERAS;
}

/* the `if not load:` branch above - only reached when the graph is unreadable */
function loadFailureReason(parsed, err) {
  if (!isRecommendedStack(parsed)) return OLD_LIBS_REASON;
  return 'Unable to Load Model : ' + (err && err.message ? err.message : String(err));
}

/* ---------------------------------------------------------------------------
 * THE SERIALISATION GATE IS GONE - it rejected models keras loads happily.
 *
 * It used to reject any file whose `keras_version` major was not 2.  That is not
 * what the python does: the version attribute is read up front but only CONSULTED
 * inside `if not load:`, so a graph keras 2.15 can deserialise reaches the layer
 * report no matter what the attribute says.  A mislabelled file, or one written
 * by 2.13, passed in python and was rejected here.
 *
 * It also substring-searched the whole config blob for `"batch_shape"` and
 * `"DTypePolicy"`, which condemned any model or layer that merely happened to
 * carry one of those names.  Both real cases are now caught structurally and for
 * the right reason: `batch_shape` is an unrecognised InputLayer keyword
 * (unknownKwargError), and a serialised DTypePolicy is an unusable `dtype`
 * (unresolvableObject).
 * ------------------------------------------------------------------------ */

/* ---------------------------------------------------------------------------
 * LAYER SETS - byte-for-byte the lists at the head of Model_Compatibility.py
 * ------------------------------------------------------------------------ */
const INPUT_LAYERS = ['InputLayer'];
const CONV_LAYERS = ['Conv2D', 'Conv1D', 'DepthwiseConv2D', 'DepthwiseConv1D', 'SeparableConv2D', 'SeparableConv1D'];
const POOL_LAYERS = ['MaxPooling2D', 'MaxPooling1D', 'AveragePooling2D', 'AveragePooling1D'];
const CORE_LAYERS = ['Dense'];
const ACT_LAYERS = ['Activation', 'Softmax', 'ReLU', 'LeakyReLU'];
const RNN_LAYERS = ['LSTM'];
const NORM_LAYERS = ['BatchNormalization'];
/* SpatialDropout3D is deliberately absent - it is commented out in the python */
const REG_LAYERS = ['Dropout', 'SpatialDropout1D', 'SpatialDropout2D', 'AlphaDropout', 'GaussianDropout', 'GaussianNoise', 'ActivityRegularization'];
const RESHAPE_LAYERS = ['Flatten', 'Reshape', 'Permute'];

const SUPP_LAYERS = [].concat(INPUT_LAYERS, CONV_LAYERS, POOL_LAYERS, CORE_LAYERS, ACT_LAYERS, RNN_LAYERS, NORM_LAYERS, REG_LAYERS, RESHAPE_LAYERS);
const PASS_LAYERS = [].concat(RESHAPE_LAYERS, REG_LAYERS, INPUT_LAYERS);
/* GaussianNoise is NOT here - the python omits it from non_input_layers */
const NON_INPUT_LAYERS = [].concat(POOL_LAYERS, ACT_LAYERS, NORM_LAYERS, [
  'Dropout', 'SpatialDropout1D', 'SpatialDropout2D', 'AlphaDropout', 'GaussianDropout', 'ActivityRegularization',
]);

/* supp_activation (Model_Compatibility_47.py:38-43) is
     [linear, relu, tf.nn.leaky_relu, softmax, sigmoid, tanh]
   and `activation in supp_activation` is an identity test against the resolved
   function.  keras.activations.get('leaky_relu') returns tf.nn.leaky_relu
   itself, so the name matches and activation_check() reports it as supported -
   leaving it out rejected a model the backend passes.

   LeakyReLU as a LAYER is separate: it is matched through SUPP_ACT_CLASSES. */
const SUPP_ACTIVATION = ['linear', 'relu', 'leaky_relu', 'softmax', 'sigmoid', 'tanh'];
/* supp_activation_classes: LeakyReLU (both module paths), ReLU, Softmax */
const SUPP_ACT_CLASSES = ['LeakyReLU', 'ReLU', 'Softmax'];
const LSTM_ACT = ['sigmoid', 'tanh'];
const LSTM_REC_ACT = ['sigmoid', 'tanh', 'hard_sigmoid'];

const ALIAS = {
  Convolution2D: 'Conv2D',
  Convolution1D: 'Conv1D',
  MaxPool2D: 'MaxPooling2D',
  MaxPool1D: 'MaxPooling1D',
  AvgPool2D: 'AveragePooling2D',
  AvgPool1D: 'AveragePooling1D',
};
const canon = (c) => ALIAS[c] || c;
const isIn = (c, arr) => arr.indexOf(canon(c)) !== -1;
const pyBool = (v) => (v ? 'True' : 'False');
const tup = (a) => (a.length === 1 ? '(' + a[0] + ',)' : '(' + a.join(', ') + ')');
/* python prints a float as 0.0, not 0 - only cosmetic, but it keeps the
   always-pass rows reading exactly the way the backend printed them */
const pyNum = (v) => {
  if (typeof v !== 'number') return String(v);
  /* repr(float('inf')) is 'inf' in python, String() gives 'Infinity' here */
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
};

/* an activation can be serialised as a name ("relu") or as a LAYER object
   ({"class_name": "LeakyReLU", "config": {...}}), which keras deserialises into
   an instance - that is the isinstance() path of activation_check(). */
function actName(a) {
  if (a === null || a === undefined) return 'linear';
  if (typeof a === 'string') return a;
  if (typeof a === 'object') {
    /* keras >= 2.13 serialises a plain activation FUNCTION as
         {"module":"keras.activations","class_name":"function",
          "config":"relu","registered_name":"relu"}
       and _retrieve_class_or_fn() resolves it through `config`, never through
       class_name.  Reading class_name here produced the literal string
       'function', which is in no allow-list. */
    if (String(a.class_name) === 'function') {
      const fn = a.config != null ? a.config : a.registered_name;
      if (fn != null) return String(fn);
    }
    if (a.class_name) return String(a.class_name);
    if (a.config && a.config.activation) return String(a.config.activation);
  }
  return String(a);
}
function actConfig(a) {
  return a && typeof a === 'object' && a.config ? a.config : null;
}

/* a scalar in a config means "the same value on every spatial axis" - keras'
   conv_utils.normalize_tuple() had already expanded it before the python read
   layer.kernel_size, so `3` on a Conv2D must become (3, 3) here too or
   kernel_check() would take its 1D branch and print the wrong message. */
function rankList(v, n, fallback) {
  if (v === null || v === undefined) return fallback;
  if (typeof v === 'number') return new Array(n).fill(v);
  return Array.isArray(v) ? v.slice() : fallback;
}

/* keras 2.x writes `batch_input_shape`; keras 3 writes `batch_shape`. */
function batchShape(cfg) {
  return (cfg && (cfg.batch_input_shape || cfg.batch_shape)) || null;
}

/* Raised where check_model_compatibility() itself raises - the model loads, but
   the backend cannot produce a report for it, so neither can this port.  Kept
   distinct from a load failure so the two are not reported with the same reason. */
class ReportError extends Error {}

/* ---------------------------------------------------------------------------
 * THE CHECK FUNCTIONS - one per  *_check()  in the python, same strings
 * ------------------------------------------------------------------------ */
const symmetry = (o) => (o[0] === o[1] ? 'Symmetric' : 'Asymmetric');
const fetchShape = (df, ips) => (df === 'channels_first' ? ips.slice(1) : ips.slice(0, -1));

const kernelCheck = (k) =>
  k.length === 1
    ? k[0] > 400
      ? 'Kernel Size Not Supported, Try Using Less than 400.'
      : 'Kernel Size Supported.'
    : k[0] * k[1] > 400
      ? 'Kernel Size Not Supported, ' + symmetry(k) + ' Kernel Supported, Try Using K1 * K2 <= 400.'
      : 'Kernel Size Supported, ' + symmetry(k) + ' Kernel Supported.';

const poolCheck = (k) =>
  k.length === 1
    ? k[0] > 400
      ? 'Pool Size Not Supported, Try Using Less than 400.'
      : 'Pool Size Supported.'
    : k[0] * k[1] > 400
      ? 'Pool Size Not Supported, ' + symmetry(k) + ' Pool Supported, Try Using P1 * P2 <= 400.'
      : 'Pool Size Supported, ' + symmetry(k) + ' Pool Supported.';

/* the body of strides_check is commented out in the python: it can only ever
   return "Supported".  Kept verbatim so the row still occupies its position. */
const stridesCheck = (s) =>
  s.length === 1 ? 'Stride Length Supported.' : 'Stride Length Supported, ' + symmetry(s) + ' Stride Supported.';

const paddingCheck = (p) =>
  ['valid', 'same'].indexOf(p) !== -1 ? 'Padding Supported.' : "Padding Type '" + p + "' Not Supported, Try 'valid' or 'same'.";

const dilationCheck = (r) =>
  (r.length === 2 && r[0] === 1 && r[1] === 1) || (r.length === 1 && r[0] === 1)
    ? 'Dilation Supported.'
    : 'Dilation ' + tup(r) + ' Not Supported, Try using 1.';

const groupCheck = (g) => (g === 1 ? 'Groups Supported.' : 'Groups ' + g + ' Not Supported, Try using 1.');
const inputCheck = (s) => (s.length === 1 ? 'Input Supported.' : symmetry(s) + ' Input Supported.');
const trainableCheck = (v) => pyBool(v) + ' Trainable Supported.';
const momentumCheck = (v) => pyNum(v) + ' Momentum Supported.';
const epsilonCheck = (v) => pyNum(v) + ' Epsilon Supported.';
const centerCheck = (v) => (v ? pyBool(v) + ' Center Supported.' : pyBool(v) + ' Center Not Supported, try True.');
const scaleCheck = (v) => (v ? pyBool(v) + ' Scale Supported.' : pyBool(v) + ' Scale Not Supported, try True.');
const ufbCheck = (v) => (v ? pyBool(v) + ' Unit Forget Bias Supported.' : pyBool(v) + ' Unit Forget Bias Not Supported, try True.');
const rsCheck = (v) => pyBool(v) + ' Return Sequences Supported.';
/* return_state_check prints "Go Backwards" - the python's own copy-paste, kept */
const returnStateCheck = (v) => (!v ? pyBool(v) + ' Go Backwards Supported.' : pyBool(v) + ' Go Backwards Not Supported, try False.');
const goBackCheck = (v) => (!v ? pyBool(v) + ' Go Backwards Supported.' : pyBool(v) + ' Go Backwards Not Supported, try False.');
const statefulCheck = (v) => (!v ? pyBool(v) + ' Stateful Supported.' : pyBool(v) + ' Stateful Not Supported, try False.');
const unrollCheck = (v) => (!v ? pyBool(v) + ' Unroll Supported.' : pyBool(v) + ' Unroll Not Supported, try False.');
const dropoutCheck = (v) => pyNum(v) + ' Dropout Supported.';
const recDropoutCheck = (v) => pyNum(v) + ' Recurrent Dropout Supported.';
const alphaCheck = (v) => pyNum(v) + ' Supported.';
const lstmNActCheck = (a) => (LSTM_ACT.indexOf(a) !== -1 ? 'Activation Supported.' : 'Activation ' + a + ' Not Supported, Try Using a Different Activation.');
const lstmRActCheck = (a) => (LSTM_REC_ACT.indexOf(a) !== -1 ? 'Activation Supported.' : 'Activation ' + a + ' Not Supported, Try Using a Different Activation.');

/* -------- shape inference (replaces what keras computed on the server) ------ */
function convOut(inp, cfg, rank, cls) {
  const df = cfg.data_format || 'channels_last';
  const k = rankList(cfg.kernel_size, rank, new Array(rank).fill(1));
  const st = rankList(cfg.strides, rank, new Array(rank).fill(1));
  const dl = rankList(cfg.dilation_rate, rank, new Array(rank).fill(1));
  const pad = cfg.padding || 'valid';
  const spatial = df === 'channels_first' ? inp.slice(2, 2 + rank) : inp.slice(1, 1 + rank);
  const out = spatial.map((d, i) => {
    if (d === null || d === undefined) return null;
    const eff = (k[i] - 1) * dl[i] + 1;
    return pad === 'same' || pad === 'causal' ? Math.ceil(d / st[i]) : Math.floor((d - eff) / st[i]) + 1;
  });
  const inCh = df === 'channels_first' ? inp[1] : inp[inp.length - 1];
  const ch =
    cls === 'DepthwiseConv2D' || cls === 'DepthwiseConv1D' ? (inCh || 1) * (cfg.depth_multiplier || 1) : cfg.filters;
  return df === 'channels_first' ? [inp[0], ch].concat(out) : [inp[0]].concat(out, [ch]);
}

function poolOut(inp, cfg, rank) {
  const df = cfg.data_format || 'channels_last';
  const ps = rankList(cfg.pool_size, rank, new Array(rank).fill(2));
  const st = rankList(cfg.strides == null ? cfg.pool_size : cfg.strides, rank, ps);
  const pad = cfg.padding || 'valid';
  const spatial = df === 'channels_first' ? inp.slice(2, 2 + rank) : inp.slice(1, 1 + rank);
  const out = spatial.map((d, i) => {
    if (d === null || d === undefined) return null;
    return pad === 'same' ? Math.ceil(d / st[i]) : Math.floor((d - ps[i]) / st[i]) + 1;
  });
  return df === 'channels_first' ? [inp[0], inp[1]].concat(out) : [inp[0]].concat(out, [inp[inp.length - 1]]);
}

function inferShapes(layers, modelInput) {
  let cur = modelInput ? modelInput.slice() : null;
  for (const L of layers) {
    const cls = canon(L.class_name);
    const cfg = L.config || {};
    const declared = batchShape(cfg);
    if (!cur && declared) cur = declared.slice();
    L.input_shape = cur ? cur.slice() : [null];
    let out = cur ? cur.slice() : [null];
    try {
      if (['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D'].indexOf(cls) !== -1) out = convOut(cur, cfg, 2, cls);
      else if (['Conv1D', 'DepthwiseConv1D', 'SeparableConv1D'].indexOf(cls) !== -1) out = convOut(cur, cfg, 1, cls);
      else if (['MaxPooling2D', 'AveragePooling2D'].indexOf(cls) !== -1) out = poolOut(cur, cfg, 2);
      else if (['MaxPooling1D', 'AveragePooling1D'].indexOf(cls) !== -1) out = poolOut(cur, cfg, 1);
      else if (cls === 'Flatten') {
        const rest = cur.slice(1);
        out = [cur[0], rest.some((d) => d === null || d === undefined) ? null : rest.reduce((a, b) => a * b, 1)];
      } else if (cls === 'Reshape') {
        /* resolve a -1 in target_shape so downstream ranks stay real */
        const target = (cfg.target_shape || []).slice();
        const known = target.filter((d) => d >= 0).reduce((a, b) => a * b, 1);
        const total = cur.slice(1).some((d) => d === null || d === undefined)
          ? null
          : cur.slice(1).reduce((a, b) => a * b, 1);
        out = [cur[0]].concat(target.map((d) => (d < 0 ? (total && known ? total / known : null) : d)));
      } else if (cls === 'Permute') out = [cur[0]].concat((cfg.dims || []).map((d) => cur[d]));
      else if (cls === 'Dense') out = cur.slice(0, -1).concat([cfg.units]);
      else if (cls === 'LSTM') out = cfg.return_sequences ? [cur[0], cur[1], cfg.units] : [cur[0], cfg.units];
    } catch (e) {
      out = cur ? cur.slice() : [null];
    }
    L.output_shape = out;
    cur = out;
  }
}

/* ============================================================================
 * 2b. THE LOAD - what keras.models.load_model() would have rejected
 *
 *   load_model() runs four stages and each one raises differently:
 *     1. cls(**config)                  -> wrapped by Layer.from_config()
 *     2. assert_input_compatibility     -> raw, and it runs BEFORE build()
 *     3. build() / compute_output_shape -> raw
 *     4. batch_set_value(weights)       -> raw, after the whole graph is built
 *
 *   The python backend got all of this free from load_model().  Every string
 *   below is copied from keras 2.15.0 / tensorflow 2.15.0, typos included.
 * ========================================================================== */

/* A JSON parse that keeps every number as its source text.  json.loads() gives
   python a float for 0.0 and repr() prints "0.0"; JSON.parse gives 0. */
const RAWNUM = '\u0000#';
function parseRawJson(text) {
  let i = 0;
  const ws = () => {
    while (i < text.length && ' \t\n\r'.indexOf(text[i]) !== -1) i++;
  };
  const str = () => {
    i++;
    let out = '';
    while (i < text.length && text[i] !== '"') {
      if (text[i] === '\\') {
        const e = text[i + 1];
        out +=
          e === 'n' ? '\n' : e === 't' ? '\t' : e === 'r' ? '\r'
            : e === 'b' ? '\b' : e === 'f' ? '\f'
              : e === 'u' ? String.fromCharCode(parseInt(text.substr(i + 2, 4), 16)) : e;
        i += e === 'u' ? 6 : 2;
        continue;
      }
      out += text[i++];
    }
    i++;
    return out;
  };
  const val = () => {
    ws();
    const c = text[i];
    if (c === '{') {
      i++;
      const o = {};
      ws();
      if (text[i] === '}') { i++; return o; }
      for (;;) {
        ws();
        const k = str();
        ws();
        i++;
        o[k] = val();
        ws();
        if (text[i] === ',') { i++; continue; }
        i++;
        return o;
      }
    }
    if (c === '[') {
      i++;
      const a = [];
      ws();
      if (text[i] === ']') { i++; return a; }
      for (;;) {
        a.push(val());
        ws();
        if (text[i] === ',') { i++; continue; }
        i++;
        return a;
      }
    }
    if (c === '"') return str();
    if (text.substr(i, 4) === 'true') { i += 4; return true; }
    if (text.substr(i, 5) === 'false') { i += 5; return false; }
    if (text.substr(i, 4) === 'null') { i += 4; return null; }
    /* json.loads() accepts the three literals python's json module emits for
       non-finite floats; JSON.parse() throws on all of them, which used to sink
       the whole config.  They are kept as raw text so pyRepr prints what python
       would have printed (inf / -inf / nan). */
    if (text.substr(i, 9) === '-Infinity') { i += 9; return RAWNUM + '-inf'; }
    if (text.substr(i, 8) === 'Infinity') { i += 8; return RAWNUM + 'inf'; }
    if (text.substr(i, 3) === 'NaN') { i += 3; return RAWNUM + 'nan'; }
    const st = i;
    while (i < text.length && '-+.eE0123456789'.indexOf(text[i]) !== -1) i++;
    return RAWNUM + text.slice(st, i);
  };
  try {
    return val();
  } catch (e) {
    return null;
  }
}

/* the literal-preserving tree turned back into ordinary values, used as the
   fallback config when JSON.parse() refused the text outright */
function fromRaw(v) {
  if (typeof v === 'string' && v.indexOf(RAWNUM) === 0) {
    const t = v.slice(RAWNUM.length);
    return t === 'inf' ? Infinity : t === '-inf' ? -Infinity : t === 'nan' ? NaN : Number(t);
  }
  if (Array.isArray(v)) return v.map(fromRaw);
  if (v && typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v)) o[k] = fromRaw(v[k]);
    return o;
  }
  return v;
}

function pyRepr(v) {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'string') {
    if (v.indexOf(RAWNUM) === 0) return v.slice(RAWNUM.length);
    return "'" + v.replace(/\\/g, '\\\\').replace(/'/g, "\\'") + "'";
  }
  if (Array.isArray(v)) return '[' + v.map(pyRepr).join(', ') + ']';
  if (typeof v === 'object') return '{' + Object.keys(v).map((k) => pyRepr(k) + ': ' + pyRepr(v[k])).join(', ') + '}';
  return String(v);
}
const pyTuple = (t) => (t.length === 1 ? '(' + t[0] + ',)' : '(' + t.join(', ') + ')');
const pyShape = (s) => pyTuple(s.map((d) => (d === null || d === undefined ? 'None' : d)));
const pyShapeList = (s) => '[' + s.map((d) => (d === null || d === undefined ? 'None' : d)).join(', ') + ']';

/* python prints the model CLASS, not its name: f"{type(model)}" */
const MODEL_CLASS_PATH = {
  Functional: 'keras.src.engine.functional.Functional',
  Model: 'keras.src.engine.training.Model',
  Sequential: 'keras.src.engine.sequential.Sequential',
};
function pyClassRepr(className) {
  const path = MODEL_CLASS_PATH[canon(className)];
  return path ? "<class '" + path + "'>" : String(className);
}

function callWrapped(name, type, msg, inputShape, extraArgs) {
  /* the footer lists every argument of the layer's call() signature, so a layer
     whose call() takes more than `inputs` gets one bullet per argument */
  let args = '  \u2022 inputs=tf.Tensor(shape=' + pyShape(inputShape) + ', dtype=float32)';
  for (const a of extraArgs || []) args += '\n  \u2022 ' + a;
  return (
    'Exception encountered when calling layer "' + name + '" (type ' + type + ').\n\n' +
    msg + '\n\nCall arguments received by layer "' + name + '" (type ' + type + '):\n' +
    args
  );
}

const CONV_RANK = {
  Conv1D: 1, SeparableConv1D: 1, DepthwiseConv1D: 1,
  Conv2D: 2, SeparableConv2D: 2, DepthwiseConv2D: 2,
};
const POOL_RANK = {
  MaxPooling1D: 1, AveragePooling1D: 1,
  MaxPooling2D: 2, AveragePooling2D: 2,
};
const DROPOUT_RATE_LAYERS = ['Dropout', 'SpatialDropout1D', 'SpatialDropout2D'];

/* A key the config does not carry is never passed to cls(**config) at all, so
   python binds the signature default.  A key present as null IS passed, and
   keras raises on it - so only `undefined` is substituted here. */
const dflt = (v, d) => (v === undefined ? d : v);

/* conv_utils.normalize_tuple -> { tuple } when it returns, { error } when it raises */
function normalizeTuple(value, n, name, allowZero) {
  let msg = 'The `' + name + '` argument must be a tuple of ' + n + ' integers. Received: ' + pyRepr(value);
  let t;
  if (typeof value === 'number') {
    t = new Array(n).fill(value);
  } else {
    if (!Array.isArray(value)) return { error: msg };
    t = value.slice();
    if (t.length !== n) return { error: msg };
  }
  const bad = t.filter((v) => (allowZero ? v < 0 : v <= 0));
  if (bad.length) {
    const uniq = bad.filter((v, i) => bad.indexOf(v) === i).sort((a, b) => a - b);
    msg += ' including {' + uniq.join(', ') + '} that does not satisfy the requirement `' + (allowZero ? '>= 0' : '> 0') + '`.';
    return { error: msg };
  }
  return { tuple: t };
}

/* ---- stage 1: Conv.__init__ then Conv._validate_init(), in keras' order ---- */
function convInitError(cls, cfg) {
  const rank = CONV_RANK[canon(cls)];
  if (!rank) return null;

  const filters = cfg.filters === undefined ? null : cfg.filters;
  if (filters !== null && filters <= 0) {
    return 'Invalid value for argument `filters`. Expected a strictly positive value. Received filters=' + filters + '.';
  }
  const groups = cfg.groups || 1;

  /* An ABSENT key is not the same as an explicit null: `cls(**config)` simply
     never passes the argument and python binds the signature default.  Reading
     `undefined` as None made every optional key mandatory, so any config not
     written by keras itself - trimmed, hand-edited, produced by a converter -
     was rejected for a value the backend fills in silently. */
  const k = normalizeTuple(cfg.kernel_size, rank, 'kernel_size', false);
  if (k.error) return k.error;
  const s = normalizeTuple(dflt(cfg.strides, 1), rank, 'strides', true);
  if (s.error) return s.error;

  const padding = String(dflt(cfg.padding, 'valid') == null ? '' : dflt(cfg.padding, 'valid')).toLowerCase();
  if (['valid', 'same', 'causal'].indexOf(padding) === -1) {
    return 'The `padding` argument must be a list/tuple or one of "valid", "same" (or "causal", only for `Conv1D). Received: ' + padding;
  }
  const df = cfg.data_format == null ? 'channels_last' : cfg.data_format;
  if (['channels_first', 'channels_last'].indexOf(String(df).toLowerCase()) === -1) {
    return 'The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + df;
  }
  const d = normalizeTuple(dflt(cfg.dilation_rate, 1), rank, 'dilation_rate', true);
  if (d.error) return d.error;

  if (filters !== null && filters % groups !== 0) {
    return 'The number of filters must be evenly divisible by the number of groups. Received: groups=' + groups + ', filters=' + filters;
  }
  if (k.tuple.indexOf(0) !== -1) {
    return 'The argument `kernel_size` cannot contain 0(s). Received: ' + pyTuple(k.tuple);
  }
  if (s.tuple.indexOf(0) !== -1) {
    /* "contains" is keras' own typo */
    return 'The argument `strides` cannot contains 0(s). Received: ' + pyTuple(s.tuple);
  }
  if (padding === 'causal' && ['Conv1D', 'SeparableConv1D'].indexOf(canon(cls)) === -1) {
    return 'Causal padding is only supported for `Conv1D`and `SeparableConv1D`.';
  }
  if (Math.max.apply(null, s.tuple) > 1 && Math.max.apply(null, d.tuple) > 1) {
    return '`strides > 1` not supported in conjunction with `dilation_rate > 1`. Received: strides=' + pyTuple(s.tuple) + ' and dilation_rate=' + pyTuple(d.tuple);
  }
  return null;
}

/* ---- stage 1: pooling.  No _validate_init, so 0 strides survive __init__ ---- */
function poolInitError(cls, cfg) {
  const rank = POOL_RANK[canon(cls)];
  if (!rank) return null;
  const poolSize = dflt(cfg.pool_size, 2);
  const p = normalizeTuple(poolSize, rank, 'pool_size', false);
  if (p.error) return p.error;
  const s = normalizeTuple(cfg.strides == null ? poolSize : cfg.strides, rank, 'strides', true);
  if (s.error) return s.error;
  const pad = String(dflt(cfg.padding, 'valid') == null ? '' : dflt(cfg.padding, 'valid')).toLowerCase();
  if (['valid', 'same', 'causal'].indexOf(pad) === -1) {
    return 'The `padding` argument must be a list/tuple or one of "valid", "same" (or "causal", only for `Conv1D). Received: ' + pad;
  }
  const df = cfg.data_format == null ? 'channels_last' : cfg.data_format;
  if (['channels_first', 'channels_last'].indexOf(String(df).toLowerCase()) === -1) {
    return 'The `data_format` argument must be one of "channels_first", "channels_last". Received: ' + df;
  }
  return null;
}

/* ---- stage 1: everything else that validates its arguments ---- */
function coreInitError(cls, cfg, rawCfg) {
  const c = canon(cls);

  /* Dense tests `< 0`, NOT `<= 0` - units=0 loads and reaches the report */
  if (c === 'Dense' && typeof cfg.units === 'number' && cfg.units < 0) {
    return 'Received an invalid value for `units`, expected a positive integer. Received: units=' + cfg.units;
  }
  /* LSTM tests `<= 0`, so units=0 fails here but not in Dense */
  if (c === 'LSTM' && 'units' in cfg) {
    /* `if units <= 0:` is the first line of LSTM.__init__.  Against a str or a
       None that comparison is a TypeError, not a ValueError - the port only
       ever ran it when the value was already a number, so a string unit count
       loaded here and died in the backend. */
    if (typeof cfg.units !== 'number') {
      return "'<=' not supported between instances of '"
        + (cfg.units === null || cfg.units === undefined ? 'NoneType' : typeof cfg.units === 'string' ? 'str' : 'dict')
        + "' and 'int'";
    }
    if (cfg.units <= 0) {
      return 'Received an invalid value for argument `units`, expected a positive integer, got ' + cfg.units + '.';
    }
  }
  if (c === 'ReLU') {
    const mv = cfg.max_value;
    if (mv !== null && mv !== undefined && mv < 0) {
      return 'max_value of a ReLU layer cannot be a negative value. Received: ' + rawNum(rawCfg, 'max_value', mv);
    }
    const ns = cfg.negative_slope === undefined ? 0.0 : cfg.negative_slope;
    if (ns === null || ns < 0) {
      return 'negative_slope of a ReLU layer cannot be a negative value. Received: ' + (ns === null ? 'None' : rawNum(rawCfg, 'negative_slope', ns));
    }
    const th = cfg.threshold === undefined ? 0.0 : cfg.threshold;
    if (th === null || th < 0) {
      return 'threshold of a ReLU layer cannot be a negative value. Received: ' + (th === null ? 'None' : rawNum(rawCfg, 'threshold', th));
    }
  }
  if (c === 'LeakyReLU' && cfg.alpha === null) {
    return 'The alpha value of a Leaky ReLU layer cannot be None, Expecting a float. Received: None';
  }
  if (c === 'Permute' && Array.isArray(cfg.dims)) {
    const sorted = cfg.dims.slice().sort((a, b) => a - b).join(',');
    const want = cfg.dims.map((_, i) => i + 1).join(',');
    if (sorted !== want) {
      return 'Invalid permutation argument `dims` for Permute Layer. The set of indices in `dims` must be consecutive and start from 1. Received dims=' + pyRepr(cfg.dims);
    }
  }
  if (c === 'SpatialDropout2D') {
    const df = cfg.data_format == null ? 'channels_last' : cfg.data_format;
    if (['channels_last', 'channels_first'].indexOf(df) === -1) {
      return '`data_format` must be "channels_last" or "channels_first". Received: data_format=' + df + '.';
    }
  }
  /* GaussianDropout stores its rate WITHOUT checking it, like AlphaDropout and
     unlike Dropout - a rate outside [0, 1] loads and reaches the report. */
  if (c === 'Softmax' && 'axis' in cfg && cfg.axis !== null && cfg.axis !== undefined) {
    const ax = Array.isArray(cfg.axis) ? cfg.axis[0] : cfg.axis;
    if (typeof ax !== 'number') {
      return 'Expected int for argument \'axis\' not ' + pyRepr(ax) + '.';
    }
  }
  /* GaussianNoise stores its stddev unchecked, exactly like GaussianDropout's
     rate - a negative stddev loads and reaches the report. */
  return null;
}

function rawNum(rawCfg, key, fallback) {
  const v = rawCfg && rawCfg[key];
  if (typeof v === 'string' && v.indexOf(RAWNUM) === 0) return v.slice(RAWNUM.length);
  return String(fallback);
}

function initError(cls, cfg, rawCfg) {
  /* Layer.__init__ rejects an unknown keyword before it validates anything */
  const kw = unknownKwargError(cls, cfg);
  if (kw) return kw;
  /* `dynamic` is accepted as a named argument of Layer.__init__, but a TRUE
     value makes the layer eager: keras can then no longer infer a static output
     shape for it while building the Sequential graph.  dynamic=False is the
     default and is harmless. */
  if (cfg.dynamic === true) {
    return 'A layer with `dynamic=True` cannot be used to build a static graph. '
      + 'Remove `dynamic=True` or implement `compute_output_shape`.';
  }
  /* base_layer.Layer.__init__ type-checks `trainable` before it touches
     anything else:  if not isinstance(trainable, bool): raise TypeError.
     JSON keeps `true` and `1` distinct, so the value is visible here - a
     numeric trainable used to sail straight through. */
  if ('trainable' in cfg && cfg.trainable !== null && cfg.trainable !== undefined
    && typeof cfg.trainable !== 'boolean') {
    return 'Expected `trainable` argument to be a boolean, but got: ' + pyRepr(cfg.trainable);
  }
  /* _init_set_name() is the very next thing Layer.__init__ runs */
  if ('name' in cfg && cfg.name !== null && cfg.name !== undefined && typeof cfg.name !== 'string') {
    return 'Expected `name` to be a string. Received: name=' + pyRepr(cfg.name)
      + ' (of type ' + pyType(cfg.name) + ')';
  }
  const declaredShape = cfg.batch_input_shape || cfg.batch_shape;
  if (Array.isArray(declaredShape)) {
    const dim = shapeDimError(declaredShape);
    if (dim) return dim;
    /* the InputLayer builds its placeholder inside __init__, so a float that
       only the source text reveals dies here rather than at build time */
    const rawShape = rawCfg && (rawCfg.batch_input_shape || rawCfg.batch_shape);
    const rf = rawFloat(rawShape);
    if (rf) return DIM_INT(rf);
  }
  if (DROPOUT_RATE_LAYERS.indexOf(canon(cls)) !== -1) {
    const r = cfg.rate;
    if (typeof r === 'number' && !(r >= 0 && r <= 1)) {
      return 'Invalid value ' + rawNum(rawCfg, 'rate', r) + ' received for `rate`, expected a value between 0 and 1.';
    }
  }
  const conv = convInitError(cls, cfg);
  if (conv) return conv;
  const pool = poolInitError(cls, cfg);
  if (pool) return pool;
  const core = coreInitError(cls, cfg, rawCfg);
  if (core) return core;
  return null;
}

/* ---- stage 2: Layer.__call__ asserts input_spec BEFORE _maybe_build ---- */
const SPEC_NDIM = {
  SpatialDropout1D: 3, SpatialDropout2D: 4,
  MaxPooling1D: 3, AveragePooling1D: 3,
  MaxPooling2D: 4, AveragePooling2D: 4,
  /* RNN.__init__ sets InputSpec(ndim=3); without this a recurrent layer handed a
     rank-2 tensor sailed through here and failed only in python */
  LSTM: 3,
};
const SPEC_MIN_NDIM = {
  Dense: 2, Flatten: 1,
  Conv1D: 3, DepthwiseConv1D: 3, SeparableConv1D: 3,
  Conv2D: 4, DepthwiseConv2D: 4, SeparableConv2D: 4,
};

function inputSpecError(cls, cfg, inputShape) {
  const c = canon(cls);
  const ndim = (inputShape || []).length;
  if (!ndim) return null;
  const name = cfg.name || c;
  const incompatible = (kind, want) =>
    'Input 0 of layer "' + name + '" is incompatible with the layer: expected ' + kind + '=' + want +
    ', found ndim=' + ndim + '. Full shape received: ' + pyShape(inputShape);

  let want = SPEC_NDIM[c];
  if (c === 'Permute' && Array.isArray(cfg.dims)) want = cfg.dims.length + 1;
  if (want !== undefined && ndim !== want) return incompatible('ndim', want);

  const min = SPEC_MIN_NDIM[c];
  if (min !== undefined && ndim < min) return incompatible('min_ndim', min);
  return null;
}

function poolStrideZero(c, cfg, full, pool, strides, pad) {
  const name = cfg.name || '';
  const op = c.indexOf('Average') === 0 ? 'AvgPool' : 'MaxPool';
  const rank = POOL_RANK[c];
  const cf = (cfg.data_format || 'channels_last') === 'channels_first';
  const df = cf ? 'NCHW' : 'NHWC';
  const spatial = cf ? full.slice(2) : full.slice(1, -1);
  const ch = cf ? full[1] : full[full.length - 1];

  const sp = rank === 1 ? [spatial[0], 1] : spatial.slice();
  const ks = rank === 1 ? [pool[0], 1] : pool.slice();
  const sd = rank === 1 ? [strides[0], 1] : strides.slice();
  const nchw = (a) => (cf ? [1, 1].concat(a) : [1].concat(a, [1]));
  const shape = cf ? ['?', ch].concat(sp) : ['?'].concat(sp, [ch]);
  const src = rank === 1 ? name + '/ExpandDims' : 'Placeholder';
  const explicit = op === 'MaxPool' ? 'explicit_paddings=[], ' : '';

  const msg =
    "Stride must be > 0, but got 0 for '{{node " + name + '/' + op + '}} = ' + op +
    '[T=DT_FLOAT, data_format="' + df + '", ' + explicit +
    'ksize=[' + nchw(ks).join(', ') + '], padding="' + String(pad).toUpperCase() +
    '", strides=[' + nchw(sd).join(', ') + ']](' + src + ")' with input shapes: [" + shape.join(',') + '].';
  return callWrapped(name, c, msg, full);
}

/* keras' own "One of the dimensions in the output is <= 0" never gets a chance to
   fire for valid padding: the TF kernel refuses to build the op first, and its
   trace is what the python surfaced.  Same shape of message for convolution and
   pooling, so both are built here. */
function negativeDimError(c, cfg, full, k, st, dl, pad, dim, eff) {
  const name = cfg.name || '';
  const isPool = !!POOL_RANK[c];
  const rank = CONV_RANK[c] || POOL_RANK[c];
  const cf = (cfg.data_format || 'channels_last') === 'channels_first';
  const df = cf ? 'NCHW' : 'NHWC';
  const spatial = cf ? full.slice(2) : full.slice(1, -1);
  const ch = cf ? full[1] : full[full.length - 1];
  /* a 1D op is run as its 2D equivalent with a length-1 axis expanded in front */
  const pre = rank === 1 ? [1] : [];
  const sp = pre.concat(spatial);
  const ks = pre.concat(k);
  const sd = pre.concat(st);
  const dls = pre.concat(dl || new Array(rank).fill(1));
  const nchw = (a) => (cf ? [1, 1].concat(a) : [1].concat(a, [1]));
  const q = (v) => (v === null || v === undefined ? '?' : v);
  const shape = (cf ? ['?', ch].concat(sp) : ['?'].concat(sp, [ch])).map(q).join(',');
  const head = 'Negative dimension size caused by subtracting ' + eff + ' from ' + dim + ' for ';

  if (isPool) {
    const op = c.indexOf('Average') === 0 ? 'AvgPool' : 'MaxPool';
    const explicit = op === 'MaxPool' ? 'explicit_paddings=[], ' : '';
    const src = rank === 1 ? name + '/ExpandDims' : 'Placeholder';
    return callWrapped(name, c,
      head + "'{{node " + name + '/' + op + '}} = ' + op + '[T=DT_FLOAT, data_format="' + df +
      '", ' + explicit + 'ksize=[' + nchw(ks).join(', ') + '], padding="' + String(pad).toUpperCase() +
      '", strides=[' + nchw(sd).join(', ') + ']](' + src + ")' with input shapes: [" + shape + '].',
      full);
  }
  const node = name + '/' + c;
  const src = rank === 1
    ? node + '/ExpandDims, ' + node + '/ExpandDims_1'
    : 'Placeholder, ' + node + '/ReadVariableOp';
  return callWrapped(name, c,
    head + "'{{node " + node + '}} = Conv2D[T=DT_FLOAT, data_format="' + df +
    '", dilations=[' + nchw(dls).join(', ') + '], explicit_paddings=[], padding="' +
    String(pad).toUpperCase() + '", strides=[' + nchw(sd).join(', ') +
    '], use_cudnn_on_gpu=true](' + src + ")' with input shapes: [" + shape + '], [' +
    ks.concat([q(ch), q(cfg.filters)]).join(',') + '].',
    full);
}

function convOutputLength(len, filt, pad, stride, dil) {
  if (len === null || len === undefined) return null;
  const d = filt + (filt - 1) * ((dil || 1) - 1);
  let o;
  if (pad === 'same' || pad === 'causal') o = len;
  else if (pad === 'valid') o = len - d + 1;
  else if (pad === 'full') o = len + d - 1;
  else return null;
  if (stride === 0) return 'ZERODIV';
  return Math.floor((o + stride - 1) / stride);
}

/* ---- stage 3: build() and compute_output_shape(), raised raw ---- */
function buildError(cls, cfg, inputShape) {
  const c = canon(cls);
  const full = inputShape || [];

  if (c === 'Reshape') {
    const inp = full.slice(1);
    if (!inp.length || inp.some((d) => d === null || d === undefined)) return null;
    const target = (cfg.target_shape || []).slice();
    if (!target.length) return null;

    const wrap = (m) => callWrapped(cfg.name, 'Reshape', m, full);
    const msg = wrap('total size of new array must be unchanged, input_shape = ' + pyShapeList(inp) + ', output_shape = ' + pyShapeList(target));

    let known = 1;
    let unknown = null;
    for (let i = 0; i < target.length; i++) {
      if (target[i] < 0) {
        if (unknown === null) unknown = i;
        else return wrap('There must be at most one unknown dimension in output_shape. Received: output_shape=' + pyShapeList(target) + '.');
      } else {
        known *= target[i];
      }
    }
    const original = inp.reduce((a, b) => a * b, 1);
    if (unknown !== null) {
      if (known === 0 || original % known !== 0) return msg;
    } else if (original !== known) {
      return msg;
    }
  }

  if ((c === 'DepthwiseConv1D' || c === 'DepthwiseConv2D') && full.length && full.length !== CONV_RANK[c] + 2) {
    return 'Inputs to `DepthwiseConv` should have rank ' + (CONV_RANK[c] + 2) + '. Received input_shape=' + pyShape(full) + '.';
  }
  if (CONV_RANK[c] && full.length && (c.indexOf('Depthwise') !== -1 || c.indexOf('Separable') !== -1)) {
    const dwDf = cfg.data_format || 'channels_last';
    const chAxis = dwDf === 'channels_first' ? -1 - CONV_RANK[c] : -1;
    const dwCh = dwDf === 'channels_first' ? full[1] : full[full.length - 1];
    if (dwCh === null || dwCh === undefined) {
      return (
        (c.indexOf('Depthwise') !== -1
          ? 'The channel dimension of the inputs to `DepthwiseConv` should be defined. '
          : 'The channel dimension of the inputs should be defined. ') +
        'The input_shape received is ' + pyShape(full) + ', where axis ' + chAxis +
        ' (0-based) is the channel dimension, which found to be `None`.'
      );
    }
  }

  const rank = CONV_RANK[c] || POOL_RANK[c];
  if (rank && full.length === rank + 2) {
    const df = cfg.data_format || 'channels_last';
    const spatial = df === 'channels_first' ? full.slice(2) : full.slice(1, -1);
    const ch = df === 'channels_first' ? full[1] : full[full.length - 1];
    const isPool = !!POOL_RANK[c];
    const k = isPool
      ? rankList(cfg.pool_size, rank, new Array(rank).fill(2))
      : rankList(cfg.kernel_size, rank, new Array(rank).fill(1));
    const st = isPool
      ? rankList(cfg.strides == null ? cfg.pool_size : cfg.strides, rank, k)
      : rankList(cfg.strides, rank, new Array(rank).fill(1));
    const dl = isPool ? new Array(rank).fill(1) : rankList(cfg.dilation_rate, rank, new Array(rank).fill(1));
    const pad = String(cfg.padding || 'valid').toLowerCase();
    const groups = cfg.groups || 1;

    if (!isPool && c.indexOf('Depthwise') === -1 && ch !== null && ch !== undefined && ch % groups !== 0) {
      return 'The number of input channels must be evenly divisible by the number of groups. Received groups=' + groups + ', but the input has ' + ch + ' channels (full input shape is ' + pyShape(full) + ').';
    }
    for (let i = 0; i < spatial.length; i++) {
      const o = convOutputLength(spatial[i], k[i], pad, st[i], dl[i]);
      if (o === 'ZERODIV') return poolStrideZero(c, cfg, full, k, st, pad);
      /* keras fails when a dimension reaches 0, not only when it goes negative */
      if (o !== null && o <= 0) {
        if (pad === 'valid') {
          const eff = k[i] + (k[i] - 1) * ((dl[i] || 1) - 1);
          return negativeDimError(c, cfg, full, k, st, dl, pad, spatial[i], eff);
        }
        return 'One of the dimensions in the output is <= 0 due to downsampling in ' + (cfg.name || '') + '. Consider increasing the input size. Received input shape ' + pyShapeList(full) + ' which would produce output shape with a zero or negative value in a dimension.';
      }
    }
  }

  if (c === 'Dense' && full.length) {
    const last = full[full.length - 1];
    if (last === null || last === undefined) {
      return 'The last dimension of the inputs to a Dense layer should be defined. Found None. Full input shape received: ' + pyShape(full);
    }
  }

  /* Softmax.call() hands self.axis to tf.nn.softmax, which range-checks it
     against the input rank.  call() -> the traceback wrapper applies, and
     Softmax.call(self, inputs, mask=None) means the footer lists mask too. */
  if (c === 'Softmax' && full.length) {
    const rank = full.length;
    const ax = cfg.axis;
    /* a 1-element list is unwrapped by call() and range-checked by tf.nn.softmax;
       a longer one takes the reduce_logsumexp path, which range-checks every
       entry of its own - the port used to skip that case entirely */
    const single = Array.isArray(ax) ? (ax.length === 1 ? ax[0] : null) : ax;
    if (typeof single === 'number' && (single < -rank || single >= rank)) {
      return callWrapped(cfg.name, 'Softmax',
        '`dim` must be in the range [-' + rank + ', ' + rank + ') where ' + rank
        + ' is the number of dimensions in the input. Received: dim=' + single,
        full, ['mask=None']);
    }
    if (Array.isArray(ax) && ax.length > 1) {
      for (const a of ax) {
        if (typeof a === 'number' && (a < -rank || a >= rank)) {
          const shp = '[' + full.map((d) => (d === null || d === undefined ? '?' : d)).join(',') + ']';
          return callWrapped(cfg.name, 'Softmax',
            'Invalid reduction dimension ' + a + ' for input with ' + rank
            + " dimensions. for '{{node " + cfg.name + '/ReduceLogSumExp/Max}} = Max[T=DT_FLOAT, '
            + 'Tidx=DT_INT32, keep_dims=true](Placeholder, ' + cfg.name
            + "/ReduceLogSumExp/Max/reduction_indices)' with input shapes: " + shp + ', ['
            + ax.length + '] and with computed input tensors: input[1] = <' + ax.join(' ') + '>.',
            full, ['mask=None']);
        }
      }
    }
  }

  if (c === 'BatchNormalization' && full.length) {
    const ax = typeof cfg.axis === 'number' ? [cfg.axis] : Array.isArray(cfg.axis) ? cfg.axis.slice() : null;
    if (ax) {
      const r = full.length;
      for (let i = 0; i < ax.length; i++) if (ax[i] < 0) ax[i] = r + ax[i];
      for (const x of ax) {
        if (x < 0 || x >= r) {
          return 'Invalid value for `axis` argument. Expected 0 <= axis < inputs.rank (with inputs.rank=' + r + '). Received: axis=' + pyTuple(ax);
        }
      }
      if (new Set(ax).size !== ax.length) return 'Duplicate axis: ' + pyTuple(ax);
      for (const x of ax) {
        if (full[x] === null || full[x] === undefined) {
          return 'Input has undefined `axis` dimension. Received input with shape ' + pyShape(full) + ' and axis=' + pyTuple(ax);
        }
      }
      if (cfg.fused === true) {
        if (r !== 4 && r !== 5) {
          return 'Batch normalization layers with `fused=True` only support 4D or 5D input tensors. Received tensor with shape: ' + pyShape(full);
        }
        if (r === 4 && !(ax.length === 1 && (ax[0] === 1 || ax[0] === 3))) {
          return 'Unsupported axis. The use of `fused=True` is only possible with `axis=1` or `axis=3` for 4D input tensors. Received: axis=' + pyTuple(ax);
        }
      }
    }
  }
  return null;
}

/* ---- stage 4: load_weights_from_hdf5_group -> batch_set_value ----
 *
 * The previous implementation started from the SAVED shape and substituted the
 * handful of axes it could name - the spatial axes and the filter/unit axis.
 * Every other axis was therefore never compared: a Conv2D kernel stored with the
 * wrong input-channel count, or with the wrong rank altogether, matched on every
 * axis the code looked at and loaded clean here while keras refused it.
 *
 * This builds the shape keras would have created from the config plus the
 * inferred input shape and compares the whole array, rank included.  An axis the
 * graph does not pin down stays null and is skipped, because keras cannot
 * disagree about a dimension it never knew either.
 *
 * The walk is POSITIONAL.  batch_set_value() zips the file's arrays against the
 * layer's symbolic weights in order, so a file whose weight_names are the wrong
 * way round hands a bias to a kernel - looking each array up by name hid that.
 * -------------------------------------------------------------------------- */
function expectedWeights(c, cfg, inputShape) {
  const inp = inputShape || [];
  const df = cfg.data_format || 'channels_last';
  const hasBias = cfg.use_bias !== false;
  const rank = CONV_RANK[c];
  const num = (v) => (typeof v === 'number' && isFinite(v) ? v : null);
  const mul = (a, b) => (a === null || b === null ? null : a * b);
  const chIn = rank ? num(df === 'channels_first' ? inp[1] : inp[inp.length - 1]) : null;
  const ks = () => rankList(cfg.kernel_size, rank, new Array(rank).fill(null)).map(num);
  const out = [];

  if (c === 'Dense') {
    out.push({ name: 'kernel', shape: [num(inp[inp.length - 1]), num(cfg.units)] });
    if (hasBias) out.push({ name: 'bias', shape: [num(cfg.units)] });
    return out;
  }
  if (rank && c.indexOf('Depthwise') === -1 && c.indexOf('Separable') === -1) {
    const groups = num(cfg.groups) || 1;
    out.push({ name: 'kernel', shape: ks().concat([chIn === null ? null : chIn / groups, num(cfg.filters)]) });
    if (hasBias) out.push({ name: 'bias', shape: [num(cfg.filters)] });
    return out;
  }
  if (c === 'DepthwiseConv1D' || c === 'DepthwiseConv2D') {
    const dm = cfg.depth_multiplier === undefined ? 1 : num(cfg.depth_multiplier);
    out.push({ name: 'depthwise_kernel', shape: ks().concat([chIn, dm]) });
    if (hasBias) out.push({ name: 'bias', shape: [mul(chIn, dm)] });
    return out;
  }
  if (c === 'SeparableConv1D' || c === 'SeparableConv2D') {
    const dm = cfg.depth_multiplier === undefined ? 1 : num(cfg.depth_multiplier);
    out.push({ name: 'depthwise_kernel', shape: ks().concat([chIn, dm]) });
    out.push({ name: 'pointwise_kernel', shape: new Array(rank).fill(1).concat([mul(chIn, dm), num(cfg.filters)]) });
    if (hasBias) out.push({ name: 'bias', shape: [num(cfg.filters)] });
    return out;
  }
  if (c === 'BatchNormalization') {
    const raw = typeof cfg.axis === 'number' ? cfg.axis : Array.isArray(cfg.axis) ? cfg.axis[0] : -1;
    const idx = raw < 0 ? inp.length + raw : raw;
    const dim = num(inp[idx]);
    if (cfg.scale !== false) out.push({ name: 'gamma', shape: [dim] });
    if (cfg.center !== false) out.push({ name: 'beta', shape: [dim] });
    out.push({ name: 'moving_mean', shape: [dim] });
    out.push({ name: 'moving_variance', shape: [dim] });
    return out;
  }
  if (c === 'LSTM') {
    const u = num(cfg.units);
    const four = u === null ? null : 4 * u;
    out.push({ name: 'kernel', shape: [num(inp[inp.length - 1]), four] });
    out.push({ name: 'recurrent_kernel', shape: [u, four] });
    if (hasBias) out.push({ name: 'bias', shape: [four] });
    return out;
  }
  return null; /* a type we cannot size - leave it to keras */
}

function weightError(layer, weights) {
  const c = canon(layer.class_name);
  const cfg = layer.config || {};
  const stored = weights && weights[cfg.name];
  if (!stored || !stored.length) return null;
  const want = expectedWeights(c, cfg, layer.input_shape);
  if (!want) return null;

  const CONV_TRANSPOSED = ['Conv1D', 'Conv2D'];
  const seenAs = (wname, shape) =>
    wname === 'kernel' && CONV_TRANSPOSED.indexOf(c) !== -1 && shape.length === 4
      ? [shape[3], shape[2], shape[0], shape[1]]
      : shape;

  for (let i = 0; i < want.length; i++) {
    const w = want[i];
    const got = stored[i];
    if (!got || !got.shape || !got.shape.length) continue;
    const have = got.shape;
    const rankDiffers = have.length !== w.shape.length;
    let differs = rankDiffers;
    if (!differs) {
      for (let n = 0; n < w.shape.length; n++) {
        const v = w.shape[n];
        if (v === null || v === undefined) continue; /* the graph never pinned this axis down */
        if (have[n] !== v) {
          differs = true;
          break;
        }
      }
    }
    if (!differs) continue;
    /* preprocess_weights_for_loading() transposes convolution kernels before the
       assignment, and numpy reports a shape it cannot transpose as an axes error
       rather than letting the assignment report a mismatch */
    if (CONV_RANK[c] && (rankDiffers || c === 'Conv1D')) return "axes don't match array";
    const shown = w.shape.map((d, n) => (d === null || d === undefined ? have[n] : d));
    return (
      "Cannot assign value to variable ' " + cfg.name + '/' + w.name + ":0': Shape mismatch." +
      'The variable shape ' + pyShape(shown) + ', and the assigned value shape '
      + pyShape(seenAs(w.name, have)) + ' are incompatible.'
    );
  }
  return null;
}

/* ---------------------------------------------------------------------------
 * load_model() also refuses a graph whose objects it cannot resolve at all:
 * an unregistered layer class, an unknown activation / initializer /
 * regularizer / constraint name, or a compile config naming a loss, metric or
 * optimizer it does not know.  In python every one of these raised inside
 * load_model() and the model was rejected with no layer table.
 * ------------------------------------------------------------------------ */
const KNOWN_LAYERS = [].concat(SUPP_LAYERS, [
  /* everything else keras 2.15 ships that our whitelist does not accept - these
     LOAD fine and must reach the report as "Try Removing This Layer" */
  'Conv3D', 'Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose', 'SeparableConv3D', 'DepthwiseConv3D',
  'MaxPooling3D', 'AveragePooling3D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalMaxPooling3D',
  'GlobalAveragePooling1D', 'GlobalAveragePooling2D', 'GlobalAveragePooling3D',
  'GRU', 'SimpleRNN', 'ConvLSTM1D', 'ConvLSTM2D', 'ConvLSTM3D', 'Bidirectional', 'TimeDistributed', 'RNN',
  'Embedding', 'Masking', 'Lambda', 'Dot', 'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum',
  'Concatenate', 'LayerNormalization', 'UnitNormalization', 'GroupNormalization', 'SpatialDropout3D',
  'ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D', 'Cropping1D', 'Cropping2D', 'Cropping3D',
  'UpSampling1D', 'UpSampling2D', 'UpSampling3D', 'RepeatVector', 'PReLU', 'ELU', 'ThresholdedReLU',
  'Attention', 'AdditiveAttention', 'MultiHeadAttention', 'Identity', 'Dropout', 'Wrapper',
  'Normalization', 'Rescaling', 'Resizing', 'CenterCrop', 'Discretization', 'Hashing', 'IntegerLookup',
  'StringLookup', 'TextVectorization', 'CategoryEncoding', 'RandomFlip', 'RandomRotation', 'RandomZoom',
  'RandomTranslation', 'RandomCrop', 'RandomContrast', 'RandomBrightness', 'RandomHeight', 'RandomWidth',
  'GaussianNoise', 'ActivityRegularization', 'Sequential', 'Functional', 'Model',
]);

const KNOWN_ACTIVATIONS = [
  'linear', 'relu', 'relu6', 'leaky_relu', 'elu', 'selu', 'gelu', 'silu', 'swish', 'mish',
  'softmax', 'softplus', 'softsign', 'sigmoid', 'hard_sigmoid', 'hard_silu', 'hard_swish',
  'tanh', 'exponential', 'log_softmax',
];

/* keras registers every one of these under BOTH its class name and its snake_case
   alias, and a model saved from code that passed strings ("glorot_uniform",
   "l2", "max_norm") stores the alias.  Listing only the class names rejected
   perfectly loadable files. */
const KNOWN_INITIALIZERS = [
  'Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling',
  'Orthogonal', 'Identity', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 'HeUniform', 'LecunNormal',
  'LecunUniform', 'OrthogonalInitializer', 'IdentityInitializer',
  'zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal',
  'variance_scaling', 'orthogonal', 'identity', 'glorot_normal', 'glorot_uniform', 'he_normal',
  'he_uniform', 'lecun_normal', 'lecun_uniform', 'zero', 'one', 'normal', 'uniform',
];
/* keras 2.15 also ships OrthogonalRegularizer, registered under both its class
   name and its snake_case alias.  Omitting it rejected models the backend loads
   without complaint - and Model_Compatibility_47.py never inspects regularizers
   at all, so a false rejection here buys nothing. */
const KNOWN_REGULARIZERS = ['L1', 'L2', 'L1L2', 'OrthogonalRegularizer',
  'l1', 'l2', 'l1_l2', 'orthogonal_regularizer'];
const KNOWN_CONSTRAINTS = [
  'MaxNorm', 'MinMaxNorm', 'NonNeg', 'UnitNorm', 'RadialConstraint',
  'max_norm', 'min_max_norm', 'non_neg', 'unit_norm', 'radial_constraint',
  /* keras/constraints.py ends with a block of LEGACY aliases - maxnorm,
     nonneg and unitnorm - bound to the same classes.  deserialize() looks the
     name up in that module's globals(), so all three resolve. */
  'maxnorm', 'nonneg', 'unitnorm',
];

/* tf.as_dtype() accepts its aliases too - "half" is float16, "double" is float64 */
const KNOWN_DTYPES = [
  'float16', 'float32', 'float64', 'bfloat16', 'half', 'float', 'double',
  'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
  'bool', 'string', 'complex64', 'complex128', 'qint8', 'qint16', 'qint32',
  'quint8', 'quint16', 'resource', 'variant', 'mixed_float16', 'mixed_bfloat16',
];

/* The keyword arguments each layer's __init__ actually accepts.  load_model()
   does `cls(**config)`, so a key outside this set is a TypeError and the model
   never loads - the port used to read only the keys it cared about and let
   everything else through.  Classes absent from the map (GRU, Conv3D, a custom
   layer) are left alone: they are outside the supported operator set anyway and
   guessing at their signatures would only invent false rejections. */
/* Layer.__init__'s own allowed_kwargs, verbatim.  These are constructor-time
   conveniences that get_config() never writes back, but validate_kwargs() still
   accepts every one of them, so a config carrying one loads normally - measured
   on input_dim (Dense and Conv1D), input_shape, batch_size, weights, autocast
   and implementation.

   `dynamic` is the exception and is NOT here: see dynamicError() below. */
const BASE_KWARGS = [
  'name', 'trainable', 'dtype', 'input_dim', 'input_shape', 'batch_input_shape',
  'batch_size', 'weights', 'activity_regularizer', 'autocast', 'implementation', 'dynamic',
];
const CONV_KWARGS = ['filters', 'kernel_size', 'strides', 'padding', 'data_format', 'dilation_rate',
  'groups', 'activation', 'use_bias', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer',
  'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint'];
/* DepthwiseConv and SeparableConv both forward **kwargs to Conv.__init__ and both
   inherit Conv.get_config(), so keras really does write `groups` and the whole
   `kernel_*` family into their configs and really does accept them back. */
const DW_KWARGS = ['kernel_size', 'strides', 'padding', 'depth_multiplier', 'data_format',
  'dilation_rate', 'activation', 'use_bias', 'depthwise_initializer', 'bias_initializer',
  'depthwise_regularizer', 'bias_regularizer', 'activity_regularizer', 'depthwise_constraint',
  'bias_constraint', 'groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint'];
const SEP_KWARGS = DW_KWARGS.concat(['filters', 'pointwise_initializer', 'pointwise_regularizer',
  'pointwise_constraint']);
const POOL_KWARGS = ['pool_size', 'strides', 'padding', 'data_format'];

const LAYER_KWARGS = {
  /* `optional` is not a keras 2.15 argument - it arrived in a later 2.x, and the
     backend this port has to agree with raises "Unrecognized keyword arguments:
     ['optional']" on a file that carries it.  Tolerating it here let a file pass
     that Model_Compatibility_47.py rejects.  Put it back only if the backend is
     moved off 2.15.0. */
  InputLayer: ['input_shape', 'batch_size', 'dtype', 'input_tensor', 'sparse', 'name', 'ragged',
    'type_spec', 'batch_input_shape'],
  Dense: ['units', 'activation', 'use_bias', 'kernel_initializer', 'bias_initializer',
    'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
    'bias_constraint'],
  Conv1D: CONV_KWARGS, Conv2D: CONV_KWARGS,
  DepthwiseConv1D: DW_KWARGS, DepthwiseConv2D: DW_KWARGS,
  SeparableConv1D: SEP_KWARGS, SeparableConv2D: SEP_KWARGS,
  MaxPooling1D: POOL_KWARGS, MaxPooling2D: POOL_KWARGS,
  AveragePooling1D: POOL_KWARGS, AveragePooling2D: POOL_KWARGS,
  Flatten: ['data_format'],
  Reshape: ['target_shape'],
  Permute: ['dims'],
  Dropout: ['rate', 'noise_shape', 'seed'],
  SpatialDropout1D: ['rate', 'noise_shape', 'seed'],
  SpatialDropout2D: ['rate', 'noise_shape', 'seed', 'data_format'],
  AlphaDropout: ['rate', 'noise_shape', 'seed'],
  GaussianDropout: ['rate', 'seed'],
  GaussianNoise: ['stddev', 'seed'],
  ActivityRegularization: ['l1', 'l2'],
  Activation: ['activation'],
  Softmax: ['axis'],
  ReLU: ['max_value', 'negative_slope', 'threshold'],
  LeakyReLU: ['alpha'],
  BatchNormalization: ['axis', 'momentum', 'epsilon', 'center', 'scale', 'beta_initializer',
    'gamma_initializer', 'moving_mean_initializer', 'moving_variance_initializer',
    'beta_regularizer', 'gamma_regularizer', 'beta_constraint', 'gamma_constraint', 'renorm',
    'renorm_clipping', 'renorm_momentum', 'fused', 'virtual_batch_size', 'adjustment',
    'synchronized'],
  LSTM: ['units', 'activation', 'recurrent_activation', 'use_bias', 'kernel_initializer',
    'recurrent_initializer', 'bias_initializer', 'unit_forget_bias', 'kernel_regularizer',
    'recurrent_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
    'recurrent_constraint', 'bias_constraint', 'dropout', 'recurrent_dropout', 'return_sequences',
    /* `implementation` is a real RNN argument and LSTM.get_config() writes it,
       so it belongs here rather than in BASE_KWARGS - on a Dense it is a key
       keras never emits. */
    'return_state', 'go_backwards', 'stateful', 'time_major', 'unroll', 'zero_output_for_mask',
    'implementation'],
};

/* The arguments each __init__ takes POSITIONALLY with no default.  load_model()
   does cls(**config), so a config missing one of these never reaches the layer
   body at all - python raises TypeError while binding the signature, before
   validate_kwargs() and before any activation or initializer is resolved.

   The port used to test only the values it found (`typeof cfg.units === 'number'`
   and friends), so an absent key slipped through and `undefined` then propagated
   into shape inference and into expectedWeights(), where it became a null axis
   the weight comparison skips - a broken model reported as clean.

   Classes absent from this map either have a default for every argument
   (pooling, BatchNormalization, Flatten, ReLU, Softmax, InputLayer) or are
   outside the supported set, where guessing at a signature would only invent
   rejections. */
const REQUIRED_ARGS = {
  Dense: ['units'],
  Conv1D: ['filters', 'kernel_size'],
  Conv2D: ['filters', 'kernel_size'],
  SeparableConv1D: ['filters', 'kernel_size'],
  SeparableConv2D: ['filters', 'kernel_size'],
  DepthwiseConv1D: ['kernel_size'],
  DepthwiseConv2D: ['kernel_size'],
  LSTM: ['units'],
  Reshape: ['target_shape'],
  Permute: ['dims'],
  Activation: ['activation'],
  Dropout: ['rate'],
  SpatialDropout1D: ['rate'],
  SpatialDropout2D: ['rate'],
  AlphaDropout: ['rate'],
  GaussianDropout: ['rate'],
  GaussianNoise: ['stddev'],
};

function missingRequiredError(cls, cfg) {
  const c = canon(cls);
  const need = REQUIRED_ARGS[c];
  if (!need) return null;
  const missing = need.filter((k) => !(k in (cfg || {})));
  if (!missing.length) return null;
  /* python's own wording, plural form included */
  const names = missing.map((m) => "'" + m + "'");
  const list = names.length === 1
    ? names[0]
    : names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
  return c + '.__init__() missing ' + missing.length + ' required positional argument'
    + (missing.length === 1 ? '' : 's') + ': ' + list;
}

/* InputLayer validates its own leftovers and words it differently from the
   generic_utils.validate_kwargs() check every other layer inherits. */
function unknownKwargError(cls, cfg) {
  const c = canon(cls);
  const allowed = LAYER_KWARGS[c];
  if (!allowed) return null;
  const ok = c === 'InputLayer' ? allowed : allowed.concat(BASE_KWARGS);
  const bad = Object.keys(cfg || {}).filter((k) => ok.indexOf(k) === -1);
  if (!bad.length) return null;
  if (c === 'InputLayer') {
    return 'Unrecognized keyword arguments: ' + pyRepr(bad);
  }
  return "('Keyword argument not understood:', " + pyRepr(bad[0]) + ')';
}

/* python's repr of a value's type, for the messages that quote it */
function pyType(v) {
  if (v === null || v === undefined) return "<class 'NoneType'>";
  if (typeof v === 'boolean') return "<class 'bool'>";
  if (typeof v === 'string') return "<class 'str'>";
  if (Array.isArray(v)) return "<class 'list'>";
  if (typeof v === 'object') return "<class 'dict'>";
  return Number.isInteger(v) ? "<class 'int'>" : "<class 'float'>";
}

/* ---------------------------------------------------------------------------
 * INTEGER ARGUMENTS
 *
 * JSON has one number type, python has two.  json.loads() hands keras a FLOAT
 * for `3.0`; JSON.parse() hands this port `3`, and nothing in the parsed tree
 * can tell the two apart.  keras carries the float into a TensorShape or an op
 * attribute and raises there, so the backend rejects a model this port passed.
 *
 * parseRawJson() already keeps the source text of every number, so the decimal
 * point is still there to be found - config_raw is the only place the
 * distinction survives.
 * ------------------------------------------------------------------------ */
function rawFloat(v) {
  if (typeof v === 'string' && v.indexOf(RAWNUM) === 0) {
    const t = v.slice(RAWNUM.length);
    /* json.loads() returns a float for any literal carrying '.', 'e' or 'E' -
       1e5 included - and an int for everything else. */
    return /[.eE]/.test(t) ? t : null;
  }
  if (Array.isArray(v)) {
    for (const x of v) {
      const r = rawFloat(x);
      if (r) return r;
    }
  }
  return null;
}

const DIM_INT = (t) => "Dimension value must be integer or None or have an __index__ method, got value '"
  + t + "' with type '<class 'float'>'";

/* The arguments a float actually breaks.  Every entry below is a MEASURED
   backend verdict, not a deduction - the two that were deduced (a convolution's
   strides and dilation_rate, left out on the grounds that they only become op
   attributes) turned out to fail, so nothing here is extrapolated any more.

   Confirmed to FAIL:  a convolution's kernel_size, strides and dilation_rate;
     Reshape.target_shape; Permute.dims; BatchNormalization.axis;
     InputLayer.batch_input_shape; LSTM.units.
   Confirmed to PASS:  a convolution's `filters` - Conv.__init__ opens with
     `if isinstance(filters, float): filters = int(filters)`;  Dense.units, for
     the same reason (`int(units)`);  and EVERYTHING a pooling layer carries,
     pool_size and strides both, which is why no pooling class appears here.

   depth_multiplier and groups are grouped with kernel_size because all three
   land in the conv kernel shape, kernel_size + (input_channel // groups,
   filters), which is built through add_weight() -> TensorShape. */
const INT_ARGS = {
  Conv1D: ['kernel_size', 'strides', 'dilation_rate', 'groups'],
  Conv2D: ['kernel_size', 'strides', 'dilation_rate', 'groups'],
  DepthwiseConv1D: ['kernel_size', 'strides', 'dilation_rate', 'depth_multiplier'],
  DepthwiseConv2D: ['kernel_size', 'strides', 'dilation_rate', 'depth_multiplier'],
  SeparableConv1D: ['kernel_size', 'strides', 'dilation_rate', 'depth_multiplier'],
  SeparableConv2D: ['kernel_size', 'strides', 'dilation_rate', 'depth_multiplier'],
  LSTM: ['units'],
  Reshape: ['target_shape'],
  Permute: ['dims'],
  BatchNormalization: ['axis'],
};

/* conv_utils.normalize_tuple(value, n, name):
     if isinstance(value, int):   value_tuple = (value,) * n
     else:                        value_tuple = tuple(value)   # TypeError -> ValueError
   A SCALAR float is neither an int nor iterable, so it dies right there - while
   a LIST of floats is iterable and survives, which is why [2.0, 2.0] loads and
   a bare 2.0 does not.  Only the raw text can tell 2.0 from 2. */
const TUPLE_ARGS = {
  Conv1D: [['kernel_size', 1], ['strides', 1], ['dilation_rate', 1]],
  Conv2D: [['kernel_size', 2], ['strides', 2], ['dilation_rate', 2]],
  DepthwiseConv1D: [['kernel_size', 1], ['strides', 1], ['dilation_rate', 1]],
  DepthwiseConv2D: [['kernel_size', 2], ['strides', 2], ['dilation_rate', 2]],
  SeparableConv1D: [['kernel_size', 1], ['strides', 1], ['dilation_rate', 1]],
  SeparableConv2D: [['kernel_size', 2], ['strides', 2], ['dilation_rate', 2]],
  MaxPooling1D: [['pool_size', 1], ['strides', 1]],
  MaxPooling2D: [['pool_size', 2], ['strides', 2]],
  AveragePooling1D: [['pool_size', 1], ['strides', 1]],
  AveragePooling2D: [['pool_size', 2], ['strides', 2]],
};

function tupleArgError(cls, rawCfg) {
  const specs = TUPLE_ARGS[canon(cls)];
  if (!specs || !rawCfg) return null;
  for (const [key, n] of specs) {
    const raw = rawCfg[key];
    if (Array.isArray(raw)) continue;          /* iterable - normalize_tuple copes */
    const t = rawFloat(raw);
    if (t) {
      return 'The `' + key + '` argument must be a tuple of ' + n + ' integers. Received: ' + t;
    }
  }
  return null;
}

/* An initializer calls _assert_float_dtype() on the dtype it is handed, so a
   layer that OWNS WEIGHTS cannot carry an int or complex dtype.  A layer with
   no weights never reaches that code. */
const FLOAT_DTYPES = ['float16', 'float32', 'float64', 'bfloat16', 'half', 'double'];

function dtypeWeightError(cls, cfg) {
  if (!INIT_RANK[canon(cls)]) return null;
  const dt = cfg.dtype;
  if (typeof dt !== 'string' || !dt) return null;
  if (FLOAT_DTYPES.indexOf(dt) !== -1) return null;
  if (KNOWN_DTYPES.indexOf(dt) === -1) return null;   /* an unknown name is reported elsewhere */
  return "Expected floating point type, got <dtype: '" + dt + "'>.";
}

/* An activation written as a nested LAYER is constructed like any other layer,
   so its own __init__ validation runs. */
function nestedActError(v) {
  if (!v || typeof v !== 'object') return null;
  if (canon(objName(v)) !== 'LeakyReLU') return null;
  const c = v.config || {};
  if ('alpha' in c && (c.alpha === null || c.alpha === undefined)) {
    return 'The alpha value of a Leaky ReLU layer cannot be None, expecting a float. '
      + 'Received: alpha=None';
  }
  return null;
}

function floatArgError(cls, rawCfg) {
  const keys = INT_ARGS[canon(cls)];
  if (!keys || !rawCfg) return null;
  for (const k of keys) {
    const t = rawFloat(rawCfg[k]);
    if (t) return DIM_INT(t);
  }
  return null;
}

/* TensorShape refuses a negative or non-integer dimension the moment the
   InputLayer is constructed; the port used to copy batch_input_shape straight
   into inferShapes() without ever looking at it. */
function shapeDimError(shape) {
  for (const d of shape) {
    if (d === null || d === undefined) continue;
    if (typeof d !== 'number' || !isFinite(d) || Math.floor(d) !== d) {
      return "Dimension value must be integer or None or have an __index__ method, got value '"
        + d + "' with type '<class '" + (typeof d === 'number' ? 'float' : 'str') + "'>'";
    }
    if (d < 0) return 'Dimension ' + d + ' must be >= 0';
  }
  return null;
}

const INITIALIZER_KEYS = [
  'kernel_initializer', 'bias_initializer', 'depthwise_initializer', 'pointwise_initializer',
  'recurrent_initializer', 'beta_initializer', 'gamma_initializer',
  'moving_mean_initializer', 'moving_variance_initializer', 'embeddings_initializer',
];
const REGULARIZER_KEYS = [
  'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'depthwise_regularizer',
  'pointwise_regularizer', 'recurrent_regularizer', 'beta_regularizer', 'gamma_regularizer',
];
const CONSTRAINT_KEYS = [
  'kernel_constraint', 'bias_constraint', 'depthwise_constraint', 'pointwise_constraint',
  'recurrent_constraint', 'beta_constraint', 'gamma_constraint',
];

/* a serialised object is {"class_name": X, "config": {...}} or a bare name */
/* ---------------------------------------------------------------------------
 * INITIALIZERS
 *
 * An initializer validates twice: VarianceScaling checks its own arguments in
 * __init__, while Identity, Orthogonal and Constant only find out what they
 * were handed when __call__ builds the weight.  Both were unchecked here, so a
 * model with an impossible initializer passed.
 *
 * The rank of each weight is fixed by the layer type, which is all these
 * checks need - no input shape required.
 * ------------------------------------------------------------------------ */
const INIT_RANK = {
  Dense: { kernel_initializer: 2, bias_initializer: 1 },
  Conv1D: { kernel_initializer: 3, bias_initializer: 1 },
  Conv2D: { kernel_initializer: 4, bias_initializer: 1 },
  DepthwiseConv1D: { depthwise_initializer: 3, bias_initializer: 1 },
  DepthwiseConv2D: { depthwise_initializer: 4, bias_initializer: 1 },
  SeparableConv1D: { depthwise_initializer: 3, pointwise_initializer: 3, bias_initializer: 1 },
  SeparableConv2D: { depthwise_initializer: 4, pointwise_initializer: 4, bias_initializer: 1 },
  LSTM: { kernel_initializer: 2, recurrent_initializer: 2, bias_initializer: 1 },
  BatchNormalization: {
    beta_initializer: 1, gamma_initializer: 1,
    moving_mean_initializer: 1, moving_variance_initializer: 1,
  },
};

/* 'IdentityInitializer', 'Identity' and 'identity' are the same thing */
function initKind(v) {
  const n = objName(v);
  if (!n) return null;
  return String(n).toLowerCase().replace(/initializer$/, '').replace(/_/g, '');
}

/* VarianceScaling.__init__ - raised while the layer is being constructed */
function initArgError(v) {
  if (initKind(v) !== 'variancescaling') return null;
  const c = (v && v.config) || {};
  if ('scale' in c && !(typeof c.scale === 'number' && c.scale > 0)) {
    return '`scale` must be positive float. Received: scale=' + pyNum(c.scale);
  }
  const allowed = ['uniform', 'truncated_normal', 'untruncated_normal', 'normal'];
  if ('distribution' in c && allowed.indexOf(String(c.distribution)) === -1) {
    return 'Invalid `distribution` argument: ' + c.distribution
      + '. Allowed distributions: ["uniform", "truncated_normal", "untruncated_normal"]';
  }
  return null;
}

/* Identity / Orthogonal / Constant - raised when __call__ builds the weight */
function initShapeError(cls, cfg) {
  const ranks = INIT_RANK[canon(cls)];
  if (!ranks) return null;
  for (const key of Object.keys(ranks)) {
    const v = cfg[key];
    const kind = initKind(v);
    if (!kind) continue;
    const rank = ranks[key];
    if (kind === 'identity' && rank !== 2) {
      return 'Identity matrix initializer can only be used for 2D matrices.';
    }
    if (kind === 'orthogonal' && rank < 2) {
      return 'The tensor to initialize must be at least two-dimensional. Received: '
        + 'shape=' + pyRepr(new Array(rank).fill(0)) + ' of rank ' + rank + '.';
    }
    if (kind === 'constant') {
      const val = v && v.config ? v.config.value : undefined;
      if (val !== undefined && typeof val !== 'number') {
        return 'Only scalar values are supported for the Constant initializer. Received: value='
          + pyRepr(val);
      }
    }
  }
  return null;
}

function objName(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'string') return v;
  if (typeof v === 'object' && v.class_name) return String(v.class_name);
  return null;
}

const OBJECT_SCOPE_TAIL =
  ' See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.';

/* An unresolvable LAYER class is raised by deserialize_keras_object() before the
   layer is ever constructed, so it reaches the user unwrapped.  Everything else
   below is raised from inside `cls(**config)` and is wrapped in the "Error when
   deserializing class" envelope by the caller. */
function unknownLayerError(cls) {
  if (KNOWN_LAYERS.indexOf(canon(cls)) !== -1) return null;
  return 'Unknown layer: ' + pyRepr(String(cls)) +
    '. Please ensure you are using a `keras.utils.custom_object_scope` and that this object is included in the scope.' +
    OBJECT_SCOPE_TAIL;
}

/* regularizers._check_penalty_number() is the ONLY place keras inspects a
   number for finiteness.  Everything else a config can carry - momentum,
   epsilon, a dropout rate, a GaussianNoise stddev, a Constant initializer -
   is stored unread, which is why those models load with an inf in them and
   an inf PENALTY does not. */
function penaltyError(regCfg) {
  if (!regCfg || typeof regCfg !== 'object') return null;
  for (const k of ['l1', 'l2']) {
    const v = regCfg[k];
    /* L1L2.__init__ does `l1 = 0.0 if l1 is None else l1` first, so a null is
       fine; _check_penalty_number() then rejects anything that is not a real
       number, and separately anything non-finite. */
    if (v === null || v === undefined) continue;
    if (typeof v !== 'number') {
      return 'Value: ' + (Array.isArray(v) ? pyRepr(v) : String(v))
        + ' is not a valid regularization penalty number, expected an int or float value.';
    }
    if (!isFinite(v)) {
      return 'Value: ' + pyNum(v) + ' is not a valid regularization penalty number, '
        + 'an infinite number or NaN are not valid values.';
    }
  }
  return null;
}

function unresolvableObject(cls, cfg) {
  const c = canon(cls);
  const unknown = (kind, n) => 'Unknown ' + kind + ': ' + pyRepr(String(n)) +
    '. Please ensure you are using a `keras.utils.custom_object_scope` and that this object is included in the scope.' +
    OBJECT_SCOPE_TAIL;

  /* an activation given as a name must resolve; given as an object it must be a
     layer keras knows (handled by the layer check above for nested classes) */
  /* One resolver for every activation slot.  An activation may be a name, a
     nested activation LAYER, or the function object above; recurrent_activation
     used to be tested only when it was a string, so an object there slipped
     past the load and failed later in the report instead. */
  const actError = (v) => {
    if (v === null || v === undefined || v === '') return null;
    /* An .h5 is loaded through legacy_serialization, which resolves an
       activation by class_name against its own table.  The new-style function
       object keeps the real name in `config` and leaves class_name as the
       literal string 'function', for which that table has no entry. */
    if (typeof v === 'object' && String(v.class_name) === 'function') {
      return unknown('activation function', 'function');
    }
    const n = actName(v);
    if (!n) return null;
    if (KNOWN_ACTIVATIONS.indexOf(String(n)) !== -1) return null;
    if (typeof v === 'object'
      && (KNOWN_LAYERS.indexOf(canon(n)) !== -1
        || KNOWN_ACTIVATIONS.indexOf(String(n).toLowerCase()) !== -1)) return null;
    return unknown('activation function', n);
  };
  const ae = actError(cfg.activation);
  if (ae) return ae;
  const na = nestedActError(cfg.activation);
  if (na) return na;
  if (c === 'LSTM') {
    const re = actError(cfg.recurrent_activation);
    if (re) return re;
  }

  for (const key of INITIALIZER_KEYS) {
    const n = objName(cfg[key]);
    if (n && KNOWN_INITIALIZERS.indexOf(n) === -1) {
      return unknown('initializer', n);
    }
    const ia = initArgError(cfg[key]);
    if (ia) return ia;
  }
  for (const key of REGULARIZER_KEYS) {
    const n = objName(cfg[key]);
    if (n && KNOWN_REGULARIZERS.indexOf(n) === -1) {
      return unknown('regularizer', n);
    }
    const pen = penaltyError(cfg[key] && cfg[key].config);
    if (pen) return pen;
  }
  /* ActivityRegularization builds an L1L2 out of its own two arguments, so the
     same penalty check applies to the layer config itself. */
  if (c === 'ActivityRegularization') {
    const pen = penaltyError(cfg);
    if (pen) return pen;
  }
  for (const key of CONSTRAINT_KEYS) {
    const n = objName(cfg[key]);
    if (n && KNOWN_CONSTRAINTS.indexOf(n) === -1) {
      return unknown('constraint', n);
    }
  }

  /* a dtype must be a real policy / dtype string.  keras 3 serialises it as a
     DTypePolicy object, which keras 2.15 cannot turn into a policy at all. */
  const dt = cfg.dtype;
  if (typeof dt === 'string' && dt) {
    if (KNOWN_DTYPES.indexOf(dt) === -1) {
      return 'Cannot convert value ' + dt + ' to a TensorFlow DType.';
    }
  }
  const dwe = dtypeWeightError(cls, cfg);
  if (dwe) return dwe;
  if (dt && typeof dt === 'object') {
    /* Layer._set_dtype_policy() has an explicit
         elif isinstance(dtype, dict): self._dtype_policy = policy.deserialize(dtype)
       branch, so a serialised keras 2.x Policy is accepted.  keras 3 writes a
       DTypePolicy instead, which 2.15 cannot deserialise - that one still
       fails, and so does any other object. */
    const dn = objName(dt);
    const inner = dt.config && dt.config.name;
    /* InputLayer does NOT go through Layer._set_dtype_policy() for the tensor
       it creates: __init__ hands `dtype` straight to backend.placeholder(),
       which calls tf.as_dtype() on it, and that takes no dict at all. */
    const policyOk = canon(cls) !== 'InputLayer'
      && (dn === 'Policy' || dn === 'PolicyV1')
      && typeof inner === 'string'
      && (KNOWN_DTYPES.indexOf(inner) !== -1
        || inner === 'mixed_float16' || inner === 'mixed_bfloat16');
    if (!policyOk) {
      return 'Cannot convert value ' + pyRepr(dt) + ' to a TensorFlow DType.';
    }
  }
  return null;
}

/* conv_utils.normalize_padding() and normalize_data_format() both .lower() the
   value and store the result, so `layer.padding` and `layer.data_format` are
   ALWAYS lower case by the time Model_Compatibility_47.py reads them - "VALID"
   and "CHANNELS_LAST" load fine there and report as supported.

   This port validated a lower-cased copy but then reported on the raw config
   string, so the same file was flagged here and clean in the backend; Flatten
   ran the mismatch the other way and passed a channels-first layer the backend
   rejects.  Normalising once, up front, is what the constructors do.

   SpatialDropout2D is deliberately NOT in this list: its __init__ compares
   data_format without lowering it, so a mixed-case value really is an error
   there, and coreInitError() already reports it. */
const PADDING_NORMALISED = [].concat(CONV_LAYERS, POOL_LAYERS);
const DATA_FORMAT_NORMALISED = [].concat(CONV_LAYERS, POOL_LAYERS, ['Flatten']);

function normalizeLayerConfigs(layers) {
  for (const l of layers || []) {
    const c = canon(l && l.class_name);
    const cfg = l && l.config;
    if (!cfg) continue;
    if (PADDING_NORMALISED.indexOf(c) !== -1 && typeof cfg.padding === 'string') {
      cfg.padding = cfg.padding.toLowerCase();
    }
    if (DATA_FORMAT_NORMALISED.indexOf(c) !== -1 && typeof cfg.data_format === 'string') {
      cfg.data_format = cfg.data_format.toLowerCase();
    }
  }
}
/* Python/Keras Sequential models require every layer name to be unique.
 * load_model() rejects duplicate layer names while reconstructing the
 * Sequential model. Keep this as a separate load-level validation so
 * existing layer compatibility checks are not changed.
 */
function duplicateLayerNameError(parsed) {
  const cfgRoot = parsed && parsed.config;
  const rawLayers = (cfgRoot && cfgRoot.config && cfgRoot.config.layers) || [];

  const seen = new Set();

  for (const layer of rawLayers) {
    const name = layer && layer.config && layer.config.name;

    if (!name) continue;

    if (seen.has(name)) {
      return "All layers added to a Sequential model should have unique names. Name '" + name + "' is used more than once.";
    }

    seen.add(name);
  }

  return null;
}

/* The ADDITIONAL KERAS 2.15 LOAD VALIDATION block that used to sit here was
 * never called - constructorFailure() reached straight past it, by design:
 * its extra checks rejected models keras loads happily.  It is removed rather
 * than left in place, because dead validation reads like live validation the
 * next time someone debugs a verdict.  No check changes: nothing ever ran it.
 * Its one live helper, isNumber(), now lives in outputShapeValidation(). */

function outputShapeValidation(layer) {
  /* Local copy of the helper of the same name in additionalKerasValidation().
     It was previously referenced here while declared only inside that
     function, so every model with a Dense layer threw a ReferenceError out
     of constructorFailure().  checkModelCompatibility()'s catch treats any
     throw as a load failure, so valid models were reported as
     'Unable to Load Model' - or, when keras_version was not 2.15.0, as the
     'Older or Newer version of Python or Libraries' message.  No check is
     changed by this; it only lets the Dense branch below actually run. */
  const isNumber = (v) => typeof v === 'number' && Number.isFinite(v);

  const c = canon(layer.class_name);
  const cfg = layer.config || {};
  const input = layer.input_shape || [];
  const output = layer.output_shape || [];

  if (!Array.isArray(input) || !Array.isArray(output)) {
    return null;
  }

  /*
   * Convolution / pooling cannot produce a negative spatial dimension.
   */
  if (
    CONV_RANK[c] ||
    POOL_RANK[c]
  ) {
    for (const d of output) {
      if (d !== null && d !== undefined && d < 0) {
        return (
          'Invalid output shape ' +
          pyShape(output) +
          ' generated from input shape ' +
          pyShape(input) +
          '.'
        );
      }
    }
  }

  /*
   * Dense must have a positive unit count.
   */
  /* Dense.__init__ stores `int(units)`, so anything int() accepts is fine -
     a float, and a string of digits too ('16' loads as 16).  Only a value
     int() refuses ('wide', None) or a negative count is a real failure. */
  const intable = (v) => {
    if (typeof v === 'number') return Number.isFinite(v);
    if (typeof v === 'string') return /^[+-]?\d+$/.test(v.trim());
    return false;
  };
  if (
    c === 'Dense' &&
    (!intable(cfg.units) || Number(cfg.units) < 0)
  ) {
    return (
      'Invalid Dense output units: ' +
      pyRepr(cfg.units) +
      '.'
    );
  }

  return null;
}
/* Sequential.from_config: deserialize each layer, then add() it, in order */
/* Sequential.from_config: deserialize each layer, then add() it, in order */
function constructorFailure(parsed) {
  const cfgRoot = parsed && parsed.config;
  const weights = parsed && parsed.weights;
  const raw = (cfgRoot && cfgRoot.config && cfgRoot.config.layers) || [];

  const rawRoot = parsed && parsed.config_raw;
  const rawLayers = (rawRoot && rawRoot.config && rawRoot.config.layers) || [];

  const seq = raw.map((l) => ({
    class_name: l.class_name,
    config: l.config || {}
  }));

  inferShapes(seq, null);

  for (let i = 0; i < seq.length; i++) {
    const shapeError = outputShapeValidation(seq[i]);

    if (shapeError) {
      return shapeError;
    }
  }

  /* RNN layers override from_config() WITHOUT the try/except that
     Layer.from_config has, so an LSTM constructor error surfaces bare. */
  const UNWRAPPED_FROM_CONFIG = ['LSTM'];

  for (let idx = 0; idx < seq.length; idx++) {
    const layer = seq[idx];
    const cls = layer.class_name;

    /* 0. Unknown layer class */
    const unknownLayer = unknownLayerError(cls);

    if (unknownLayer) {
      return unknownLayer;
    }

    const rawCfg =
      (rawLayers[idx] && rawLayers[idx].config) || null;

    /*
     * Stage 1:
     * Match the existing Python/Keras constructor checks.
     */
    const inner =
      missingRequiredError(cls, layer.config)
      || unresolvableObject(cls, layer.config)
      || initError(cls, layer.config, rawCfg);

    if (inner) {
      if (UNWRAPPED_FROM_CONFIG.indexOf(canon(cls)) !== -1) {
        return inner;
      }

      const printable = rawCfg || layer.config;

      return "Error when deserializing class '" + cls +
        "' using config=" + pyRepr(printable) +
        '.\n\nException encountered: ' + inner;
    }

    /*
     * IMPORTANT:
     *
     * Do NOT run additionalKerasValidation() here.
     *
     * Python's Model_Compatibility.py does not have this
     * additional validation layer. Its extra checks were causing
     * valid Keras models to be rejected by the JSX port.
     */

    /* Stage 2: InputSpec validation */
    const spec = inputSpecError(
      cls,
      layer.config,
      layer.input_shape
    );

    if (spec) {
      return spec;
    }

    /* Stage 3: build / shape validation.  A float that survived __init__ is
       only rejected once it reaches a shape or an op attribute, and that error
       is raised from model.add(), outside the deserialize envelope. */
    const tup = tupleArgError(cls, rawCfg);
    if (tup) return tup;

    const flt = floatArgError(cls, rawCfg);
    if (flt) return flt;

    const ish = initShapeError(cls, layer.config);
    if (ish) return ish;

    const built = buildError(
      cls,
      layer.config,
      layer.input_shape
    );

    if (built) {
      return built;
    }
  }

  /*
   * Stage 4:
   * Only after every layer has been constructed and built
   * does Keras assign the saved weights.
   */
  if (weights === null) {
    return "Unable to open object (object 'model_weights' doesn't exist)";
  }

  const wErr = weightsGroupError(seq, weights);

  if (wErr) {
    return wErr;
  }

  for (const layer of seq) {
    const wt = weightError(layer, weights);

    if (wt) {
      return wt;
    }
  }

  return null;
}

/* load_weights_from_hdf5_group() compares the saved layer list against the graph
   BEFORE it assigns anything, then checks the array count per layer.  Both of
   those raise their own messages, neither of which is the assignment error. */
function weightsGroupError(seq, weights) {
  if (!weights) return null;
  /* No early return on an empty map.  keras compares the saved layer list with
     the model's weighted layers BEFORE assigning anything, so a file that
     declares nothing and a model that expects weights is a count mismatch -
     bailing out here reported such a file as clean. */
  const savedNames = Object.keys(weights).filter((k) => k !== '__missing__');

  /* Layer types we can size.  It is only used to spot a layer the config gained
     that the file has no weights for - never to decide that a layer is
     weightless, because any type missing from this list (GRU, SimpleRNN, a
     custom layer) would then be miscounted. */
  const WEIGHT_COUNT = {
    Dense: 1, Conv1D: 1, Conv2D: 1, DepthwiseConv1D: 1, DepthwiseConv2D: 1,
    SeparableConv1D: 2, SeparableConv2D: 2, LSTM: 2,
  };
  const expectedCount = (c, cfg) => {
    if (c === 'BatchNormalization') {
      /* gamma and beta only exist when scale / center are on */
      return 2 + (cfg.center === false ? 0 : 1) + (cfg.scale === false ? 0 : 1);
    }
    if (c === 'Embedding' || c === 'PReLU') return 1;
    const base = WEIGHT_COUNT[c];
    if (base !== undefined) return base + (cfg.use_bias === false ? 0 : 1);
    /* every other layer in the supported set is weightless, and saying so is the
       point: returning undefined here let a weight group bolted onto a Dropout
       or a GaussianNoise count itself into the model's own tally, so keras saw a
       file with more weighted layers than the graph has and the port did not */
    if (SUPP_LAYERS.indexOf(c) !== -1) return 0;
    return undefined;
  };

  /* keras' filtered_layers: the layers that actually carry weights.  A type we
     cannot size falls back to "does the file have a group for it", so a layer
     added to the config is still caught. */
  const named = (l) => (l.config && l.config.name) || '';
  const filtered = seq.filter((l) => {
    const want = expectedCount(canon(l.class_name), l.config || {});
    if (want === undefined) return savedNames.indexOf(named(l)) !== -1;
    return want > 0;
  });

  if (filtered.length !== savedNames.length) {
    return 'Layer count mismatch when loading weights from file. Model expected '
      + filtered.length + ' layers, found ' + savedNames.length + ' saved layers.';
  }

  for (let k = 0; k < filtered.length; k++) {
    const l = filtered[k];
    const cfg = l.config || {};
    const want = expectedCount(canon(l.class_name), cfg);
    if (want === undefined) continue;            /* unknown type - cannot size it */
    const stored = weights[named(l)];
    if (!stored) continue;
    if (stored.length !== want) {
      return 'Weight count mismatch for layer #' + k + ' (named ' + named(l)
        /* the save-file name is looked up by name, not by position: the h5 group
           listing is alphabetical, keras' layer_names attr is in save order */
        + ' in the current model, ' + named(l) + ' in the save file). Layer expects '
        + want + ' weight(s). Received ' + stored.length + ' saved weight(s)';
    }
  }
  return null;
}

/* ============================================================================
 * 2c. THE LAYER REPORT - the body of check_model_compatibility()
 * ==========================================================================*/
function analyse(cfgRoot) {
  /* module-level globals in the python; per-call here so two validations of two
     different files can never bleed into each other */
  let modelFormat = null;
  const modelFormats = [];

  function channelCheck(ch) {
    if (ch === 'channels_first' || ch === 'channels_last') {
      modelFormat = ch;
      if (modelFormats.length === 0 || modelFormats.indexOf(ch) !== -1) {
        modelFormats.push(ch);
        return 'Data Format Supported.';
      }
      return 'Data Format Not Supported, All Layers should have the same Data Format, except the Flatten layer.';
    }
    return "Data Format '" + ch + "' Not Supported, Try 'channels_first' or 'channels_last'.";
  }

  /* activation_check(activations):
       in supp_activation                      -> Supported          (a name)
       isinstance(supp_activation_classes)     -> Supported          (a layer)
         + ReLU with max_value not None        -> max_value message
       else                                    -> Not Supported
     the class path keys off the ACTIVATION value, never off the host layer. */
  function activationCheck(a, aCfg) {
    if (SUPP_ACTIVATION.indexOf(a) !== -1) return 'Activation Supported.';
    const asClass = SUPP_ACT_CLASSES.filter((k) => k.toLowerCase() === String(a).toLowerCase())[0];
    if (asClass) {
      if (asClass === 'ReLU' && aCfg && aCfg.max_value !== null && aCfg.max_value !== undefined) {
        return 'Activation Supported, but max_value Not Supported, try None.';
      }
      return 'Activation Supported.';
    }
    return 'Activation ' + a + ' Not Supported, Try Using a Different Activation.';
  }

  function axisCheck(axisValue, modelType, inputLen, rank) {
    let av = Array.isArray(axisValue) ? axisValue[0] : axisValue;
    if (av === null || av === undefined) av = -1;
    /* keras stores the axis normalised after build(); a config written before
       build keeps -1, so normalise it or the default axis reads as a failure */
    if (av < 0 && rank) av = rank + av;

    if (modelType === 1) {
      if (modelFormat === 'channels_first') {
        return av === 1 ? av + ' Axis Supported.' : av + ' Axis Not Supported with Channels First in CNN 1D, Try Changing Data Format or Axis.';
      }
      return av === 2 ? av + ' Axis Supported.' : av + ' Axis Not Supported with Channels Last in CNN 1D, Try Changing Data Format or Axis.';
    }
    if (modelType === 2) {
      if (modelFormat === 'channels_first') {
        return av === 1 ? av + ' Axis Supported.' : av + ' Axis Not Supported with Channels First in CNN 2D, Try Changing Data Format or Axis.';
      }
      return av === 3 ? av + ' Axis Supported.' : av + ' Axis Not Supported with Channels Last in CNN 2D, Try Changing Data Format or Axis.';
    }
    if (modelType === 3) {
      if (inputLen === 4) return av + ' Axis Not Supported with Multi Dimension Dense, Try Changing Dense Shape.';
      return av === 1 ? av + ' Axis Supported.' : av + ' Axis Not Supported.';
    }
    if (modelType === 4) {
      if (inputLen === 3) {
        return av === 2 ? av + ' Axis Supported.' : av + ' Axis Not Supported with LSTM return_sequences True, Try Changing return_sequences or Axis.';
      }
      return av === 1 ? av + ' Axis Supported.' : av + ' Axis Not Supported with LSTM return_sequences False, Try Changing return_sequences or Axis.';
    }
    /* the python's `else: print("Unknown Layer Found")` fall-through */
    return av + ' Axis Supported.';
  }

  let rawLayers = (cfgRoot.config && cfgRoot.config.layers) || [];
  let modelInput = null;
  if (rawLayers.length && canon(rawLayers[0].class_name) === 'InputLayer') {
    modelInput = batchShape(rawLayers[0].config);
    rawLayers = rawLayers.slice(1); /* keras drops the implicit InputLayer from model.layers */
  }
  if (!modelInput && rawLayers.length) modelInput = batchShape(rawLayers[0].config);

  const layers = rawLayers.map((l) => ({ class_name: l.class_name, config: l.config || {} }));
  inferShapes(layers, modelInput);

  const op = [];
  let lstmDenseFlag = 0;
  let modelStartFlag = false;
  let firstLayerFlag = false;

  const hdr = (l, s) => (l.config.name || canon(l.class_name)) + '!!!' + l.class_name + '!!!' + s;
  const prevOf = (i) => layers.slice(0, i).reverse();

  layers.forEach((layer, idx) => {
    const ops = [];
    const cls = canon(layer.class_name);
    const cfg = layer.config;

    /* --- layer type whitelist -------------------------------------------- */
    if (!isIn(cls, SUPP_LAYERS)) {
      op.push([hdr(layer, 'Not Supported')]);
      return;
    }

    /* --- first-layer / pre-weighted gates -------------------------------- */
    if (isIn(cls, NON_INPUT_LAYERS) && !modelStartFlag) {
      ops.push(hdr(layer, 'Not Supported'));
      ops.push(
        firstLayerFlag
          ? 'This Layer type is Not Supported as the prior to Weighted Layers for the Model, Try using a Weighted layer before.'
          : 'This Layer type is Not Supported as the First Layer for the Model, try CNN, DNN, LSTM, Reshape or Input Layers as First Layer.'
      );

      /* --- convolution ---------------------------------------------------- */
    } else if (isIn(cls, CONV_LAYERS)) {
      modelStartFlag = true;
      firstLayerFlag = true;
      const rank = cls.indexOf('1D') !== -1 ? 1 : 2;
      if (prevOf(idx).some((p) => canon(p.class_name) === 'LSTM')) {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('Convolution Not Supported after LSTM, try using before LSTM or Pure LSTM Architecture.');
      } else {
        ops.push(hdr(layer, 'Supported'));
        const channel = cfg.data_format || 'channels_last';
        ops.push(channelCheck(channel));
        const ishape = fetchShape(channel, layer.input_shape.slice(1));
        const k = rankList(cfg.kernel_size, rank, new Array(rank).fill(1));
        const st = rankList(cfg.strides, rank, new Array(rank).fill(1));
        const dl = rankList(cfg.dilation_rate, rank, new Array(rank).fill(1));
        if (idx === 0) ops.push(inputCheck(ishape));
        /* Model_Compatibility.py emits the strides/activation pair in a
           different order per class: Conv2D and DepthwiseConv2D report strides
           first, every 1D variant and SeparableConv2D report the activation
           first.  The order decides which problem heads the suggestion list. */
        if (cls.indexOf('1D') !== -1 || cls === 'SeparableConv2D') {
          ops.push(kernelCheck(k));
          ops.push(activationCheck(actName(cfg.activation), actConfig(cfg.activation)));
          ops.push(stridesCheck(st));
        } else {
          ops.push(kernelCheck(k));
          ops.push(stridesCheck(st));
          ops.push(activationCheck(actName(cfg.activation), actConfig(cfg.activation)));
        }
        ops.push(paddingCheck(cfg.padding || 'valid'));
        ops.push(dilationCheck(dl));
        /* only Conv2D / Conv1D expose `groups` */
        if (cls === 'Conv2D' || cls === 'Conv1D') {
          ops.push(groupCheck(cfg.groups === undefined || cfg.groups === null ? 1 : cfg.groups));
        }
      }

      /* --- pooling -------------------------------------------------------- */
    } else if (isIn(cls, POOL_LAYERS)) {
      const rank = cls.indexOf('1D') !== -1 ? 1 : 2;
      let poolFlag = 0;
      /* the back-scan stops at a convolution only, so Pool -> Dense -> Pool is
         still reported as consecutive - the python's own behaviour */
      for (const p of prevOf(idx)) {
        if (isIn(p.class_name, CONV_LAYERS)) break;
        if (isIn(p.class_name, POOL_LAYERS)) {
          poolFlag = 1;
          break;
        }
      }
      if (poolFlag) {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('Consecutive Pooling Layers are Not Supported, Please Use Only 1.');
      } else {
        ops.push(hdr(layer, 'Supported'));
        /* dead in practice - a pooling layer at index 0 is caught by the
           first-layer gate above - but it holds its position in ops[] */
        if (idx === 0) ops.push(inputCheck(layer.input_shape));
        const channel = cfg.data_format || 'channels_last';
        ops.push(channelCheck(channel));
        const ps = rankList(cfg.pool_size, rank, new Array(rank).fill(2));
        ops.push(poolCheck(ps));
        ops.push(stridesCheck(rankList(cfg.strides == null ? cfg.pool_size : cfg.strides, rank, ps)));
        ops.push(paddingCheck(cfg.padding || 'valid'));
      }

      /* --- Dense ---------------------------------------------------------- */
    } else if (cls === 'Dense') {
      modelStartFlag = true;
      firstLayerFlag = true;
      /* Model_Compatibility_47.py:603-626 enters this branch whenever an LSTM
         exists ANYWHERE in the model, but only appends a header row when one is
         found BEFORE this Dense.  A Dense that precedes every LSTM therefore
         leaves ops[] with no header and the python raises IndexError in the
         tail (py:1079).  The port used to invent a header here, which reported
         a clean model for a file the backend cannot report on at all; the tail
         below now reproduces the failure instead. */
      if (!lstmDenseFlag) {
        if (layers.some((l) => canon(l.class_name) === 'LSTM')) {
          const prev = prevOf(idx);
          for (let i = 0; i < prev.length; i++) {
            if (canon(prev[i].class_name) === 'LSTM') {
              lstmDenseFlag = 1;
              if (prev[i].config.return_sequences) {
                const btw = prev.slice(0, i).map((x) => canon(x.class_name));
                if (btw.indexOf('Flatten') === -1 && btw.indexOf('Reshape') === -1) {
                  ops.push(hdr(layer, 'Not Supported'));
                  ops.push('Location of Dense Layer Not Supported, try adding Reshape or Flatten Layer before Dense or change return_sequences to False for previous LSTM Layer.');
                } else ops.push(hdr(layer, 'Supported'));
              } else ops.push(hdr(layer, 'Supported'));
              break;
            }
          }
        } else {
          ops.push(hdr(layer, 'Supported'));
        }
      } else {
        ops.push(hdr(layer, 'Supported'));
      }
      ops.push(activationCheck(actName(cfg.activation), actConfig(cfg.activation)));
      if (idx === 0) ops.push(inputCheck(layer.input_shape));

      /* --- Activation / Softmax / ReLU / LeakyReLU ------------------------ */
    } else if (isIn(cls, ACT_LAYERS)) {
      let done = false;
      /* the back-scan does NOT stop at BatchNorm / Dropout / Flatten, so
         Conv(relu) -> BN -> Activation reads as consecutive activations */
      for (const p of prevOf(idx)) {
        const pc = canon(p.class_name);
        if (isIn(pc, [].concat(CONV_LAYERS, CORE_LAYERS))) {
          done = true;
          if (actName(p.config.activation) !== 'linear') {
            ops.push(hdr(layer, 'Not Supported'));
            ops.push('Consecutive Activations Not Supported, Please Use Only 1.');
          } else {
            ops.push(hdr(layer, 'Supported'));
            if (cls === 'Activation') ops.push(activationCheck(actName(cfg.activation), actConfig(cfg.activation)));
            if (cls === 'LeakyReLU') ops.push(alphaCheck(cfg.alpha !== undefined ? cfg.alpha : cfg.negative_slope));
          }
          break;
        }
        if (isIn(pc, RNN_LAYERS)) {
          done = true;
          ops.push(hdr(layer, 'Not Supported'));
          ops.push('Activations After LSTM Not Supported, Please Remove Activation Layer.');
          break;
        }
        if (isIn(pc, ACT_LAYERS)) {
          done = true;
          ops.push(hdr(layer, 'Not Supported'));
          ops.push('Consecutive Activations Not Supported, Please Use Only 1.');
          break;
        }
        if (isIn(pc, POOL_LAYERS)) {
          done = true;
          ops.push(hdr(layer, 'Not Supported'));
          ops.push('Activations After Pooling Not Supported, Please Remove Activation Layer.');
          break;
        }
      }
      if (!done) {
        ops.push(hdr(layer, 'Supported'));
        if (cls === 'Activation') ops.push(activationCheck(actName(cfg.activation), actConfig(cfg.activation)));
        if (cls === 'LeakyReLU') ops.push(alphaCheck(cfg.alpha !== undefined ? cfg.alpha : cfg.negative_slope));
      }
      /* the three ReLU extras are appended regardless of the ordering verdict */
      if (cls === 'ReLU') {
        if (cfg.max_value !== null && cfg.max_value !== undefined) {
          ops.push('Activation Supported, but max_value Not Supported, try None.');
        }
        if (cfg.negative_slope !== undefined && cfg.negative_slope !== null && Number(cfg.negative_slope) !== 0) {
          ops.push('Activation Supported, but negative_slope Not Supported, try 0.0 or use LeakyReLU.');
        }
        if (cfg.threshold !== undefined && cfg.threshold !== null && Number(cfg.threshold) !== 0) {
          ops.push('Activation Supported, but threshold Not Supported, try 0.0.');
        }
      }

      /* --- pass-through layers -------------------------------------------- */
    } else if (isIn(cls, PASS_LAYERS)) {
      if (isIn(cls, RESHAPE_LAYERS) || isIn(cls, INPUT_LAYERS) || cls === 'GaussianNoise') firstLayerFlag = true;
      if (cls === 'Flatten' && cfg.data_format === 'channels_first') {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('Channels First Data Format in Flatten Layer is Not Supported, Try Channels Last.');
      } else ops.push(hdr(layer, 'Supported'));

      /* --- LSTM ----------------------------------------------------------- */
    } else if (cls === 'LSTM') {
      modelStartFlag = true;
      firstLayerFlag = true;
      ops.push(hdr(layer, 'Supported'));
      /* actName() maps a missing key to 'linear', which is in neither LSTM
         list - fall back to the layer's real defaults so a config that omits
         them does not invent two "Activation Not Supported" rows */
      ops.push(lstmNActCheck(actName(cfg.activation === undefined ? 'tanh' : cfg.activation)));
      ops.push(lstmRActCheck(actName(cfg.recurrent_activation === undefined ? 'sigmoid' : cfg.recurrent_activation)));
      ops.push(ufbCheck(cfg.unit_forget_bias === undefined ? true : cfg.unit_forget_bias));
      if (!cfg.return_sequences) {
        const prev = prevOf(idx);
        if (!prev.some((y) => isIn(y.class_name, [].concat(CONV_LAYERS, CORE_LAYERS)))) {
          /* the python never breaks out of this loop, so prev_rs ends up
             holding the EARLIEST LSTM's flag, not the nearest one */
          let prevRs = true;
          for (const j of prev) if (canon(j.class_name) === 'LSTM') prevRs = !!j.config.return_sequences;
          ops.push(
            prevRs
              ? rsCheck(cfg.return_sequences)
              : '2 Consecutive LSTM Layers with return_sequences as False is Not Supported, try changing return_sequences to True for Initial Layer.'
          );
        } else ops.push(rsCheck(cfg.return_sequences));
      } else ops.push(rsCheck(cfg.return_sequences));
      ops.push(returnStateCheck(cfg.return_state));
      ops.push(goBackCheck(cfg.go_backwards));
      ops.push(statefulCheck(cfg.stateful));
      ops.push(unrollCheck(cfg.unroll));
      /* LSTMCell.__init__ stores min(1., max(0., dropout)); python's max()
         returns its FIRST argument when the comparison is False, so a NaN
         becomes 0.0 there and the backend reports 0.0, not nan. */
      const clampRate = (v) => {
        if (v === undefined) return 0.0;
        if (typeof v !== 'number' || Number.isNaN(v)) return 0.0;
        return Math.min(1.0, Math.max(0.0, v));
      };
      ops.push(dropoutCheck(clampRate(cfg.dropout)));
      ops.push(recDropoutCheck(clampRate(cfg.recurrent_dropout)));

      /* --- BatchNormalization --------------------------------------------- */
    } else if (cls === 'BatchNormalization') {
      let modelType = null;
      let inputLen = null;
      const rank = layer.input_shape.length;
      let multiBn = 0;
      let poolBn = 0;
      for (const p of prevOf(idx)) {
        const pc = canon(p.class_name);
        if (['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D'].indexOf(pc) !== -1) {
          modelType = 2;
          break;
        }
        if (['Conv1D', 'DepthwiseConv1D', 'SeparableConv1D'].indexOf(pc) !== -1) {
          modelType = 1;
          break;
        }
        if (pc === 'Dense') {
          modelType = 3;
          inputLen = p.input_shape.length;
          break;
        }
        if (pc === 'LSTM') {
          modelType = 4;
          inputLen = p.output_shape.length;
          break;
        }
        if (isIn(pc, NORM_LAYERS)) {
          multiBn = 1;
          break;
        }
        if (isIn(pc, POOL_LAYERS)) {
          poolBn = 1;
          break;
        }
      }
      /* both loops break on the first match, so this pair is unreachable -
         kept because the python keeps it */
      if (multiBn && poolBn) {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('Consecutive BatchNormalization Layers Not Supported, Please Use Only 1.');
        ops.push('BatchNormalization after Pooling Layers is Not Supported, Please Change Order.');
      } else if (multiBn) {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('Consecutive BatchNormalization Layers Not Supported, Please Use Only 1.');
      } else if (poolBn) {
        ops.push(hdr(layer, 'Not Supported'));
        ops.push('BatchNormalization after Pooling Layers is Not Supported, Please Change Order.');
      } else {
        ops.push(hdr(layer, 'Supported'));
        ops.push(axisCheck(cfg.axis, modelType, inputLen, rank));
        ops.push(trainableCheck(cfg.trainable === undefined ? true : cfg.trainable));
        ops.push(momentumCheck(cfg.momentum === undefined ? 0.99 : cfg.momentum));
        ops.push(epsilonCheck(cfg.epsilon === undefined ? 0.001 : cfg.epsilon));
        ops.push(centerCheck(cfg.center === undefined ? true : cfg.center));
        ops.push(scaleCheck(cfg.scale === undefined ? true : cfg.scale));
      }
    }

    op.push(ops);
  });

  /* ---- the heap / memory budget --------------------------------------------
     wts_check(), get_max_inter_output() and heap_check() are all commented out
     at the tail of check_model_compatibility(), so nothing today tells the user
     the model will not fit in GPX-10's 256 KB.  Left disabled to match.

     const wtsTotal = wtsCheck(layers);
     const [maxInter, , firstInput] = maxInterOutput(layers);
     op.push(heapCheck(wtsTotal, firstInput, maxInter));
     ---------------------------------------------------------------------- */

  /* ---- post-processing : identical to the tail of check_model_compatibility */
  const NS = (s) => String(s).indexOf('Not Supported') !== -1;
  const filtered = [];
  for (let i = 0; i < op.length; i++) {
    /* py:1079 and py:1084 both index filtered_op[i][0][0] and [0][1], which only
       exist when the row opens with a "name!!!Class!!!status" header.  A row
       without one - the Dense case above, or the python's unreachable
       `else: pass` - raises IndexError there and the model gets no report. */
    if (!op[i].length || String(op[i][0]).indexOf('!!!') === -1) {
      throw new ReportError('IndexError: list index out of range');
    }
    let flag = 0;
    let status = [];
    if (op[i].length === 1 && NS(op[i][0])) {
      status = [null];
      flag = 2;
    } else if (NS(op[i][0])) {
      for (const s of op[i]) {
        if (NS(s)) {
          flag = 3;
          status.push(s);
        }
      }
    } else {
      for (const s of op[i]) {
        if (NS(s)) {
          flag = 1;
          status.push(s);
        }
      }
    }

    if (flag === 1) {
      status.unshift(op[i][0]);
      status[0] = op[i][0].split('!!!').slice(1);
      filtered.push(status);
    } else if (flag === 2) {
      status.unshift(op[i][0]);
      status[0] = op[i][0].split('!!!').slice(1);
      status[status.length - 1] = 'Try Removing This Layer';
      filtered.push(status);
    } else if (flag === 3) {
      status[0] = status[0].split('!!!').slice(1);
      filtered.push(status);
    } else {
      filtered.push([op[i][0].split('!!!').slice(1)]);
    }
  }

  /* num counts MESSAGES, not layers - the python's own arithmetic, which is
     what makes "Issues found in X out of Y layers" read the way it does */
  const num = filtered.map((fo) => fo.length - 1);
  const layerDetails = [];
  const suggestions = [];
  for (let i = 0; i < filtered.length; i++) {
    if (layers[i]) {
      layers[i].supported = filtered[i].length === 1;
      layers[i].issues = filtered[i].slice(1).filter((s) => typeof s === 'string');
    }
    if (filtered[i].length === 1) {
      filtered[i][0] = filtered[i][0][0] + ' ' + filtered[i][0][1] + '.';
      filtered[i].push('Operators Supported.');
      layerDetails.push(filtered[i]);
    } else {
      /* only the FIRST problem of the layer becomes a suggestion */
      const problem = filtered[i][1];
      suggestions.push('In Layer ' + (i + 1) + ' of type ' + filtered[i][0][0] + ', ' + problem);
      filtered[i][0] = filtered[i][0][0] + ' ' + filtered[i][0][1] + '.';
      filtered[i][1] = 'Operators Not Supported.';
      layerDetails.push(filtered[i]);
    }
  }

  return {
    layer_info: [num.reduce((a, b) => a + b, 0), filtered.length],
    layer_details: layerDetails,
    tips: suggestions.length ? suggestions : ['No Suggestions to Give.'],
    layers,
  };
}

/* Same order of business as the head of check_model_compatibility():
     read the file -> load the model -> reject non-Sequential -> report,
   with any failure to read the graph falling through to failure_reason.      */
function checkModelCompatibility(parsed) {
  const cfgRoot = parsed && parsed.config;

  const reject = (tip) => ({
    layer_info: [null, null],
    layer_details: null,
    layers: [],
    tips: [tip],
  });

  try {
    if (!cfgRoot || !cfgRoot.class_name) {
      throw new Error('model_config missing or unreadable');
    }
    /* h5py raises before keras gets to look at the graph at all */
    if (parsed.weights && parsed.weights.__missing__) {
      return reject(
        "Unable to Load Model : 'Unable to synchronously open object "
        + "(component not found)'"
      );
    }
    /* `if type(model) != keras.models.Sequential` - the per-layer emulation
       below walks the layer list as a linear stack, which is meaningless for a
       DAG, so the model-type message must win before it runs. */
    if (canon(cfgRoot.class_name) !== 'Sequential') {
      return reject(
        pyClassRepr(cfgRoot.class_name) +
        ' found, Only Sequential Model Types are Supported, Please change the Model Type.'
      );
    }

/* what the layer constructors do to these two strings, done once up front so
 * the load checks and the report both read the same value keras stored */
    normalizeLayerConfigs(
      cfgRoot.config && cfgRoot.config.layers
    );
/* Sequential.from_config rejects duplicate layer names during load_model().
 * This is an additional load-level check; all existing compatibility checks
 * remain unchanged. */
const duplicateNameError = duplicateLayerNameError(parsed);
if (duplicateNameError) throw new Error(duplicateNameError);

/* stages 1-4 of load_model() on a linear stack */
const ctor = constructorFailure(parsed);
if (ctor) throw new Error(ctor);

    /* an empty Sequential loads in python and reports as compatible */
    return analyse(cfgRoot);
  } catch (err) {
    /* the model loaded and check_model_compatibility() itself raised - that is
       not a load failure and must not be reported as an old-libraries problem */
    if (err instanceof ReportError) return reject('Unable to Analyse Model : ' + err.message);
    return reject(loadFailureReason(parsed, err));
  }
}

/* ============================================================================
 * 3. GRAPH VIEWER
 *    ../components/NetronViewer.jsx renders the real Netron web app, vendored
 *    at public/netron (see public/netron/VENDOR.md) and loaded from our own
 *    origin.  The uploaded File goes straight into Netron's own file-open path.
 *    (Replaces both netron_view.py and the netron server that ran on :8181.)
 * ==========================================================================*/

/* ============================================================================
 * 5. MAIN TAB
 * ==========================================================================*/

const ModelCompatibility = () => {
  const fileInputRef = useRef(null);
  const [modelName, setModelName] = useState('');
  const [fileUploaded, setFileUploaded] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showThreeBoxes, setShowThreeBoxes] = useState(false);
  const [animatingOut, setAnimatingOut] = useState(false);
  const [isCanceling, setIsCanceling] = useState(false);
  const [isUploadScreenOperatorsOpen, setUploadScreenOperatorsOpen] = useState(false);
  const [isLeftOverlayActive, setLeftOverlayActive] = useState(false);

  const [modelFile, setModelFile] = useState(null);
  const [validationResults, setValidationResults] = useState(null);
  const [isValidating, setIsValidating] = useState(false);

  const dragCounterRef = useRef(0);
  const seqRef = useRef(0);

  const splitNameAndExt = (filename) => {
    const name = String(filename ?? '');
    const dotIdx = name.lastIndexOf('.');
    if (dotIdx <= 0 || dotIdx === name.length - 1) return { base: name, ext: '' };
    return { base: name.slice(0, dotIdx), ext: name.slice(dotIdx) };
  };
  const { base: modelBaseName, ext: modelExt } = splitNameAndExt(modelName);

  /* ---------------------------- upload + analyse ----------------------------
     Only the compatibility verdict is computed here; the graph is Netron's job
     and NetronViewer gets the same File independently.                        */
  /* ---------------------------- upload + analyse ----------------------------
     Only the compatibility verdict is computed here; the graph is Netron's job
     and NetronViewer gets the same File independently.                        */
  const processFile = useCallback(async (file) => {
    const seq = ++seqRef.current;
    setIsValidating(true);
    try {
      const buf = await file.arrayBuffer();
      const parsed = readH5(buf, file.name);
      if (seq !== seqRef.current) return;

      const analysis = checkModelCompatibility(parsed);
      if (seq !== seqRef.current) return;
      setValidationResults(analysis);
    } catch (err) {
      if (seq !== seqRef.current) return;
      /* No toast here.  processFile runs the moment the file is picked, which is
         BEFORE the user presses Continue, so a toast would pop on the upload
         screen.  The reason is carried in tips instead, and ModelValidation
         opens the suggestions panel by itself when layer_details is null. */
      const msg = 'Model is not a valid Tensorflow/Keras Model Format, Please Verify Model.';
      setValidationResults({ layer_info: [null, null], layer_details: null, tips: [msg], layers: [] });
    } finally {
      if (seq === seqRef.current) setIsValidating(false);
    }
  }, []);

  const handleFiles = useCallback(
    (file) => {
      if (!file) return;
      const fileName = file?.name || '';
      if (!fileName.toLowerCase().endsWith('.h5')) {
        alert('Please upload a valid .h5 model file');
        return;
      }
      setModelName(file.name);
      setModelFile(file);
      setFileUploaded(true);
      setValidationResults(null);
      processFile(file);
    },
    [processFile]
  );

  const handleUploadClick = () => fileInputRef.current.click();
  const handleFileChange = (event) => handleFiles(event.target.files[0]);
  const handleContinue = () => {
    setAnimatingOut(true);
    setTimeout(() => setShowThreeBoxes(true), 1000);
  };

  const handleCancelUpload = () => {
    setIsCanceling(true);
    seqRef.current++;
    setTimeout(() => {
      setModelName('');
      setModelFile(null);
      setFileUploaded(false);
      setShowThreeBoxes(false);
      setAnimatingOut(false);
      setValidationResults(null);
      setIsValidating(false);
      if (fileInputRef.current) fileInputRef.current.value = null;
      setIsCanceling(false);
    }, 300);
  };

  const rerunValidation = () => {
    if (modelFile) processFile(modelFile);
  };

  /* ------------------------------ drag & drop ------------------------------ */
  useEffect(() => {
    const stop = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };
    const handleDragIn = (e) => {
      stop(e);
      if (showThreeBoxes) return;
      dragCounterRef.current++;
      if (e.dataTransfer.items && e.dataTransfer.items.length > 0) setIsDragging(true);
    };
    const handleDragOut = (e) => {
      stop(e);
      if (showThreeBoxes) return;
      dragCounterRef.current--;
      if (dragCounterRef.current <= 0) {
        dragCounterRef.current = 0;
        setIsDragging(false);
      }
    };
    const handleDrop = (e) => {
      stop(e);
      if (showThreeBoxes) return;
      setIsDragging(false);
      dragCounterRef.current = 0;
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files[0]);
        e.dataTransfer.clearData();
      }
    };

    window.addEventListener('dragenter', handleDragIn);
    window.addEventListener('dragleave', handleDragOut);
    window.addEventListener('dragover', stop);
    window.addEventListener('drop', handleDrop);
    return () => {
      window.removeEventListener('dragenter', handleDragIn);
      window.removeEventListener('dragleave', handleDragOut);
      window.removeEventListener('dragover', stop);
      window.removeEventListener('drop', handleDrop);
    };
  }, [showThreeBoxes, handleFiles]);

  /* --------------------------------- render -------------------------------- */
  return (
    <div className="modelcompatible-upload-wrapper">
      <ToastWrapper />
      {isDragging && (
        <div className="drag-overlay">
          <div className="corner top-left"></div>
          <div className="corner top-right"></div>
          <div className="corner bottom-left"></div>
          <div className="corner bottom-right"></div>
          <span className="mc-drag-scan" aria-hidden="true" />
          <div className="drag-overlay-text">Drop here</div>
        </div>
      )}

      {!showThreeBoxes ? (
        <div className="modelcompatible-upload-screen">
          {/* decorative backdrop only - no interaction, no content */}
          <div className="mc-bg" aria-hidden="true">
            <span className="mc-bg-grid" />
            <span className="mc-bg-orb mc-bg-orb--a" />
            <span className="mc-bg-orb mc-bg-orb--b" />
          </div>

          <button
            type="button"
            className={`upload-screen-operators-tab ${isUploadScreenOperatorsOpen ? 'show' : ''}`}
            onClick={() => setUploadScreenOperatorsOpen((v) => !v)}
          >
            {isUploadScreenOperatorsOpen ? 'Close Operators' : 'Supported Operators'}
          </button>
          <div className={`upload-screen-operators-panel ${isUploadScreenOperatorsOpen ? 'show' : ''}`}>
            <div className="upload-screen-operators-panel-inner">
              <SupportedOperators />
            </div>
          </div>

          <div
            className={`modelcompatible-upload-box ${animatingOut ? 'animate-disappear' : ''} ${fileUploaded ? 'is-loaded' : ''}`}
          >
            {!fileUploaded ? (
              <div className="modelcompatible-upload-content is-idle" onClick={handleUploadClick}>
                <div className="modelcompatible-upload-icon">
                  <span className="mc-icon-ring" aria-hidden="true" />
                  <span className="mc-icon-ring mc-icon-ring--2" aria-hidden="true" />
                  <FaUpload />
                </div>
                <p className="modelcompatible-upload-text">Click to upload or drag and drop</p>
                <p className="modelcompatible-upload-subtext">Support file type(s): model (.h5)</p>
                <input type="file" accept=".h5,.H5" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileChange} />
              </div>
            ) : (
              <div className={`modelcompatible-upload-content ${isCanceling ? 'fade-out' : ''}`}>
                <div className="mc-upload-success" aria-hidden="true">
                  <span className="mc-check-ring" />
                  <FaCheck className="mc-check" />
                </div>
                <p className="file-uploaded-heading">Model Uploaded</p>
                <p className="model-name-display">{modelName}</p>
                {/* <div className="mc-upload-progress" aria-hidden="true">
                  <span />
                </div> */}
                <div className="modelcompatible-action-buttons">
                  <button className="continue-btn" onClick={handleContinue}>
                    Continue
                  </button>
                  <button className="cancel-btn" onClick={handleCancelUpload}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="three-boxes-container three-boxes-outer">
          <div className="model-info-top-fixed">
            <span className="model-name">
              <span className="model-name-base">{modelBaseName}</span>
              {modelExt && <span className="model-name-ext">{modelExt}</span>}
            </span>
            <button className="cancel-btn-small" onClick={handleCancelUpload}>
              Cancel
            </button>
          </div>

          <button
            type="button"
            className={`left-overlay-tab ${isLeftOverlayActive ? 'show' : ''}`}
            onClick={() => setLeftOverlayActive((v) => !v)}
          >
            {isLeftOverlayActive ? 'Close Operators' : 'Supported Operators'}
          </button>
          <div className={`left-overlay-panel ${isLeftOverlayActive ? 'show' : ''}`}>
            <div className="left-overlay-panel-inner">
              <SupportedOperators />
            </div>
          </div>

          <div className="three-boxes-container">
            <div className="model-boxes">
              <div className="model-box">
                <span className="mc-panel-accent" aria-hidden="true" />
                {modelFile ? (
                  <NetronViewer file={modelFile} />
                ) : (
                  <>
                    <FaRocket className="box-icon" />
                    <h3>Model Netron View</h3>
                    <p>Visualize architecture and flow</p>
                  </>
                )}
              </div>

              <div className="model-box">
                <span className="mc-panel-accent" aria-hidden="true" />
                <ModelValidation
                  modelFile={modelFile}
                  results={validationResults}
                  isValidating={isValidating}
                  onValidate={rerunValidation}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelCompatibility;
