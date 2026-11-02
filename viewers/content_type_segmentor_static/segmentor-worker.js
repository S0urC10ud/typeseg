const STATIC_BASE = './';

// Reuse the cache-busting query the page passed on the worker URL.
const ASSET_VERSION = self.location.search || '';

importScripts(STATIC_BASE + 'js/mamba.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/webgpu_matmul.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/postprocess.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/unet.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/security-postprocess.js' + ASSET_VERSION);

const STATE = {
  manifest: null,
  // Mamba state
  mambaModel: null,
  mambaGpu: null,
  mambaLoaded: false,
  mambaLoading: null,
  // U-Net state
  unetModel: null,
  unetLoaded: false,
  unetLoading: null,
  // Current selection
  activeModel: 'mamba',
  activeSet: 'general',
  models: new Map(),
  loading: new Map(),
};

async function fetchOrThrow(url, type='text') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url} (${response.status}).`);
  return type === 'arrayBuffer' ? response.arrayBuffer() : response.text();
}

async function loadManifest() {
  if (STATE.manifest) return STATE.manifest;
  const manifestStr = await fetchOrThrow(`${STATIC_BASE}assets/model_manifest.json${ASSET_VERSION}`, 'text');
  STATE.manifest = JSON.parse(manifestStr);
  return STATE.manifest;
}

async function loadMamba() {
  if (STATE.mambaLoaded) return;
  if (STATE.mambaLoading) return STATE.mambaLoading;
  
  STATE.mambaLoading = (async () => {
    const manifest = await loadManifest();
    
    const weightsBuffer = await fetchOrThrow(`${STATIC_BASE}assets/sfullfiles4_weights.bin`, 'arrayBuffer');
    const metaStr = await fetchOrThrow(`${STATIC_BASE}assets/sfullfiles4_weights_meta.json`, 'text');
    const meta = JSON.parse(metaStr);
    
    const weightsDict = {};
    for (const [key, info] of Object.entries(meta)) {
      weightsDict[key] = new Float32Array(weightsBuffer, info.offset * 4, info.length);
    }
    STATE.mambaModel = new MambaSegmentor(manifest, weightsDict);
    
    const gpu = new WebGPUMatMul();
    await gpu.init();
    STATE.mambaGpu = gpu;
    
    STATE.mambaLoaded = true;
  })();
  
  await STATE.mambaLoading;
}

async function loadSelected(modelName, modelSet) {
  if (!['unet', 'mamba'].includes(modelName)) throw new Error('Unknown model.');
  if (!['general', 'security'].includes(modelSet)) throw new Error('Unknown model set.');
  const manifest = await loadManifest();
  if (modelSet === 'general') {
    await (modelName === 'unet' ? loadUNet() : loadMamba());
    return { model: modelName === 'unet' ? STATE.unetModel : STATE.mambaModel, gpu: STATE.mambaGpu };
  }
  const config = manifest.model_sets?.security?.[modelName];
  if (!config) throw new Error(`Security ${modelName} is unavailable in this build.`);
  const key = `${modelSet}/${modelName}/${config.id}`;
  if (STATE.models.has(key)) return STATE.models.get(key);
  if (!STATE.loading.has(key)) {
    const pending = (async () => {
      const buffer = await fetchOrThrow(`${STATIC_BASE}assets/${config.weights}`, 'arrayBuffer');
      const metaText = await fetchOrThrow(`${STATIC_BASE}assets/${config.metadata}`, 'text');
      for (const [kind, bytes] of [['bin', buffer], ['metadata', new TextEncoder().encode(metaText)]]) {
        const actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
        if (actual !== config.sha256[kind]) throw new Error(`Security ${kind} checksum mismatch.`);
      }
      const metadata = JSON.parse(metaText), weights = {};
      for (const [name, info] of Object.entries(metadata)) weights[name] = new Float32Array(buffer, info.offset * 4, info.length);
      const modelManifest = { ...manifest, model: { ...manifest.model, ...config } };
      const model = modelName === 'unet' ? new UNetSegmentor(modelManifest, weights) : new MambaSegmentor(modelManifest, weights);
      let gpu = null;
      if (modelName === 'mamba') { gpu = new WebGPUMatMul(); await gpu.init(); }
      const value = { model, gpu, config }; STATE.models.set(key, value); return value;
    })();
    STATE.loading.set(key, pending);
  }
  try { return await STATE.loading.get(key); }
  finally { STATE.loading.delete(key); }
}

async function loadUNet() {
  if (STATE.unetLoaded) return;
  if (STATE.unetLoading) return STATE.unetLoading;
  
  STATE.unetLoading = (async () => {
    const manifest = await loadManifest();
    
    const weightsBuffer = await fetchOrThrow(`${STATIC_BASE}assets/unet_al_weights.bin`, 'arrayBuffer');
    const metaStr = await fetchOrThrow(`${STATIC_BASE}assets/unet_al_weights_meta.json`, 'text');
    const meta = JSON.parse(metaStr);
    
    const weightsDict = {};
    for (const [key, info] of Object.entries(meta)) {
      weightsDict[key] = new Float32Array(weightsBuffer, info.offset * 4, info.length);
    }
    STATE.unetModel = new UNetSegmentor(manifest, weightsDict);
    
    STATE.unetLoaded = true;
  })();
  
  await STATE.unetLoading;
}

async function ensureModelLoaded(modelName = 'mamba', modelSet = 'general') {
  await loadSelected(modelName, modelSet);
  STATE.activeModel = modelName;
  STATE.activeSet = modelSet;
  const manifest = await loadManifest();
  return {
    loaded: true,
    model_id: `${modelSet === 'security' ? 'Security' : 'General'} ${modelName === 'unet' ? 'U-Net' : 'Mamba'}`,
    num_classes: manifest.num_classes,
    window_bytes: manifest.window_bytes,
    window_stride_bytes: manifest.window_stride_bytes,
    other_threshold: manifest.other_threshold,
    max_input_bytes: manifest.max_input_bytes,
    postprocess_min_run_chars: manifest.postprocess_min_run_chars,
    postprocess_boundary_snap_max_shift: manifest.postprocess_boundary_snap_max_shift,
    label_order: manifest.label_order,
    display_labels: manifest.display_labels,
    runtime: STATE.activeModel === 'unet' ? 'unet-js' : 'hybrid-webgpu',
    activeModel: STATE.activeModel,
    model_set: STATE.activeSet,
  };
}

async function runSegment(payloadObj) {
  const text = payloadObj?.text || '';
  // The UI preserves source offsets by normalizing to visible ASCII first.
  // Reject direct unsupported worker input instead of confusing UTF-8 bytes
  // with JavaScript character offsets.
  if (/[^\x20-\x7E\n\t]/.test(text)) throw new Error('Normalize input to ASCII before browser segmentation.');
  const threshold = payloadObj?.threshold;
  const ppOptions = payloadObj?.ppOptions || {};
  
  const manifest = await loadManifest();
  
  const nextThreshold = Number.isFinite(Number(threshold))
    ? Number(threshold)
    : Number(manifest?.other_threshold ?? 0.3);
    
  const started = performance.now();
  
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  const seq = bytes.length;
  if (seq > manifest.max_input_bytes) throw new Error(`Browser input exceeds ${manifest.max_input_bytes} bytes.`);
  
  const inputArr = new Int32Array(seq);
  for (let i = 0; i < seq; i++) inputArr[i] = bytes[i];
  
  if (seq === 0) {
    return {
      text, segments: [], char_top_probs: [], char_confidences: [], stats: [],
      input_bytes: 0, window_count: 0, other_threshold: nextThreshold,
      elapsed_ms: 0, runtime: STATE.activeModel
    };
  }

  // Run inference based on active model
  let probs;
  const selected = await loadSelected(STATE.activeModel, STATE.activeSet);
  if (STATE.activeSet === 'security') {
    // Demo input is capped at6144 bytes: a single full-context research bucket.
    // Fixed1536 windows changed GroupNorm statistics and lost security accuracy.
    let width = STATE.activeModel === 'unet' ? Math.ceil(seq / 256) * 256 : 1024;
    if (STATE.activeModel === 'mamba') while (width < seq) width *= 2;
    const padded = new Int32Array(width).fill(256); padded.set(inputArr);
    const full = STATE.activeModel === 'unet'
      ? selected.model._predict_single_window(padded)
      : await selected.model.predict_window_probs(padded, selected.gpu);
    probs = full.slice(0,seq * manifest.num_classes);
  } else if (STATE.activeModel === 'unet') probs = selected.model.predict_window_probs(inputArr);
  else probs = await selected.model.predict_window_probs(inputArr, selected.gpu);
  
  const num_classes = manifest.num_classes;
  const raw_labels = new Int32Array(seq);
  for(let i = 0; i < seq; i++) {
    let best_j = 0;
    let best_p = -1;
    for(let j=0; j < num_classes; j++) {
      if (probs[i*num_classes+j] > best_p) {
        best_p = probs[i*num_classes+j];
        best_j = j;
      }
    }
    raw_labels[i] = best_j;
  }

  const decoder = STATE.activeSet === 'security' ? securityPostprocessTraced : postprocessCharLabelsTraced;
  const ppResult = decoder(text, raw_labels, probs, {
    threshold: nextThreshold,
    otherId: num_classes,
    ppOptions: ppOptions
  });
  const final_labels = ppResult.labels;

  // Per-char provenance: which post-processing steps changed the label,
  // as [{step, from, to}, ...] per character (null when untouched).
  const char_pp_trace = new Array(seq).fill(null);
  let prev_stage_labels = raw_labels;
  for (const stage of ppResult.stages) {
    const cur = stage.labels;
    for (let i = 0; i < seq; i++) {
      if (cur[i] !== prev_stage_labels[i]) {
        if (!char_pp_trace[i]) char_pp_trace[i] = [];
        char_pp_trace[i].push({ step: stage.step, from: prev_stage_labels[i], to: cur[i] });
      }
    }
    prev_stage_labels = cur;
  }

  const segments = [];
  let currentLabel = -1;
  let currentStart = 0;
  for (let i = 0; i < seq; i++) {
    const L = final_labels[i];
    if (L !== currentLabel) {
      if (currentLabel !== -1) {
        segments.push({ start: currentStart, end: i, label_id: currentLabel });
      }
      currentLabel = L;
      currentStart = i;
    }
  }
  if (currentLabel !== -1) {
    segments.push({ start: currentStart, end: seq, label_id: currentLabel });
  }

  const statsMap = new Map();
  for(let i=0; i<seq; i++) {
    const L = final_labels[i];
    statsMap.set(L, (statsMap.get(L) || 0) + 1);
  }
  const stats = [];
  for(const [id, count] of statsMap.entries()) {
    stats.push({ id, count, pct: (count / seq) * 100 });
  }

  const char_top_probs = [];
  const char_confidences = [];
  for(let i=0; i<seq; i++) {
    const p = [];
    for(let j=0; j<num_classes; j++) {
      p.push({ id: j, prob: probs[i*num_classes+j] });
    }
    p.sort((a,b) => b.prob - a.prob);
    char_top_probs.push(p.slice(0, 5));
    char_confidences.push(p[0].prob);
  }

  return {
    text,
    segments,
    char_top_probs,
    char_confidences,
    char_pp_trace,
    stats,
    input_bytes: seq,
    window_count: STATE.activeModel === 'unet' && STATE.activeSet === 'general' ? Math.max(1, Math.ceil((seq - 1536) / 768) + 1) : 1,
    other_threshold: nextThreshold,
    elapsed_ms: Number((performance.now() - started).toFixed(1)),
    runtime: STATE.activeModel === 'unet' ? 'unet-js' : 'hybrid-webgpu',
    model_id: `${STATE.activeSet === 'security' ? 'Security' : 'General'} ${STATE.activeModel === 'unet' ? 'U-Net' : 'Mamba'}`,
    activeModel: STATE.activeModel,
    model_set: STATE.activeSet,
    model_checkpoint_sha256: selected.config?.checkpoint_sha256 || null,
  };
}

self.addEventListener('message', async event => {
  const { type, requestId, payload } = event.data || {};
  try {
    if (type === 'load') {
      const result = await ensureModelLoaded(payload?.model, payload?.modelSet);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    if (type === 'switch_model') {
      const result = await ensureModelLoaded(payload?.model, payload?.modelSet);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    if (type === 'segment') {
      const result = await runSegment(payload || {});
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    throw new Error(`Unknown worker message type: ${String(type)}`);
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId,
      payload: { message: error instanceof Error ? error.message : String(error) },
    });
  }
});
