"""Weight loading, windowed inference, and result assembly."""
from __future__ import annotations

import io
import platform
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import _cupy_backend as cub
from . import _numpy_backend as nb
from . import _onnx_backend as ob
from ._numpy_backend import Weights, mamba_forward, unet_forward
from ._options import Options
from ._profiles import manifest as _manifest, model_config, profile, verify_asset
from ._security import decode as security_decode
from ._postprocess import (
    build_segments,
    confidence_gate,
    normalize_short_runs,
    paired_delimiter_fill,
    relabel_whitespace,
    snap_boundaries,
)
from ._segmentation import Segmentation
from ._tokenize import PAD_BYTE_ID, byte_probs_to_char, text_to_bytes

try:  # Python 3.9+
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover
    from importlib_resources import files as _files  # type: ignore

DEFAULT_CHUNK = 1536

_warned_no_gpu = False


def _maybe_warn_no_gpu() -> None:
    """On macOS, warn once that GPU backends are unavailable (CPU/ONNX only).

    Wheel installs run no install-time hook, so we cannot print during
    ``pip install``; this is the first runtime opportunity to tell a Mac user why
    ``typeseg[gpu]`` resolved to the plain CPU package. Only fires in auto mode —
    an explicit ``TYPESEG_BACKEND`` (numpy forced, or gpu/cuda fail-fast) is the
    user's own choice and is left to the backend to honour. Suppressible via the
    standard ``warnings`` filters or ``TYPESEG_BACKEND=numpy``.
    """
    global _warned_no_gpu
    if _warned_no_gpu or platform.system() != "Darwin":
        return
    if ob._mode():  # explicit backend choice — don't second-guess it
        _warned_no_gpu = True
        return
    _warned_no_gpu = True
    warnings.warn(
        "typeseg: running on CPU (ONNX). GPU backends are unavailable on macOS — "
        "neither onnxruntime-gpu nor cupy-cuda12x publishes macOS wheels and Apple "
        "has no NVIDIA CUDA, so 'typeseg[gpu]' installs the same CPU package. "
        "This is expected; the CPU ONNX backend is fast. Silence this warning with "
        "TYPESEG_BACKEND=numpy or Python's warnings filters.",
        RuntimeWarning,
        stacklevel=2,
    )


@lru_cache(maxsize=4)
def _weights(npz_name: str) -> Weights:
    blob = (_files("typeseg") / "data" / npz_name).read_bytes()
    verify_asset(npz_name, blob)
    with np.load(io.BytesIO(blob)) as data:
        flat = {k.replace("__", "/"): np.asarray(data[k], dtype=np.float32) for k in data.files}
    return Weights(flat)


def _window_spans(length: int, chunk: int) -> List[Tuple[int, int]]:
    if length <= 0:
        return []
    win = max(64, chunk)
    if length <= win:
        return [(0, length)]
    stride = max(1, win // 2)
    spans, start = [], 0
    while True:
        end = min(start + win, length)
        spans.append((start, end))
        if end >= length:
            break
        start += stride
    return spans


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 0),), dtype=np.float32)
    pos = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    return np.exp(-0.5 * (pos / 0.5) ** 2).astype(np.float32)


def _unet_windows(byte_tokens: np.ndarray, spans) -> Tuple[np.ndarray, List[int]]:
    """Stack each span into a fixed-width (1536) PAD-filled row.

    The U-Net pools 7x; every window runs at the fixed model width (padding short
    windows with PAD) so they don't collapse, then output is sliced back. Mirrors
    the viewer."""
    rows = np.full((len(spans), DEFAULT_CHUNK), PAD_BYTE_ID, dtype=np.int64)
    lens: List[int] = []
    for i, (s, e) in enumerate(spans):
        win_len = e - s
        rows[i, :win_len] = byte_tokens[s:e]
        lens.append(win_len)
    return rows, lens


def _unet_byte_probs(byte_tokens: np.ndarray, channels, num_classes: int, cfg=None) -> np.ndarray:
    cfg = cfg or _manifest()["unet"]
    if cfg.get("inference_policy") == "research-context-v2":
        return _security_unet_byte_probs(byte_tokens, cfg, num_classes)
    onnx_name = cfg.get("onnx", "unet_al.onnx")
    length = int(byte_tokens.shape[0])
    spans = _window_spans(length, DEFAULT_CHUNK)
    rows, _lens = _unet_windows(byte_tokens, spans)
    if ob.available(onnx_name):
        logits_all = ob.unet_window_logits(rows, name=onnx_name)      # (N, 1536, C)
    else:
        w = _weights(cfg["file"])
        logits_all = np.stack(
            [unet_forward(w, rows[i], channels=tuple(channels)) for i in range(rows.shape[0])],
            axis=0,
        )
    accum = np.zeros((length, num_classes), dtype=np.float32)
    wsum = np.zeros((length,), dtype=np.float32)
    for i, (s, e) in enumerate(spans):
        win_len = e - s
        probs = nb._softmax(logits_all[i, :win_len], axis=-1)
        wt = _window_weights(win_len)
        accum[s:e] += probs * wt[:, None]
        wsum[s:e] += wt
    wsum = np.where(wsum <= 0, 1.0, wsum)
    return accum / wsum[:, None]


def _security_unet_byte_probs(tokens, cfg, num_classes):
    """Match research context and PAD geometry; GroupNorm depends on both.

    Full sequence up to 200000 bytes, rounded to 256. Larger sources use1536
    windows in groups of12, each group padded to its largest rounded length.
    """
    length = len(tokens)
    spans = [(0, length)] if length <= cfg['full_context_max_bytes'] else _window_spans(length,1536)
    accum = np.zeros((length,num_classes),np.float32)
    sums = np.zeros(length,np.float32)
    for offset in range(0,len(spans),12):
        group = spans[offset:offset+12]
        width = ((max(b-a for a,b in group)+255)//256)*256
        rows = np.full((len(group),width),PAD_BYTE_ID,np.int64)
        for i,(a,b) in enumerate(group): rows[i,:b-a] = tokens[a:b]
        if ob.available(cfg['onnx']): logits = ob.unet_window_logits(rows,name=cfg['onnx'])
        else:
            weights = _weights(cfg['file'])
            logits = np.stack([unet_forward(weights,row,channels=tuple(cfg['channels'])) for row in rows])
        for i,(a,b) in enumerate(group):
            probabilities = nb._softmax(logits[i,:b-a],axis=-1)
            weights = _window_weights(b-a)
            accum[a:b] += probabilities*weights[:,None]; sums[a:b] += weights
    return accum / np.maximum(sums[:,None],np.finfo(np.float32).tiny)


def _mamba_logits(byte_tokens: np.ndarray, cfg: dict) -> np.ndarray:
    # CuPy GPU (parallel scan) > ONNX CPU > pure numpy. The ONNX `Scan` op is
    # slower on GPU than CPU, so GPU acceleration for Mamba comes via CuPy.
    npz_name, onnx_name = cfg["file"], cfg.get("onnx", "mamba_al.onnx")
    if cub.available(npz_name):
        logits = cub.mamba_logits(byte_tokens, name=npz_name, config=cfg)
    elif ob.available(onnx_name):
        logits = ob.mamba_logits(byte_tokens, name=onnx_name)
    else:
        w = _weights(npz_name)
        logits = mamba_forward(w, byte_tokens, n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                               dt_rank=cfg["dt_rank"], d_conv=cfg["d_conv"])
    return logits


def _mamba_byte_probs(byte_tokens: np.ndarray, cfg: dict) -> np.ndarray:
    # Preserve the security models' full-context padding and large-file windows.
    length = len(byte_tokens)
    chunk = cfg.get("window_bytes")
    if cfg.get('inference_policy') == 'research-context-v2':
        spans = [(0,length)] if length <= cfg['full_context_max_bytes'] else _window_spans(length,int(chunk))
        accum = None
        sums = np.zeros(length,np.float32)
        for a,b in spans:
            width = 1024
            while width < b-a: width *= 2
            tokens = np.full(width,PAD_BYTE_ID,np.int64);tokens[:b-a] = byte_tokens[a:b]
            probabilities = nb._softmax(_mamba_logits(tokens,cfg)[:b-a],axis=-1)
            if accum is None: accum = np.zeros((length,probabilities.shape[1]),np.float32)
            weights = _window_weights(b-a)
            accum[a:b] += probabilities*weights[:,None]; sums[a:b] += weights
        return accum / np.maximum(sums[:,None],np.finfo(np.float32).tiny)
    return nb._softmax(_mamba_logits(byte_tokens, cfg), axis=-1)


def _assemble(text: str, byte_probs: np.ndarray, options: Options, model_set="general") -> Segmentation:
    man = _manifest()
    labels_names: List[str] = man["labels"]
    other_index = man["other_index"]
    other_label = man["other_label"]

    char_probs, char_labels = byte_probs_to_char(text, byte_probs)
    labels = [int(x) for x in char_labels]

    # Order follows the thesis four steps (whitespace -> confidence -> boundary
    # snap -> min-run); the extra paired-delimiter fill runs last, after min-run
    # has reabsorbed short low-confidence runs that would otherwise split a
    # delimiter-wrapped region.
    if model_set == "security":
        labels = security_decode(text, char_probs, options, other_index).tolist()
        # Optional caller overrides. The security defaults disable these two
        # General refinements so the canonical research policy stays explicit.
        if options.boundary_snap:
            labels = snap_boundaries(text, labels, char_probs, options.boundary_snap_max_shift,
                                     options.min_run_chars)
        if options.paired_delimiter_fill:
            labels = paired_delimiter_fill(text, labels, char_probs, options.paired_delimiter_max_shift)
    else:
        labels = _general_decode(text, labels, char_probs, options, other_index)

    segments, char_label_names, char_conf = build_segments(
        text, labels, char_probs, labels_names, other_index, other_label
    )
    return Segmentation(text=text, segments=segments, char_labels=char_label_names,
                        char_confidence=char_conf, char_probs=char_probs, labels=labels_names)


def _general_decode(text, labels, char_probs, options, other_index):
    if options.whitespace_relabel:
        labels = relabel_whitespace(text, labels)
    if options.confidence_gating:
        labels = confidence_gate(char_probs, labels, options.other_threshold, other_index)
    if options.boundary_snap:
        labels = snap_boundaries(text, labels, char_probs, options.boundary_snap_max_shift,
                                 options.min_run_chars)
    if options.min_run_normalize:
        labels = normalize_short_runs(labels, char_probs, options.min_run_chars, other_index)
    if options.paired_delimiter_fill:
        labels = paired_delimiter_fill(text, labels, char_probs, options.paired_delimiter_max_shift)
    if options.whitespace_relabel:
        labels = relabel_whitespace(text, labels)

    return labels


def _empty(text: str) -> Segmentation:
    man = _manifest()
    char_probs = np.zeros((0, int(man["num_classes"])), dtype=np.float32)
    return Segmentation(text=text, segments=[], char_labels=[], char_confidence=[],
                        char_probs=char_probs, labels=list(man["labels"]))


def run(model: str, text: str, options: Optional[Options], *, model_set="general") -> Segmentation:
    cfg = model_config("unet" if model == "fast" else "mamba", model_set)
    if options is None:
        options = Options.security() if model_set == "security" else Options()
    _maybe_warn_no_gpu()
    if not text:
        return _empty(text)
    byte_tokens = text_to_bytes(text)
    if byte_tokens.size == 0:
        return _empty(text)
    man = _manifest()
    num_classes = man["num_classes"]
    if model == "fast":
        byte_probs = _unet_byte_probs(byte_tokens, cfg["channels"], num_classes, cfg)
    else:
        byte_probs = _mamba_byte_probs(byte_tokens, cfg)
    return _assemble(text, byte_probs, options, model_set)


def backend_info(*, model_set="general") -> dict:
    """Report the active inference backend (for diagnostics).

    Providers reflect what each model's session *actually loaded* — a CUDA provider
    that fails to initialise (and falls back to CPU) is reported as CPU. ``fast``
    (U-Net) may run on CUDA while ``precise`` (Mamba) stays on CPU, because the
    Mamba selective-scan is slower on GPU than CPU.
    """
    # fast() = U-Net (ONNX, CUDA-capable). precise() = Mamba: CuPy GPU > ONNX CPU > numpy.
    selected = profile(model_set)
    u, m = selected.get("unet"), selected.get("mamba")
    unet_onnx = u.get("onnx", "unet_al.onnx") if u else None
    mamba_onnx = m.get("onnx", "mamba_al.onnx") if m else None
    fast_p = ob.active_providers(unet_onnx) if u and ob.available(unet_onnx) else []
    backend = "onnx" if fast_p else "numpy"
    try:
        cupy_on = bool(m and cub.available(m["file"]))
    except Exception:
        cupy_on = False
    if cupy_on:
        precise_p = cub.active_providers(m["file"])
    elif m and ob.available(mamba_onnx):
        precise_p = ob.active_providers(mamba_onnx)
    else:
        precise_p = []
    return {
        "backend": backend,
        "gpu": ("CUDAExecutionProvider" in fast_p) or cupy_on,
        "fast_providers": fast_p,
        "precise_providers": precise_p,
        "precise_gpu": cupy_on,
        "model_set": model_set,
        "fast_model": u["file"] if u else None,
        "precise_model": m["file"] if m else None,
    }
