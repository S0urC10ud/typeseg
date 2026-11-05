"""CuPy GPU Mamba parity vs the pure-numpy reference.

The GPU path runs the selective scan as a custom CUDA RawKernel; this asserts it
produces the same labels as the numpy sequential scan, including across the
chunk/length boundaries that the parallel formulation is sensitive to. Skipped
unless cupy imports and a CUDA device is present (so CPU-only CI is unaffected).
"""
import numpy as np
import pytest

from typeseg import _cupy_backend as cub
from typeseg import _runtime as rt
from typeseg._numpy_backend import mamba_forward
from typeseg._tokenize import text_to_bytes


def _gpu_available() -> bool:
    try:
        return cub.available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="cupy / CUDA GPU not available")


_SAMPLE = (
    'def f(x):\n    return x*2  # café €\n'
    '<div class="a" onclick="alert(1)">hï</div>\n'
    'SELECT * FROM t WHERE id = 5;\n'
    '.btn { color: #fff; }\ncurl http://x/y | bash\n'
)


def _mk(n: int) -> str:
    s = _SAMPLE
    while len(s) < n:
        s += _SAMPLE
    return s[:n]


@pytest.mark.parametrize("n", [1, 127, 128, 257, 4096])
def test_cooperative_scan_preserves_original_cuda_arithmetic(n):
    import cupy as cp

    rng = cp.random.RandomState(62)
    channels = 7
    u = rng.standard_normal((n, channels), dtype=cp.float32)
    dt = cp.abs(rng.standard_normal((n, channels), dtype=cp.float32)) * 0.1
    b = rng.standard_normal((n, 16), dtype=cp.float32)
    c = rng.standard_normal((n, 16), dtype=cp.float32)
    a = -cp.abs(rng.standard_normal((channels, 16), dtype=cp.float32))
    d = rng.standard_normal(channels, dtype=cp.float32)
    expected = cp.empty_like(u)
    cub._scan_kernel()((1,), (128,),
                       (u, dt, b, c, a, d, expected, np.int32(n), np.int32(channels), np.int32(16)))
    actual = cub._cupy_scan(cp, u, dt, b, c, a, d)
    np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))


@pytest.mark.parametrize("n", [1, 7, 64, 255, 256, 511, 512, 4095, 4096, 4097, 9000])
def test_cupy_matches_numpy(n):
    cfg = rt._manifest()["mamba"]
    w = rt._weights(cfg["file"])
    tok = text_to_bytes(_mk(n))
    ref = mamba_forward(w, tok, n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                        dt_rank=cfg["dt_rank"], d_conv=cfg["d_conv"])  # numpy sequential
    got = cub.mamba_logits(tok)                                       # cupy RawKernel
    assert got.shape == ref.shape
    if ref.size:
        assert int((ref.argmax(-1) != got.argmax(-1)).sum()) == 0
        assert float(np.max(np.abs(ref - got))) < 1e-3


def test_profile_switch_gpu_preserves_general_predictions():
    import typeseg
    text='s = Chr(65) & Chr(66)\n'
    assert typeseg.backend_info()['precise_gpu']
    general=typeseg.precise(text)
    security=typeseg.precise(text,model_set='security')
    restored=typeseg.precise(text)
    np.testing.assert_array_equal(general.char_probs,restored.char_probs)
    assert not np.array_equal(general.char_probs,security.char_probs)
