"""CUDA graph replay must consume each input and return independent results."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pytest
import typeseg
from typeseg import _cuda_graph
from typeseg._profiles import model_config


pytestmark = pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="ONNX CUDA provider unavailable",
)


@pytest.mark.parametrize("profile", ["general", "security"])
def test_cuda_graph_matches_ordinary_execution_and_handles_concurrent_inputs(monkeypatch, profile):
    snippets = ["def square(x): return x*x\n", ".button { color: red; }\n", "SELECT * FROM events;\n"]
    texts = [(text * (4096 // len(text) + 1))[:4096] for text in snippets]
    with monkeypatch.context() as patch:
        patch.setattr(_cuda_graph, "window_logits", lambda ids, name: None)
        expected = [typeseg.fast(text, model_set=profile) for text in texts]
    actual = [typeseg.fast(text, model_set=profile) for text in texts]
    name = model_config("unet", profile).get("onnx", "unet_al.onnx")
    if _cuda_graph.active_providers(name) is None:
        pytest.skip("CUDA graph capture unavailable on this device")
    with ThreadPoolExecutor(max_workers=3) as pool:
        concurrent = list(pool.map(lambda text: typeseg.fast(text, model_set=profile), texts))
    for reference, result, parallel in zip(expected, actual, concurrent):
        np.testing.assert_array_equal(result.char_probs, reference.char_probs)
        np.testing.assert_array_equal(parallel.char_probs, reference.char_probs)
        assert result.char_labels == reference.char_labels == parallel.char_labels
