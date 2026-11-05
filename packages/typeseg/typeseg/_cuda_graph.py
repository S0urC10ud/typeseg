"""Bounded CUDA graph cache for repeated U-Net window shapes."""
from functools import lru_cache
from importlib.resources import files
from threading import Lock
from weakref import WeakValueDictionary

import numpy as np
import onnxruntime as ort

from ._profiles import verify_asset


_active = WeakValueDictionary()


class _WindowGraph:
    def __init__(self, name, shape):
        self.lock = Lock()
        content = (files("typeseg") / "data" / name).read_bytes()
        verify_asset(name, content)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            content, options,
            providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA graph execution provider could not load")
        self.input = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.int64, "cuda", 0)
        classes = self.session.get_outputs()[0].shape[-1]
        self.output = ort.OrtValue.ortvalue_from_shape_and_type((*shape, classes), np.float32, "cuda", 0)
        self.binding = self.session.io_binding()
        self.binding.bind_ortvalue_input("ids", self.input)
        self.binding.bind_ortvalue_output("logits", self.output)

    def run(self, ids):
        # Captured device addresses are shared by calls using this shape.
        with self.lock:
            if self.session is None:
                return None
            try:
                self.input.update_inplace(ids)
                self.session.run_with_iobinding(self.binding)
                return self.binding.copy_outputs_to_cpu()[0]
            except Exception:
                # Release the graph arena before the caller retries ordinary CUDA.
                self.binding = self.input = self.output = self.session = None
                return None


@lru_cache(maxsize=4)
def _graph(name, shape):
    try:
        return _WindowGraph(name, shape)
    except Exception:
        return None


def window_logits(ids, name):
    # Limit retained activation memory for long full-context Security inputs.
    if ids.shape[1] > 32768 or ids.size > 131072:
        return None
    graph = _graph(name, tuple(ids.shape))
    if graph is None:
        return None
    result = graph.run(ids)
    if result is not None:
        _active[name] = graph
    return result


def active_providers(name):
    graph = _active.get(name)
    session = graph.session if graph is not None else None
    if session is not None:
        return session.get_providers()
    return None
