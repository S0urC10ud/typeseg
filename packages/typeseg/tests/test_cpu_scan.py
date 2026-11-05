"""CPU fusion must preserve the shipped models and the NumPy recurrence."""
from pathlib import Path
import sys

import numpy as np
import pytest
import typeseg
from typeseg import _onnx_backend as ob
from typeseg import _runtime as rt
from typeseg._profiles import model_config


def test_gpu_mode_explains_how_to_repair_a_shadowed_cuda_runtime(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setenv("TYPESEG_BACKEND", "gpu")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="force-reinstall"):
        ob._providers("unet")


@pytest.mark.parametrize("profile", ["general", "security"])
@pytest.mark.parametrize("model", ["fast", "precise", None])
def test_diagnostics_load_only_the_requested_models(monkeypatch, profile, model):
    loaded = []
    monkeypatch.setattr(ob, "available", lambda name: True)
    monkeypatch.setattr(rt.cub, "available", lambda name: False)
    monkeypatch.setattr(ob, "scan_backend", lambda name: "native-sse2")

    def providers(name):
        loaded.append(name)
        return ["CPUExecutionProvider"]

    monkeypatch.setattr(ob, "active_providers", providers)
    info = typeseg.backend_info(model_set=profile, model=model)
    assert info["backend"] == "onnx"
    assert info["gpu"] is False
    assert len(loaded) == (2 if model is None else 1)
    if model is not None:
        assert ("unet" in loaded[0]) == (model == "fast")


def test_onnx_fallback_works_without_native_extension(monkeypatch):
    text = "print(1)"
    expected = typeseg.precise(text)
    monkeypatch.delattr(typeseg, "_scan_cpu", raising=False)
    monkeypatch.setitem(sys.modules, "typeseg._scan_cpu", None)
    ob._session.cache_clear()
    try:
        actual = typeseg.precise(text)
        assert not ob._session("mamba_al.onnx")._typeseg_fused_scan
        np.testing.assert_allclose(actual.char_probs, expected.char_probs, atol=0.00001)
        assert actual.char_labels == expected.char_labels
    finally:
        ob._session.cache_clear()


@pytest.mark.parametrize("profile", ["general", "security"])
def test_cpu_graph_preserves_weights_and_type_information(profile):
    onnx = pytest.importorskip("onnx")
    from typeseg._onnx_cpu import fuse_mamba_scans

    config = model_config("mamba", profile)
    name = config.get("onnx", "mamba_al.onnx")
    content = (Path(typeseg.__file__).parent / "data" / name).read_bytes()
    original = onnx.load_model_from_string(content)
    fused = onnx.load_model_from_string(fuse_mamba_scans(content))
    initial_states = {node.input[0] for node in original.graph.node if node.op_type == "Scan"}
    assert [value.SerializeToString() for value in fused.graph.initializer] == [
        value.SerializeToString() for value in original.graph.initializer if value.name not in initial_states
    ]
    assert [value.SerializeToString() for value in fused.graph.input] == [
        value.SerializeToString() for value in original.graph.input
    ]
    assert [value.SerializeToString() for value in fused.graph.output] == [
        value.SerializeToString() for value in original.graph.output
    ]
    scans = [node for node in fused.graph.node if node.domain == "typeseg.cpu"]
    assert len(scans) == 12
    assert all(node.op_type == "SelectiveScan" for node in scans)


@pytest.mark.parametrize("profile", ["general", "security"])
def test_cpu_model_logits_match_original_onnx(profile):
    pytest.importorskip("typeseg._scan_cpu")
    import onnxruntime as ort

    config = model_config("mamba", profile)
    name = config.get("onnx", "mamba_al.onnx")
    path = Path(typeseg.__file__).parent / "data" / name
    plain = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    fused = ob._session(name)
    assert fused._typeseg_fused_scan
    rng = np.random.default_rng(144)
    for length in (1, 17, 255, 256, 1023, 1024, 4095, 4096, 4097):
        tokens = rng.integers(0, 257, length, dtype=np.int64)
        inputs = {"ids": ob._compact_ids(tokens)}
        expected = plain.run(None, inputs)[0]
        actual = fused.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, atol=0.0003, rtol=0.0003)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("reverse", [False, True])
def test_native_scan_matches_recurrence_and_rejects_bad_shapes(reverse):
    native = pytest.importorskip("typeseg._scan_cpu")
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort
    from typeseg._mamba_kernel import _selective_scan_seq

    shapes = {"dt": [None, 7], "u": [None, 7], "b": [None, None],
              "c": [None, 16], "a": [7, 16], "d": [7]}
    inputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
              for name, shape in shapes.items()]
    output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, 7])
    node = onnx.helper.make_node("SelectiveScan", list(shapes), ["y"],
                                 domain="typeseg.cpu", reverse=int(reverse))
    model = onnx.helper.make_model(
        onnx.helper.make_graph([node], "scan", inputs, [output]),
        opset_imports=[onnx.helper.make_opsetid("", 17),
                      onnx.helper.make_opsetid("typeseg.cpu", 1)],
        ir_version=9,
    )
    options = ort.SessionOptions()
    options.register_custom_ops_library(native.__file__)
    session = ort.InferenceSession(model.SerializeToString(), options,
                                  providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(331)
    for length in (0, 1, 31, 256):
        data = {name: rng.standard_normal([length if v is None else v for v in shape]).astype(np.float32)
                for name, shape in shapes.items()}
        data["b"] = rng.standard_normal((length, 16)).astype(np.float32)
        data["dt"] = np.abs(data["dt"]) * 0.1
        data["a"] = -np.abs(data["a"]) * 1000
        arrays = [data[name][::-1] if reverse else data[name] for name in ("u", "dt", "b", "c")]
        expected = _selective_scan_seq(np, *arrays, data["a"], data["d"])
        if reverse:
            expected = expected[::-1]
        actual = session.run(None, data)[0]
        np.testing.assert_allclose(actual, expected, atol=0.00001, rtol=0.00001)
    data["b"] = np.zeros((256, 17), dtype=np.float32)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="tensor shapes"):
        session.run(None, data)
