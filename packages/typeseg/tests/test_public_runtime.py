"""Public installs must support predictable CPU selection and useful diagnostics."""
import json
import platform
import re
import sys
import warnings

import onnxruntime as ort
import pytest
import typeseg
from typeseg import _cli, _cupy_backend as cub, _onnx_backend as ob


def unexpected_gpu_probe(*args, **kwargs):
    raise AssertionError("explicit CPU mode must not inspect or initialize CUDA")


def test_cpu_mode_ignores_even_a_broken_gpu_install(monkeypatch):
    monkeypatch.setenv("TYPESEG_BACKEND", "cpu")
    monkeypatch.setattr(ort, "get_available_providers", unexpected_gpu_probe)
    monkeypatch.setattr(cub, "_has_device", unexpected_gpu_probe)
    monkeypatch.setattr(cub, "_probe_compile", unexpected_gpu_probe)
    assert ob._providers("unet") == ["CPUExecutionProvider"]
    assert ob._providers("mamba") == ["CPUExecutionProvider"]
    assert not cub.available()


@pytest.mark.parametrize("model", ["fast", "precise"])
def test_explicit_cpu_reports_the_backend_that_actually_loaded(monkeypatch, model):
    monkeypatch.setenv("TYPESEG_BACKEND", "cpu")
    info = typeseg.backend_info(model=model, model_set="security")
    assert info["backend"] == "onnx"
    assert not info["gpu"]
    if model == "precise":
        assert info["precise_scan"] in (
            "native-avx512", "native-sse2", "native-scalar", "onnx-scan"
        )
    else:
        assert info["precise_scan"] is None


def test_diagnose_does_not_read_input_or_run_segmentation(monkeypatch, capsys):
    class NoInput:
        def read(self):
            raise AssertionError("diagnose must not wait for stdin")

    monkeypatch.setattr(sys, "stdin", NoInput())
    monkeypatch.setattr(_cli, "fast", unexpected_gpu_probe)
    monkeypatch.setattr(_cli, "precise", unexpected_gpu_probe)
    monkeypatch.setenv("TYPESEG_BACKEND", "cpu")
    _cli.main(["--diagnose", "--model-set", "security"])
    info = json.loads(capsys.readouterr().out)
    assert info["requested_backend"] == "cpu"
    assert info["model"] == "precise"
    assert info["model_set"] == "security"
    assert info["version"] == typeseg.__version__
    assert info["backend"] == "onnx"


def test_diagnose_runtime_failure_has_an_actionable_message(monkeypatch, capsys):
    def unavailable(**kwargs):
        raise RuntimeError("CUDA unavailable; use TYPESEG_BACKEND=cpu")

    monkeypatch.setattr(_cli, "backend_info", unavailable)
    with pytest.raises(SystemExit) as exc:
        _cli.main(["--diagnose"])
    assert exc.value.code == 1
    assert "TYPESEG_BACKEND=cpu" in capsys.readouterr().err


def test_ordinary_mac_cpu_use_does_not_emit_a_missing_gpu_warning(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setenv("TYPESEG_BACKEND", "cpu")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        typeseg.fast("print('hello')")
    assert not any("GPU backends are unavailable" in str(item.message) for item in emitted)


@pytest.mark.parametrize("color_mode", [None, "never", "always"])
def test_cli_escapes_controls_and_honors_color_settings(monkeypatch, capsys, tmp_path, color_mode):
    monkeypatch.setenv("TYPESEG_BACKEND", "cpu")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    if color_mode == "never":
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", "1")
    if color_mode is None:
        monkeypatch.delenv("TYPESEG_COLOR", raising=False)
    else:
        monkeypatch.setenv("TYPESEG_COLOR", color_mode)
    text = "x\x1b[31my\x07\x08\x9bz\n"
    source = tmp_path / "controls.txt"
    source.write_text(text, encoding="utf-8")

    _cli.main(["--model", "fast", str(source)])

    output = capsys.readouterr().out
    assert ("\x1b[0m" in output) == (color_mode == "always")
    plain = re.sub(r"\x1b\[(?:0|1|2|(?:38|48);2;\d+;\d+;\d+)m", "", output)
    assert all(ch not in plain for ch in ("\x1b", "\x07", "\x08", "\x9b"))
    for escaped in (r"\x1b", r"\x07", r"\x08", r"\x9b"):
        assert plain.count(escaped) >= 2  # Both the text body and segment previews.
    assert f"{len(text)} chars" in plain


def test_colored_segment_repr_escapes_controls(monkeypatch):
    monkeypatch.setenv("TYPESEG_COLOR", "always")
    segment = typeseg.Segment(0, 5, "text", 1.0, "x\x1b[2J")
    rendered = repr(segment)
    assert r"\x1b[2J" in rendered
    assert "\x1b[2J" not in rendered
