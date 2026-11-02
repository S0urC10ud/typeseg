"""Versioned model profiles. General remains the backward-compatible default."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((files("typeseg") / "data" / "manifest.json").read_text())


def profile(model_set: str = "general") -> dict:
    if model_set not in ("general", "security"):
        raise ValueError("model_set must be 'general' or 'security'")
    root = manifest()
    if model_set == "general":
        return root
    value = root.get("model_sets", {}).get(model_set)
    if value is None:
        raise ValueError("The security model profile is not included in this distribution")
    return {**root, **value}


def model_config(architecture: str, model_set: str = "general") -> dict:
    value = profile(model_set).get(architecture)
    if not value:
        raise ValueError(f"No {architecture} model is available in the {model_set} profile")
    return value


def verify_asset(name: str, content: bytes) -> None:
    """Check versioned security assets once when their backend cache loads them."""
    for value in manifest().get("model_sets", {}).values():
        for architecture in ("unet", "mamba"):
            model = value.get(architecture) or {}
            for kind, filename in (("npz", model.get("file")), ("onnx", model.get("onnx"))):
                if filename == name:
                    expected = model.get("sha256", {}).get(kind)
                    if expected and hashlib.sha256(content).hexdigest() != expected:
                        raise ValueError(f"Model asset checksum mismatch: {name}")
