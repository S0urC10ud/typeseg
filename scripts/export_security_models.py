"""Export the two frozen Security checkpoints for Python and the browser.

Run after export_typeseg_weights.py and export_typeseg_onnx.py. Requires numpy,
flax and onnx at build time. No training data or training code is needed.
"""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import onnx
from flax.serialization import msgpack_restore

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/typeseg/typeseg/data"
BROWSER = ROOT / "viewers/content_type_segmentor_static/assets"
OUTPUT = ROOT / "build/security-release"
CHECKPOINTS = {
    "unet": "12f7cddf8286be865f633a835dcd7f0e2c925d41cf09a5d8af49a001f34ff50f",
    "mamba": "aab2662e09e05278a1f42e4a77e0636a1dee46e5be50090759a97449a8bf506f",
}
MAMBA_KEYS = {
    "ln_scale": "LayerNorm_0/scale", "ln_bias": "LayerNorm_0/bias",
    "in_proj_kernel": "Dense_0/kernel", "in_proj_bias": "Dense_0/bias",
    "conv_kernel": "Conv_0/kernel", "conv_bias": "Conv_0/bias",
    "x_proj_kernel": "Dense_1/kernel", "x_proj_bias": "Dense_1/bias",
    "dt_proj_kernel": "Dense_2/kernel", "dt_proj_bias": "Dense_2/bias",
    "out_proj_kernel": "Dense_3/kernel", "out_proj_bias": "Dense_3/bias",
    "a_log": "A_log", "d": "D",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "/"))
        else:
            result[name] = np.asarray(value, dtype=np.float32)
    return result


def browser_arrays(arch, flat):
    # Browser token IDs are raw bytes plus padding. The compact vocabulary uses
    # one shared row for all non-ASCII bytes; expand that lookup without rounding.
    table = np.r_[np.arange(128), np.full(128, 128), 129].astype(int)
    embedding = flat["Embed_0/embedding"][table]
    if arch == "unet":
        return {**flat, "Embed_0/embedding": embedding}

    arrays = {
        "embed/embedding": embedding,
        "final/ln_scale": flat["LayerNorm_0/scale"],
        "final/ln_bias": flat["LayerNorm_0/bias"],
        "final/dense_kernel": flat["Dense_0/kernel"],
        "final/dense_bias": flat["Dense_0/bias"],
    }
    for i in range(6):
        prefix = f"CheckpointMambaBlock1D_{i}/"
        for target, source in MAMBA_KEYS.items():
            value = flat[prefix + source]
            arrays[f"blocks/{i}/{target}"] = value[:, 0, :] if target == "conv_kernel" else value
    return arrays


def write_browser_weights(name, arrays):
    binary = OUTPUT / (name + ".bin")
    metadata_path = OUTPUT / (name + "_meta.json")
    metadata, offset = {}, 0
    with binary.open("wb") as destination:
        for key, value in arrays.items():
            metadata[key] = {"offset": offset, "length": int(value.size), "shape": list(value.shape)}
            destination.write(value.astype("<f4").tobytes())
            offset += value.size
    write_json(metadata_path, metadata)

    restored = np.fromfile(binary, dtype="<f4")
    for key, value in arrays.items():
        entry = metadata[key]
        actual = restored[entry["offset"]:entry["offset"] + entry["length"]].reshape(entry["shape"])
        np.testing.assert_array_equal(actual, value)
    return binary, metadata_path


def export(arch, manifest, browser_manifest, exporter):
    checkpoint = ROOT / "checkpoints" / (arch + "_sec.msgpack")
    if sha256(checkpoint) != CHECKPOINTS[arch]:
        raise ValueError(f"Unexpected frozen {arch} Security checkpoint checksum")
    tree = msgpack_restore(checkpoint.read_bytes())
    flat = flatten(tree.get("params", tree))
    if not all(np.isfinite(value).all() for value in flat.values()):
        raise ValueError(f"Non-finite {arch} parameters")

    name = arch + "_sec_format_20261101"
    npz = OUTPUT / (name + ".npz")
    np.savez_compressed(npz, **{key.replace("/", "__"): value for key, value in flat.items()})
    with np.load(npz) as restored:
        for key, value in flat.items():
            np.testing.assert_array_equal(restored[key.replace("/", "__")], value)

    config = {
        **manifest[arch],
        "file": npz.name,
        "onnx": name + "_context_v2.onnx",
        "checkpoint_sha256": CHECKPOINTS[arch],
        "id": name,
        "dtype": "float32",
        "num_token_embeddings": 130,
        "d_model": 256,
        "inference_policy": "research-context-v2",
        "full_context_max_bytes": 200000,
        "pad_bucket_bytes": 256 if arch == "unet" else 1024,
    }
    if arch == "unet":
        graph = exporter.build_unet(flat, config["channels"], 35, seq=None)
    else:
        graph = exporter.build_mamba(flat, 6, 16, 16, 4, 35)
        config.update({
            "n_layers": 6, "d_state": 16, "expand": 1, "dt_rank": 16,
            "d_conv": 4, "bidirectional": True,
            "window_bytes": 10000, "window_stride": 5000,
            "variant": "Bidirectional Mamba1D, Mamba-1-style selective SSM, 6 layers, "
                       "d_model=256, d_state=16, expand=1, dt_rank=16, d_conv=4",
        })
    onnx_path = OUTPUT / config["onnx"]
    onnx.save(graph, onnx_path)
    onnx.checker.check_model(str(onnx_path))
    config["sha256"] = {"npz": sha256(npz), "onnx": sha256(onnx_path)}

    binary, metadata = write_browser_weights(name, browser_arrays(arch, flat))
    manifest["model_sets"]["security"][arch] = config
    browser_manifest["model_sets"]["security"][arch] = {
        **config, "weights": binary.name, "metadata": metadata.name,
        "sha256": {**config["sha256"], "bin": sha256(binary), "metadata": sha256(metadata)},
        "layer_norm_eps": 1e-6,
    }
    for source in (npz, onnx_path):
        shutil.copy2(source, PACKAGE / source.name)
    for source in (binary, metadata):
        shutil.copy2(source, BROWSER / source.name)
    print(f"Exported {arch}-Sec: {sum(value.size for value in flat.values()):,} FP32 parameters")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(PACKAGE / "manifest.json")
    browser_manifest = read_json(BROWSER / "model_manifest.json")
    profile = {"id": "security-format-20261101", "decoder": "security-canonical-v1"}
    manifest.setdefault("model_sets", {})["security"] = {
        **profile,
        "defaults": {
            "other_threshold": 0.3, "min_run_chars": 3,
            "boundary_snap": False, "paired_delimiter_fill": False,
        },
    }
    browser_manifest.setdefault("model_sets", {})["security"] = profile.copy()
    spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts/export_typeseg_onnx.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    for arch in CHECKPOINTS:
        export(arch, manifest, browser_manifest, exporter)
    write_json(PACKAGE / "manifest.json", manifest)
    write_json(BROWSER / "model_manifest.json", browser_manifest)


if __name__ == "__main__":
    main()
