"""Restore the unchanged TypeSeg model assets from the pinned 0.3.0 wheel."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


SEED_URL = (
    "https://files.pythonhosted.org/packages/3a/51/"
    "d62edd204593b28c8c6be616332fed974b96ed52b9559ad887c48dac2e13/"
    "typeseg-0.3.0-py3-none-any.whl"
)
SEED_SHA256 = "7b576781c033b8ff6d413f9f41ee33dbff373e898df3bf2279cfb23ebcf3333b"
DATA_DIR = Path(__file__).resolve().parents[2] / "packages/typeseg/typeseg/data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="use a local copy of the pinned seed wheel")
    args = parser.parse_args()
    if args.wheel:
        blob = args.wheel.read_bytes()
    else:
        with urlopen(SEED_URL, timeout=60) as response:
            blob = response.read()
    if hashlib.sha256(blob).hexdigest() != SEED_SHA256:
        raise ValueError("Seed wheel checksum mismatch")

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    profiles = [manifest, manifest["model_sets"]["security"]]
    names = {
        name
        for profile in profiles
        for architecture in ("unet", "mamba")
        for name in (
            profile[architecture]["file"],
            profile[architecture].get("onnx", f"{architecture}_al.onnx"),
        )
    }
    if len(names) != 8 or any(Path(name).name != name for name in names):
        raise ValueError("Expected eight model asset basenames")
    with ZipFile(io.BytesIO(blob)) as wheel:
        if json.loads(wheel.read("typeseg/data/manifest.json")) != manifest:
            raise ValueError("The source and seed wheel must use the same model manifest")
        assets = {name: wheel.read("typeseg/data/" + name) for name in names}
    for name, asset in assets.items():
        destination = DATA_DIR / name
        if destination.is_file() and destination.read_bytes() != asset:
            raise ValueError(f"Existing asset differs from the pinned wheel: {name}")
    for name, asset in assets.items():
        (DATA_DIR / name).write_bytes(asset)
    print(f"Verified and restored {len(assets)} unchanged TypeSeg 0.3.0 model assets")


if __name__ == "__main__":
    main()
