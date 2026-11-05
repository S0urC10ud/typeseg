# Releasing TypeSeg

Use a new version for each release. The package should install with bundled
weights and a prebuilt CPU extension on its supported platforms. CUDA is an
optional setup with the same model weights and Python API.

1. Update `pyproject.toml` to the new version and review the package diff.
   `.github/scripts/prepare_typeseg_assets.py` restores the eight unchanged
   model assets from the checksum-pinned 0.3.0 wheel. If releasing new weights,
   update the asset provenance and manifest explicitly.
2. Run [the package workflow](../../.github/workflows/package.yml) on the release
   commit. It runs on package changes pushed to `codex/` branches and supports
   manual dispatch. It builds a source distribution and native wheels for
   Windows x86-64, manylinux x86-64, and macOS Intel and Apple Silicon.
3. Check all jobs. Wheel builds require the native extension, audit the stable
   ABI, and test clean installations on Python 3.9, 3.12 and 3.14 (3.13 instead
   of 3.14 on Intel Macs, matching ONNX Runtime's wheels). Verify the
   extension loads, parity checks pass, and the ONNX scan still works without
   the extension. `TYPESEG_REQUIRE_NATIVE=1` also enables strict builds locally.
4. Download artifacts from that exact commit. Run `twine check --strict` and
   confirm the version, bundled models, and licenses. Distribution files must
   exclude datasets, augmentation outputs, credentials, and malware samples.
5. Push the reviewed commit and version tag, upload only the audited artifacts,
   and compare their SHA-256 hashes against PyPI's published files. Run a clean
   PyPI install and check `typeseg --diagnose` and inference for both profiles.

The CUDA setup is tested on NVIDIA hardware under WSL2. See the package README
for compatible libraries and installation order. The CPU release workflow does
not require CUDA dependencies.
