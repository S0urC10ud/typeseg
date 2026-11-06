---
title: Installation
description: Install the TypeSeg Python package for local CPU inference, with optional CUDA support.
---
Install [TypeSeg from PyPI](https://pypi.org/project/typeseg/) with Python 3.9 or later:

```bash
pip install typeseg
```

The package includes general and security U-Net and Mamba models. NumPy and ONNX Runtime are installed automatically. No API key, model download step, or GPU is needed for CPU inference. Use Python 3.9–3.13 on Intel Macs for ONNX Runtime compatibility.

For throughput-oriented text analysis, start with `typeseg.fast(text)` or `typeseg --model fast file.txt`.

## Optional GPU support

The [CUDA setup guide](https://github.com/S0urC10ud/typeseg/blob/main/packages/typeseg/README.md#running-on-gpu) covers the `gpu` extra, matching CUDA and cuDNN libraries, and the ONNX Runtime installation order. The setup is tested on Linux and WSL2 with NVIDIA hardware. macOS uses CPU inference.

After setup, check that your chosen model reports `"gpu": true`.

```bash
typeseg --diagnose --model fast
typeseg --diagnose --model precise
```

The same Python API selects an available GPU backend. CUDA support uses ONNX Runtime for U-Net and CuPy for Mamba.

## Check the installation

```bash
python -c "import typeseg; print(typeseg.fast('SELECT name FROM users;'))"
```

Continue with [Quick Start](../quick-start/) for CLI and Python examples, or try the [browser demo](../../introduction/web-demo/) without installing anything.
