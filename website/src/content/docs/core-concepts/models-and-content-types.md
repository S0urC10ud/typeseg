---
title: Models & Content Types
description: Choose U-Net for throughput or Mamba for context, across 35 learned textual labels.
---
## Choose a model

| Model | Python API | Good starting point for | Learned parameters |
| --- | --- | --- | --- |
| U-Net | `typeseg.fast(text)` | CPU deployment and bulk text inspection | 1.52M |
| Mamba | `typeseg.precise(text)` | Inspecting where content types change | 1.36M |

Both are small deep learning models. Parameters are the learned weights that determine their predictions. U-Net uses overlapping 1,536-character sections; Mamba carries context across the file.

Start with **U-Net for bulk processing or CPU speed**. **Mamba can improve labels near the starts and ends of regions**, but is much slower on CPU in the research benchmark. Overall label F1 is 94.9% for U-Net and 95.1% for Mamba; F1 near type changes is 45.3% and 57.2%, respectively. [Metric definitions and speed measurements](../../security/performance/) explain the comparison.

The **CLI and browser demo default to Mamba**. Use `typeseg --model fast file.txt` or choose U-Net in the demo’s Settings for faster inspection. Python calls choose the model explicitly through `fast()` or `precise()`.

**General** remains the default model set. Select the **Security** specialist with `model_set="security"` in Python, `--model-set security` in the CLI, or **U-Net-Sec / Mamba-Sec** in the demo’s visible model selector. It specializes in mixed scripts and encodings and can be less accurate on ordinary languages. These checkpoints differ from the paper’s security models.

Security Mamba is the custom bidirectional **Mamba1D**, a Mamba-1-style selective SSM with six layers, `d_model=256`, `d_state=16`, `expand=1`, `dt_rank=16`, `d_conv=4`, 130 compact embeddings and 35 classes. Portable inference uses FP32.

## Supported content types

The model learns 35 textual labels:

| Category | Labels |
| --- | --- |
| Programming languages (17) | `python`, `javascript_typescript`, `java`, `c_family`, `csharp`, `go`, `rust`, `ruby`, `php`, `swift`, `kotlin`, `scala`, `dart`, `visual_basic`, `shell`, `powershell`, `sql` |
| Markup and documentation (7) | `html`, `xml`, `svg`, `css`, `markdown`, `restructuredtext`, `tex` |
| Data and configuration (6) | `json`, `yaml`, `csv`, `text`, `dockerfile`, `gettext_catalog` |
| Text encodings (5) | `encoding_hex`, `encoding_base64`, `encoding_base32`, `encoding_base58`, `encoding_base85` |

`c_family` groups C, C++, and Objective-C. `javascript_typescript` groups JavaScript and TypeScript. The browser uses the shorter display name `js_ts`.

Confidence gating adds an auxiliary `other` output for uncertain positions. The model learns 35 classes; `other` is assigned during post-processing. Unsupported languages can still receive a known label.

TypeSeg handles textual content. Binary executables, archives, images, and overlapping interpretations of the same bytes are outside its current scope.
