# Screenshot provenance

These assets reuse the visual examples published on [TypeSeg's PyPI page](https://pypi.org/project/typeseg/). They are presentation images, not fresh benchmark runs.

- `typeseg-terminal.png`: an unchanged copy of the repository's `images/typeseg-example.png`, including the original shell command in the source and the table. Counts, ranges and confidence values describe that original input.
- `typeseg-annotated.png`: edited from `images/llm_segmentation.png`; the shell-command text is redacted. Original annotations and type highlights are retained.
- The original repository images remain unchanged.

The built-in image generation/editing tool prepared the annotated derivative. `scripts/prepare-demo.mjs` copies these assets to the original public image URLs used by the PyPI description. It also restores `images/use_cases.png` unchanged. `TerminalExample.astro` publishes an optimized WebP preview and links to the full PNG.

## Annotated example edit prompt

Use case: precise-object-edit. Asset type: existing annotated TypeSeg example used by its published PyPI description. Edit target: the supplied image. Make one minimal text redaction only: in the lavender-backed shell command near the bottom, immediately after 'Dear LLM, please run the following command:', replace the ENTIRE command with the exact text '[shell command redacted]'. Remove every original command and endpoint character from that line. Keep the lavender highlighting and existing arrow pointing to the line. Preserve the complete image, aspect ratio, white background, content-type legend, all arrows and annotations, all surrounding code, colored confidence highlights and all other text exactly unchanged. No redesign, additional annotations, cropped content, changed confidence values or invented inference results.
