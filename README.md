# paper-ocr

Born-digital-aware PDF to Markdown pipeline using olmOCR via DeepInfra.

## Quick start

```bash
paper-ocr run data/LISA out
```

Set `DEEPINFRA_API_KEY` in `.env` or your environment.

Optional:
- `--text-only` enables text-layer extraction when high-quality text is detected (skips VLM).

## Output layout
For each PDF:
```
out/<pdf_name>/
  <pdf_name>.md
  metadata/
    manifest.json
    document.jsonl
    pages/
      0001.md
      0002.md
    assets/
    debug/
```
