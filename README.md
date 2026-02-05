# paper-ocr

Born-digital-aware PDF to Markdown pipeline using olmOCR via DeepInfra.

## Quick start

```bash
uv run paper-ocr run data/LISA out
```

Set `DEEPINFRA_API_KEY` in `.env` or your environment.

Key options:
- `--text-only` enables text-layer extraction when high-quality text is detected (skips VLM).
- `--metadata-model` sets the post-OCR bibliographic extraction model used for naming outputs.

## Output layout
For each PDF:
```
out/<input_parent_folder>/<author_year>/
  <paper_title>.md
  pages/
    0001.md
    0002.md
  metadata/
    manifest.json
    bibliography.json
    document.jsonl
    assets/
    debug/
```

Notes:
- `author_year` is metadata-derived (snake case, e.g. `Carreau_Pierre_J_1972`) using the first page.
- The consolidated markdown filename is derived from extracted paper title.
