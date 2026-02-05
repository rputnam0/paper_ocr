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
  - Also used for discovery metadata extraction from the first few pages (abstract + topics).

## Output layout
For each PDF:
```
out/<input_parent_folder>/
  README.md
out/<input_parent_folder>/<author_year>/
  <paper_title>.md
  pages/
    0001.md
    0002.md
  metadata/
    manifest.json
    bibliography.json
    discovery.json
    sections.json
    document.jsonl
    assets/
    debug/
```

Notes:
- `author_year` is metadata-derived (snake case, e.g. `Carreau_Pierre_J_1972`) using the first page.
- The consolidated markdown filename is derived from extracted paper title.
- Each parent output folder also gets a generated `README.md` paper index for quick LLM discovery.
- Discovery summary is abstract-first: extracted from the opening pages instead of whole-document summarization.
