# paper-ocr

Born-digital-aware PDF to Markdown pipeline using olmOCR via DeepInfra.

## Quick start

```bash
uv run paper-ocr run data/LISA out
```

Set `DEEPINFRA_API_KEY` in `.env` or your environment.

## Telegram DOI fetch workflow

1. Fetch PDFs from your Telegram bot into an OCR input folder:
```bash
uv run paper-ocr fetch-telegram papers.csv data/inbox
```
2. Run OCR on fetched PDFs:
```bash
uv run paper-ocr run data/inbox out
```

Required env vars for fetch:
- `TG_API_ID`
- `TG_API_HASH`

Optional env vars for fetch defaults:
- `TARGET_BOT` (default `@your_bot_username`)
- `MIN_DELAY` (default `4`)
- `MAX_DELAY` (default `8`)

CLI timeout default for bot response:
- `--response-timeout` (default `25` seconds)

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
