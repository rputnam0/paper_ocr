# paper-ocr

Born-digital-aware PDF -> Markdown OCR pipeline with two ingestion paths:
- Local PDFs (`paper-ocr run`)
- DOI CSV via Telegram bot (`paper-ocr fetch-telegram`)

The pipeline is optimized for technical papers and produces per-paper Markdown, structured metadata, and folder-level discovery artifacts.

## What This Project Does

- Routes each page through anchored or unanchored prompting based on text-layer heuristics.
- Supports text-only extraction when high-quality text layers are available.
- Renders pages with OCR-safe sizing constraints.
- Parses model YAML front matter and writes normalized page outputs.
- Extracts bibliography metadata from first-page content.
- Names output folders/files from extracted paper metadata when possible.
- Extracts discovery metadata (`paper_summary`, `key_topics`, `sections`) from early pages.
- Generates group-level `README.md` indexes for discoverability.

## High-Level Architecture

1. **Ingest**: find PDFs and compute stable identifiers/hashes.
2. **Inspect**: determine page route (`anchored` vs `unanchored`) from text heuristics.
3. **Render**: convert pages to model-ready images.
4. **OCR Call**: send page + prompt to DeepInfra OpenAI-compatible endpoint.
5. **Postprocess**: parse YAML front matter and normalize page markdown/metadata.
6. **Document Assembly**: merge pages, produce `document.jsonl`, write consolidated markdown.
7. **Metadata Enrichment**: extract bibliography + discovery JSON.
8. **Store**: emit deterministic output bundle + group readmes.

Telegram DOI fetch is a pre-ingest source step that fills a PDF job folder used by OCR.

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/) for environment and command execution
- DeepInfra API key for OCR calls
- Telegram API credentials for DOI fetch flow

## Installation

```bash
uv sync
```

## Environment Configuration

Create `.env` in project root.

### Required for OCR (`paper-ocr run`)

```ini
DEEPINFRA_API_KEY=...
```

### Required for Telegram fetch (`paper-ocr fetch-telegram`)

```ini
TG_API_ID=...
TG_API_HASH=...
TARGET_BOT=@your_bot_username
```

Optional fetch defaults:

```ini
MIN_DELAY=4
MAX_DELAY=8
```

## CLI Overview

### 1) OCR existing PDFs

```bash
uv run paper-ocr run <in_dir> <out_dir> [options]
```

Core options:
- `--workers` default `32`
- `--model` default `allenai/olmOCR-2-7B-1025`
- `--base-url` default `https://api.deepinfra.com/v1/openai`
- `--max-tokens` default `8192`
- `--force`
- `--mode auto|anchored|unanchored` default `auto`
- `--debug`
- `--scan-preprocess`
- `--text-only` / `--no-text-only` (default: `--text-only`)
- `--metadata-model` default `nvidia/Nemotron-3-Nano-30B-A3B`

### 2) Fetch PDFs from DOI CSV via Telegram bot

```bash
uv run paper-ocr fetch-telegram <doi_csv> [output_root] [options]
```

Arguments:
- `doi_csv`: CSV with DOI column
- `output_root`: default `data/telegram_jobs`

Fetch options:
- `--doi-column` default `DOI`
- `--target-bot` required unless `TARGET_BOT` is set in env
- `--session-name` default `nexus_session`
- `--min-delay` default `4`
- `--max-delay` default `8`
- `--response-timeout` default `15` (seconds per poll)
- `--search-timeout` default `40` (seconds total after `searching...`)
- `--debug`
- `--report-file` override report path
- `--failed-file` override failed-report path

## Recommended Workflows

### Workflow A: Local PDF folder -> OCR

```bash
uv run paper-ocr run data/LISA out
```

### Workflow B: DOI CSV -> Telegram fetch -> OCR

1. Put CSV in `input/` (ignored by git):

```bash
input/papers.csv
```

2. Fetch PDFs:

```bash
uv run paper-ocr fetch-telegram input/papers.csv
```

3. OCR fetched PDFs:

```bash
uv run paper-ocr run data/telegram_jobs/papers/pdfs data/telegram_jobs/papers/ocr_out
```

## Telegram Job Layout

For `input/papers.csv` (CSV stem = `papers`):

```text
data/telegram_jobs/papers/
  input/
    papers.csv
  pdfs/
    <bot_paper_title>.pdf
  reports/
    telegram_download_report.csv
    telegram_failed_papers.csv
    download_index.json
  ocr_out/
```

Notes:
- `pdfs/` is the OCR input stage for fetched content.
- `ocr_out/` is reserved for final OCR outputs from `paper-ocr run`.
- PDF names are derived from bot-provided titles (DOI fallback if title unavailable).
- `download_index.json` keeps DOI -> filename mapping stable across reruns.

## OCR Output Layout

For each processed PDF:

```text
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

Behavior notes:
- Folder naming prefers extracted author/year metadata; falls back safely when missing.
- Consolidated markdown filename is derived from extracted title.
- Group-level readmes are generated for folder-level discoverability.

## Development

Run tests:

```bash
uv run pytest
```

Project layout:
- `src/paper_ocr/cli.py` CLI entrypoints
- `src/paper_ocr/telegram_fetch.py` DOI/Telegram retrieval flow
- `src/paper_ocr/ingest.py` PDF discovery + hashing
- `src/paper_ocr/inspect.py` routing heuristics
- `src/paper_ocr/render.py` page rendering
- `src/paper_ocr/client.py` model API calls/retries
- `src/paper_ocr/postprocess.py` YAML front matter parsing
- `src/paper_ocr/store.py` output helpers
- `src/paper_ocr/schemas.py` manifest schema

## Troubleshooting

- `Missing DEEPINFRA_API_KEY`: set OCR key in `.env`.
- `Missing TG_API_ID/TG_API_HASH`: set Telegram credentials in `.env`.
- `Missing target bot`: set `TARGET_BOT` or pass `--target-bot`.
- First Telegram run prompts login/2FA and creates local `.session` file.
- `database is locked`: stop concurrent fetch processes using same session.
- Frequent `Timeout`: increase `--search-timeout` (e.g. `30-60`).
- `FloodWaitError`: reduce request rate by increasing `--min-delay/--max-delay`.
- High DeepInfra usage on born-digital PDFs: ensure `--no-text-only` is not set (text-only is default) and use `--force` when re-running old outputs so pages are reprocessed under current settings.

## Security and Privacy

- Do not commit `.env`, Telegram `.session` files, or private input CSVs.
- `input/*.csv` is ignored by default for local datasets.
- Rotate credentials immediately if exposed.

## Contributing

- Use `uv` commands (`uv sync`, `uv run ...`, `uv run pytest`).
- Add/adjust tests with behavior changes.
- Keep output structure deterministic and idempotent.
- Prefer small, reviewable commits.
