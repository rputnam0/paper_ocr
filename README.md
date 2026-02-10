# paper-ocr

Born-digital-aware PDF -> Markdown OCR pipeline with two ingestion paths:
- Local PDFs (`paper-ocr run`)
- DOI CSV via Telegram bot (`paper-ocr fetch-telegram`)

The pipeline is optimized for technical papers and produces per-paper Markdown, structured metadata, and folder-level discovery artifacts.

## What This Project Does

- Routes each page through anchored or unanchored prompting based on text-layer heuristics.
- Supports text-only extraction when high-quality text layers are available.
- Supports structured born-digital extraction via external Marker (OCR disabled) with fallback to existing OCR/text paths.
- Exports machine-readable table/figure artifacts from markdown and Marker assets for downstream data pipelines.
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
4. **Structured Born-Digital Path (optional)**: run external Marker page extraction (with `OCR_ENGINE=None`) and optionally enrich metadata/sections with GROBID TEI.
5. **OCR Call**: send page + prompt to DeepInfra OpenAI-compatible endpoint when structured extraction is disabled or a structured page falls back.
6. **Postprocess**: parse YAML front matter and normalize page markdown/metadata.
7. **Document Assembly**: merge pages, produce `document.jsonl`, write consolidated markdown.
8. **Metadata Enrichment**: extract bibliography + discovery JSON.
9. **Store**: emit deterministic output bundle + group readmes.
10. **Structured Data Export**: parse markdown tables/figures into JSON/CSV and optionally run a DePlot-compatible command on figure images.

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

Optional born-digital structured defaults:

```ini
PAPER_OCR_DIGITAL_STRUCTURED=auto
PAPER_OCR_MARKER_COMMAND=marker_single
PAPER_OCR_GROBID_URL=
PAPER_OCR_MARKER_TIMEOUT=120
PAPER_OCR_GROBID_TIMEOUT=60
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
- `--digital-structured off|auto|on` default `auto`
- `--structured-backend marker|hybrid` default `hybrid`
- `--marker-command` default `marker_single`
- `--marker-timeout` default `120`
- `--grobid-url` optional URL (enables TEI enrichment)
- `--grobid-timeout` default `60`
- `--structured-max-workers` default `4`
- `--structured-asset-level standard|full` default `standard`
- `--extract-structured-data` / `--no-extract-structured-data` (default: enabled)
- `--deplot-command` optional external command with `{image}` placeholder
- `--deplot-timeout` default `90`

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

### 3) Export structured data from existing OCR outputs

```bash
uv run paper-ocr export-structured-data <ocr_out_dir> [options]
```

Options:
- `--deplot-command` optional external command with `{image}` placeholder
- `--deplot-timeout` default `90`

This command scans existing OCR document folders, regenerates:
- `metadata/assets/structured/extracted/tables/*`
- `metadata/assets/structured/extracted/figures/*`

and updates each document manifest with `structured_data_extraction`.

## Recommended Workflows

### Workflow A: Local PDF folder -> OCR

```bash
uv run paper-ocr run data/LISA out
```

### Workflow A2: Born-digital structured extraction with optional GROBID

```bash
uv run paper-ocr run data/LISA out \
  --digital-structured auto \
  --structured-backend hybrid \
  --marker-command marker_single \
  --grobid-url http://<wsl-host>:8070 \
  --deplot-command "deplot-cli --image {image}"
```

Notes:
- `--digital-structured auto` applies document-level eligibility rules and falls back safely.
- Marker is invoked as an external command and OCR is forced off by default (`--disable_ocr` is auto-added and `OCR_ENGINE=None` is set).
- If GROBID is unavailable, run continues without TEI enrichment.

WSL GPU note:
- Marker auto-selects CUDA when available in its runtime environment.
- For non-interactive SSH sessions, you may need to export `PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"` so both `marker_single` and `nvidia-smi` are discoverable.

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
- Fetch progress is persisted incrementally, so rerunning the same CSV resumes safely:
  existing papers are skipped, remaining DOIs continue.

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
      structured/
        marker/
          page_0001.md
          page_0001.json
          page_0001_assets/
        grobid/
          fulltext.tei.xml
        extracted/
          manifest.json
          tables/
            manifest.jsonl
            p0001_t01.csv
            p0001_t01.json
          figures/
            manifest.jsonl
            deplot/
              p0001_f01.json
    debug/
```

Behavior notes:
- Folder naming prefers extracted author/year metadata; falls back safely when missing.
- Consolidated markdown filename is derived from extracted title.
- Group-level readmes are generated for folder-level discoverability.
- `manifest.json` includes a `structured_extraction` block with `enabled`, `backend`, `grobid_used`, `fallback_count`, and `structured_page_count`.
- `manifest.json` includes `structured_data_extraction` with table/figure/deplot counts and extraction errors.
- Figure records are resolved against Marker page asset folders, making embedded figure files addressable for downstream ML extraction.

## Development

Run tests:

```bash
uv run pytest
```

Project layout:
- `src/paper_ocr/cli.py` CLI entrypoints
- `src/paper_ocr/telegram_fetch.py` DOI/Telegram retrieval flow
- `src/paper_ocr/structured_extract.py` Marker/GROBID integration + markdown normalization
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
- `marker_single: command not found`: install Marker separately or pass a valid `--marker-command`.
- Marker extraction fails on some pages: run continues via fallback; inspect page-level `status=structured_fallback` in `manifest.json`.
- GROBID connection errors: verify `--grobid-url` points to reachable service endpoint from this machine.

## Security and Privacy

- Do not commit `.env`, Telegram `.session` files, or private input CSVs.
- `input/*.csv` is ignored by default for local datasets.
- Rotate credentials immediately if exposed.

## Contributing

- Use `uv` commands (`uv sync`, `uv run ...`, `uv run pytest`).
- Add/adjust tests with behavior changes.
- Keep output structure deterministic and idempotent.
- Prefer small, reviewable commits.
