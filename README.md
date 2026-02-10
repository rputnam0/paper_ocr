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
PAPER_OCR_MARKER_URL=
PAPER_OCR_GROBID_URL=
PAPER_OCR_MARKER_TIMEOUT=120
PAPER_OCR_GROBID_TIMEOUT=60
```

## Repository Organization

Keep generated artifacts out of project root and use explicit contracts:

- `data/corpora/<slug>/source_pdfs/`: curated source PDFs grouped by corpus/topic.
- `data/jobs/<job_slug>/`: ingestion job folders with `input/`, `pdfs/`, `reports/`, optional `ocr_out/`, optional `logs/`.
- `data/archive/`: old runs and legacy layouts kept for traceability.
- `data/cache/`: disposable caches.
- `data/tmp/`: ephemeral scratch.
- `input/`: local CSV inputs (gitignored).
- `out/`: canonical OCR output target when chosen by CLI invocation.
- `docs/`: design/architecture/operation docs (tracked).

Audit layout any time with:

```bash
uv run paper-ocr data-audit data --strict
```

Detailed contract reference: `docs/data_layout_contract.md`.

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
- `--marker-url` optional Marker API base URL (uses service mode when set)
- `--marker-timeout` default `120`
- `--grobid-url` optional URL (enables TEI enrichment)
- `--grobid-timeout` default `60`
- `--structured-max-workers` default `4`
- `--structured-asset-level standard|full` default `standard`
- `--extract-structured-data` / `--no-extract-structured-data` (default: enabled)
- `--deplot-command` optional external command with `{image}` placeholder
- `--deplot-timeout` default `90`
- `--marker-localize` / `--no-marker-localize` default enabled
- `--marker-localize-profile localize_only|full_json` default `full_json`
- `--layout-fallback none|surya` default `surya`
- `--table-source marker-first|markdown-only` default `marker-first`
- `--table-quality-gate` / `--no-table-quality-gate` default enabled
- `--table-escalation off|auto|always` default `auto`
- `--table-escalation-max` default `20`
- `--table-qa-mode off|warn|strict` default `warn`

### 2) Fetch PDFs from DOI CSV via Telegram bot

```bash
uv run paper-ocr fetch-telegram <doi_csv> [output_root] [options]
```

Arguments:
- `doi_csv`: CSV with DOI column
- `output_root`: default `data/jobs`

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
- `--table-source marker-first|markdown-only` default `marker-first`
- `--table-quality-gate` / `--no-table-quality-gate` default enabled
- `--table-escalation off|auto|always` default `auto`
- `--table-escalation-max` default `20`
- `--table-qa-mode off|warn|strict` default `warn`

This command scans existing OCR document folders, regenerates:
- `metadata/assets/structured/extracted/tables/*`
- `metadata/assets/structured/extracted/figures/*`

and updates each document manifest with `structured_data_extraction`.

### 4) Audit data layout contract

```bash
uv run paper-ocr data-audit [data_dir] [--strict] [--json]
```

Behavior:
- validates top-level `data/` contract (`corpora/jobs/cache/archive/tmp`)
- validates job and corpus folder conventions
- flags misplaced PDFs
- `--strict` returns non-zero on errors

### 5) Evaluate table pipeline against gold set

```bash
uv run paper-ocr eval-table-pipeline <gold_dir> <pred_dir>
```

This command reports baseline table detection metrics (precision/recall) using page-level matching.

## Recommended Workflows

### Workflow A: Local PDF folder -> OCR

```bash
uv run paper-ocr run data/corpora/lisa/source_pdfs out
```

### Workflow A2: Born-digital structured extraction with optional GROBID

```bash
uv run paper-ocr run data/corpora/lisa/source_pdfs out \
  --digital-structured auto \
  --structured-backend hybrid \
  --marker-url http://<marker-host>:8008 \
  --grobid-url http://<grobid-host>:8070 \
  --deplot-command "deplot-cli --image {image}"
```

Notes:
- `--digital-structured auto` applies document-level eligibility rules and falls back safely.
- Marker supports both command mode (`--marker-command`) and service mode (`--marker-url`).
- Marker OCR is forced off by default in command mode (`--disable_ocr` is auto-added and `OCR_ENGINE=None` is set).
- If GROBID is unavailable, run continues without TEI enrichment.

## Routing Semantics (Two "Auto" Modes)

There are two independent auto decisions:

1. Page route auto (`--mode auto`):
- decides per page: `anchored` or `unanchored`.
- this is prompt routing for OCR/text pipeline internals.

2. Document route auto (`--digital-structured auto`):
- decides once per document whether to attempt Marker structured extraction first.
- eligibility requires:
  - at least 60% pages pass `is_structured_page_candidate`
    (clean digital text layer, but less strict than `text_only`, so table/math-heavy pages are included),
  - and either:
    - first page is auto-routed `anchored`, or
    - if first page is not auto-routed `anchored`, at least 70% of body pages pass `is_structured_page_candidate`.
- Structured eligibility always uses auto-derived route signals, even if OCR route mode is forced with `--mode anchored|unanchored`.

Execution flow in plain terms:

- If document is structured-eligible:
  - attempt Marker per page (`structured_ok` on success).
  - if Marker fails for a page: fallback for that page (`structured_fallback`), then continue through normal page pipeline.
- If document is not structured-eligible:
  - use normal page pipeline directly.

Normal page pipeline outcomes:
- `text_only`: born-digital page with strong text layer; no DeepInfra OCR call.
- `ok`: OCR call path succeeded.
- `skipped`: existing output reused (unless `--force`).

Structured page outcomes:
- `structured_ok`: Marker output written for that page.
- `structured_fallback`: Marker failed, page handled by normal pipeline.

## Marker-First Table Pipeline

The table pipeline now runs Marker localization artifacts for all PDFs by default and uses Marker outputs as the first extraction source.

Order of operations:
1. Extract page/document markdown via existing OCR/text flow.
2. Run Marker document localization (`full_document.md`, chunks/blocks/json artifacts).
3. Parse table fragments from Marker artifacts first.
4. Fallback to markdown table parsing only when Marker table fragments are unavailable.
5. Run quality gates and selective escalation for low-quality table fragments.
6. Compare against GROBID table index and emit QA flags (`warn` by default).

Coordinate normalization:
- Canonical internal space is render pixel coordinates (`top_left`, `y-down`).
- Manifest stores per-page transforms:
  - `pdf_page_w_pt`, `pdf_page_h_pt`
  - `render_page_w_px`, `render_page_h_px`
  - `px_per_pt_x`, `px_per_pt_y`
  - `pdf_to_px_transform`, `px_to_pdf_transform`
- GROBID coordinates are transformed into pixel-space before QA matching.

See:
- `docs/table_extraction_pipeline.md`
- `docs/table_qa_metrics.md`

### Service Mode (General)

You can run Marker and GROBID as independent services (local or remote) and point `paper-ocr` at their URLs.

Marker service:
- Set `--marker-url <base_url>` (for example `http://127.0.0.1:8008`)
- Health check: `GET <marker_url>/openapi.json`

GROBID service:
- Set `--grobid-url <base_url>` (for example `http://127.0.0.1:8070`)
- Health check: `GET <grobid_url>/api/isalive`

This keeps `paper-ocr` orchestration separate from service hosting and allows heavy compute to run on dedicated hardware.

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
uv run paper-ocr run data/jobs/papers/pdfs data/jobs/papers/ocr_out
```

## Telegram Job Layout

For `input/papers.csv` (CSV stem = `papers`):

```text
data/jobs/papers/
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
          figures_tables.jsonl
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
- `metadata/assets/structured/grobid/figures_tables.jsonl` provides a GROBID-derived figure/table index with labels, captions, page, and coords for QA/fallback targeting.
- `manifest.json` includes `structured_data_extraction` with table/figure/deplot counts and extraction errors.
- `metadata/assets/structured/marker/full_document.md`, `chunks.jsonl`, and `blocks.jsonl` provide Marker localization artifacts.
- `metadata/assets/structured/extracted/table_fragments.jsonl` and `tables/canonical.jsonl` capture table fragment lineage and merged canonical tables.
- `metadata/assets/structured/qa/table_flags.jsonl` captures Marker-vs-GROBID QA disagreements and low-confidence joins.
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
- `--marker-url` connection errors: verify Marker API is reachable and `openapi.json` loads.
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
