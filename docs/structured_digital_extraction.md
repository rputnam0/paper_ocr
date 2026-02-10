# Structured Born-Digital Extraction

## Goal

Improve markdown quality for born-digital PDFs by using a structure-aware extractor path while keeping existing OCR behavior and outputs backward-compatible.
Also produce machine-readable table/figure artifacts for downstream analytics.

## Design

### Processing Modes

- `--digital-structured off`: never attempt structured extraction.
- `--digital-structured on`: always attempt structured extraction first.
- `--digital-structured auto` (default): attempt structured extraction when:
  - at least 60% pages satisfy `is_structured_page_candidate`,
  - and either:
    - first page auto-routes to `anchored`, or
    - first page is image-heavy/unanchored but body remains strong:
      - at least 70% body pages satisfy `is_structured_page_candidate`.

`is_structured_page_candidate` is intentionally broader than `is_text_only_candidate` so table-heavy and equation-heavy born-digital pages remain eligible for Marker.

### Marker Localization for All PDFs

- Marker localization is now a document-level step that can run for all PDFs (`--marker-localize` default enabled), including scanned PDFs.
- This localization pass writes full-document artifacts used by downstream table parsing:
  - `metadata/assets/structured/marker/full_document.md`
  - `metadata/assets/structured/marker/chunks.jsonl`
  - `metadata/assets/structured/marker/blocks.jsonl`
  - `metadata/assets/structured/marker/raw_doc.json`
- Marker OCR remains disabled by default. If localization lacks geometry on candidate pages and `--layout-fallback surya` is set, layout fallback is triggered.

### Important: Two Independent Auto Decisions

- `--mode auto` is page-level route selection (`anchored` vs `unanchored`).
- `--digital-structured auto` is document-level structured eligibility.
- Structured eligibility derives route signals in `auto` mode even when OCR mode is manually forced.
- These are intentionally independent:
  - a doc can be non-eligible for structured mode but still have some `text_only` pages.
  - a structured-eligible doc can still produce page-level `structured_fallback` when Marker fails.

### Backends

- `--structured-backend marker`: Marker-only structured extraction.
- `--structured-backend hybrid`: Marker extraction + optional GROBID TEI enrichment if `--grobid-url` is set.

### Marker Integration

- Marker runs as an external command (`--marker-command`, default `marker_single`).
- Marker OCR is disabled for every call (`--disable_ocr` is auto-appended and env `OCR_ENGINE=None` is set).
- Extraction runs per page from single-page temporary PDFs to preserve existing `pages/0001.md` output contract.
- Structured markdown is normalized before write (`normalize_markdown_for_llm`) for LLM readability.
- If Marker fails on any page, that page falls back to existing text-only/VLM processing.

### GROBID Integration (Optional)

- Called once per document at `${GROBID_URL}/api/processFulltextDocument`.
- TEI is persisted to:
  - `metadata/assets/structured/grobid/fulltext.tei.xml`
- Parsed TEI can patch missing bibliography values (`title`, `authors`, `year`) and provide section candidates.
- Parsed TEI also emits:
  - `metadata/assets/structured/grobid/figures_tables.jsonl`
- Each JSONL record includes:
  - `doc_id`
  - `type` (`figure` or `table`)
  - `label`
  - `caption_text`
  - `page`
  - `coords[]` entries parsed from TEI coordinate strings (`page,x,y,w,h`)
- GROBID failures are non-fatal and do not stop OCR.
- When geometry QA is enabled, GROBID is requested with `teiCoordinates=figure` and transformed into canonical pixel space before comparison.

## Output Contract

Existing outputs remain unchanged:

- `pages/*.md`
- `metadata/manifest.json`
- `metadata/bibliography.json`
- `metadata/discovery.json`
- `metadata/sections.json`

New structured artifacts are additive:

- `metadata/assets/structured/marker/page_0001.md`
- `metadata/assets/structured/marker/page_0001.json`
- `metadata/assets/structured/marker/page_0001_assets/*`
- `metadata/assets/structured/grobid/fulltext.tei.xml`
- `metadata/assets/structured/grobid/figures_tables.jsonl`
- `metadata/assets/structured/extracted/manifest.json`
- `metadata/assets/structured/extracted/table_fragments.jsonl`
- `metadata/assets/structured/extracted/tables/canonical.jsonl`
- `metadata/assets/structured/extracted/tables/manifest.jsonl`
- `metadata/assets/structured/extracted/tables/p0001_t01.csv`
- `metadata/assets/structured/extracted/tables/p0001_t01.json`
- `metadata/assets/structured/extracted/figures/manifest.jsonl`
- `metadata/assets/structured/extracted/figures/deplot/p0001_f01.json` (optional)
- `metadata/assets/structured/qa/table_reconciliation.json`
- `metadata/assets/structured/qa/table_flags.jsonl`

Manifest includes:

```json
{
  "structured_extraction": {
    "enabled": true,
    "backend": "hybrid",
    "grobid_used": true,
    "fallback_count": 2,
    "structured_page_count": 14
  }
}
```

And:

```json
{
  "structured_data_extraction": {
    "enabled": true,
    "table_count": 6,
    "figure_count": 12,
    "deplot_count": 4,
    "unresolved_figure_count": 1,
    "errors": []
  }
}
```

Page entries may include:

- `status: structured_ok`
- `status: structured_fallback`
- `status: text_only`
- `status: ok`
- `status: skipped`
- `structured: { backend, artifacts, fallback_reason }`

## Safety and Fallback

- Structured extraction never blocks run completion.
- Any page-level failure automatically uses the existing pipeline.
- If `--digital-structured off`, behavior is effectively unchanged from prior OCR flow.
- GROBID disagreement handling is mode-driven:
  - `off`: skip QA flags
  - `warn`: write flags and continue
  - `strict`: fail the document after writing diagnostics

## Configuration

Environment variables:

- `PAPER_OCR_DIGITAL_STRUCTURED`
- `PAPER_OCR_MARKER_COMMAND`
- `PAPER_OCR_MARKER_URL`
- `PAPER_OCR_GROBID_URL`
- `PAPER_OCR_MARKER_TIMEOUT`
- `PAPER_OCR_GROBID_TIMEOUT`

CLI options:

- `--digital-structured off|auto|on`
- `--structured-backend marker|hybrid`
- `--marker-command <cmd>`
- `--marker-url <url>`
- `--marker-timeout <sec>`
- `--grobid-url <url>`
- `--grobid-timeout <sec>`
- `--structured-max-workers <n>`
- `--structured-asset-level standard|full`
- `--extract-structured-data/--no-extract-structured-data`
- `--deplot-command "<cmd with {image}>"`
- `--deplot-timeout <sec>`

## Recommended WSL GROBID Setup

Run GROBID in WSL/Docker and expose port `8070`, then pass host URL from macOS:

```bash
uv run paper-ocr run data/in out --grobid-url http://<wsl-host>:8070 --structured-backend hybrid
```

Use firewall/network settings so the service is reachable from the machine running `paper-ocr`.
