# AGENTS.md

## Tooling and Workflow Preferences
- Use `uv` for Python workflows instead of `pip`.
  - Install deps: `uv sync`
  - Run CLI: `uv run paper-ocr run <in_dir> <out_dir>`
  - Run tests: `uv run pytest`
- Make **frequent git commits** after meaningful additions to the codebase.
- Use **test-driven development** for new features.
  - Write tests first, then implement.
  - Tests must be exercised and passing before considering work complete.

## Project Overview
This project implements a born-digital–aware PDF → Markdown OCR pipeline using `allenai/olmOCR-2-7B-1025` via DeepInfra’s OpenAI-compatible API.

Key behaviors:
- Routes pages to **anchored** or **unanchored** OCR prompts based on text-layer heuristics.
- Renders pages to images with the **longest dimension capped at 1288 px**.
- Parses YAML front matter from model output and writes markdown + metadata per page.
- Extracts bibliographic metadata from first-page markdown using `nvidia/Nemotron-3-Nano-30B-A3B` by default.
- Names each output document folder from extracted author/year metadata.
- Names each consolidated markdown file from extracted paper title.
- Extracts discovery metadata (`paper_summary`, `key_topics`) from the first few pages (abstract-first).
- Writes group-level paper index README files for fast folder-level discoverability.
- Produces a full output bundle per PDF, including manifest, debug artifacts, and assembled document files.

## CLI Usage
Command:
- `paper-ocr run <in_dir> <out_dir> [options]`

Options:
- `--workers <int>` default `32`
- `--model <str>` default `allenai/olmOCR-2-7B-1025`
- `--base-url <str>` default `https://api.deepinfra.com/v1/openai`
- `--max-tokens <int>` default `8192`
- `--force` reprocess existing outputs
- `--mode auto|anchored|unanchored` default `auto`
- `--debug` write request/response payloads per page
- `--scan-preprocess` enable mild scan preprocessing
- `--text-only` enable high-quality text-layer extraction (skips VLM)
- `--metadata-model <str>` default `nvidia/Nemotron-3-Nano-30B-A3B`

Example:
- `uv run paper-ocr run data/LISA out`

## Environment
Required env var:
- `DEEPINFRA_API_KEY` (loaded automatically from `.env`)

## Output Layout
For each PDF:
```
out/<input_parent>/
  README.md
out/<input_parent>/<author_year>/
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
      page_0001.request.json
      page_0001.response.json
      page_0001.metadata.json
```

## Code Structure
- `src/paper_ocr/cli.py` CLI entrypoint
- `src/paper_ocr/ingest.py` PDF discovery + hashing
- `src/paper_ocr/inspect.py` text-layer heuristics + routing
- `src/paper_ocr/render.py` PDF→image rendering
- `src/paper_ocr/anchoring.py` anchor extraction + prompt building
- `src/paper_ocr/client.py` DeepInfra request + retry logic
- `src/paper_ocr/postprocess.py` YAML front matter parsing
- `src/paper_ocr/store.py` file IO helpers
- `src/paper_ocr/schemas.py` manifest construction

## Testing
- Run tests with: `uv run pytest`
- Add tests in `tests/` for each new feature or bugfix.
- Tests are required for new additions and must pass.

## Best Practices
- Preserve stable output (idempotent writes unless `--force`).
- Keep prompts deterministic and versioned.
- Keep debug artifacts small and JSON-serializable.
- When changing routing heuristics or rendering settings, update tests accordingly.
- If adding new dependencies, update `pyproject.toml` and ensure `uv sync` remains clean.
- Avoid committing secrets or API keys; rely on `.env` and `.gitignore`.
- Keep ingestion/state artifacts under `data/` and final OCR outputs under `out/`.

## Repository Folder Contracts
- `data/corpora/<slug>/source_pdfs/`: canonical source PDFs by topic/corpus.
- `data/jobs/<job_slug>/`: pipeline jobs with required `pdfs/`, `reports/`; optional `logs/`.
- `data/archive/`: legacy outputs and historical runs.
- `data/cache/`: disposable cache-only artifacts.
- `data/tmp/`: transient scratch.
- `input/`: local CSV intake only.
- `out/`: generated OCR outputs only.
- `docs/`: tracked documentation/specs only.
- `src/`: production code only.
- `tests/`: automated tests only.

Validation command:
- `uv run paper-ocr data-audit data --strict`
- Do not write final `paper-ocr run` outputs under `data/jobs`; use `out/<...>`.

## Remote Service Guidance (Marker/GROBID)
- Prefer service URLs for heavy structured extraction when running from low-resource clients.
- `paper-ocr run` supports both:
  - Marker CLI mode: `--marker-command ...`
  - Marker service mode: `--marker-url <base_url>`
- GROBID service mode: `--grobid-url <base_url>`
- Environment defaults for service mode:
  - `PAPER_OCR_MARKER_URL`
  - `PAPER_OCR_GROBID_URL`
- Service health checks before long runs:
  - Marker: `GET <marker_url>/openapi.json`
  - GROBID: `GET <grobid_url>/api/isalive`
- If services are remote, agents may use SSH local forwarding to bind remote service ports to localhost, then pass localhost URLs to CLI options/env vars.
- Marker OCR must remain disabled by default; do not remove no-OCR safeguards in structured extraction path.

## Table Pipeline Guardrails
- Keep table extraction `marker-first`; markdown table parsing is fallback only.
- Keep OCR table merge enabled by default and scoped to headers (`--table-ocr-merge --table-ocr-merge-scope header`) to recover symbols without corrupting body rows.
- Preserve canonical coordinate normalization (PDF-space to render-pixel transforms) before QA matching.
- Do not compare raw GROBID coordinates directly against marker/crop pixel coordinates.
- GROBID is a QA comparator, not a hard dependency; gate disagreement flags when GROBID parse quality is invalid.
- Preserve fragment lineage (`fragment_id`, block IDs, page, polygons) and merged canonical table outputs.
