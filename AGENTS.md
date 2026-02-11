# AGENTS.md

## Tooling and Workflow Preferences
- Use `uv` for Python workflows instead of `pip`.
  - Install deps: `uv sync`
  - Run CLI: `uv run paper-ocr run <in_dir> <out_dir>`
  - Run tests (fast lane, default): `uv run pytest`
  - Run full suite (pre-merge/CI parity): `uv run pytest --run-integration --run-slow --run-network --run-service --run-gpu`
- Run normal/lightweight tasks locally on this machine by default.
- Use WSL only for heavy structured workloads that need GPU-backed services:
  - Marker service workloads
  - GROBID document parsing workloads
- Access WSL services through SSH alias `wsl` with local port forwarding.
- Do not run local Marker CLI extraction unless explicitly requested by the user.
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
- Names each output document folder as `doc_<doc_id>` (deterministic from source hash).
- Uses a stable consolidated markdown filename: `document.md`.
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
out/<input_parent>/doc_<doc_id>/
  document.md
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
- Fast lane (default local): `uv run pytest`
  - Skips tests marked `integration`, `slow`, `network`, `service`, `gpu`.
- Full lane (required before merge): `uv run pytest --run-integration --run-slow --run-network --run-service --run-gpu`
- Optional full-lane env toggle: `PAPER_OCR_TEST_FULL=1 uv run pytest`
- Add tests in `tests/` for each new feature or bugfix.
- Tests are required for new additions and must pass.
- Useful local iteration commands:
  - Last failed only: `uv run pytest --lf`
  - Failed-first ordering: `uv run pytest --ff`
  - Subset filter: `uv run pytest -k "<expr>"`

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
- This machine is a low-resource MacBook Air; keep non-heavy processing local.
- Use WSL-hosted services only for heavy GPU-oriented structured extraction and GROBID document parsing.
- Treat local CLI extraction as fallback only when the user explicitly requests it.
- `paper-ocr run` enforces a resource guard for structured extraction:
  - If `--digital-structured auto|on` and `--marker-url` is missing on a low-resource host, run fails fast and instructs WSL service usage.
  - Override only when explicitly requested: `--allow-local-heavy` or `PAPER_OCR_ALLOW_LOCAL_HEAVY=1`.
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
- Agents should call WSL as a service endpoint via SSH local forwarding and pass localhost URLs to CLI options/env vars.
  - Example tunnel:
    - `ssh -N -L 8008:127.0.0.1:8008 -L 8070:127.0.0.1:8070 wsl`
  - Example env:
    - `export PAPER_OCR_MARKER_URL=http://127.0.0.1:8008`
    - `export PAPER_OCR_GROBID_URL=http://127.0.0.1:8070`
  - Example preflight checks:
    - `curl -fsS http://127.0.0.1:8008/openapi.json >/dev/null`
    - `curl -fsS http://127.0.0.1:8070/api/isalive`
  - Example run:
    - `uv run paper-ocr run <in_dir> <out_dir> --marker-url http://127.0.0.1:8008 --grobid-url http://127.0.0.1:8070`
- Marker OCR must remain disabled by default; do not remove no-OCR safeguards in structured extraction path.

## Table Pipeline Guardrails
- Keep table extraction `marker-first`; markdown table parsing is fallback only.
- Keep OCR table merge enabled by default and scoped to headers (`--table-ocr-merge --table-ocr-merge-scope header`) to recover symbols without corrupting body rows.
- Preserve canonical coordinate normalization (PDF-space to render-pixel transforms) before QA matching.
- Do not compare raw GROBID coordinates directly against marker/crop pixel coordinates.
- GROBID is a QA comparator, not a hard dependency; gate disagreement flags when GROBID parse quality is invalid.
- Preserve fragment lineage (`fragment_id`, block IDs, page, polygons) and merged canonical table outputs.
