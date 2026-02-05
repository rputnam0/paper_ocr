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
out/<input_parent>/<author_year>/
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
