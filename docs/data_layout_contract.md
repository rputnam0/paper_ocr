# Data Layout Contract

This document defines the required structure for `/Users/rexputnam/Documents/projects/paper_ocr/data`.

## Top-Level

Allowed top-level folders only:

- `corpora/`
- `jobs/`
- `cache/`
- `archive/`
- `tmp/`

Anything else under `data/` is considered a contract violation.

## Corpora

Path format:

```text
data/corpora/<corpus_slug>/source_pdfs/
```

Rules:

- `<corpus_slug>` must be lowercase slug format: `a-z`, `0-9`, `_`, `-`.
- Source PDFs for curated datasets belong in `source_pdfs/`.
- Optional corpus metadata may live in `metadata/` or `notes/`.

## Jobs

Path format:

```text
data/jobs/<job_slug>/
```

Required subfolders:

- `input/`
- `pdfs/`
- `reports/`

Optional subfolders:

- `ocr_out/`
- `logs/`

Rules:

- `<job_slug>` must be lowercase slug format: `a-z`, `0-9`, `_`, `-`.
- Job-acquired PDFs must be written under `pdfs/`.
- CSV manifests and fetch reports belong under `reports/`.

## Archive / Cache / Tmp

- `archive/`: historical runs and deprecated layouts preserved for traceability.
- `cache/`: disposable cache artifacts.
- `tmp/`: ephemeral scratch data.

## Validation

Run the built-in audit:

```bash
uv run paper-ocr data-audit data --strict
```

Use `--json` for machine-readable output.
