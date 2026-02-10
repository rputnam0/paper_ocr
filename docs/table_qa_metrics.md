# Table QA Metrics

## Gold Set Format

Recommended starter set: 30-50 papers split across:

- born-digital simple tables
- born-digital complex/multi-page tables
- scanned/OCR-heavy tables

Minimum gold annotations:

- table presence per page
- table bbox/polygon per page
- expected canonical CSV for selected target tables
- expected caption text and table number when available

## Evaluation Inputs

`paper-ocr eval-table-pipeline <gold_dir> <pred_dir>` expects:

- `<gold_dir>/tables.jsonl`
- `<pred_dir>/tables.jsonl`

Each JSONL row should include at least:

- `page`
- `caption_text` (optional but recommended)
- stable table identifier (recommended)

## Metrics

Core metrics:

- `table_detection_precision`
- `table_detection_recall`

Pipeline-level metrics (from manifests/flags):

- caption binding low-confidence rate
- continued-table merge count and merge-confidence distribution
- quality-gate failure rate
- escalation hit rate
- QA disagreement rate (Marker vs GROBID)

Extraction quality metrics (for CSV-backed gold subset):

- row-count match rate
- column-count match rate
- numeric parse success rate
- cell-level correctness on key material-property columns

## Regression Policy

Suggested policy in CI:

- fail when precision or recall drops by more than `0.03` from baseline
- fail when QA disagreement rate increases by more than `0.05` without approved baseline update
- fail when numeric parse success drops below project threshold

Track baseline metrics per pipeline version in source control.
