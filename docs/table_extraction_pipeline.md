# Table Extraction Pipeline

## Objective

Produce deterministic, geometry-aware table artifacts for both born-digital and scanned PDFs while keeping OCR cost bounded.

## Pipeline Stages

1. Page/document markdown extraction using existing text-layer/OCR flow.
2. Marker document localization (`--marker-localize`) to emit structured block/chunk artifacts.
3. Marker-first table fragment parsing from:
   - `metadata/assets/structured/marker/tables_raw.jsonl` (if present)
   - `metadata/assets/structured/marker/chunks.jsonl`
   - `metadata/assets/structured/marker/blocks.jsonl`
4. Fallback markdown table parsing (`pages/*.md`) only when Marker table fragments are unavailable.
5. Caption binding + confidence scoring.
6. Multi-page continued-table merge into canonical table objects.
7. Quality gate checks and selective escalation.
8. GROBID QA reconciliation and flags.

## Canonical Coordinate Contract

All comparisons and crop joins use render-pixel coordinates (`top_left`, `y-down`).

Per-page contract fields:

- `pdf_page_w_pt`
- `pdf_page_h_pt`
- `render_page_w_px`
- `render_page_h_px`
- `px_per_pt_x`
- `px_per_pt_y`
- `rotation_degrees`
- `pdf_to_px_transform`
- `px_to_pdf_transform`

GROBID coords are converted to pixel-space before matching.

## Schemas

### TableFragment

- `fragment_id`
- `table_group_id`
- `table_block_ids[]`
- `caption_block_id`
- `note_block_ids[]`
- `page`
- `polygons[]`
- `bbox`
- `header_rows[]`
- `data_rows[]`
- `caption_text`
- `caption_confidence`
- `source_format` (`html|markdown|ocr_crop`)
- `quality_metrics`

### CanonicalTable

- `table_id`
- `table_group_id`
- `fragment_ids[]`
- `pages[]`
- `caption_text`
- `header_rows[]`
- `data_rows[]`
- `source_format`
- `merge_confidence`
- `quality_metrics`

### QAFlag

- `flag_id`
- `severity`
- `type`
- `page`
- `table_ref`
- `details`

## Caption Binding Heuristics

Scoring combines:

- explicit caption block types
- lexical anchors (`Table 3`, `Tab. 3`, roman numerals)
- spatial adjacency to table polygons
- continuation markers (`continued`, `cont.`)

Low-confidence joins are flagged (`caption_low_confidence`).

## Multi-Page Merge Rules

Merge candidates when one of:

- matching normalized table number
- adjacent pages with high header similarity and continuation markers

Deduplicate repeated header rows during merge and preserve per-fragment lineage.

## Quality Gates

Default fail thresholds:

- `empty_cell_ratio > 0.35`
- `repeated_text_ratio > 0.45`
- `column_instability_ratio > 0.20`

Escalation (`--table-escalation auto`) applies only to failing fragments up to `--table-escalation-max`.
Escalation QA flags:
- `escalation_missing_ocr`
- `escalation_no_improvement`
- `escalation_applied`

## Output Artifacts

- `metadata/assets/structured/extracted/table_fragments.jsonl`
- `metadata/assets/structured/extracted/tables/canonical.jsonl`
- `metadata/assets/structured/extracted/tables/*.csv`
- `metadata/assets/structured/qa/table_reconciliation.json`
- `metadata/assets/structured/qa/table_flags.jsonl`
