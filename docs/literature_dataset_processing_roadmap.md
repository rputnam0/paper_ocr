# Literature Dataset Processing Roadmap

## Objective
Build a reliable pipeline that converts large literature corpora (1000s of papers) into a clean, normalized, provenance-rich material-property dataset suitable for modeling and analytics.

## Current Pipeline Status

### What is working
- DOI ingestion via Telegram bot with resumable reports and deduplication.
- Born-digital structured extraction path (Marker-first) with safe per-page fallback.
- Optional service-mode structured extraction (`--marker-url`) and GROBID enrichment (`--grobid-url`).
- Marker no-OCR safety defaults in CLI mode (`--disable_ocr` + `OCR_ENGINE=None`).
- Structured artifacts written deterministically:
  - Marker page artifacts
  - GROBID TEI XML
  - GROBID `figures_tables.jsonl` index
  - Extracted tables/figures manifests and table CSV/JSON outputs
- Manifest-level observability for structured extraction and structured-data extraction counts.

### Output artifacts available today
For each processed paper, the pipeline can produce:
- Canonical page markdown (`pages/*.md`)
- Structured extraction metadata (`metadata/manifest.json`)
- Marker artifacts (`metadata/assets/structured/marker/...`)
- GROBID TEI (`metadata/assets/structured/grobid/fulltext.tei.xml`)
- GROBID figure/table index (`metadata/assets/structured/grobid/figures_tables.jsonl`)
- Table/figure exports (`metadata/assets/structured/extracted/...`)

## Gaps to Reach Dataset-Grade Extraction

### 1. Missing canonical property schema
There is no final normalized record schema for material-property facts. We need a strict schema before scaling extraction.

### 2. Table semantics are shallow
Current table extraction is structural markdown parsing; complex headers, merged cells, footnotes, uncertainty fields, and condition columns need semantic reconstruction.

### 3. Figure-to-data extraction is incomplete
The DePlot hook exists but is not yet a standardized, confidence-scored chart extraction stage integrated into dataset outputs.

### 4. No datum-level confidence/provenance contract
Every extracted value should carry provenance (doc/page/table/cell or figure region), parser/model source, and confidence.

### 5. No benchmark harness for extraction quality
Need labeled evaluation set and automated precision/recall metrics by property type.

### 6. Scale operations are not fully productionized
Need queue-based orchestration, retry/dead-letter handling, throughput controls, and QA sampling at corpus scale.

## Proposed Phased Plan

### Phase 0 — Dataset Contract (must happen first)
Deliverables:
- JSON schema for `property_record`.
- Controlled vocab for property names/material classes.
- Unit canonicalization policy.

Minimum `property_record` fields:
- `record_id`
- `doc_id`
- `material_name_raw`
- `material_name_normalized`
- `property_name_raw`
- `property_name_normalized`
- `value_raw`
- `value_numeric`
- `unit_raw`
- `unit_normalized`
- `conditions` (temperature, pressure, concentration, solvent, etc.)
- `method_context` (table/figure/text)
- `source` (page/table label/row/col OR figure label/coords)
- `confidence`
- `extraction_stage`
- `created_at`

### Phase 1 — Table Fact Extraction
Deliverables:
- Table semantic parser that maps exported table CSV/JSON into `property_record` rows.
- Header hierarchy reconstruction.
- Cell-level provenance and confidence scoring.

Use GROBID figure/table index as:
- validation cross-check against Marker table detections
- fallback targeting signal for pages where Marker table extraction fails

### Phase 2 — Figure/Graph Fact Extraction
Deliverables:
- Unified figure extraction stage (DePlot or equivalent) producing typed points/series.
- Post-processing to map chart output into `property_record` fields.
- Confidence and anomaly checks for axis/unit parsing.

### Phase 3 — Normalization and Entity Resolution
Deliverables:
- Material name normalization and synonym resolution.
- Property synonym mapping (e.g., viscosity variants).
- Unit conversion and dimensional validation.
- Duplicate/near-duplicate record consolidation across papers.

### Phase 4 — Quality and Evaluation
Deliverables:
- Gold-labeled benchmark set (100–300 papers initially).
- Automated evaluation metrics:
  - extraction precision/recall/F1 by property
  - table detection recall
  - unit normalization accuracy
- Regression tests gated in CI for extraction quality.

### Phase 5 — Scale and Operations
Deliverables:
- Queue-based batch processing (document-level idempotent jobs).
- Retry and dead-letter strategy for failed papers/pages.
- Throughput/resource tuning for remote service backends.
- Sampling-based QA dashboard for manual review.

## Success Metrics
- >=95% document completion rate (pipeline robustness)
- >=90% table detection recall on benchmark
- >=85% fact-level precision on target properties before broad corpus export
- 100% extracted facts include provenance + confidence
- Reproducible reruns with stable output contracts

## Immediate Next Steps
1. Create `property_record` schema and add schema validation utilities.
2. Implement first-pass table-to-fact mapper using current extracted table artifacts.
3. Add benchmark dataset scaffolding and evaluation script.
4. Add manifest links to generated dataset records per document.

## Notes on Service-Oriented Execution
The pipeline now supports remote structured services via URLs (`--marker-url`, `--grobid-url`).
This allows orchestration on lightweight clients while heavy extraction runs on dedicated compute hosts.
