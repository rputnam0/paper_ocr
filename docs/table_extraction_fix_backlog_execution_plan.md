# Robust Table Extraction Backlog Plan (P0-first, Generalized Rules)

Status: In Progress

## Summary
This plan executes the prioritized backlog in gated sprints, starting with highest-impact extraction failures and using Gemini rubric metrics as hard acceptance criteria.
Chosen defaults:
- Phasing: P0-first sprints
- Scope: Generalized rules first (avoid corpus overfit)

Goal:
- Improve structural correctness, completeness, and notation fidelity while keeping false positives controlled.
- Use the existing Gemini QA outputs (including failure_modes, rubric, and llm_extraction_instructions) as the evaluation contract.

Success criteria at end of plan:
- reject count reduced from 3 to <=1 on out/polymer_viscosity_lit_20260211_090425.
- recommended_accept increases by >=10 tables.
- Failure-mode reductions meeting sprint gates below.
- No regression in table-presence precision (table_present_no remains 0 on this corpus).

## Public API / Interface / Type Changes
1. Extend canonical extracted table JSON schema at out/.../metadata/assets/structured/extracted/tables/<table_id>.json with stable fields:
- header_rows_full
- header_hierarchy
- row_lineage
- context_mappings
- required_fields_missing
2. Extend QA flags taxonomy in src/paper_ocr/structured_data.py:
- multi_level_header_recovered
- row_column_topology_corrected
- completeness_guardrail_triggered
- legend_code_resolved
3. Add/extend reporting artifact:
- data/jobs/polymer_viscosity_lit/reports/table_fix_backlog_metrics_<date>.json
4. No breaking CLI change required for extraction command signatures.
5. Optional additive CLI for reporting only:
- paper-ocr summarize-gemini-failures <ocr_out_dir>

## Implementation Plan

### Phase 0: Baseline and Harness Lock
1. Freeze baseline metrics from:
- data/jobs/polymer_viscosity_lit/reports/table_validation_failure_summary_gemini_20260211_rubric.json
2. Save target gates per failure mode in a checked-in fixture config.
3. Add a deterministic evaluation script that computes:
- action counts
- failure mode counts
- rubric averages
- robustness counts
4. Gate each sprint with this script plus Gemini validation rerun.

### Sprint P0-A: Header Hierarchy Recovery (highest evidence)
Code targets:
- src/paper_ocr/structured_data.py (_collapse_header_rows, cross-page header logic, canonical assembly)
Work:
1. Preserve multi-row headers as structured tiers before flattening.
2. Improve flattening to keep parent-child header relationships deterministic.
3. Ensure continuation merges retain all header rows, not only first-row header.
4. Emit header_hierarchy in table JSON.
Acceptance gates:
- multi_level_header_loss down >=50%.
- missing_required_columns down >=50%.
- No increase in reject.
Tests:
- Multi-tier header with colspan/rowspan.
- Continuation table with header row on first page and sub-header on second page.
- Adjacent non-continuation tables remain unmerged.

### Sprint P0-B: Row/Column Topology Correction
Code targets:
- src/paper_ocr/structured_data.py (_patch_table_grid, _match_tables_for_page, _quality_metrics)
Work:
1. Add topology checks for transposition signatures.
2. Improve marker/OCR match scoring to penalize row/column inversion.
3. Add correction pass for row-shift and column-shift when confidence is sufficient.
4. Add conservative fail-safe to avoid hallucinated structural repairs.
Acceptance gates:
- row_shift_or_merge down >=40%.
- column_shift_or_merge down >=40%.
- No obvious transposed table in reject/review tail.
Tests:
- Known transposed case fixture.
- Row-identifier column preservation.
- Sparse tables with empty cells remain aligned.

### Sprint P1-A: Completeness Guardrails
Code targets:
- src/paper_ocr/structured_data.py (canonical write path, continuation row coercion)
Work:
1. Enforce row/column completeness checks before export.
2. Detect partial extraction by expected-row continuity and fragment lineage.
3. Add duplicate/hallucination suppression with deterministic thresholds.
4. Populate required_fields_missing per table.
Acceptance gates:
- partial_table_extracted down >=40%.
- Data completeness rubric average +0.05 absolute.
Tests:
- Missing middle rows in continuation.
- Duplicate row hallucination prevention.
- Empty cell preservation without row drops.

### Sprint P1-B: Symbol / Unit / Notation Fidelity
Code targets:
- src/paper_ocr/structured_data.py (_normalize_cell_text, _cell_patch_decision, _parse_ocr_html_table)
Work:
1. Preserve scientific notation and chemical formulas more strictly.
2. Preserve superscript/subscript semantics in canonical text form.
3. Patch symbol loss only when OCR alignment confidence passes threshold.
Acceptance gates:
- symbol_or_unit_loss down >=50%.
- Unit/symbol rubric average >=0.90.
Tests:
- Chemical formulas (K2CO3, SiO2), powers, ±, Greek symbols.
- Mixed HTML/text OCR inputs.

### Sprint P2-A: Context / Code / Legend Resolution
Code targets:
- New resolver module under src/paper_ocr/ used by structured export path.
Work:
1. Add lightweight context resolver that links table aliases/codes to nearby legend paragraphs.
2. Write context_mappings into canonical table JSON.
3. Surface unresolved mappings into required_fields_missing.
Acceptance gates:
- Context-resolution rubric +0.08.
- Zero critical unresolved alias/code issues in targeted docs.
Tests:
- Alias polymer code table fixture with legend in nearby text.
- Ambiguous mappings remain unresolved (do not hallucinate).

### Sprint P2-B: Continuation Merge QA Hardening
Code targets:
- src/paper_ocr/structured_data.py continuation logic
- src/paper_ocr/table_validation.py continuation evaluation path
Work:
1. Strengthen continuation merge to retain all data rows from trailing pages.
2. Require header+data cross-checks when combining continuation fragments.
3. Emit explicit continuation lineage in table JSON.
Acceptance gates:
- split_table_continuation issues eliminated in this corpus.
- No continuation table missing first/last continuation rows.
Tests:
- Two-page and three-page continuation fixtures.
- False merge prevention with adjacent independent tables.

### Distilled LLM Extraction Buckets to Reuse Later
Use these 6 stable instruction clauses when building LLM-aware extraction:
1. Preserve full header hierarchy (multi-row/colspan/subheaders) before flattening.
2. Enforce row/column topology and prevent transposition.
3. Enforce completeness and anti-hallucination checks.
4. Merge split/continuation tables with full header and row retention.
5. Preserve symbols/units/notation verbatim.
6. Resolve alias/code mappings using legend and nearby context.

## Test Cases and Scenarios
1. Regression corpus replay on polymer run with Gemini rubric evaluation.
2. Synthetic fixtures:
- Multi-level headers
- Transposed table
- Split tables with delayed headers
- Chemical notation-heavy cells
- Alias code + legend mapping
3. Safety scenarios:
- Adjacent independent tables on consecutive pages
- OCR-noisy tables
- Sparse and partially empty tables
4. Output integrity checks:
- CSV/JSON consistency
- Canonical lineage completeness
- QA flags and pipeline status correctness

## Rollout and Monitoring
1. Land sprint changes behind additive logic only.
2. Run full export-structured-data + validate-tables-gemini after each sprint.
3. Update report files:
- data/jobs/polymer_viscosity_lit/reports/table_fix_backlog_metrics_<date>.json
- data/jobs/polymer_viscosity_lit/reports/table_validation_failure_summary_gemini_<date>.json
4. Promotion rule:
- Move to next sprint only if current sprint gates pass.

## Assumptions and Defaults
1. Gemini rubric outputs remain the primary quality oracle for this backlog phase.
2. No changes to external service contracts (Marker/GROBID endpoints) are required.
3. Fixes prioritize generalized behavior over corpus-specific overrides.
4. If a conflict appears between recall and precision, default to preserving precision and flagging for review rather than hallucinating table content.
