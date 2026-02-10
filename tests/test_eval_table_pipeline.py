from pathlib import Path

from paper_ocr.table_eval import evaluate_table_pipeline


def _write_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(__import__("json").dumps(r, ensure_ascii=True) for r in rows) + "\n"
    path.write_text(payload)


def test_evaluate_table_pipeline_basic_metrics(tmp_path: Path):
    gold = tmp_path / "gold"
    pred = tmp_path / "pred"
    _write_lines(
        gold / "tables.jsonl",
        [
            {"table_id": "t1", "page": 1, "caption_text": "Table 1"},
            {"table_id": "t2", "page": 3, "caption_text": "Table 2"},
        ],
    )
    _write_lines(
        pred / "tables.jsonl",
        [
            {"table_id": "x1", "page": 1, "caption_text": "Table 1"},
            {"table_id": "x2", "page": 2, "caption_text": "Table 2"},
        ],
    )
    metrics = evaluate_table_pipeline(gold, pred)
    assert metrics["gold_table_count"] == 2
    assert metrics["pred_table_count"] == 2
    assert 0.0 <= metrics["table_detection_recall"] <= 1.0
    assert 0.0 <= metrics["table_detection_precision"] <= 1.0
