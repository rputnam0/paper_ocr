from pathlib import Path

from paper_ocr.data_audit import format_audit_report, run_data_audit


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_data_audit_happy_path(tmp_path: Path):
    data_dir = tmp_path / "data"
    _mkdir(data_dir / "corpora" / "polymer-viscosity" / "source_pdfs")
    _mkdir(data_dir / "jobs" / "doi-smoke" / "input")
    _mkdir(data_dir / "jobs" / "doi-smoke" / "pdfs")
    _mkdir(data_dir / "jobs" / "doi-smoke" / "reports")
    _mkdir(data_dir / "cache")
    _mkdir(data_dir / "archive")
    _mkdir(data_dir / "tmp")

    (data_dir / "corpora" / "polymer-viscosity" / "source_pdfs" / "paper.pdf").write_bytes(b"pdf-1")
    (data_dir / "jobs" / "doi-smoke" / "pdfs" / "downloaded.pdf").write_bytes(b"pdf-2")

    report = run_data_audit(data_dir)
    rendered = format_audit_report(report)

    assert report.issue_count == 0
    assert report.total_pdf_count == 2
    assert "issues=0" in rendered


def test_data_audit_flags_unexpected_top_level(tmp_path: Path):
    data_dir = tmp_path / "data"
    _mkdir(data_dir / "telegram_jobs")
    _mkdir(data_dir / "corpora")
    _mkdir(data_dir / "jobs")
    _mkdir(data_dir / "cache")
    _mkdir(data_dir / "archive")
    _mkdir(data_dir / "tmp")

    report = run_data_audit(data_dir)
    codes = {item.code for item in report.issues}

    assert "unexpected_top_level_dir" in codes


def test_data_audit_flags_corpus_and_job_contract_issues(tmp_path: Path):
    data_dir = tmp_path / "data"
    _mkdir(data_dir / "corpora" / "Bad Name")
    _mkdir(data_dir / "jobs" / "job_one" / "pdfs")
    _mkdir(data_dir / "cache")
    _mkdir(data_dir / "archive")
    _mkdir(data_dir / "tmp")

    report = run_data_audit(data_dir)
    codes = {item.code for item in report.issues}

    assert "invalid_slug" in codes
    assert "missing_source_pdfs_dir" in codes
    assert "missing_job_subdir" in codes


def test_data_audit_flags_misplaced_pdf(tmp_path: Path):
    data_dir = tmp_path / "data"
    _mkdir(data_dir / "corpora" / "polymer")
    _mkdir(data_dir / "jobs" / "job-1" / "input")
    _mkdir(data_dir / "jobs" / "job-1" / "pdfs")
    _mkdir(data_dir / "jobs" / "job-1" / "reports")
    _mkdir(data_dir / "cache")
    _mkdir(data_dir / "archive")
    _mkdir(data_dir / "tmp")

    (data_dir / "corpora" / "polymer" / "bad.pdf").write_bytes(b"oops")
    report = run_data_audit(data_dir)
    codes = {item.code for item in report.issues}

    assert "misplaced_pdf" in codes
