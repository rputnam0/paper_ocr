from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALLOWED_TOP_LEVEL_DIRS = {"corpora", "jobs", "cache", "archive", "tmp"}
ALLOWED_TOP_LEVEL_FILES = {"README.md"}
REQUIRED_JOB_SUBDIRS = ("input", "pdfs", "reports")
ALLOWED_JOB_SUBDIRS = {"input", "pdfs", "reports", "ocr_out", "logs"}
ALLOWED_CORPUS_SUBDIRS = {"source_pdfs", "metadata", "notes"}


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class AuditReport:
    data_dir: str
    issues: list[AuditIssue]
    total_pdf_count: int

    @property
    def issue_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.issues if item.severity != "error")

    def to_dict(self) -> dict[str, object]:
        return {
            "data_dir": self.data_dir,
            "issue_count": self.issue_count,
            "warning_count": self.warning_count,
            "total_pdf_count": self.total_pdf_count,
            "issues": [asdict(item) for item in self.issues],
        }


def _is_slug(name: str) -> bool:
    return bool(SLUG_RE.match(name))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:  # noqa: BLE001
        return str(path)


def run_data_audit(data_dir: Path) -> AuditReport:
    root = data_dir.resolve()
    issues: list[AuditIssue] = []
    pdf_paths: list[Path] = []

    if not root.exists():
        issues.append(
            AuditIssue(
                code="missing_data_dir",
                path=str(root),
                message="Data directory does not exist.",
            )
        )
        return AuditReport(data_dir=str(root), issues=issues, total_pdf_count=0)

    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and entry.name not in ALLOWED_TOP_LEVEL_DIRS:
            issues.append(
                AuditIssue(
                    code="unexpected_top_level_dir",
                    path=_rel(entry, root),
                    message=f"Unexpected top-level folder '{entry.name}'.",
                )
            )
        if entry.is_file() and entry.name not in ALLOWED_TOP_LEVEL_FILES:
            issues.append(
                AuditIssue(
                    code="unexpected_top_level_file",
                    path=_rel(entry, root),
                    message=f"Unexpected top-level file '{entry.name}'.",
                )
            )

    for required in sorted(ALLOWED_TOP_LEVEL_DIRS):
        if not (root / required).is_dir():
            issues.append(
                AuditIssue(
                    code="missing_top_level_dir",
                    path=required,
                    message=f"Missing top-level folder '{required}'.",
                    severity="warning",
                )
            )

    corpora_dir = root / "corpora"
    if corpora_dir.is_dir():
        for corpus_dir in sorted((d for d in corpora_dir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
            if not _is_slug(corpus_dir.name):
                issues.append(
                    AuditIssue(
                        code="invalid_slug",
                        path=_rel(corpus_dir, root),
                        message="Corpus folder name must be lowercase slug (a-z, 0-9, _, -).",
                    )
                )
            source_dir = corpus_dir / "source_pdfs"
            if not source_dir.is_dir():
                issues.append(
                    AuditIssue(
                        code="missing_source_pdfs_dir",
                        path=_rel(corpus_dir, root),
                        message="Corpus folder is missing required 'source_pdfs/' directory.",
                    )
                )
            for child in sorted(corpus_dir.iterdir(), key=lambda d: d.name.lower()):
                if child.is_dir() and child.name not in ALLOWED_CORPUS_SUBDIRS:
                    issues.append(
                        AuditIssue(
                            code="unexpected_corpus_subdir",
                            path=_rel(child, root),
                            message=f"Unexpected corpus subdirectory '{child.name}'.",
                            severity="warning",
                        )
                    )

    jobs_dir = root / "jobs"
    if jobs_dir.is_dir():
        for job_dir in sorted((d for d in jobs_dir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
            if not _is_slug(job_dir.name):
                issues.append(
                    AuditIssue(
                        code="invalid_slug",
                        path=_rel(job_dir, root),
                        message="Job folder name must be lowercase slug (a-z, 0-9, _, -).",
                    )
                )
            for required in REQUIRED_JOB_SUBDIRS:
                if not (job_dir / required).is_dir():
                    issues.append(
                        AuditIssue(
                            code="missing_job_subdir",
                            path=_rel(job_dir, root),
                            message=f"Job folder is missing required '{required}/' directory.",
                        )
                    )
            for child in sorted(job_dir.iterdir(), key=lambda d: d.name.lower()):
                if child.is_dir() and child.name not in ALLOWED_JOB_SUBDIRS:
                    issues.append(
                        AuditIssue(
                            code="unexpected_job_subdir",
                            path=_rel(child, root),
                            message=f"Unexpected job subdirectory '{child.name}'.",
                            severity="warning",
                        )
                    )
                if child.is_file():
                    issues.append(
                        AuditIssue(
                            code="unexpected_job_file",
                            path=_rel(child, root),
                            message=f"Unexpected file under job root: '{child.name}'.",
                            severity="warning",
                        )
                    )

    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            pdf_paths.append(candidate)
            rel_parts = candidate.relative_to(root).parts
            if len(rel_parts) < 2:
                issues.append(
                    AuditIssue(
                        code="misplaced_pdf",
                        path=_rel(candidate, root),
                        message="PDF is not under an allowed data contract path.",
                    )
                )
                continue
            top_level = rel_parts[0]
            if top_level == "corpora":
                if len(rel_parts) < 4 or rel_parts[2] != "source_pdfs":
                    issues.append(
                        AuditIssue(
                            code="misplaced_pdf",
                            path=_rel(candidate, root),
                            message="Corpus PDF must live under corpora/<slug>/source_pdfs/.",
                        )
                    )
            elif top_level == "jobs":
                if len(rel_parts) < 4 or rel_parts[2] != "pdfs":
                    issues.append(
                        AuditIssue(
                            code="misplaced_pdf",
                            path=_rel(candidate, root),
                            message="Job PDF must live under jobs/<slug>/pdfs/.",
                        )
                    )
            elif top_level not in {"archive", "cache", "tmp"}:
                issues.append(
                    AuditIssue(
                        code="misplaced_pdf",
                        path=_rel(candidate, root),
                        message="PDF is under unsupported top-level data folder.",
                    )
                )

    return AuditReport(
        data_dir=str(root),
        issues=issues,
        total_pdf_count=len(pdf_paths),
    )


def format_audit_report(report: AuditReport) -> str:
    lines = [
        (
            "[data-audit] "
            f"data_dir={report.data_dir} "
            f"issues={report.issue_count} "
            f"warnings={report.warning_count} "
            f"pdfs={report.total_pdf_count}"
        )
    ]
    for item in report.issues:
        lines.append(f"- [{item.severity}] {item.code}: {item.path} ({item.message})")
    return "\n".join(lines)

