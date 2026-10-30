import pytest

from rag_lab.evaluation import summarize, validate_report
from rag_lab.io import write_json, write_jsonl


def test_report_roundtrip_and_range(tmp_path):
    provenance = {
        "commit": "a" * 40,
        "dataset": "fixture",
        "scope": "fixture",
        "split": "train",
        "dataset_manifest_sha256": "b" * 64,
        "index_version": "c" * 64,
        "config": {"chunk_size": 256},
        "models": {"fixture": True},
    }
    rows = [
        {
            "question_id": "q",
            "method": "bm25",
            "cold": False,
            "recall@5": 1.0,
            "total_ms": 10.0,
            "ranking": ["a"],
        }
    ]
    write_json(tmp_path / "provenance.json", provenance)
    write_jsonl(tmp_path / "results.jsonl", rows)
    summarize(rows, tmp_path, provenance)
    validate_report(tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert "not sentence-level factuality" in (tmp_path / "report.md").read_text()
    rows[0]["recall@5"] = 2
    write_jsonl(tmp_path / "results.jsonl", rows)
    with pytest.raises(ValueError):
        validate_report(tmp_path)
