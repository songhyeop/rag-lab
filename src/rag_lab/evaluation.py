import csv
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rag_lab.auth import settings
from rag_lab.budget import Ledger
from rag_lab.generation import PROMPT, Generator
from rag_lab.io import digest, read_json, read_jsonl, sha, verify_files, write_json, write_jsonl
from rag_lab.metrics import answer_metrics, citation_metrics, retrieval_metrics
from rag_lab.retrieval import SearchEngine, unique_docs

METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def evaluate(
    data: Path,
    index: Path,
    models: Path,
    split: str,
    dest: Path,
    generate: bool = False,
    subject: str = "",
) -> None:
    if (dest / "results.jsonl").exists():
        raise ValueError("Results already exist; choose a new run destination")
    load_started = time.perf_counter()
    engine = SearchEngine(index, models)
    engine_load_ms = (time.perf_counter() - load_started) * 1000
    manifest = read_json(data / "manifest.json")
    if engine.manifest["dataset_manifest_sha256"] != sha(data / "manifest.json"):
        raise ValueError("Evaluation/index data mismatch")
    if generate and manifest["dataset"] != "hotpotqa":
        raise ValueError("SciFact is retrieval-only")
    generator = None
    if generate:
        if not subject:
            raise ValueError("Explicit Google sub required for shared evaluation ledger")
        cfg = settings()
        generator = Generator(Ledger(cfg.usage_table, evaluation=True), Path(cfg.price_file))
    labels_path = data / f"labels.{split}.jsonl"
    if sha(labels_path) != manifest["files"][labels_path.name]:
        raise ValueError("Labels checksum mismatch")
    labels = read_jsonl(labels_path)
    ids = {c["doc_id"] for c in engine.chunks}
    if any(set(q["relevant"]) - ids for q in labels):
        raise ValueError("Gold references missing documents")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except subprocess.CalledProcessError:
        commit, dirty = "uncommitted", True
    provenance = {
        "commit": commit,
        "dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": manifest["dataset"],
        "scope": manifest["scope"],
        "split": split,
        "questions": len(labels),
        "dataset_manifest_sha256": sha(data / "manifest.json"),
        "labels_sha256": sha(labels_path),
        "index_version": engine.manifest["version"],
        "config": engine.config,
        "models": engine.manifest["models"],
        "prompt_sha256": digest(PROMPT),
        "uv_lock_sha256": sha(Path("uv.lock")),
        "generation": generate,
        "methods": METHODS + (["no_retrieval", "oracle"] if generate else []),
        "latency_note": "first query per method cold; subsequent queries warm",
        "engine_load_ms": engine_load_ms,
        "index_seconds": engine.manifest["index_seconds"],
        "index_bytes": engine.manifest["index_bytes"],
    }
    write_json(dest / "provenance.json", provenance)
    rows = []
    # Persist each row as it completes; interrupted paid runs retain audit evidence.
    dest.mkdir(parents=True, exist_ok=True)
    for qi, q in enumerate(labels):
        for method in METHODS + (["no_retrieval", "oracle"] if generator else []):
            row = {
                "question_id": q["id"],
                "method": method,
                "cold": qi == 0,
            }
            if method in METHODS:
                search = engine.search(q["question"], method)
                ranking = unique_docs(search["hits"])
                row.update(retrieval_metrics(ranking, q["relevant"]))
                row.update(ranking=ranking, **search["timings"])
                hits = search["hits"]
            else:
                hits = (
                    []
                    if method == "no_retrieval"
                    else sorted(
                        [c for c in engine.chunks if c["doc_id"] in q["relevant"]],
                        key=lambda c: (int(c["id"].rsplit(":", 1)[1]), c["doc_id"]),
                    )
                )
            if generator:
                answer = generator.answer(subject, q["question"], hits, engine.config, mode=method)
                row.update(answer_metrics(answer["answer"], q["answer"]))
                row.update(citation_metrics(answer["citations"], q["relevant"]))
                row.update(answer)
            rows.append(row)
            with (dest / "results.jsonl").open("a") as stream:
                import json

                stream.write(json.dumps(row) + "\n")
    summarize(rows, dest, provenance)
    sample = sorted(labels, key=lambda q: q["id"])
    random.Random(42).shuffle(sample)
    write_jsonl(
        dest / "manual_review.jsonl",
        [
            {
                "question_id": q["id"],
                "question": q["question"],
                "gold": q.get("answer"),
                "relevant": q["relevant"],
                "reviewed": False,
                "failure_category": "",
                "notes": "",
            }
            for q in sample[:30]
        ],
    )


def summarize(rows: list[dict], dest: Path, provenance: dict) -> None:
    keys = sorted(
        {
            k
            for r in rows
            for k, v in r.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    )
    summary = []
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        record = {"method": method, "n": len(group)}
        for key in keys:
            values = [r[key] for r in group if key in r]
            if values:
                record[key] = float(np.mean(values))
        warm = [r["total_ms"] for r in group if not r["cold"] and "total_ms" in r]
        if warm:
            record["warm_search_p95_ms"] = float(np.percentile(warm, 95))
        warm_answer = [
            r["generation_ms"] + r.get("total_ms", 0)
            for r in group
            if not r["cold"] and "generation_ms" in r
        ]
        if warm_answer:
            record["warm_answer_p95_ms"] = float(np.percentile(warm_answer, 95))
        if any("usage" in r for r in group):
            record["input_tokens_total"] = sum(
                r.get("usage", {}).get("inputTokens", 0) for r in group
            )
            record["output_tokens_total"] = sum(
                r.get("usage", {}).get("outputTokens", 0) for r in group
            )
            record["cost_nano_usd_total"] = sum(r.get("cost_nano_usd", 0) for r in group)
        summary.append(record)
    write_json(dest / "summary.json", {"provenance": provenance, "metrics": summary})
    columns = ["method", "n"] + sorted(set().union(*(r.keys() for r in summary)) - {"method", "n"})
    with (dest / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Experiment report",
        "",
        f"Dataset: {provenance['dataset']} ({provenance['scope']})",
        f"Split: {provenance['split']}; commit: {provenance['commit']}",
        "",
        "HotpotQA reduced-corpus results are not comparable with official fullwiki results.",
        "Citation metrics measure document overlap, not sentence-level factuality.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join(str(r.get(k, "")) for k in columns) + " |" for r in summary)
    lines += [
        "",
        "Manual review: complete manual_review.jsonl for 30 examples before claiming analysis.",
        "No improvement is required. Record null or negative findings and causes.",
    ]
    (dest / "report.md").write_text("\n".join(lines) + "\n")


def validate_report(path: Path, data: Path | None = None) -> None:
    provenance = read_json(path / "provenance.json")
    rows = read_jsonl(path / "results.jsonl")
    for key in ["commit", "dataset_manifest_sha256", "index_version", "config", "models", "split"]:
        if not provenance.get(key):
            raise ValueError(f"Missing provenance: {key}")
    seen = set()
    for row in rows:
        result_key = row["question_id"], row["method"]
        if result_key in seen:
            raise ValueError("Duplicate result")
        seen.add(result_key)
        for name, value in row.items():
            if name.startswith(("recall@", "ndcg@", "mrr@", "answer_", "citation_")):
                if isinstance(value, (int, float)) and not 0 <= value <= 1:
                    raise ValueError("Metric outside [0, 1]")
    if not rows:
        raise ValueError("Empty report")

    if data is not None:
        manifest = read_json(data / "manifest.json")
        verify_files(data, manifest["files"])
        if sha(data / "manifest.json") != provenance["dataset_manifest_sha256"]:
            raise ValueError("Report dataset identity mismatch")
        docs = {d["id"] for d in read_jsonl(data / "corpus.jsonl")}
        labels = {q["id"]: q for q in read_jsonl(data / f"labels.{provenance['split']}.jsonl")}
        by_method: dict[str, set[str]] = {}
        for row in rows:
            if row["question_id"] not in labels:
                raise ValueError("Unknown question ID in report")
            by_method.setdefault(row["method"], set()).add(row["question_id"])
            q = labels[row["question_id"]]
            if (
                set(q["relevant"]) - docs
                or set(row.get("ranking", [])) - docs
                or set(row.get("citations", [])) - docs
            ):
                raise ValueError("Report references unknown document ID")
            expected = retrieval_metrics(row["ranking"], q["relevant"]) if "ranking" in row else {}
            if "answer" in row:
                expected.update(answer_metrics(row["answer"], q["answer"]))
                expected.update(citation_metrics(row["citations"], q["relevant"]))
            for key, value in expected.items():
                if abs(row[key] - value) > 1e-9:
                    raise ValueError(f"Evaluation calculation mismatch: {key}")
        if any(qids != set(labels) for qids in by_method.values()):
            raise ValueError("Incomplete experiment questions")

        if "methods" in provenance and set(by_method) != set(provenance["methods"]):
            raise ValueError("Incomplete experiment methods")
