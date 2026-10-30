"""Deterministic source preparation. Labels never enter corpus records."""

import csv
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError

from rag_lab.io import digest, read_json, read_jsonl, sha, write_json, write_jsonl

SCIFACT = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
HOTPOT = "https://curtis.ml.cmu.edu/datasets/hotpot/"


HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def hotpot_mirror(raw: Path) -> tuple[dict, list]:
    """Pinned publisher namespace mirror; explicitly recorded, not silently substituted."""
    import pyarrow.parquet as pq

    rows: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for split, files in [
        ("train", ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]),
        ("dev", ["validation-00000-of-00001.parquet"]),
    ]:
        rows[split] = []
        for name in files:
            url = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{HOTPOT_REVISION}/distractor/{name}"
            path = raw / name
            download(url, path)
            for row in pq.read_table(path).to_pylist():
                rows[split].append(
                    {
                        "_id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "context": list(
                            zip(row["context"]["title"], row["context"]["sentences"], strict=True)
                        ),
                        "supporting_facts": list(
                            zip(
                                row["supporting_facts"]["title"],
                                row["supporting_facts"]["sent_id"],
                                strict=True,
                            )
                        ),
                    }
                )
            sources.append(
                {
                    "url": url,
                    "sha256": sha(path),
                    "split": split,
                    "license": "CC-BY-SA-4.0",
                    "revision": HOTPOT_REVISION,
                    "mirror_reason": "Official curtis.ml.cmu.edu host unavailable",
                }
            )
    return rows, sources


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.rename(path)


def document(title: str, sentences: list[str]) -> dict[str, str]:
    text = "".join(sentences).strip()
    title = title.strip()
    return {"id": digest([title, text]), "title": title, "text": text}


def prepare_hotpot(
    train: list[dict[str, Any]], dev: list[dict[str, Any]], count: int = 200, size: int = 20000
) -> tuple[list, dict]:
    if {x["_id"] for x in train} & {x["_id"] for x in dev}:
        raise ValueError("Overlapping train/dev question IDs")
    selected = {}
    for split, rows in [("train", train), ("dev", dev)]:
        rows = sorted(rows, key=lambda x: x["_id"])
        random.Random(42).shuffle(rows)
        if len(rows) < count:
            raise ValueError("Insufficient questions")
        selected[split] = rows[:count]
    corpus: dict[str, dict] = {}
    labels: dict[str, list] = {}
    for split, rows in selected.items():
        labels[split] = []
        for row in rows:
            docs = [document(title, sentences) for title, sentences in row["context"]]
            corpus.update({d["id"]: d for d in docs})
            support = {title for title, _ in row["supporting_facts"]}
            relevant = {d["id"]: 1 for d in docs if d["title"] in support}
            if len(relevant) != len(support) or len(relevant) != 2:
                raise ValueError("Missing or ambiguous supporting documents")
            labels[split].append(
                {
                    "id": row["_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "relevant": relevant,
                }
            )
    if len(corpus) > size:
        raise ValueError("Required contexts exceed corpus size")
    extra = sorted(train, key=lambda x: x["_id"])
    random.Random(42).shuffle(extra)
    for row in extra:
        for title, sentences in row["context"]:
            if len(corpus) == size:
                break
            d = document(title, sentences)
            corpus[d["id"]] = d
        if len(corpus) == size:
            break
    if len(corpus) != size:
        raise ValueError("Not enough unique documents")
    return sorted(corpus.values(), key=lambda d: d["id"]), labels


def prepare(dataset: str, root: Path) -> None:
    dest = root / dataset
    raw = root / "raw"
    frozen_path = Path("configs/manifests") / f"{dataset}.json"
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    sources = []
    if dataset == "scifact":
        archive = raw / "scifact.zip"
        download(SCIFACT, archive)
        with zipfile.ZipFile(archive) as z:
            # Only known files, no arbitrary archive extraction.
            for name in ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"]:
                out = raw / "scifact" / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read("scifact/" + name))
        corpus = [
            {"id": str(d["_id"]), "title": d["title"], "text": d["text"]}
            for d in read_jsonl(raw / "scifact/corpus.jsonl")
        ]
        queries = {str(q["_id"]): q["text"] for q in read_jsonl(raw / "scifact/queries.jsonl")}
        labels = {}
        for split in ["train", "test"]:
            grouped: dict[str, dict] = {}
            with (raw / f"scifact/qrels/{split}.tsv").open() as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    qid = row["query-id"]
                    grouped.setdefault(qid, {"id": qid, "question": queries[qid], "relevant": {}})
                    if int(row["score"]) > 0:
                        grouped[qid]["relevant"][row["corpus-id"]] = int(row["score"])
            labels[split] = sorted(grouped.values(), key=lambda x: x["id"])
        sources.append(
            {
                "url": SCIFACT,
                "sha256": sha(archive),
                "split": "train/test",
                "license": {
                    "abstracts": "original publisher rights",
                    "annotations": "CC-BY-NC-2.0",
                },
            }
        )
    elif dataset == "hotpotqa":
        rows = {}
        try:
            if frozen and "huggingface.co" in frozen["sources"][0]["url"]:
                raise URLError("Using frozen publisher mirror")
            for split, filename in [
                ("train", "hotpot_train_v1.1.json"),
                ("dev", "hotpot_dev_distractor_v1.json"),
            ]:
                path = raw / filename
                download(HOTPOT + filename, path)
                rows[split] = read_json(path)
                sources.append(
                    {
                        "url": HOTPOT + filename,
                        "sha256": sha(path),
                        "split": split,
                        "license": "CC-BY-SA-4.0",
                    }
                )
        except URLError:
            rows, sources = hotpot_mirror(raw)
        corpus, labels = prepare_hotpot(rows["train"], rows["dev"])
    else:
        raise ValueError(dataset)
    write_jsonl(dest / "corpus.jsonl", corpus)
    for split, qs in labels.items():
        write_jsonl(dest / f"labels.{split}.jsonl", qs)
    files = {p.name: sha(p) for p in sorted(dest.glob("*.jsonl"))}
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": 42,
        "normalization": "title-body-v1",
        "sources": sources,
        "scope": "reduced-corpus" if dataset == "hotpotqa" else "BEIR",
        "files": files,
        "document_ids_sha256": digest(sorted(d["id"] for d in corpus)),
        "counts": {"documents": len(corpus), **{s: len(q) for s, q in labels.items()}},
    }
    if frozen and frozen != manifest:
        raise ValueError(
            "Prepared data differs from frozen manifest; do not silently replace the baseline"
        )
    write_json(dest / "manifest.json", manifest)
