import pytest

from rag_lab.data import document, prepare_hotpot


def question(i):
    return {
        "_id": str(i),
        "question": "Which?",
        "answer": "A",
        "context": [[f"A{i}", ["Alpha."]], [f"B{i}", ["Beta."]]],
        "supporting_facts": [[f"A{i}", 0], [f"B{i}", 0]],
    }


def test_deterministic_reduced_corpus_label_separation():
    train, dev = [question(i) for i in range(10)], [question(100 + i) for i in range(10)]
    first = prepare_hotpot(train, dev, count=2, size=12)
    assert first == prepare_hotpot(list(reversed(train)), list(reversed(dev)), count=2, size=12)
    corpus, labels = first
    assert len(corpus) == 12
    assert all(set(d) == {"id", "title", "text"} for d in corpus)
    assert {q["id"] for q in labels["train"]}.isdisjoint(q["id"] for q in labels["dev"])
    assert all(
        set(q["relevant"]) <= {d["id"] for d in corpus} for qs in labels.values() for q in qs
    )
    assert document("A", ["x"])["id"] != document("A", ["y"])["id"]


def test_overlap_and_missing_evidence_fail():
    with pytest.raises(ValueError, match="Overlapping"):
        prepare_hotpot([question(1)], [question(1)], count=1, size=2)
    broken = question(1)
    broken["supporting_facts"] = [["missing", 0]]
    with pytest.raises(ValueError, match="supporting"):
        prepare_hotpot([broken], [question(2)], count=1, size=4)


def test_index_rejects_label_fields_even_with_valid_checksums(tmp_path):
    from rag_lab.io import sha, write_json, write_jsonl
    from rag_lab.retrieval import build_index

    write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": "a", "title": "A", "text": "text", "answer": "leaked"}]
    )
    write_json(
        tmp_path / "manifest.json",
        {
            "dataset": "fixture",
            "scope": "fixture",
            "files": {"corpus.jsonl": sha(tmp_path / "corpus.jsonl")},
        },
    )
    with pytest.raises(ValueError, match="label leakage"):
        build_index(
            tmp_path,
            tmp_path / "index",
            tmp_path / "missing-models",
            {"chunk_size": 256, "overlap": 32},
        )
