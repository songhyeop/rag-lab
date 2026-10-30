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
