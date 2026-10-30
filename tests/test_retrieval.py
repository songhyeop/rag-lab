import math

import numpy as np
import pytest

from rag_lab.metrics import answer_metrics, citation_metrics, retrieval_metrics
from rag_lab.retrieval import context, rrf, unique_docs


def test_rrf_and_document_collapse():
    ranked = rrf([[2, 1, 0], [0, 2, 1]])
    assert [i for i, _ in ranked] == [2, 0, 1]
    assert ranked[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert unique_docs([{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]) == ["a", "b"]


def test_fixed_vector_ranking_and_metrics():
    # No network/model dependency in PR CI.
    import faiss

    index = faiss.IndexFlatIP(2)
    vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype="float32")
    index.add(vectors)
    _, ids = index.search(np.array([[1, 0]], dtype="float32"), 3)
    assert ids.tolist() == [[0, 1, 2]]
    metrics = retrieval_metrics(["a", "a", "b", "c"], {"a": 1, "c": 1})
    assert metrics["recall@5"] == 1
    assert metrics["mrr@10"] == 1
    assert metrics["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


@pytest.mark.parametrize(
    ("prediction", "gold", "em", "f1"),
    [
        ("The Mars!", "Mars", 1, 1),
        ("yes", "yes indeed", 0, 0),
        ("New York City", "New York", 0, 0.8),
        ("", "", 1, 0),
        ("no", "yes", 0, 0),
    ],
)
def test_hotpot_answer_scoring(prediction, gold, em, f1):
    assert answer_metrics(prediction, gold) == {"answer_em": em, "answer_f1": f1}


def test_citations_are_document_metrics():
    assert citation_metrics(["a", "a", "wrong"], {"a": 1, "b": 1}) == {
        "citation_precision": 0.5,
        "citation_recall": 0.5,
        "all_support_cited": 0,
    }


def test_context_token_budget():
    from rag_lab.retrieval import ENC

    hits = [{"doc_id": str(i), "title": "Title", "text": "evidence " * 2000} for i in range(8)]
    output = context(hits, {"context_tokens": 2000, "context_k": 5})
    assert len(output) <= 5
    assert (
        sum(len(ENC.encode(f"[{h['doc_id']}] {h['title']}\n" + h["text"])) for h in output) <= 2000
    )
