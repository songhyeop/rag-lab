"""Document metrics and independently implemented HotpotQA answer normalization.

Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""

import math
import re
import string
from collections import Counter


def retrieval_metrics(ranking: list[str], relevant: dict[str, int]) -> dict[str, float]:
    ranking = list(dict.fromkeys(ranking))
    gold = {key for key, grade in relevant.items() if grade > 0}
    if not gold:
        raise ValueError("Evaluation question has no relevant documents")
    out = {f"recall@{k}": len(set(ranking[:k]) & gold) / len(gold) for k in [5, 10]}
    out["mrr@10"] = next((1 / i for i, d in enumerate(ranking[:10], 1) if d in gold), 0.0)
    dcg = sum(
        (2 ** relevant.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranking[:10], 1)
    )
    ideal = sum(
        (2**g - 1) / math.log2(i + 1)
        for i, g in enumerate(sorted(relevant.values(), reverse=True)[:10], 1)
    )
    out["ndcg@10"] = dcg / ideal
    out["all_support@5"] = float(gold <= set(ranking[:5]))
    out["all_support@10"] = float(gold <= set(ranking[:10]))
    return out


def normalize(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    p, g = normalize(prediction), normalize(gold)
    em = float(p == g)
    overlap = sum((Counter(p.split()) & Counter(g.split())).values())
    if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
        overlap = 0
    f1 = 2 * overlap / (len(p.split()) + len(g.split())) if overlap else 0.0
    return {"answer_em": em, "answer_f1": f1}


def citation_metrics(citations: list[str], relevant: dict[str, int]) -> dict[str, float]:
    cited = set(citations)
    gold = {d for d, g in relevant.items() if g > 0}
    common = len(cited & gold)
    return {
        "citation_precision": common / len(cited) if cited else 0.0,
        "citation_recall": common / len(gold) if gold else 0.0,
        "all_support_cited": float(bool(gold) and gold <= cited),
    }
