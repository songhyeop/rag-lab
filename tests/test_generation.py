from unittest.mock import Mock

import pytest

from rag_lab.generation import MODEL, Generator
from rag_lab.io import write_json


@pytest.fixture
def generator(tmp_path):
    path = tmp_path / "price.json"
    write_json(
        path,
        {
            "model_id": MODEL,
            "region": "ap-southeast-2",
            "verified_at": "test-only",
            "source_url": "https://aws.amazon.com/bedrock/pricing/",
            "input_usd_per_million": 0.1,
            "output_usd_per_million": 0.4,
        },
    )
    client = Mock()
    client.count_tokens.return_value = {"inputTokens": 20}
    client.converse.return_value = {
        "usage": {"inputTokens": 20, "outputTokens": 10},
        "output": {
            "message": {
                "content": [{"text": '{"answer":"Mars","citations":["mars"],"abstained":false}'}]
            }
        },
    }
    ledger = Mock()
    ledger.reserve.return_value = {"amount": 999999}
    return Generator(ledger, path, client)


HITS = [{"doc_id": "mars", "title": "Mars", "text": "Mars is the red planet."}]
CONFIG = {"context_tokens": 2000, "context_k": 5}


def test_generation_records_usage_and_citations(generator):
    result = generator.answer("sub", "Which planet?", HITS, CONFIG)
    assert result["answer"] == "Mars"
    generator.client.count_tokens.assert_not_called()
    generator.ledger.reserve.assert_called_once()
    generator.ledger.settle.assert_called_once()


def test_bad_citations_are_errors_but_usage_settles(generator):
    generator.client.converse.return_value["output"]["message"]["content"][0]["text"] = (
        '{"answer":"Mars","citations":["fake"],"abstained":false}'
    )
    with pytest.raises(ValueError, match="citation"):
        generator.answer("sub", "?", HITS, CONFIG)
    assert generator.ledger.settle.call_args.args[1] is not None


def test_unknown_usage_keeps_reservation(generator):
    generator.client.converse.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        generator.answer("sub", "?", HITS, CONFIG)
    assert generator.ledger.settle.call_args.args[1] is None


def test_budget_denial_happens_before_model_call(generator):
    generator.ledger.reserve.side_effect = ValueError("limit")
    with pytest.raises(ValueError):
        generator.answer("sub", "?", HITS, CONFIG)
    generator.client.converse.assert_not_called()


def test_missing_evidence_abstains_without_call(generator):
    assert generator.answer("sub", "?", [], CONFIG)["abstained"]
    generator.client.converse.assert_not_called()
    generator.ledger.reserve.assert_not_called()
