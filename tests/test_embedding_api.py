import io
import json
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from rag_lab.embedding_api import (
    BedrockEmbeddings,
    MeteredEmbeddings,
    OpenAIEmbeddings,
    create_embedder,
)
from rag_lab.embeddings import EmbeddingBatch, EmbeddingSpec
from rag_lab.io import write_json


def test_openai_batch_response_order_without_network():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}],
                "usage": {"prompt_tokens": 2},
            },
            headers={"x-request-id": "fixture-request"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIEmbeddings(
        EmbeddingSpec("openai", "text-embedding-3-small", 2, "epoch1", region="global"),
        client,
        api_key="fake-test-key",
    )
    batch = provider.generate(["a", "b"])
    np.testing.assert_array_equal(batch.vectors, [[1, 0], [0, 1]])
    assert batch.input_tokens == 2


def test_bedrock_payload_without_network():
    client = Mock()
    client.invoke_model.return_value = {
        "body": io.StringIO(json.dumps({"embedding": [1] * 256, "inputTextTokenCount": 2}))
    }
    provider = BedrockEmbeddings(
        EmbeddingSpec("bedrock", "amazon.titan-embed-text-v2:0", 256, "epoch1"), client
    )
    assert provider.generate(["a"]).vectors.shape == (1, 256)
    assert json.loads(client.invoke_model.call_args.kwargs["body"])["normalize"]
    with pytest.raises(ValueError, match="one inputText"):
        provider.generate(["a", "b"])


def metered(tmp_path, enabled=True):
    spec = EmbeddingSpec("bedrock", "amazon.titan-embed-text-v2:0", 256, "epoch1")
    raw = Mock()
    raw.spec = spec
    raw.generate.return_value = EmbeddingBatch(np.ones((1, 256)), 3, spec.model)
    price = tmp_path / "price.json"
    write_json(
        price,
        {
            "provider": spec.provider,
            "model_id": spec.model,
            "region": spec.region,
            "verified_at": "test-only",
            "source_url": "https://aws.amazon.com/bedrock/pricing/",
            "input_usd_per_million": 1,
        },
    )
    ledger = Mock()
    provider = MeteredEmbeddings(
        raw, enabled=enabled, subject="test", price_file=price, ledger=ledger
    )
    return provider, raw, ledger


def test_disabled_api_does_not_reserve_or_call(tmp_path):
    provider, raw, ledger = metered(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="disabled"):
        provider.generate(["a"])
    raw.generate.assert_not_called()
    ledger.reserve.assert_not_called()


def test_embedding_reserves_before_request_and_settles_actual(tmp_path):
    provider, raw, ledger = metered(tmp_path)
    provider.generate(["a"])
    assert ledger.reserve.call_args.kwargs == {"kind": "embedding"}
    assert ledger.settle.call_args.args[1] == 3000
    assert ledger.settle.call_args.args[2]["inputTokens"] == 3


def test_unknown_response_keeps_embedding_reservation(tmp_path):
    provider, raw, ledger = metered(tmp_path)
    raw.generate.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        provider.generate(["a"])
    assert ledger.settle.call_args.args[1] is None


def test_no_local_provider_fallback_or_implicit_default(tmp_path):
    with pytest.raises(ValueError, match="CPU embedding is disabled"):
        create_embedder({"embedding": {"provider": "unconfigured"}}, cache_path=tmp_path / "cache")
