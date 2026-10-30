"""Lazy API adapters. No network call until explicitly enabled and budget-reserved."""

import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import boto3
import httpx
import numpy as np
from botocore.config import Config

from rag_lab.auth import settings
from rag_lab.budget import Ledger
from rag_lab.embeddings import CachedEmbedder, EmbeddingBatch, EmbeddingSpec, VectorCache
from rag_lab.io import read_json


class OpenAIEmbeddings:
    def __init__(self, spec: EmbeddingSpec, client: Any = None, api_key: str = ""):
        if spec.provider != "openai" or spec.model != "text-embedding-3-small":
            raise ValueError("Unsupported OpenAI embedding model")
        self.spec, self.client, self.api_key = spec, client, api_key

    def generate(self, texts: list[str]) -> EmbeddingBatch:
        key = self.api_key or settings().openai_api_key
        if not key:
            raise ValueError("RAG_OPENAI_API_KEY required; no API call was made")
        client = self.client or httpx.Client(timeout=60)
        response = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": "Bearer " + key},
            json={
                "model": self.spec.model,
                "input": texts,
                "dimensions": self.spec.dimensions,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        result = response.json()
        data = sorted(result["data"], key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("Embedding API response indexes do not match the batch")
        if result["model"] != self.spec.model:
            raise ValueError("Embedding API returned a different model")
        return EmbeddingBatch(
            np.array([item["embedding"] for item in data]),
            result["usage"]["prompt_tokens"],
            result["model"],
            response.headers.get("x-request-id", ""),
        )


class BedrockEmbeddings:
    def __init__(self, spec: EmbeddingSpec, client: Any = None):
        if (
            spec.provider != "bedrock"
            or spec.model != "amazon.titan-embed-text-v2:0"
            or spec.dimensions not in [256, 512, 1024]
            or spec.region != "ap-southeast-2"
        ):
            raise ValueError(
                "Supported Bedrock embedding: Sydney Titan v2, 256/512/1024 dimensions"
            )
        self.spec, self.client = spec, client

    def generate(self, texts: list[str]) -> EmbeddingBatch:
        if len(texts) != 1:
            raise ValueError("Titan v2 accepts one inputText per request")
        client = self.client or boto3.client(
            "bedrock-runtime",
            region_name=self.spec.region,
            config=Config(retries={"total_max_attempts": 1}, read_timeout=60, connect_timeout=5),
        )
        response = client.invoke_model(
            modelId=self.spec.model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": texts[0],
                    "dimensions": self.spec.dimensions,
                    "normalize": True,
                    "embeddingTypes": ["float"],
                }
            ),
        )
        data = json.load(response["body"])
        return EmbeddingBatch(
            np.array([data["embedding"]]),
            data["inputTextTokenCount"],
            self.spec.model,
            response.get("ResponseMetadata", {}).get("RequestId", ""),
        )


class MeteredEmbeddings:
    def __init__(
        self, provider: Any, *, enabled: bool, subject: str, price_file: Path, ledger: Any = None
    ):
        self.provider, self.spec = provider, provider.spec
        self.enabled, self.subject = enabled, subject
        self.price_file, self.ledger = price_file, ledger

    def for_subject(self, subject: str):
        return MeteredEmbeddings(
            self.provider,
            enabled=self.enabled,
            subject=subject,
            price_file=self.price_file,
            ledger=self.ledger,
        )

    def generate(self, texts: list[str]) -> EmbeddingBatch:
        if not self.enabled:
            raise ValueError(
                "Embedding API disabled. Use --allow-api explicitly after configuration"
            )
        if any(not t or len(t.encode("utf-8")) > 32768 for t in texts):
            raise ValueError("Embedding input is empty or exceeds safety limit")
        price = (
            json.loads(settings().embedding_price_json)
            if settings().embedding_price_json
            else read_json(self.price_file)
        )
        prefix = (
            "https://aws.amazon.com/bedrock/pricing"
            if self.spec.provider == "bedrock"
            else "https://openai.com/api/pricing"
        )
        if (
            price.get("provider") != self.spec.provider
            or price.get("model_id") != self.spec.model
            or price.get("region") != self.spec.region
            or not price.get("verified_at")
            or not price.get("source_url", "").startswith(prefix)
            or price.get("input_usd_per_million", 0) <= 0
        ):
            raise ValueError("Verified embedding price matching provider/model/region required")
        rate = price["input_usd_per_million"]
        ledger = self.ledger or Ledger(settings().usage_table)
        # Both supported models cap input at 8192 tokens per text. Reserve this full ceiling.
        reservation = ledger.reserve(
            self.subject, str(uuid4()), math.ceil(8192 * len(texts) * rate * 1000), kind="embedding"
        )
        actual, usage = None, None
        try:
            result = self.provider.generate(texts)
            usage = {"inputTokens": result.input_tokens, "outputTokens": 0}
            actual = math.ceil(result.input_tokens * rate * 1000)
            return result
        finally:
            ledger.settle(reservation, actual, usage)


def create_embedder(
    config: dict,
    *,
    cache_path: Path | None = None,
    offline: bool = True,
    allow_api: bool = False,
    subject: str = "local-indexer",
) -> CachedEmbedder:
    if "embedding" not in config or config["embedding"].get("provider") == "unconfigured":
        raise ValueError("Select an API embedding configuration; local CPU embedding is disabled")
    spec = EmbeddingSpec(**config["embedding"])
    raw: Any
    if spec.provider == "openai":
        raw = OpenAIEmbeddings(spec)
    elif spec.provider == "bedrock":
        raw = BedrockEmbeddings(spec)
    else:
        raise ValueError("Only openai/bedrock embedding APIs are supported; no local fallback")
    cfg = settings()
    provider = MeteredEmbeddings(
        raw, enabled=allow_api, subject=subject, price_file=Path(cfg.embedding_price_file)
    )
    return CachedEmbedder(
        provider,
        VectorCache(cache_path or Path(cfg.embedding_cache)),
        batch_size=1 if spec.provider == "bedrock" else 64,
        offline=offline,
    )
