import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rag_lab.embeddings import CachedEmbedder, EmbeddingBatch, EmbeddingSpec, VectorCache


class FakeProvider:
    def __init__(self, spec=None):
        self.spec = spec or EmbeddingSpec("fixture", "fixture-api", 2, "v1")
        self.calls = []
        self.fail_on = None

    def generate(self, texts):
        self.calls.append(texts)
        if len(self.calls) == self.fail_on:
            raise TimeoutError("injected interruption")
        return EmbeddingBatch(np.array([[len(t), 1] for t in texts]), len(texts), "fixture-api")


def test_restart_rebuild_and_duplicate_content_reuse(tmp_path):
    path = tmp_path / "vectors.sqlite3"
    first = FakeProvider()
    expected = CachedEmbedder(first, VectorCache(path), batch_size=2).embed(
        ["alpha", "beta", "alpha"]
    )
    assert first.calls == [["alpha", "beta"]]
    second = FakeProvider()
    actual = CachedEmbedder(second, VectorCache(path)).embed(["alpha", "beta", "alpha"])
    assert not second.calls
    np.testing.assert_array_equal(actual, expected)
    CachedEmbedder(second, VectorCache(path)).embed(["alpha", "gamma"])
    assert second.calls == [["gamma"]]
    assert VectorCache(path).info()["vectors"] == 3


def test_interrupted_batch_resumes_from_committed_checkpoint(tmp_path):
    path = tmp_path / "vectors.sqlite3"
    first = FakeProvider()
    first.fail_on = 2
    with pytest.raises(TimeoutError):
        CachedEmbedder(first, VectorCache(path), batch_size=2).embed(["a", "b", "c", "d"])
    resumed = FakeProvider()
    CachedEmbedder(resumed, VectorCache(path), batch_size=2).embed(["a", "b", "c", "d"])
    assert resumed.calls == [["c", "d"]]


def test_model_revision_and_dimensions_have_separate_namespaces(tmp_path):
    cache = VectorCache(tmp_path / "vectors.sqlite3")
    old = FakeProvider()
    CachedEmbedder(old, cache).embed(["alpha"])
    new = FakeProvider(replace(old.spec, revision="v2"))
    CachedEmbedder(new, cache).embed(["alpha"])
    assert new.calls == [["alpha"]]
    assert old.spec.key("alpha") != replace(old.spec, dimensions=3).key("alpha")
    assert len(cache.info()["namespaces"]) == 2


def test_offline_miss_does_not_call_api_and_export_is_portable(tmp_path):
    cache = VectorCache(tmp_path / "vectors.sqlite3")
    provider = FakeProvider()
    CachedEmbedder(provider, cache).embed(["alpha"])
    cache.export(tmp_path / "portable.sqlite3")
    offline = FakeProvider()
    embedder = CachedEmbedder(offline, VectorCache(tmp_path / "portable.sqlite3"), offline=True)
    embedder.embed(["alpha"])
    with pytest.raises(ValueError, match="Offline"):
        embedder.embed(["missing"])
    assert not offline.calls


def test_corruption_is_not_silently_rebilled(tmp_path):
    path = tmp_path / "vectors.sqlite3"
    provider = FakeProvider()
    CachedEmbedder(provider, VectorCache(path)).embed(["alpha"])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE vectors SET vector=?", (b"corrupt",))
    fresh = FakeProvider()
    with pytest.raises(ValueError, match="Corrupt"):
        CachedEmbedder(fresh, VectorCache(path)).embed(["alpha"])
    assert not fresh.calls


def test_queries_default_to_memory_only(tmp_path):
    provider = FakeProvider()
    cache = VectorCache(tmp_path / "vectors.sqlite3")
    embedder = CachedEmbedder(provider, cache)
    embedder.embed_query("private query")
    embedder.embed_query("private query")
    assert len(provider.calls) == 1
    assert cache.info()["vectors"] == 0


def test_rebuilding_faiss_uses_saved_vectors_without_network_or_local_model(tmp_path):
    from unittest.mock import patch

    from rag_lab.io import read_json, sha, write_json
    from rag_lab.retrieval import SearchEngine, build_index

    provider = FakeProvider()
    cache = VectorCache(tmp_path / "cache.sqlite3")
    first = CachedEmbedder(provider, cache)
    config = read_json(Path("configs/default.json"))
    config["embedding"] = provider.spec.identity()
    models = tmp_path / "models"
    write_json(models / "reranker/artifact.json", {"revision": "fixture", "files": {}})
    query = "Which planet is the red planet?"
    with patch("rag_lab.retrieval.OnnxModel", side_effect=AssertionError("No local model needed")):
        build_index(Path("tests/fixtures/tiny"), tmp_path / "first", models, config, embedder=first)
        first.embed(
            [query]
        )  # Public evaluation query, explicitly persisted for offline reproduction.
        fresh = FakeProvider()
        offline = CachedEmbedder(fresh, VectorCache(tmp_path / "cache.sqlite3"), offline=True)
        build_index(
            Path("tests/fixtures/tiny"), tmp_path / "second", models, config, embedder=offline
        )
        engine = SearchEngine(tmp_path / "second", models, embedder=offline)
        assert engine.search(query, "dense")["hits"]
    assert not fresh.calls
    assert sha(tmp_path / "first/vectors.faiss") == sha(tmp_path / "second/vectors.faiss")


def test_export_checksum_covers_portable_database(tmp_path):
    from rag_lab.io import sha

    cache = VectorCache(tmp_path / "cache.sqlite3")
    CachedEmbedder(FakeProvider(), cache).embed(["alpha"])
    manifest = cache.export(tmp_path / "snapshot.sqlite3")
    assert manifest["cache_sha256"] == sha(tmp_path / "snapshot.sqlite3")
