import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from rank_bm25 import BM25Okapi

from rag_lab.embedding_api import create_embedder
from rag_lab.embeddings import CachedEmbedder
from rag_lab.io import digest, read_json, read_jsonl, sha, verify_files, write_json, write_jsonl
from rag_lab.models import OnnxModel

ENC = tiktoken.get_encoding("cl100k_base")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def unique_docs(hits: list[dict]) -> list[str]:
    return list(dict.fromkeys(h["doc_id"] for h in hits))


def build_index(
    data: Path,
    dest: Path,
    model_dir: Path,
    config: dict,
    *,
    embedder: CachedEmbedder | None = None,
    cache: Path | None = None,
    offline: bool = True,
    allow_api: bool = False,
) -> None:
    started = time.perf_counter()
    source = read_json(data / "manifest.json")
    verify_files(data, source["files"])
    docs = read_jsonl(data / "corpus.jsonl")
    if any(set(d) != {"id", "title", "text"} for d in docs):
        raise ValueError("Unexpected corpus fields: potential label leakage")
    splitter = TokenTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["overlap"], tokenizer=ENC.encode
    )
    chunks = []
    for doc in docs:
        # Explicit text only: never attach labels or query metadata to LlamaIndex nodes.
        nodes = splitter.get_nodes_from_documents(
            [Document(text=doc["title"] + "\n" + doc["text"])]
        )
        for i, node in enumerate(nodes):
            chunks.append(
                {
                    "id": f"{doc['id']}:{i}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": node.get_content(),
                }
            )
    embed = embedder or create_embedder(
        config, cache_path=cache, offline=offline, allow_api=allow_api
    )
    vectors = embed.embed([c["text"] for c in chunks])
    dest.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(dest / "vectors.faiss"))
    write_jsonl(dest / "chunks.jsonl", chunks)
    models = {
        "embedding": embed.spec.identity(),
        "reranker": read_json(model_dir / "reranker" / "artifact.json"),
    }
    manifest = {
        "embedding_cache": embed.last_stats,
        "schema_version": 2,
        "dataset": source["dataset"],
        "scope": source["scope"],
        "dataset_manifest_sha256": sha(data / "manifest.json"),
        "config": config,
        "models": models,
        "files": {n: sha(dest / n) for n in ["vectors.faiss", "chunks.jsonl"]},
        "documents": len(docs),
        "chunks": len(chunks),
        "index_seconds": time.perf_counter() - started,
        "index_bytes": sum(p.stat().st_size for p in dest.iterdir() if p.is_file()),
    }
    manifest["version"] = digest(manifest)
    write_json(dest / "manifest.json", manifest)


class SearchEngine:
    def __init__(
        self,
        index_dir: Path,
        model_dir: Path,
        expected: str = "",
        *,
        embedder: CachedEmbedder | None = None,
        cache: Path | None = None,
        offline: bool = True,
        allow_api: bool = False,
        persist_queries: bool = False,
    ):
        self.manifest = read_json(index_dir / "manifest.json")
        base = {k: v for k, v in self.manifest.items() if k != "version"}
        if digest(base) != self.manifest["version"]:
            raise ValueError("Index manifest identity mismatch")
        if expected and self.manifest["version"] != expected:
            raise ValueError("Index version mismatch")
        verify_files(index_dir, self.manifest["files"])
        self.chunks = read_jsonl(index_dir / "chunks.jsonl")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.index = faiss.read_index(str(index_dir / "vectors.faiss"))
        self.config = self.manifest["config"]
        self.embedding = embedder or create_embedder(
            self.config,
            cache_path=cache,
            offline=offline,
            allow_api=allow_api,
            subject="query-embedding",
        )
        if self.embedding.spec.identity() != self.manifest["models"]["embedding"]:
            raise ValueError("API/index embedding specification mismatch")
        if self.index.d != self.embedding.spec.dimensions:
            raise ValueError("FAISS/API embedding dimensions mismatch")
        self.model_dir = model_dir
        self.persist_queries = persist_queries
        self.reranker: OnnxModel | None = None

    def search(
        self,
        query: str,
        method: str = "hybrid_rerank",
        limit: int = 50,
        *,
        subject: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        timings = {}
        k = min(self.config["candidate_k"], len(self.chunks))
        ranks = []
        if method in ["bm25", "hybrid", "hybrid_rerank"]:
            tick = time.perf_counter()
            scores = self.bm25.get_scores(tokenize(query))
            # Zero-evidence lexical matches are omitted.
            lexical = sorted(
                (i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i)
            )[:k]
            ranks.append(lexical)
            timings["bm25_ms"] = (time.perf_counter() - tick) * 1000
        if method in ["dense", "hybrid", "hybrid_rerank"]:
            tick = time.perf_counter()
            vector = self.embedding.embed_query(
                query, persist=self.persist_queries, subject=subject
            )
            _, indices = self.index.search(vector, k)
            ranks.append([int(i) for i in indices[0] if i >= 0])
            timings["dense_ms"] = (time.perf_counter() - tick) * 1000
        if not ranks:
            raise ValueError("Unknown retrieval method")
        ranked = rrf(ranks, self.config["rrf_k"])
        if method == "hybrid_rerank" and ranked:
            tick = time.perf_counter()
            candidates = ranked[: self.config["rerank_k"]]
            if self.reranker is None:
                self.reranker = OnnxModel(self.model_dir / "reranker")
                if self.reranker.manifest != self.manifest["models"]["reranker"]:
                    raise ValueError("Local/index reranker artifact mismatch")
            scores = self.reranker.rerank(query, [self.chunks[i]["text"] for i, _ in candidates])
            reranked = sorted(
                zip([i for i, _ in candidates], scores, strict=True), key=lambda x: (-x[1], x[0])
            )
            ranked = reranked + ranked[self.config["rerank_k"] :]
            timings["rerank_ms"] = (time.perf_counter() - tick) * 1000
        hits = [{**self.chunks[i], "score": float(s)} for i, s in ranked[:limit]]
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return {
            "hits": hits,
            "timings": timings,
            "method": method,
            "index_version": self.manifest["version"],
            "dataset": self.manifest["dataset"],
            "config_sha256": digest(self.config),
            "data_version": self.manifest["dataset_manifest_sha256"],
            "model_revisions": {name: m["revision"] for name, m in self.manifest["models"].items()},
        }


def context(hits: list[dict], config: dict) -> list[dict]:
    out = []
    remaining = config["context_tokens"]
    for hit in hits[: config["context_k"]]:
        header = f"[{hit['doc_id']}] {hit['title']}\n"
        budget = remaining - len(ENC.encode(header))
        if budget <= 0:
            break
        text = ENC.decode(ENC.encode(hit["text"])[:budget])
        out.append({**hit, "text": text})
        remaining -= len(ENC.encode(header + text))
    return out
