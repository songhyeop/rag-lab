"""API-only embeddings with a durable, content-addressed SQLite vector cache.

The cache stores hashes and vectors, never source text, questions or API credentials.
A cross-process lock covers lookup/call/commit to avoid duplicate paid calls on one host.
"""

import fcntl
import hashlib
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_lab.io import digest, sha, write_json

logger = logging.getLogger("rag_lab.embeddings")


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    dimensions: int
    revision: str
    region: str = "ap-southeast-2"
    preprocessing: str = "strip-v1"
    normalize: bool = True

    def __post_init__(self):
        if (
            not self.model
            or not self.revision
            or self.dimensions <= 0
            or self.preprocessing != "strip-v1"
            or not self.normalize
        ):
            raise ValueError("Explicit embedding model/revision/dimensions/normalization required")

    def identity(self) -> dict:
        return asdict(self)

    def key(self, text: str) -> str:
        # Chunk position, source ID and chunk size are intentionally absent: identical text reuses
        # its vector across datasets, chunk configurations and rebuilt FAISS indexes.
        return digest({"schema": 1, "spec": self.identity(), "text_sha256": digest(text.strip())})


@dataclass
class EmbeddingBatch:
    vectors: np.ndarray
    input_tokens: int
    response_model: str
    request_id: str = ""


class Provider(Protocol):
    spec: EmbeddingSpec

    def generate(self, texts: list[str]) -> EmbeddingBatch: ...


def normalized(vectors: Any, count: int, dimensions: int) -> np.ndarray:
    array = np.asarray(vectors, dtype="<f4")
    if array.shape != (count, dimensions) or not np.isfinite(array).all():
        raise ValueError("Embedding API returned invalid dimensions or non-finite values")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Embedding API returned a zero vector")
    return (array / lengths).astype("<f4")


class VectorCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS vectors (
                key TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, text_hash TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector BLOB NOT NULL, checksum TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("""CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY, spec_hash TEXT NOT NULL, count INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL, response_model TEXT NOT NULL, request_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=120)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def get(self, db: sqlite3.Connection, spec: EmbeddingSpec, text: str) -> np.ndarray | None:
        row = db.execute(
            "SELECT dimensions,vector,checksum FROM vectors WHERE key=?", (spec.key(text),)
        ).fetchone()
        if row is None:
            return None
        dimensions, raw, checksum = row
        if dimensions != spec.dimensions or hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("Corrupt cached vector; refusing to silently regenerate and charge")
        vector = np.frombuffer(raw, dtype="<f4").copy()
        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
            raise ValueError("Invalid cached vector")
        return vector

    def put_batch(
        self, db: sqlite3.Connection, spec: EmbeddingSpec, texts: list[str], batch: EmbeddingBatch
    ) -> None:
        vectors = normalized(batch.vectors, len(texts), spec.dimensions)
        with db:
            for text, vector in zip(texts, vectors, strict=True):
                raw = vector.tobytes()
                db.execute(
                    "INSERT INTO vectors(key,spec_hash,text_hash,dimensions,vector,checksum) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        spec.key(text),
                        digest(spec.identity()),
                        digest(text),
                        spec.dimensions,
                        raw,
                        hashlib.sha256(raw).hexdigest(),
                    ),
                )
            db.execute(
                "INSERT INTO batches(spec_hash,count,input_tokens,response_model,request_id) "
                "VALUES(?,?,?,?,?)",
                (
                    digest(spec.identity()),
                    len(texts),
                    batch.input_tokens,
                    batch.response_model,
                    batch.request_id,
                ),
            )

    def info(self) -> dict:
        with self.connect() as db:
            return {
                "vectors": db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
                "namespaces": [
                    {"spec_hash": r[0], "dimensions": r[1], "vectors": r[2]}
                    for r in db.execute(
                        "SELECT spec_hash,dimensions,COUNT(*) FROM vectors "
                        "GROUP BY spec_hash,dimensions"
                    )
                ],
                "api_batches": db.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
                "input_tokens": db.execute(
                    "SELECT COALESCE(SUM(input_tokens),0) FROM batches"
                ).fetchone()[0],
            }

    def export(self, destination: Path) -> dict:
        if destination.resolve() == self.path.resolve() or destination.exists():
            raise ValueError("Export destination must be a new, separate file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # SQLite backup includes committed WAL pages and produces a self-contained portable file.
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        stats = VectorCache(destination).info()
        with sqlite3.connect(destination) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {"schema_version": 1, "cache_sha256": sha(destination), "stats": stats}
        write_json(destination.with_suffix(destination.suffix + ".manifest.json"), manifest)
        return manifest


class CachedEmbedder:
    def __init__(
        self, provider: Provider, cache: VectorCache, batch_size: int = 32, offline: bool = False
    ):
        self.provider, self.spec, self.cache = provider, provider.spec, cache
        self.batch_size, self.offline = batch_size, offline
        if not 1 <= batch_size <= 128:
            raise ValueError("Embedding batch size must be 1..128")
        self.last_stats: dict = {}
        self.query_vectors: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.spec.dimensions), dtype="float32")
        texts = [t.strip() for t in texts]
        if any(not t for t in texts):
            raise ValueError("Cannot embed empty text")
        unique = list(dict.fromkeys(texts))
        found = {}
        api_batches = 0
        # Every process using this cache participates in this lock. No network call precedes it.
        with self.cache.path.with_suffix(self.cache.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.cache.connect() as db:
                missing = []
                for text in unique:
                    vector = self.cache.get(db, self.spec, text)
                    if vector is None:
                        missing.append(text)
                    else:
                        found[text] = vector
                cached = len(found)
                logger.warning(
                    "Embedding cache: %d reused, %d missing (of %d unique)",
                    cached,
                    len(missing),
                    len(unique),
                )
                if self.offline and missing:
                    raise ValueError(f"Offline cache misses: {len(missing)}; no API was called")
                for start in range(0, len(missing), self.batch_size):
                    batch_texts = missing[start : start + self.batch_size]
                    response = self.provider.generate(batch_texts)
                    self.cache.put_batch(db, self.spec, batch_texts, response)
                    for text in batch_texts:
                        committed = self.cache.get(db, self.spec, text)
                        assert committed is not None
                        found[text] = committed
                    api_batches += 1
                    logger.warning(
                        "Embedding checkpoint committed: %d/%d new vectors",
                        start + len(batch_texts),
                        len(missing),
                    )
        self.last_stats = {
            "total": len(texts),
            "unique": len(unique),
            "reused": cached,
            "created": len(missing),
            "api_batches": api_batches,
            "spec_sha256": digest(self.spec.identity()),
        }
        return np.stack([found[text] for text in texts]).astype("float32")

    def embed_query(
        self, text: str, *, persist: bool = False, subject: str | None = None
    ) -> np.ndarray:
        if persist:
            return self.embed([text])
        key = self.spec.key(text)
        if key not in self.query_vectors:
            if self.offline:
                with self.cache.connect() as db:
                    vector = self.cache.get(db, self.spec, text.strip())
                if vector is None:
                    raise ValueError("Query is not in offline cache")
                self.query_vectors[key] = vector.reshape(1, -1)
            else:
                scoped = getattr(self.provider, "for_subject", None)
                provider = scoped(subject) if scoped and subject else self.provider
                response = provider.generate([text.strip()])
                self.query_vectors[key] = normalized(response.vectors, 1, self.spec.dimensions)
            if len(self.query_vectors) > 256:
                del self.query_vectors[next(iter(self.query_vectors))]
        return self.query_vectors[key].copy()
