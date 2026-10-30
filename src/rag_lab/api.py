import json
import logging
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import boto3
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

from rag_lab.auth import settings, user
from rag_lab.budget import BudgetDenied, Ledger
from rag_lab.generation import Generator
from rag_lab.io import read_json, sha
from rag_lab.retrieval import SearchEngine

app = FastAPI(title="RAG Lab", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings().cors_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
logger = logging.getLogger("rag_lab")
logger.setLevel(logging.INFO)
cold = True


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    global cold
    started = time.perf_counter()
    request.state.request_id = str(uuid4())
    was_cold = cold
    cold = False
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["Cache-Control"] = "no-store"
    # Never log question, prompt, tokens, email or authorization headers.
    logger.info(
        "request id=%s path=%s status=%s duration_ms=%.1f cold=%s",
        request.state.request_id,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
        was_cold,
    )
    return response


@lru_cache
def engine() -> SearchEngine:
    cfg = settings()
    directory = Path(cfg.index_dir)
    if cfg.index_bucket:
        if not all([cfg.index_key, cfg.index_s3_version, cfg.index_zip_sha256, cfg.index_version]):
            raise ValueError("Pinned S3 index version and checksum required")
        directory = Path("/tmp/index")
        if not (directory / "manifest.json").exists():
            archive = Path("/tmp/index.zip")
            boto3.client("s3", region_name="ap-southeast-2").download_file(
                cfg.index_bucket,
                cfg.index_key,
                str(archive),
                ExtraArgs={"VersionId": cfg.index_s3_version},
            )
            if sha(archive) != cfg.index_zip_sha256:
                raise ValueError("Index ZIP checksum mismatch")
            with zipfile.ZipFile(archive) as z:
                allowed = {"manifest.json", "chunks.jsonl", "vectors.faiss"}
                if set(z.namelist()) != allowed:
                    raise ValueError("Unexpected index archive files")
                if sum(i.file_size for i in z.infolist()) > 1_500_000_000:
                    raise ValueError("Index exceeds ephemeral storage budget")
                z.extractall(directory)
    return SearchEngine(directory, Path(cfg.model_dir), cfg.index_version)


@lru_cache
def ledger() -> Ledger:
    return Ledger(settings().usage_table)


@lru_cache
def generator() -> Generator:
    return Generator(
        ledger(),
        json.loads(settings().price_json) if settings().price_json else Path(settings().price_file),
    )


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    method: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(query: Query, request: Request, account: dict = Depends(user)):
    try:
        return {
            "request_id": request.state.request_id,
            **engine().search(query.question, query.method),
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(503, "Pinned search artifacts unavailable") from exc


@app.post("/answer")
def answer(query: Query, request: Request, account: dict = Depends(user)):
    try:
        retrieval = engine().search(query.question, query.method)
        result = generator().answer(
            account["sub"],
            query.question,
            retrieval["hits"],
            engine().config,
            request_id=request.state.request_id,
        )
        return {**retrieval, **result}
    except BudgetDenied as exc:
        raise HTTPException(429, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, "Generation unavailable or response validation failed") from exc


@app.get("/usage")
def usage(account: dict = Depends(user)):
    try:
        return ledger().usage(account["sub"])
    except ValueError as exc:
        raise HTTPException(503, "Shared budget not configured") from exc


@app.get("/experiments")
def experiments(account: dict = Depends(user)):
    # Only aggregate summaries are exposed, never prompts, user logs or gold answers.
    return {
        "experiments": [
            read_json(p) for p in sorted(Path(settings().report_dir).glob("*/summary.json"))
        ]
    }


handler = Mangum(app, lifespan="off")
