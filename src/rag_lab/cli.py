import argparse
import json
from pathlib import Path

from rag_lab.data import prepare
from rag_lab.embeddings import VectorCache
from rag_lab.evaluation import evaluate, validate_report
from rag_lab.io import read_json
from rag_lab.models import OnnxModel, fetch_models
from rag_lab.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Lab; all paid model calls are opt-in")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("dataset", choices=["scifact", "hotpotqa"])
    p.add_argument("--root", type=Path, default=Path("data"))
    p = sub.add_parser("models")
    p.add_argument("--lock", type=Path, default=Path("configs/models.lock.json"))
    p.add_argument("--dest", type=Path, default=Path("models"))
    p = sub.add_parser("index")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--config", type=Path, default=Path("configs/default.json"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/embedding-cache.sqlite3"))
    p.add_argument("--allow-api", action="store_true", help="Explicitly permit paid cache misses")
    p = sub.add_parser("evaluate")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--split", choices=["train", "test", "dev"], required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--subject", default="")
    p.add_argument("--allow-embedding-api", action="store_true")
    p = sub.add_parser("validate-report")
    p.add_argument("path", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("smoke-reranker", aliases=["smoke-models"])
    p.add_argument("--models", type=Path, default=Path("models"))
    p = sub.add_parser("cache-info")
    p.add_argument("--cache", type=Path, default=Path("artifacts/embedding-cache.sqlite3"))
    p = sub.add_parser("cache-export")
    p.add_argument("--cache", type=Path, default=Path("artifacts/embedding-cache.sqlite3"))
    p.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.root)
    elif args.command == "models":
        fetch_models(args.lock, args.dest)
    elif args.command == "index":
        build_index(
            args.data,
            args.dest,
            args.models,
            read_json(args.config),
            cache=args.cache,
            offline=not args.allow_api,
            allow_api=args.allow_api,
        )
    elif args.command == "evaluate":
        evaluate(
            args.data,
            args.index,
            args.models,
            args.split,
            args.dest,
            args.generate,
            args.subject,
            allow_embedding_api=args.allow_embedding_api,
        )
    elif args.command == "validate-report":
        validate_report(args.path, args.data)
    elif args.command in ["smoke-reranker", "smoke-models"]:
        model = OnnxModel(args.models / "reranker")
        scores = model.rerank(
            "Which planet is the red planet?",
            ["Mars is known as the red planet.", "The Moon orbits Earth."],
        )
        if not scores[0] > scores[1]:
            raise RuntimeError("Real ONNX reranker smoke failed")
        print("Real ONNX reranker smoke passed; no embedding model or API call")
    elif args.command == "cache-info":
        print(json.dumps(VectorCache(args.cache).info(), indent=2))
    elif args.command == "cache-export":
        print(json.dumps(VectorCache(args.cache).export(args.dest), indent=2))


if __name__ == "__main__":
    main()
