import argparse
from pathlib import Path

from rag_lab.data import prepare
from rag_lab.evaluation import evaluate, validate_report
from rag_lab.io import read_json
from rag_lab.models import fetch_models
from rag_lab.retrieval import SearchEngine, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Lab local experiments; generation is opt-in")
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
    p = sub.add_parser("evaluate")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--split", choices=["train", "test", "dev"], required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--subject", default="")
    p = sub.add_parser("validate-report")
    p.add_argument("path", type=Path)
    p = sub.add_parser("smoke-models")
    p.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.root)
    elif args.command == "models":
        fetch_models(args.lock, args.dest)
    elif args.command == "index":
        build_index(args.data, args.dest, args.models, read_json(args.config))
    elif args.command == "evaluate":
        evaluate(
            args.data, args.index, args.models, args.split, args.dest, args.generate, args.subject
        )
    elif args.command == "validate-report":
        validate_report(args.path)
    elif args.command == "smoke-models":
        # Real model/index regression from self-authored, non-benchmark fixture.
        dest = Path("artifacts/smoke-index")
        build_index(
            Path("tests/fixtures/tiny"), dest, args.models, read_json(Path("configs/default.json"))
        )
        engine = SearchEngine(dest, args.models)
        for method in ["dense", "hybrid", "hybrid_rerank"]:
            result = engine.search("Which planet is the red planet?", method)
            if result["hits"][0]["doc_id"] != "mars":
                raise RuntimeError(f"Real ONNX search smoke failed: {method}")
        print("Real ONNX embedding/reranker search smoke passed")


if __name__ == "__main__":
    main()
