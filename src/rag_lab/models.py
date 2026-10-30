"""Native ONNX reranker inference. Document/query embeddings are API-only."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from rag_lab.io import read_json, sha, verify_files, write_json


def fetch_models(lock: Path, dest: Path) -> None:
    for name, item in read_json(lock).items():
        path = dest / name
        snapshot_download(
            repo_id=item["repo"],
            revision=item["revision"],
            local_dir=path,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "*LICENSE*",
                "*README*",
                item["onnx"],
            ],
        )
        files = {
            str(p.relative_to(path)): sha(p)
            for p in path.rglob("*")
            if p.is_file() and ".cache" not in p.parts and p.name != "artifact.json"
        }
        write_json(path / "artifact.json", {**item, "files": files})


class OnnxModel:
    def __init__(self, path: Path):
        self.manifest = read_json(path / "artifact.json")
        verify_files(path, self.manifest["files"])
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path / self.manifest["onnx"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def run(self, texts: list) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        return self.session.run(None, {i.name: arrays[i.name] for i in self.session.get_inputs()})[
            0
        ]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return self.run([(query, t) for t in texts]).reshape(-1).tolist()
