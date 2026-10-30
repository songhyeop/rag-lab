import os
from pathlib import Path

from rag_lab.io import sha, write_json

root = Path("artifacts/candidate")
write_json(
    root / "checks.json", {"backend": "passed", "frontend": "passed", "real_model_smoke": "passed"}
)
write_json(
    root / "candidate.json",
    {
        "schema_version": 1,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"],
        "platform": "linux/amd64",
        "image_tag": f"rag-lab:{os.environ['GITHUB_SHA']}",
        "image_id": Path("artifacts/image-id.txt").read_text().strip(),
        "files": {
            name: sha(root / name) for name in ["image.tar.gz", "frontend.tar.gz", "checks.json"]
        },
    },
)
