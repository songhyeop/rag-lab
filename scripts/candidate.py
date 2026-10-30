"""Validate provenance before downloading a successful main candidate. Never uses AWS."""

import argparse
import json
import subprocess
from pathlib import Path

from rag_lab.io import read_json, verify_files

REPO = "songhyeop/rag-lab"
WORKFLOW = ".github/workflows/candidate.yml"
FILES = {"image.tar.gz", "frontend.tar.gz", "checks.json"}


def gh_json(endpoint: str):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def validate_run(run: dict, run_id: int) -> None:
    if (
        run["id"] != run_id
        or run["repository"]["full_name"] != REPO
        or run["head_repository"]["full_name"] != REPO
        or run["head_branch"] != "main"
        or run["event"] != "push"
        or run["status"] != "completed"
        or run["conclusion"] != "success"
        or run["path"] != WORKFLOW
    ):
        raise ValueError(
            "Only successful main push candidate runs from the expected repository qualify"
        )


def validate_bundle(root: Path, run: dict) -> dict:
    manifest = read_json(root / "candidate.json")
    if (
        manifest["run_id"] != run["id"]
        or manifest["commit"] != run["head_sha"]
        or manifest["repository"] != REPO
        or manifest["platform"] != "linux/amd64"
        or manifest.get("image_tag") != "rag-lab:" + run["head_sha"]
        or not manifest.get("image_id", "").startswith("sha256:")
        or len(manifest.get("image_id", "")) != 71
        or set(manifest["files"]) != FILES
    ):
        raise ValueError("Candidate provenance mismatch")
    verify_files(root, manifest["files"])
    checks = read_json(root / "checks.json")
    if checks != {"backend": "passed", "frontend": "passed", "real_model_smoke": "passed"}:
        raise ValueError("Incomplete candidate checks")
    return manifest


def download(run_id: int, dest: Path) -> tuple[dict, dict]:
    run = gh_json(f"repos/{REPO}/actions/runs/{run_id}")
    validate_run(run, run_id)
    artifacts = gh_json(f"repos/{REPO}/actions/runs/{run_id}/artifacts")["artifacts"]
    matches = [a for a in artifacts if a["name"] == "candidate" and not a["expired"]]
    if len(matches) != 1:
        raise ValueError("One non-expired candidate artifact is required")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            REPO,
            "--name",
            "candidate",
            "--dir",
            str(dest),
        ],
        check=True,
    )
    manifest = validate_bundle(dest, run)
    return run, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    _, manifest = download(args.run_id, args.dest)
    print(json.dumps(manifest, indent=2))
