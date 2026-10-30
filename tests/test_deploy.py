import io
import tarfile

import pytest

from rag_lab.io import sha, write_json
from scripts.candidate import REPO, validate_bundle, validate_run
from scripts.deploy import check_config, safe_frontend


def run():
    return {
        "id": 42,
        "repository": {"full_name": REPO},
        "head_repository": {"full_name": REPO},
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/candidate.yml",
        "head_sha": "a" * 40,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("conclusion", "failure"),
        ("head_branch", "feature"),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_reject_invalid_ci_run(key, value):
    data = run()
    data[key] = value
    with pytest.raises(ValueError):
        validate_run(data, 42)


def test_bundle_checksum_and_commit(tmp_path):
    checks = {"backend": "passed", "frontend": "passed", "real_model_smoke": "passed"}
    write_json(tmp_path / "checks.json", checks)
    for name in ["image.tar.gz", "frontend.tar.gz"]:
        (tmp_path / name).write_bytes(b"artifact")
    manifest = {
        "run_id": 42,
        "commit": "a" * 40,
        "repository": REPO,
        "platform": "linux/amd64",
        "image_tag": "rag-lab:" + "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "files": {p.name: sha(p) for p in tmp_path.iterdir()},
    }
    write_json(tmp_path / "candidate.json", manifest)
    validate_bundle(tmp_path, run())
    (tmp_path / "image.tar.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(tmp_path, run())


def test_wrong_aws_account_or_missing_index_fails():
    with pytest.raises(ValueError, match="account"):
        check_config({"expected_account": "123", "region": "ap-southeast-2"}, "456")
    with pytest.raises(ValueError, match="Pinned"):
        check_config(
            {
                "expected_account": "123",
                "region": "ap-southeast-2",
                "google_client_id": "test",
                "invited_emails": ["x@gmail.com"],
            },
            "123",
        )


def test_frontend_path_traversal_rejected(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        safe_frontend(path)


def test_index_archive_identity_and_checksum(tmp_path):
    import hashlib
    import json
    import zipfile

    from rag_lab.io import digest
    from scripts.deploy import validate_index_archive

    files = {"chunks.jsonl": b"{}\n", "vectors.faiss": b"vectors"}
    manifest = {"files": {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}, "models": {}}
    manifest["version"] = digest(manifest)
    archive = tmp_path / "index.zip"

    def write():
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest))
            for n, v in files.items():
                z.writestr(n, v)

    write()
    validate_index_archive(archive, manifest["version"])
    with pytest.raises(ValueError, match="version"):
        validate_index_archive(archive, "wrong")
    files["vectors.faiss"] = b"tampered"
    write()
    with pytest.raises(ValueError, match="checksum"):
        validate_index_archive(archive, manifest["version"])
