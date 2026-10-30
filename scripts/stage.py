"""Initial deploy staging: exact CI image and checksummed immutable index only."""

import argparse
import zipfile
from pathlib import Path

import boto3

from rag_lab.io import digest, read_json, sha, verify_files, write_json
from scripts.candidate import download
from scripts.deploy import push_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--expected-account", required=True)
    p.add_argument("--storage-stack", default="rag-lab-storage")
    p.add_argument("--index", type=Path, required=True)
    args = p.parse_args()
    session = boto3.Session(region_name="ap-southeast-2")
    account = session.client("sts").get_caller_identity()["Account"]
    if account != args.expected_account:
        raise ValueError("AWS account mismatch")
    if session.client("freetier").get_account_plan_state()["accountPlanType"] != "FREE":
        raise ValueError("Free plan expected")
    root = Path("artifacts") / f"stage-{args.run_id}"
    _, candidate = download(args.run_id, root)
    index = read_json(args.index / "manifest.json")
    if digest({k: v for k, v in index.items() if k != "version"}) != index["version"]:
        raise ValueError("Index manifest mismatch")
    verify_files(args.index, index["files"])
    stack = session.client("cloudformation").describe_stacks(StackName=args.storage_stack)[
        "Stacks"
    ][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}
    image = push_image(session, root, candidate, outputs["RepositoryUri"], account, index)
    archive = root / "index.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ["manifest.json", "chunks.jsonl", "vectors.faiss"]:
            z.write(args.index / name, name)
    checksum = sha(archive)
    key = f"indexes/{index['version']}/{checksum}.zip"
    s3 = session.client("s3")
    response = s3.put_object(
        Bucket=outputs["IndexBucketName"],
        Key=key,
        Body=archive.read_bytes(),
        Metadata={"sha256": checksum, "index-version": index["version"]},
    )
    write_json(
        Path("configs/staged.local.json"),
        {
            "ImageUri": image,
            "IndexBucket": outputs["IndexBucketName"],
            "IndexKey": key,
            "IndexVersion": index["version"],
            "IndexS3Version": response["VersionId"],
            "IndexZipSha256": checksum,
        },
    )
    print(
        "Staged exact CI image and index. Complete configs/staged.local.json into service parameters locally."
    )


if __name__ == "__main__":
    main()
