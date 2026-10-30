"""Deploy exact CI bytes. Writes recovery manifest before changing the live alias."""

import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import boto3
import requests

from rag_lab.io import read_json, sha, write_json
from scripts.candidate import download


def check_config(config: dict, account: str) -> None:
    if config["expected_account"] != account or config["region"] != "ap-southeast-2":
        raise ValueError("AWS account/region mismatch")
    if (
        not config["google_client_id"]
        or not 1 <= len(config["invited_emails"]) <= 5
        or not all(
            config.get(k)
            for k in [
                "index_version",
                "index_bucket",
                "index_key",
                "index_s3_version",
                "index_zip_sha256",
            ]
        )
    ):
        raise ValueError("Pinned index and Google configuration required")


def wait_function(client, name):
    client.get_waiter("function_updated_v2").wait(FunctionName=name)


def event(path: str, method: str = "GET", token: str = "", body: dict | None = None) -> dict:
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"authorization": f"Bearer {token}", "content-type": "application/json"},
        "requestContext": {
            "http": {"method": method, "path": path, "sourceIp": "127.0.0.1"},
            "stage": "$default",
            "domainName": "smoke.local",
        },
        "body": json.dumps(body) if body else None,
        "isBase64Encoded": False,
    }


def invoke(client, function: str, version: str, payload: dict) -> dict:
    response = client.invoke(
        FunctionName=function, Qualifier=version, Payload=json.dumps(payload).encode()
    )
    data = json.load(response["Payload"])
    if response.get("FunctionError"):
        raise ValueError("Candidate Lambda invocation failed")
    return data


def smoke(
    client, function: str, version: str, token: str, index_version: str, generate: bool = False
) -> None:
    if invoke(client, function, version, event("/health"))["statusCode"] != 200:
        raise ValueError("Candidate health failed")
    if (
        invoke(client, function, version, event("/search", "POST", body={"question": "test"}))[
            "statusCode"
        ]
        != 401
    ):
        raise ValueError("Candidate must block unauthenticated requests")
    for path in ["/search"] + (["/answer"] if generate else []):
        result = invoke(
            client,
            function,
            version,
            event(
                path,
                "POST",
                token,
                {"question": "Which planet is known as the red planet?", "method": "hybrid_rerank"},
            ),
        )
        if (
            result["statusCode"] != 200
            or json.loads(result["body"])["index_version"] != index_version
        ):
            raise ValueError("Authenticated candidate search/version check failed")


def safe_frontend(archive: Path) -> dict[str, bytes]:
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            name = member.name.removeprefix("./")
            if (
                not member.isfile()
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or member.size > 20_000_000
            ):
                raise ValueError("Unsafe frontend archive")
            stream = tar.extractfile(member)
            assert stream is not None
            files[name] = stream.read()
    if "index.html" not in files or "runtime-config.js" not in files:
        raise ValueError("Frontend artifact incomplete")
    return files


def update_origin(client, distribution: str, path: str) -> None:
    response = client.get_distribution_config(Id=distribution)
    config = response["DistributionConfig"]
    for origin in config["Origins"]["Items"]:
        if origin["Id"] == "web":
            origin["OriginPath"] = path
    client.update_distribution(Id=distribution, IfMatch=response["ETag"], DistributionConfig=config)
    client.get_waiter("distribution_deployed").wait(Id=distribution)
    client.create_invalidation(
        DistributionId=distribution,
        InvalidationBatch={
            "Paths": {"Quantity": 1, "Items": ["/*"]},
            "CallerReference": str(time.time_ns()),
        },
    )


def push_image(session, root: Path, candidate: dict, repository_uri: str, account: str) -> str:
    # Authenticate Docker via stdin. Credentials are not command arguments or logs.
    auth = session.client("ecr").get_authorization_token()["authorizationData"][0]
    import base64

    _, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    registry = repository_uri.split("/")[0]
    if registry != f"{account}.dkr.ecr.ap-southeast-2.amazonaws.com":
        raise ValueError("ECR account/region mismatch")
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password.encode(),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["docker", "load", "--input", str(root / "image.tar.gz")], check=True)
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", candidate["image_tag"]])
    )[0]
    if (
        image["Id"] != candidate["image_id"]
        or image["Architecture"] != "amd64"
        or image["Os"] != "linux"
    ):
        raise ValueError("Loaded image identity/platform mismatch")
    target = repository_uri + ":" + candidate["commit"]
    subprocess.run(["docker", "tag", candidate["image_tag"], target], check=True)
    subprocess.run(["docker", "push", target], check=True)
    repo = repository_uri.split("/", 1)[1]
    digest = session.client("ecr").describe_images(
        repositoryName=repo, imageIds=[{"imageTag": candidate["commit"]}]
    )["imageDetails"][0]["imageDigest"]
    return repository_uri + "@" + digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generate-smoke", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config)
    session = boto3.Session(region_name="ap-southeast-2")
    account = session.client("sts").get_caller_identity()["Account"]
    check_config(cfg, account)
    token = os.environ.get(cfg["smoke_token_env"], "")
    if not token:
        raise ValueError("A valid invited Google ID token is required for candidate search smoke")
    if session.client("freetier").get_account_plan_state()["accountPlanType"] != "FREE":
        raise ValueError("Free plan expected")
    root = Path("artifacts") / f"run-{args.run_id}"
    run, candidate = download(args.run_id, root)
    web = safe_frontend(root / "frontend.tar.gz")
    s3 = session.client("s3")
    archive = root / "index.zip"
    s3.download_file(
        cfg["index_bucket"],
        cfg["index_key"],
        str(archive),
        ExtraArgs={"VersionId": cfg["index_s3_version"]},
    )
    if sha(archive) != cfg["index_zip_sha256"]:
        raise ValueError("Index artifact checksum mismatch")
    import zipfile

    with zipfile.ZipFile(archive) as z:
        index = json.loads(z.read("manifest.json"))
    if index["version"] != cfg["index_version"]:
        raise ValueError("Index version mismatch")
    cf = session.client("cloudformation")
    stack = cf.describe_stacks(StackName=cfg["stack"])["Stacks"][0]
    outputs = {x["OutputKey"]: x["OutputValue"] for x in stack["Outputs"]}
    function = outputs["FunctionName"]
    lam = session.client("lambda")
    previous = lam.get_alias(FunctionName=function, Name="live")
    dist = session.client("cloudfront")
    previous_dist = dist.get_distribution_config(Id=outputs["DistributionId"])["DistributionConfig"]
    previous_path = next(
        o["OriginPath"] for o in previous_dist["Origins"]["Items"] if o["Id"] == "web"
    )
    image_uri = push_image(session, root, candidate, cfg["repository_uri"], account)
    # Save a complete rollback record before mutating the function or alias.
    prior_config = lam.get_function_configuration(
        FunctionName=function, Qualifier=previous["FunctionVersion"]
    )
    release = {
        "run_id": args.run_id,
        "commit": run["head_sha"],
        "account": account,
        "region": cfg["region"],
        "stack": cfg["stack"],
        "function": function,
        "distribution": outputs["DistributionId"],
        "web_bucket": outputs["WebBucketName"],
        "previous_version": previous["FunctionVersion"],
        "previous_frontend_path": previous_path,
        "previous_environment": prior_config["Environment"],
        "image": image_uri,
        "index_version": cfg["index_version"],
        "index_s3_version": cfg["index_s3_version"],
        "status": "candidate",
    }
    record = Path("releases") / f"{args.run_id}.json"
    write_json(record, release)
    env = lam.get_function_configuration(FunctionName=function)["Environment"]["Variables"]
    env.update(
        RAG_GOOGLE_CLIENT_ID=cfg["google_client_id"],
        RAG_INVITED_EMAILS=json.dumps(cfg["invited_emails"]),
        RAG_INDEX_BUCKET=cfg["index_bucket"],
        RAG_INDEX_KEY=cfg["index_key"],
        RAG_INDEX_S3_VERSION=cfg["index_s3_version"],
        RAG_INDEX_ZIP_SHA256=cfg["index_zip_sha256"],
        RAG_INDEX_VERSION=cfg["index_version"],
        RAG_CORS_ORIGIN=outputs["DistributionUrl"],
        RAG_PRICE_JSON=json.dumps(read_json(Path(cfg["price_file"]))),
    )
    lam.update_function_configuration(FunctionName=function, Environment={"Variables": env})
    wait_function(lam, function)
    lam.update_function_code(FunctionName=function, ImageUri=release["image"])
    wait_function(lam, function)
    version = lam.publish_version(
        FunctionName=function, Description=f"CI {args.run_id} {candidate['commit']}"
    )["Version"]
    release["version"] = version
    write_json(record, release)
    smoke(lam, function, version, token, cfg["index_version"], args.generate_smoke)
    prefix = f"releases/{args.run_id}-{candidate['commit']}"
    web["runtime-config.js"] = (
        "window.RAG_CONFIG = "
        + json.dumps({"googleClientId": cfg["google_client_id"], "apiBase": ""})
        + ";\n"
    ).encode()
    import mimetypes

    for name, data in web.items():
        s3.put_object(
            Bucket=outputs["WebBucketName"],
            Key=prefix + "/" + name,
            Body=data,
            ContentType=mimetypes.guess_type(name)[0] or "application/octet-stream",
            CacheControl="no-cache"
            if name in ["index.html", "runtime-config.js"]
            else "max-age=31536000,immutable",
        )
    try:
        lam.update_alias(
            FunctionName=function,
            Name="live",
            FunctionVersion=version,
            RevisionId=previous["RevisionId"],
        )
        update_origin(dist, outputs["DistributionId"], "/" + prefix)
        for path, expected in [("/health", 200), ("/usage", 401)]:
            if requests.get(outputs["DistributionUrl"] + path, timeout=60).status_code != expected:
                raise ValueError("Public post-deployment smoke failed")
        smoke(lam, function, "live", token, cfg["index_version"])
    except Exception:
        lam.update_alias(
            FunctionName=function, Name="live", FunctionVersion=previous["FunctionVersion"]
        )
        update_origin(dist, outputs["DistributionId"], previous_path)
        release["status"] = "failed-restored"
        write_json(record, release)
        raise
    release.update(status="live", frontend_path="/" + prefix)
    write_json(record, release)
    print("Deployed:", outputs["DistributionUrl"], "release:", record)


if __name__ == "__main__":
    main()
