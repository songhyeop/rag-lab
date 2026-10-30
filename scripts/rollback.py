import argparse
import os
from pathlib import Path

import boto3

from rag_lab.io import read_json, write_json
from scripts.deploy import smoke, update_origin

p = argparse.ArgumentParser()
p.add_argument("--release", type=Path, required=True)
p.add_argument("--token-env", default="RAG_SMOKE_GOOGLE_TOKEN")
args = p.parse_args()
release = read_json(args.release)
session = boto3.Session(region_name="ap-southeast-2")
if session.client("sts").get_caller_identity()["Account"] != release["account"]:
    raise ValueError("Rollback AWS account mismatch")

token = os.environ.get(args.token_env, "")
if not token:
    raise ValueError("Invited Google ID token required to verify rollback search")
lam = session.client("lambda")
prior = lam.get_function_configuration(
    FunctionName=release["function"], Qualifier=release["previous_version"]
)
index_version = prior["Environment"]["Variables"]["RAG_INDEX_VERSION"]
smoke(lam, release["function"], release["previous_version"], token, index_version)
lam.update_alias(
    FunctionName=release["function"], Name="live", FunctionVersion=release["previous_version"]
)
update_origin(
    session.client("cloudfront"), release["distribution"], release["previous_frontend_path"]
)
smoke(lam, release["function"], "live", token, index_version)
release["status"] = "rolled-back-verified"
write_json(args.release, release)
print("Rollback verified:", release["previous_version"])
