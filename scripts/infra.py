"""Infrastructure changes are always reviewed as change sets before execution."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import boto3


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("plan")
    c.add_argument("--stack", required=True)
    c.add_argument("--template", type=Path, required=True)
    c.add_argument("--parameters", type=Path)
    e = sub.add_parser("apply")
    e.add_argument("--change-set", required=True)
    e.add_argument("--expected-account", required=True)
    args = p.parse_args()
    session = boto3.Session(region_name="ap-southeast-2")
    cf = session.client("cloudformation")
    account = session.client("sts").get_caller_identity()["Account"]
    if session.client("freetier").get_account_plan_state()["accountPlanType"] != "FREE":
        raise ValueError("Expected existing Free plan")
    if args.action == "plan":
        try:
            cf.describe_stacks(StackName=args.stack)
            kind = "UPDATE"
        except cf.exceptions.ClientError as exc:
            if "does not exist" not in str(exc):
                raise
            kind = "CREATE"
        parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
        result = cf.create_change_set(
            StackName=args.stack,
            ChangeSetName="review-" + str(uuid4()),
            ChangeSetType=kind,
            TemplateBody=args.template.read_text(),
            Parameters=[
                {"ParameterKey": k, "ParameterValue": str(v)} for k, v in parameters.items()
            ],
            Capabilities=["CAPABILITY_IAM", "CAPABILITY_AUTO_EXPAND"],
        )
        print("Account:", account, "Region: ap-southeast-2")
        print("Change set:", result["Id"])
        cf.get_waiter("change_set_create_complete").wait(ChangeSetName=result["Id"])
        description = cf.describe_change_set(ChangeSetName=result["Id"])
        # Avoid printing parameter values (contains private invitations/settings).
        print(json.dumps(description["Changes"], indent=2, default=str))
        print("Review this change set before running apply with its exact ID and account.")
    else:
        if account != args.expected_account:
            raise ValueError("AWS account mismatch")
        description = cf.describe_change_set(ChangeSetName=args.change_set)
        if (
            description["Status"] != "CREATE_COMPLETE"
            or description["ExecutionStatus"] != "AVAILABLE"
        ):
            raise ValueError("Change set is not available")
        cf.execute_change_set(ChangeSetName=args.change_set)
        print("Execution started:", description["StackId"])


if __name__ == "__main__":
    main()
