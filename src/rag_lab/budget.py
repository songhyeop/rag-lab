"""Fail-closed shared ledger. Amounts are integer nano-USD, including reservations.

No automatic reservation expiry: unknown Bedrock outcomes retain the maximum charge.
Active locks are deliberately retained after a process crash until an operator reconciles.
"""

from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

LIMIT = 30_000_000_000


class BudgetDenied(Exception):
    pass


class Ledger:
    def __init__(self, table: str, client: Any = None, *, evaluation: bool = False):
        if not table:
            raise ValueError("Shared budget table required")
        self.evaluation = evaluation
        self.table = table
        self.client = client or boto3.client(
            "dynamodb",
            region_name="ap-southeast-2",
            config=Config(retries={"total_max_attempts": 1}),
        )

    def reserve(
        self, subject: str, request_id: str, amount: int, *, kind: str = "generation"
    ) -> dict:
        if kind not in {"generation", "embedding"}:
            raise ValueError("Unsupported budget request kind")
        if amount <= 0 or amount > LIMIT:
            raise BudgetDenied("Invalid reservation amount")
        day = datetime.now(UTC).date().isoformat()
        user = f"user#{subject}"
        # User active lock spans UTC days; daily counters are separate records.
        operations: list[dict[str, Any]] = [
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": "global"}},
                    "UpdateExpression": "SET active = if_not_exists(active, :zero) + :one, "
                    "charged = if_not_exists(charged, :zero) + :cost",
                    "ConditionExpression": "(attribute_not_exists(active) OR active < :two) AND "
                    "(attribute_not_exists(charged) OR charged <= :available)",
                    "ExpressionAttributeValues": {
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                        ":two": {"N": "2"},
                        ":cost": {"N": str(amount)},
                        ":available": {"N": str(LIMIT - amount)},
                    },
                }
            },
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": user}},
                    "UpdateExpression": "SET active = :one, charged = if_not_exists(charged, :zero) + :cost",
                    "ConditionExpression": "attribute_not_exists(active) OR active = :zero",
                    "ExpressionAttributeValues": {
                        ":one": {"N": "1"},
                        ":zero": {"N": "0"},
                        ":cost": {"N": str(amount)},
                    },
                }
            },
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"{user}#{day}"}},
                    "UpdateExpression": "SET calls = if_not_exists(calls, :zero) + :one",
                    "ConditionExpression": "attribute_not_exists(calls) OR calls < :limit",
                    "ExpressionAttributeValues": {
                        ":one": {"N": "1"},
                        ":zero": {"N": "0"},
                        ":limit": {"N": "30"},
                    },
                }
            },
            {
                "Put": {
                    "TableName": self.table,
                    "Item": {
                        "pk": {"S": f"request#{request_id}"},
                        "subject": {"S": subject},
                        "amount": {"N": str(amount)},
                        "state": {"S": "reserved"},
                        "day": {"S": day},
                        "kind": {"S": kind},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
        ]
        if kind == "embedding":
            operations.pop(2)  # Embedding calls share dollars, not the 30 generations/day counter.
        elif self.evaluation:
            # Local explicit evaluation only. Shares global cost/concurrency; web keeps 30/day.
            daily = operations[2]["Update"]
            daily.pop("ConditionExpression")
            daily["ExpressionAttributeValues"].pop(":limit")
        try:
            self.client.transact_write_items(
                TransactItems=operations, ClientRequestToken=request_id
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "TransactionCanceledException":
                raise BudgetDenied("Budget, daily quota or concurrency limit reached") from exc
            raise
        return {"subject": subject, "request_id": request_id, "amount": amount}

    def settle(self, reservation: dict, actual: int | None, usage: dict | None = None) -> None:
        reserved = reservation["amount"]
        # A violated bound is never silently refunded. Alert/reconcile via failed settlement.
        if actual is not None and (actual < 0 or actual > reserved):
            raise ValueError("Provider usage exceeded reservation")
        retained = reserved if actual is None else actual
        item = {"pk": {"S": f"request#{reservation['request_id']}"}}
        values = {
            ":reserved": {"S": "reserved"},
            ":done": {"S": "unknown" if actual is None else "settled"},
            ":actual": {"N": str(retained)},
            ":input": {"N": str((usage or {}).get("inputTokens", 0))},
            ":output": {"N": str((usage or {}).get("outputTokens", 0))},
        }
        self.client.transact_write_items(
            TransactItems=[
                {
                    "Update": {
                        "TableName": self.table,
                        "Key": item,
                        "UpdateExpression": "SET #state = :done, actual = :actual, "
                        "input_tokens = :input, output_tokens = :output",
                        "ConditionExpression": "#state = :reserved",
                        "ExpressionAttributeNames": {"#state": "state"},
                        "ExpressionAttributeValues": values,
                    }
                },
                {
                    "Update": {
                        "TableName": self.table,
                        "Key": {"pk": {"S": "global"}},
                        "UpdateExpression": "ADD active :minus, charged :refund",
                        "ExpressionAttributeValues": {
                            ":minus": {"N": "-1"},
                            ":refund": {"N": str(retained - reserved)},
                        },
                    }
                },
                {
                    "Update": {
                        "TableName": self.table,
                        "Key": {"pk": {"S": f"user#{reservation['subject']}"}},
                        "UpdateExpression": "SET active = :zero ADD charged :refund, input_tokens :input, output_tokens :output",
                        "ExpressionAttributeValues": {
                            ":zero": {"N": "0"},
                            ":refund": {"N": str(retained - reserved)},
                            ":input": values[":input"],
                            ":output": values[":output"],
                        },
                    }
                },
            ]
        )

    def usage(self, subject: str) -> dict:
        day = datetime.now(UTC).date().isoformat()
        result = {}
        for label, key in [
            ("user", f"user#{subject}"),
            ("today", f"user#{subject}#{day}"),
            ("service", "global"),
        ]:
            item = self.client.get_item(
                TableName=self.table, Key={"pk": {"S": key}}, ConsistentRead=True
            ).get("Item", {})
            result[label] = {k: int(v["N"]) for k, v in item.items() if "N" in v}
        return {**result, "day_utc": day, "currency_unit": "nano-USD", "limit": LIMIT}
