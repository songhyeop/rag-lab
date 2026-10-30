import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from uuid import uuid4

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from rag_lab.budget import LIMIT, BudgetDenied, Ledger


@pytest.fixture
def ledger():
    endpoint = os.environ.get("RAG_TEST_DYNAMODB_ENDPOINT")
    if endpoint and not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("Integration tests only target local DynamoDB")
    with nullcontext() if endpoint else mock_aws():
        client = boto3.client(
            "dynamodb",
            region_name="ap-southeast-2",
            endpoint_url=endpoint,
            aws_access_key_id="localtest",
            aws_secret_access_key="localtest",
        )
        table = "usage-" + uuid4().hex
        client.create_table(
            TableName=table,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        try:
            yield Ledger(table, client)
        finally:
            client.delete_table(TableName=table)


def reserve(ledger, subject="alice", amount=100):
    return ledger.reserve(subject, str(uuid4()), amount)


def test_user_global_concurrency_and_unknown_usage(ledger):
    a = reserve(ledger)
    with pytest.raises(BudgetDenied):
        reserve(ledger)
    b = reserve(ledger, "bob")
    with pytest.raises(BudgetDenied):
        reserve(ledger, "charlie")
    ledger.settle(a, 40, {"inputTokens": 4, "outputTokens": 2})
    ledger.settle(b, None)
    assert ledger.usage("alice")["service"] == {"active": 0, "charged": 140}
    with pytest.raises(ClientError):
        ledger.settle(a, 40)
    assert ledger.usage("alice")["service"]["charged"] == 140


def test_daily_limit_and_shared_budget(ledger):
    for _ in range(30):
        r = reserve(ledger)
        ledger.settle(r, 1)
    with pytest.raises(BudgetDenied):
        reserve(ledger)
    r = reserve(ledger, "bob", LIMIT - 30)
    ledger.settle(r, None)
    with pytest.raises(BudgetDenied):
        reserve(ledger, "charlie")
    assert ledger.usage("bob")["service"]["charged"] == LIMIT


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_DYNAMODB_ENDPOINT"),
    reason="DynamoDB Local required; Moto is not thread-safe",
)
def test_parallel_reservation_preserves_cost_ceiling(ledger):
    def attempt(i):
        try:
            return reserve(ledger, str(i), LIMIT)
        except BudgetDenied:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt, range(6)))
    assert sum(x is not None for x in results) == 1
    assert ledger.usage("0")["service"]["charged"] == LIMIT


def test_explicit_local_evaluation_shares_total_but_not_web_daily_limit(ledger):
    evaluation = Ledger(ledger.table, ledger.client, evaluation=True)
    for _ in range(31):
        r = reserve(evaluation)
        evaluation.settle(r, 1, {"inputTokens": 1, "outputTokens": 1})
    with pytest.raises(BudgetDenied):
        reserve(ledger)
    usage = evaluation.usage("alice")
    assert usage["user"]["charged"] == 31
    assert usage["user"]["input_tokens"] == 31
    assert usage["service"]["charged"] == 31


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_DYNAMODB_ENDPOINT"), reason="DynamoDB Local required"
)
def test_parallel_same_user_is_single_flight(ledger):
    def attempt(_):
        try:
            return reserve(ledger, "alice", 1)
        except BudgetDenied:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        winners = [r for r in pool.map(attempt, range(6)) if r is not None]
    assert len(winners) == 1
    assert ledger.usage("alice")["today"]["calls"] == 1
    assert ledger.usage("alice")["service"] == {"active": 1, "charged": 1}


def test_embedding_cost_shares_budget_without_consuming_generation_quota(ledger):
    reservation = ledger.reserve("alice", str(uuid4()), 100, kind="embedding")
    ledger.settle(reservation, 50, {"inputTokens": 5, "outputTokens": 0})
    usage = ledger.usage("alice")
    assert usage["service"]["charged"] == 50
    assert usage["user"]["input_tokens"] == 5
    assert usage["today"].get("calls", 0) == 0
