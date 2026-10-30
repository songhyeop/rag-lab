from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from rag_lab.budget import LIMIT, BudgetDenied, Ledger


@pytest.fixture
def ledger():
    with mock_aws():
        client = boto3.client("dynamodb", region_name="ap-southeast-2")
        client.create_table(
            TableName="usage",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield Ledger("usage", client)


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
