import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rag_lab.api import app
from rag_lab.auth import Settings, authenticate

CONFIG = Settings(google_client_id="test-client", invited_emails=["invited@gmail.com"])
CLAIMS = {
    "iss": "https://accounts.google.com",
    "aud": "test-client",
    "sub": "123",
    "email": "invited@gmail.com",
    "email_verified": True,
    "exp": int(time.time()) + 1000,
}


def test_business_routes_block_before_models():
    client = TestClient(app)
    with patch("rag_lab.api.engine") as engine, patch("rag_lab.api.generator") as generator:
        for path in ["/search", "/answer"]:
            assert client.post(path, json={"question": "test"}).status_code == 401
        for path in ["/experiments", "/usage"]:
            assert client.get(path).status_code == 401
        assert client.get("/health").json() == {"status": "ok"}
        engine.assert_not_called()
        generator.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"aud": "wrong"},
        {"email_verified": False},
        {"iss": "wrong"},
        {"email": "stranger@gmail.com"},
        {"sub": ""},
        {"email": "invited@example.org"},
    ],
)
def test_uninvited_or_invalid_claims(change):
    with patch("rag_lab.auth.id_token.verify_oauth2_token", return_value={**CLAIMS, **change}):
        with pytest.raises(HTTPException) as exc:
            authenticate("opaque", CONFIG)
        assert exc.value.status_code == 403


def test_signature_and_expiry_errors_are_unauthorized():
    with patch("rag_lab.auth.id_token.verify_oauth2_token", side_effect=ValueError("expired")):
        with pytest.raises(HTTPException) as exc:
            authenticate("expired", CONFIG)
        assert exc.value.status_code == 401


def test_allowed_account_uses_sub():
    with patch("rag_lab.auth.id_token.verify_oauth2_token", return_value=CLAIMS):
        assert authenticate("token", CONFIG)["sub"] == "123"
