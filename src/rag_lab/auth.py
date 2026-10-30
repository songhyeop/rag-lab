from functools import lru_cache

import requests
from cachecontrol import CacheControl
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    google_client_id: str = ""
    invited_emails: list[str] = []
    index_dir: str = "artifacts/index"
    index_version: str = ""
    model_dir: str = "models"
    usage_table: str = ""
    price_file: str = "configs/pricing.local.json"
    price_json: str = ""
    embedding_api_enabled: bool = False
    embedding_cache: str = "artifacts/embedding-cache.sqlite3"
    embedding_price_file: str = "configs/embedding-price.local.json"
    embedding_price_json: str = ""
    openai_api_key: str = ""
    cors_origin: str = "http://localhost:5173"
    report_dir: str = "results"
    index_bucket: str = ""
    index_key: str = ""
    index_s3_version: str = ""
    index_zip_sha256: str = ""


@lru_cache
def settings() -> Settings:
    return Settings()


transport = Request(session=CacheControl(requests.Session()))
security = HTTPBearer(auto_error=False)


def authenticate(token: str, config: Settings) -> dict:
    if not config.google_client_id or not 1 <= len(config.invited_emails) <= 5:
        raise HTTPException(503, "Google authentication is not configured")
    try:
        claims = id_token.verify_oauth2_token(token, transport, config.google_client_id)
    except (ValueError, requests.RequestException) as exc:
        raise HTTPException(401, "Invalid or expired Google token") from exc
    email = claims.get("email", "").lower()
    if (
        claims.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]
        or claims.get("aud") != config.google_client_id
        or claims.get("email_verified") is not True
        or not claims.get("sub")
        or (not email.endswith("@gmail.com") and not claims.get("hd"))
        or email not in {e.lower() for e in config.invited_emails}
    ):
        raise HTTPException(403, "Invited Google account required")
    return {"sub": claims["sub"], "email": email}


def user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Google login required")
    return authenticate(credentials.credentials, settings())
