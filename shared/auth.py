import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

from shared.config import get_settings

ACCESS_TOKEN_COOKIE_NAME = "access_token"

MAX_PASSWORD_BYTES = 72


class TokenPayload(BaseModel):
    sub: str  # user id
    email: str
    exp: int


class GoogleProfile(BaseModel):
    google_id: str
    email: str
    name: str
    picture: Optional[str] = None


def _jwt_secret() -> str:
    return get_settings().get("JWT_SECRET", "dev-secret-change-me")


def _jwt_algorithm() -> str:
    return get_settings().get("JWT_ALGORITHM", "HS256")


def _jwt_expire_minutes() -> int:
    return int(get_settings().get("JWT_EXPIRE_MINUTES", "43200") or "43200")


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_jwt_expire_minutes())
    payload = {"sub": user_id, "email": email, "exp": int(expire.timestamp())}
    return jwt.encode(payload, _jwt_secret(), algorithm=_jwt_algorithm())


def decode_access_token(token: str) -> TokenPayload:
    data = jwt.decode(token, _jwt_secret(), algorithms=[_jwt_algorithm()])
    return TokenPayload(**data)


def verify_google_id_token(token: str) -> GoogleProfile:
    client_id = get_settings().get("GOOGLE_CLIENT_ID")
    idinfo = google_id_token.verify_oauth2_token(
        token, google_requests.Request(), client_id, clock_skew_in_seconds=10,
    )
    return GoogleProfile(
        google_id=idinfo["sub"],
        email=idinfo["email"],
        name=idinfo.get("name", idinfo["email"]),
        picture=idinfo.get("picture"),
    )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def generate_email_token() -> str:
    return secrets.token_urlsafe(32)


def generate_temporary_password() -> str:
    return secrets.token_urlsafe(12)
