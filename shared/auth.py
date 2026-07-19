"""JWT issuing/verification + Google ID token verification, shared so both
api_service (issues/verifies) and any future service can decode the same
token without duplicating the secret-handling logic.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

from shared.config import get_settings

ACCESS_TOKEN_COOKIE_NAME = "access_token"


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
    """Raises jwt.PyJWTError (expired/invalid signature/etc.) on failure -
    callers turn that into a 401, they don't need to inspect the exception."""
    data = jwt.decode(token, _jwt_secret(), algorithms=[_jwt_algorithm()])
    return TokenPayload(**data)


def verify_google_id_token(token: str) -> GoogleProfile:
    """Verifies a Google-issued ID token (from the frontend's client-side
    Google Sign-In flow) against Google's public keys + our client ID.
    Raises ValueError (via google-auth) on an invalid/expired/wrong-audience
    token - callers turn that into a 401."""
    client_id = get_settings().get("GOOGLE_CLIENT_ID")
    print(client_id)
    idinfo = google_id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
    print(idinfo)
    return GoogleProfile(
        google_id=idinfo["sub"],
        email=idinfo["email"],
        name=idinfo.get("name", idinfo["email"]),
        picture=idinfo.get("picture"),
    )
