"""JWT issuing/verification + Google ID token verification + email/password credential
handling, shared so both api_service (issues/verifies) and any future service can decode the
same token / hash passwords the same way without duplicating the secret-handling logic.
"""
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

# bcrypt truncates silently past 72 bytes - reject anything longer up front (in
# MIN/MAX_PASSWORD_LENGTH-consuming call sites, see api_service/routers/auth.py) rather than let
# two different long passwords that share the same first 72 bytes hash identically.
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
    """Raises jwt.PyJWTError (expired/invalid signature/etc.) on failure -
    callers turn that into a 401, they don't need to inspect the exception."""
    data = jwt.decode(token, _jwt_secret(), algorithms=[_jwt_algorithm()])
    return TokenPayload(**data)


def verify_google_id_token(token: str) -> GoogleProfile:
    """Verifies a Google-issued ID token (from the frontend's client-side
    Google Sign-In flow) against Google's public keys + our client ID.
    Raises ValueError (via google-auth) on an invalid/expired/wrong-audience
    token - callers turn that into a 401.

    clock_skew_in_seconds=10: google-auth's verify_oauth2_token() defaults this to 0 - no
    tolerance at all for the verifying machine's clock running behind Google's. In a
    container (or a Docker Desktop/WSL2 VM, whose clock is a common source of drift after
    the host sleeps/resumes) even a few seconds of skew then hard-fails every login with
    "Token used too early" (the `iat` claim, timestamped by Google's clock, looks like it's
    in the future relative to ours). 10s matches google-auth's own long-standing internal
    default skew elsewhere (_CLOCK_SKEW_SECS) and is what this parameter exists for -
    this doesn't fix a genuinely wrong clock, it just stops small/transient drift from
    taking login down. If this error keeps recurring, check the host/VM's clock sync
    (`wsl --shutdown` + restart Docker Desktop resyncs the WSL2 VM's clock on Windows)."""
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


# ---------------------------------------------------------------- email/password credentials

def hash_password(password: str) -> str:
    """bcrypt with a fresh random salt per call (gensalt()'s default cost factor, 12) - the
    resulting hash string carries its own salt/cost, so verify_password below needs nothing
    else to check against it later."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """False (never raises) for a malformed/foreign hash - a User whose password_hash somehow
    isn't a valid bcrypt hash should fail a login attempt, not 500 the request."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def generate_email_token() -> str:
    """URL-safe, unguessable token for the email-verification link - see
    User.email_verification_token."""
    return secrets.token_urlsafe(32)


def generate_temporary_password() -> str:
    """Random password emailed by POST /auth/forgot-password - long/random enough to be safe to
    send in plaintext over email for the short time until the user changes it, comfortably over
    the signup form's own minimum length."""
    return secrets.token_urlsafe(12)
