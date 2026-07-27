import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, field_validator, model_validator
from pymongo.errors import DuplicateKeyError

from api_service.deps import get_current_user
from shared.auth import (
    ACCESS_TOKEN_COOKIE_NAME,
    MAX_PASSWORD_BYTES,
    create_access_token,
    generate_email_token,
    generate_temporary_password,
    hash_password,
    verify_google_id_token,
    verify_password,
)
from shared.config import get_settings
from shared.db import get_db
from shared.email import send_password_changed_email, send_temporary_password_email, send_verification_email
from shared.models.user import COLLECTION as USERS
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.models.workspace import Workspace

logger = logging.getLogger("api.auth")

router = APIRouter(tags=["auth"])

MIN_PASSWORD_LENGTH = 8
# Deliberately generic - never reveal which half was wrong, or whether the email exists at all.
INVALID_CREDENTIALS_MESSAGE = "Invalid email or password"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _validate_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError("Password is too long")
    return password


class GoogleLoginRequest(BaseModel):
    id_token: str


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    confirm_password: str

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str) -> str:
        v = _normalize_email(v)
        if not _EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def _password_rules(cls, v: str) -> str:
        return _validate_password(v)

    @model_validator(mode="after")
    def _passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    email: str
    password: str


class ResendVerificationRequest(BaseModel):
    email: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ChangePasswordRequest(BaseModel):
    # Omit current_password when the account has no password yet (a Google-only account
    # setting one for the first time) - see change_password below, which checks that against
    # the real user record rather than trusting a flag in the request body.
    current_password: Optional[str] = None
    new_password: str
    confirm_new_password: str

    @field_validator("new_password")
    @classmethod
    def _password_rules(cls, v: str) -> str:
        return _validate_password(v)

    @model_validator(mode="after")
    def _passwords_match(self):
        if self.new_password != self.confirm_new_password:
            raise ValueError("Passwords do not match")
        return self


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    picture: str | None = None
    email_verified: bool
    has_password: bool


class MessageOut(BaseModel):
    message: str


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id, email=user.email, name=user.name, picture=user.picture,
        email_verified=user.email_verified, has_password=user.password_hash is not None,
    )


def _cookie_max_age() -> int:
    minutes = int(get_settings().get("JWT_EXPIRE_MINUTES", "43200") or "43200")
    return minutes * 60


def _set_auth_cookie(response: Response, token: str) -> None:
    # secure=True in production (HTTPS) - set via env-driven flag if you also
    # deploy dev over http. Left True here since prod is the deploy target
    # in Phase 11; flip to settings-driven if you need local http testing.
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_cookie_max_age(),
        path="/",
    )


async def _create_default_workspace(user_id: str) -> None:
    workspace = Workspace(user_id=user_id, name="Workspace1")
    await get_db()[WORKSPACES].insert_one(workspace.to_mongo())


def _verification_token_expiry() -> datetime:
    hours = int(get_settings().get("EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS", "24") or "24")
    return datetime.now(timezone.utc) + timedelta(hours=hours)


@router.post("/auth/google", response_model=UserOut)
async def google_login(body: GoogleLoginRequest, response: Response):
    try:
        profile = verify_google_id_token(body.id_token)
    except Exception as exc:
        # The client only ever sees the generic 401 below (don't leak verifier
        # internals) - but log the real reason, since "invalid token" collapses
        # a handful of very different root causes (wrong/missing
        # GOOGLE_CLIENT_ID, expired token, clock skew, wrong token type) into
        # one message that's impossible to debug blind.
        logger.warning("Google ID token verification failed: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google token")

    db = get_db()
    existing = await db[USERS].find_one({"google_id": profile.google_id})

    if existing is None:
        # A password account may already own this email (signed up manually, then hits "Log in
        # with Google" later) - link Google onto that SAME User doc instead of failing on the
        # email unique index / silently creating a confusing second account for one person.
        existing_by_email = await db[USERS].find_one({"email": profile.email})
        if existing_by_email is not None:
            user = User.from_mongo(existing_by_email)
            user.google_id = profile.google_id
            user.picture = user.picture or profile.picture
            # Google already verified this email before issuing the ID token we just checked -
            # trust that even if the password-signup verification link was never clicked.
            user.email_verified = True
            await db[USERS].update_one(
                {"_id": user.id},
                {"$set": {"google_id": user.google_id, "picture": user.picture, "email_verified": True}},
            )
        else:
            user = User(
                google_id=profile.google_id, email=profile.email, name=profile.name,
                picture=profile.picture, email_verified=True,
            )
            await db[USERS].insert_one(user.to_mongo())
            await _create_default_workspace(user.id)  # first login: create the default workspace
    else:
        user = User.from_mongo(existing)

    token = create_access_token(user.id, user.email)
    _set_auth_cookie(response, token)
    return _user_out(user)


@router.post("/auth/signup", response_model=MessageOut)
async def signup(body: SignupRequest):
    """Creates the account but does NOT log the user in - see /auth/verify-email. Returns a
    generic confirmation regardless of outcome details beyond the one real failure case (email
    already registered) so the response never has to distinguish "email sent" from "email
    delivery failed" (send_verification_email never raises - see shared/email.py)."""
    db = get_db()

    user = User(
        email=body.email, name=body.name, password_hash=hash_password(body.password),
        email_verified=False, email_verification_token=generate_email_token(),
        email_verification_token_expires_at=_verification_token_expiry(),
    )
    try:
        await db[USERS].insert_one(user.to_mongo())
    except DuplicateKeyError:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    await _create_default_workspace(user.id)
    await send_verification_email(user.email, user.name, user.email_verification_token)
    return MessageOut(message="Account created - check your email to verify your address before logging in.")


@router.get("/auth/verify-email", response_model=MessageOut)
async def verify_email(token: str):
    """Idempotent by design: the token is never cleared once used (see User.
    email_verification_token's docstring), so clicking the same link twice - or a user re-
    opening an old email - looks the token up either way and just reports which state it's in,
    rather than the second click erroring with "invalid token"."""
    db = get_db()
    doc = await db[USERS].find_one({"email_verification_token": token})
    if doc is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid verification link")

    user = User.from_mongo(doc)
    if user.email_verified:
        return MessageOut(message="Your email is already verified - you can log in.")

    if user.email_verification_token_expires_at and user.email_verification_token_expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This verification link has expired - request a new one from the login page.",
        )

    await db[USERS].update_one({"_id": user.id}, {"$set": {"email_verified": True}})
    return MessageOut(message="Email verified - you can log in now.")


@router.post("/auth/resend-verification", response_model=MessageOut)
async def resend_verification(body: ResendVerificationRequest):
    # Same generic response whether or not the account exists/is already verified - avoids
    # leaking which emails are registered.
    generic = MessageOut(message="If that account exists and isn't verified yet, we've sent a new link.")
    db = get_db()
    doc = await db[USERS].find_one({"email": _normalize_email(body.email)})
    if doc is None:
        return generic

    user = User.from_mongo(doc)
    if user.email_verified or user.password_hash is None:
        return generic

    token = generate_email_token()
    await db[USERS].update_one(
        {"_id": user.id},
        {"$set": {"email_verification_token": token, "email_verification_token_expires_at": _verification_token_expiry()}},
    )
    await send_verification_email(user.email, user.name, token)
    return generic


@router.post("/auth/login", response_model=UserOut)
async def login(body: LoginRequest, response: Response):
    db = get_db()
    doc = await db[USERS].find_one({"email": _normalize_email(body.email)})
    user = User.from_mongo(doc) if doc else None

    if user is None or user.password_hash is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS_MESSAGE)

    if not user.email_verified:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Please verify your email before logging in - check your inbox, or request a new link.",
        )

    token = create_access_token(user.id, user.email)
    _set_auth_cookie(response, token)
    return _user_out(user)


@router.post("/auth/forgot-password", response_model=MessageOut)
async def forgot_password(body: ForgotPasswordRequest):
    # Same generic response regardless of whether the account exists - avoids leaking which
    # emails are registered. The actual rotation only happens when a real account is found.
    generic = MessageOut(message="If that email is registered, we've sent a new password to it.")
    db = get_db()
    doc = await db[USERS].find_one({"email": _normalize_email(body.email)})
    if doc is None:
        return generic

    user = User.from_mongo(doc)
    temporary_password = generate_temporary_password()
    await db[USERS].update_one(
        {"_id": user.id},
        # Receiving this email proves ownership of the address either way (Google-linked or
        # not) - mark it verified too, so a Google-only account that just gained a password
        # this way isn't immediately blocked from using it by the /auth/login check above.
        {"$set": {"password_hash": hash_password(temporary_password), "email_verified": True}},
    )
    await send_temporary_password_email(user.email, user.name, temporary_password)
    return generic


@router.post("/auth/change-password", response_model=MessageOut)
async def change_password(body: ChangePasswordRequest, user: User = Depends(get_current_user)):
    if user.password_hash is not None:
        if not body.current_password or not verify_password(body.current_password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Current password is incorrect")
    # else: Google-only account setting a password for the first time - nothing to check
    # against yet.

    await get_db()[USERS].update_one(
        {"_id": user.id}, {"$set": {"password_hash": hash_password(body.new_password)}},
    )
    await send_password_changed_email(user.email, user.name)
    return MessageOut(message="Password updated.")


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return _user_out(user)


@router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME, path="/")
    return {"ok": True}
