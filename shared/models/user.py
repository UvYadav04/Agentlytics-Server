from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "users"


class User(MongoModel):
    email: str
    name: str
    picture: Optional[str] = None
    # Google Sign-In identity - None for an account created via email/password signup that has
    # never linked Google. Optional (used to be required) so both auth methods can coexist on
    # the same User doc/collection - see shared/db.py's ensure_indexes() for why the unique
    # index on this field has to be partial once it's nullable.
    google_id: Optional[str] = None
    # bcrypt hash (see shared/auth.py's hash_password/verify_password) - None for a Google-only
    # account that has never set a password (e.g. via POST /auth/forgot-password, which doubles
    # as "add a password to my Google account").
    password_hash: Optional[str] = None
    # True once the owner has clicked a verification link for `email` (see POST
    # /auth/verify-email), or unconditionally for a Google account - Google itself already
    # verified the email before handing us the ID token (see google_login). Login for a
    # password account is blocked until this is True.
    email_verified: bool = False
    # Proves ownership of `email` for a not-yet-verified password account. Deliberately NOT
    # cleared once used (see /auth/verify-email) - re-visiting the same link after it's already
    # verified the account looks the token up and returns "already verified" instead of a
    # confusing "invalid token" error.
    email_verification_token: Optional[str] = None
    email_verification_token_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
