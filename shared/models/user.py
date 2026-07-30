from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "users"


class User(MongoModel):
    email: str
    name: str
    picture: Optional[str] = None
    google_id: Optional[str] = None
    password_hash: Optional[str] = None
    email_verified: bool = False
    email_verification_token: Optional[str] = None
    email_verification_token_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
