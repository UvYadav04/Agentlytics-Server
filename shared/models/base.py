"""Common base for Mongo-backed documents.

We deliberately do not use a full ODM (beanie/mongoengine). Documents are
plain pydantic models; shared/db.py provides thin collection accessors and
each router does explicit motor calls. This keeps the Mongo <-> pydantic
boundary obvious and easy to debug, matching the "plain PyMongo" option
called out in the build plan.

Ids are plain strings (uuid4 hex), not ObjectId - this avoids ObjectId
JSON-serialization headaches in FastAPI responses and matches the id style
already used in the engine (InvestigationState.session_id).
"""
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoModel(BaseModel):
    """Base class for all documents stored in Mongo.

    `id` is stored as Mongo's `_id` field. Subclasses just declare their own
    fields; use `.to_mongo()` when writing and `MyModel(**doc)` when reading
    (motor returns `_id` as the raw field name, so `to_mongo()`/`from_mongo`
    handle the `_id` <-> `id` rename).
    """

    id: str = Field(default_factory=new_id)

    def to_mongo(self) -> dict:
        data = self.model_dump()
        data["_id"] = data.pop("id")
        return data

    @classmethod
    def from_mongo(cls, doc: dict | None):
        if doc is None:
            return None
        doc = dict(doc)
        doc["id"] = doc.pop("_id")
        return cls(**doc)
