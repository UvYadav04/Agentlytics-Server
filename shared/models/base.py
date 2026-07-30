import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoModel(BaseModel):


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
