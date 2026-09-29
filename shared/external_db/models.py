from dataclasses import dataclass, field
from typing import Optional

MAX_OBJECTS = 200
MAX_FIELDS_PER_OBJECT = 150


@dataclass
class InspectionResult:
    database: Optional[str]
    server_version: Optional[str]
    objects: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False

    def schema_document(self) -> dict:
        return {
            "database": self.database,
            "server_version": self.server_version,
            "objects": self.objects,
            "truncated": self.truncated,
        }
