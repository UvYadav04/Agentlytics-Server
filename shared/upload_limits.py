from shared.config import get_settings

DEFAULT_MAX_DOCUMENT_MB = 25

_LIMITED_TYPES = {"pdf", "txt"}

# DOCX/DOC uploads aren't supported - there's no ingestor for them under
# analyzerEngine/ingestion/file_types/ (PDF and TXT both have one and are accepted below).
UNSUPPORTED_UPLOAD_EXTENSIONS = {"docx", "doc"}


def is_supported_upload_extension(ext: str) -> bool:
    return ext.lower() not in UNSUPPORTED_UPLOAD_EXTENSIONS


def max_size_bytes(file_type: str) -> int | None:
    """Byte cap for this file_type, or None if it isn't a size-limited type."""
    if file_type not in _LIMITED_TYPES:
        return None
    settings = get_settings()
    mb = float(settings.get("MAX_DOCUMENT_UPLOAD_MB", DEFAULT_MAX_DOCUMENT_MB) or DEFAULT_MAX_DOCUMENT_MB)
    return int(mb * 1024 * 1024)


def describe_limit(file_type: str) -> str:
    limit = max_size_bytes(file_type)
    mb = (limit or 0) / (1024 * 1024)
    return f"{file_type.upper()} files are limited to {mb:.0f}MB"
