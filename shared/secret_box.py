import base64
import hashlib
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from shared.config import get_settings

logger = logging.getLogger("shared.secret_box")

KEY_SETTING = "DB_CONNECTION_ENCRYPTION_KEY"


class SecretBoxError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    settings = get_settings()
    configured = (settings.get(KEY_SETTING) or "").strip()
    if configured:
        try:
            return Fernet(configured.encode())
        except (ValueError, TypeError) as exc:
            raise SecretBoxError(f"{KEY_SETTING} is not a valid Fernet key") from exc

    fallback_seed = (settings.get("JWT_SECRET") or "").strip()
    if not fallback_seed:
        raise SecretBoxError(f"{KEY_SETTING} is not configured")
    logger.warning(
        "%s is not set - deriving the connection-string encryption key from JWT_SECRET. "
        "Set a dedicated key in production.", KEY_SETTING,
    )
    derived = hashlib.sha256(f"database-connections:{fallback_seed}".encode()).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise SecretBoxError("stored secret could not be decrypted") from exc
