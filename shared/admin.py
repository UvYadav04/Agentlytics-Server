from shared.config import get_settings

_DEFAULT_ADMIN_EMAILS = "dineshnirban01@gmail.com"


def _admin_emails() -> set[str]:
    raw = get_settings().get("ADMIN_EMAILS", "") or _DEFAULT_ADMIN_EMAILS
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin_email(email: str | None) -> bool:
    if not email:
        return False
    return email.strip().lower() in _admin_emails()
