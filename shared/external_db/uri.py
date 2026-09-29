import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qsl, unquote

from shared.external_db.errors import ConnectionStringError

MAX_CONNECTION_STRING_LENGTH = 4096

SCHEMES = {
    "mongodb": ("mongodb", "mongodb+srv"),
    "postgresql": ("postgresql", "postgres"),
}

LABELS = {"mongodb": "MongoDB", "postgresql": "PostgreSQL"}

DEFAULT_PORTS = {"mongodb": 27017, "postgresql": 5432}

_PLACEHOLDER = re.compile(r"<[^<>/@:]+>")

_ALLOWED_MONGO_AUTH = {"", "scram-sha-1", "scram-sha-256", "default"}

_BLOCKED_MONGO_PARAMS = {"tlscafile", "tlscertificatekeyfile", "tlscertificatekeyfilepassword", "tlscrlfile", "authmechanismproperties"}

_BLOCKED_POSTGRES_PARAMS = {
    "passfile", "service", "servicefile", "sslcert", "sslkey", "sslrootcert", "sslcrl", "sslcrldir",
    "sslpassword", "krbsrvname", "gsslib", "hostaddr", "host",
}


@dataclass
class ParsedConnection:
    db_type: str
    scheme: str
    username: Optional[str]
    password: Optional[str]
    hosts: list[tuple[str, int]] = field(default_factory=list)
    database: Optional[str] = None
    display_uri: str = ""
    host_label: str = ""

    @property
    def is_srv(self) -> bool:
        return self.scheme == "mongodb+srv"


def _split_host(entry: str, default_port: int) -> tuple[str, int]:
    entry = entry.strip()
    if not entry:
        raise ConnectionStringError("The connection string has an empty host.")
    if entry.startswith("["):
        end = entry.find("]")
        if end == -1:
            raise ConnectionStringError("The connection string has an invalid IPv6 host.")
        host = entry[1:end]
        rest = entry[end + 1:]
        port_text = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port_text = entry.partition(":")
    host = unquote(host).strip().lower()
    if not host or host.startswith("/"):
        raise ConnectionStringError("The connection string must include a network host name.")
    if port_text:
        if not port_text.isdigit() or not 0 < int(port_text) < 65536:
            raise ConnectionStringError("The connection string has an invalid port.")
        return host, int(port_text)
    return host, default_port


def _display_host(host: str, port: int, default_port: int, is_srv: bool) -> str:
    shown = f"[{host}]" if ":" in host else host
    if is_srv or port == default_port:
        return shown
    return f"{shown}:{port}"


def _check_params(db_type: str, query: str) -> None:
    params = {key.lower(): value for key, value in parse_qsl(query, keep_blank_values=True)}
    if db_type == "mongodb":
        blocked = sorted(set(params) & _BLOCKED_MONGO_PARAMS)
        if blocked:
            raise ConnectionStringError(f"Unsupported connection option: {blocked[0]}.")
        mechanism = params.get("authmechanism", "").strip().lower()
        if mechanism not in _ALLOWED_MONGO_AUTH:
            raise ConnectionStringError("Only username/password (SCRAM) authentication is supported.")
    else:
        blocked = sorted(set(params) & _BLOCKED_POSTGRES_PARAMS)
        if blocked:
            raise ConnectionStringError(f"Unsupported connection option: {blocked[0]}.")


def parse_connection_string(db_type: str, raw: str) -> ParsedConnection:
    if db_type not in SCHEMES:
        raise ConnectionStringError("Unsupported database type.")
    label = LABELS[db_type]
    value = (raw or "").strip()
    if not value:
        raise ConnectionStringError(f"Paste your {label} connection string.")
    if len(value) > MAX_CONNECTION_STRING_LENGTH:
        raise ConnectionStringError("The connection string is too long.")
    if any(ch.isspace() for ch in value):
        raise ConnectionStringError("The connection string must not contain spaces or line breaks.")
    if _PLACEHOLDER.search(value):
        raise ConnectionStringError("Replace the <placeholders> (like <password>) with your real values.")

    scheme, sep, rest = value.partition("://")
    scheme = scheme.lower()
    if not sep or scheme not in SCHEMES[db_type]:
        expected = " or ".join(f"{s}://" for s in SCHEMES[db_type])
        raise ConnectionStringError(f"A {label} connection string must start with {expected}")

    split_at = len(rest)
    for marker in ("/", "?"):
        index = rest.find(marker)
        if index != -1:
            split_at = min(split_at, index)
    authority, remainder = rest[:split_at], rest[split_at:]

    userinfo, at, hostpart = authority.rpartition("@")
    if not at:
        userinfo, hostpart = "", authority
    raw_user, has_password, raw_password = userinfo.partition(":")
    username = unquote(raw_user) or None
    password = unquote(raw_password) if has_password else None
    if not username or not password:
        raise ConnectionStringError(
            "Include the read-only user's username and password in the connection string."
        )

    path, _, query = remainder.partition("?")
    database = unquote(path.lstrip("/")) or None
    if database and "/" in database:
        raise ConnectionStringError("The database name in the connection string is invalid.")
    _check_params(db_type, query)

    default_port = DEFAULT_PORTS[db_type]
    hosts = [_split_host(entry, default_port) for entry in hostpart.split(",")]
    is_srv = scheme == "mongodb+srv"
    if is_srv and (len(hosts) != 1 or ":" in hostpart):
        raise ConnectionStringError("A mongodb+srv:// connection string must have exactly one host and no port.")
    if db_type == "postgresql" and len(hosts) != 1:
        raise ConnectionStringError("Use a PostgreSQL connection string with a single host.")

    shown_hosts = ",".join(_display_host(h, p, default_port, is_srv) for h, p in hosts)
    display = f"{scheme}://{raw_user}:****@{shown_hosts}"
    if database:
        display += f"/{database}"
    first_host = _display_host(hosts[0][0], hosts[0][1], default_port, is_srv)
    host_label = first_host if len(hosts) == 1 else f"{first_host} (+{len(hosts) - 1} more)"

    return ParsedConnection(
        db_type=db_type, scheme=scheme, username=username, password=password, hosts=hosts,
        database=database, display_uri=display, host_label=host_label,
    )


def scrub_secrets(message: str, raw_connection_string: str, parsed: Optional[ParsedConnection]) -> str:
    text = str(message or "")
    candidates = [raw_connection_string.strip()] if raw_connection_string else []
    if parsed and parsed.password:
        candidates.append(parsed.password)
        candidates.append(parsed.password.replace("@", "%40"))
    for secret in sorted({c for c in candidates if c}, key=len, reverse=True):
        text = text.replace(secret, "****")
    text = re.sub(r"(mongodb(?:\+srv)?|postgres(?:ql)?)://[^\s'\"]+", r"\1://****", text)
    text = re.sub(r"password=\S+", "password=****", text, flags=re.IGNORECASE)
    return text.strip()[:300]
