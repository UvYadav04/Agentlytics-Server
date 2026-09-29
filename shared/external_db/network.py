import ipaddress
import socket

import dns.exception
import dns.resolver

from shared.config import get_settings
from shared.external_db.errors import ConnectionStringError
from shared.external_db.uri import ParsedConnection

_DNS_TIMEOUT_SECONDS = 5
_MAX_SRV_TARGETS = 20


def _private_hosts_allowed() -> bool:
    return str(get_settings().get("ALLOW_PRIVATE_DATABASE_HOSTS", "")).strip().lower() in ("1", "true", "yes")


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _resolve(host: str, port: int) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ConnectionStringError(f"Couldn't resolve the host '{host}'. Check the address in the connection string.")
    return {info[4][0] for info in infos}


def _srv_targets(host: str) -> list[tuple[str, int]]:
    try:
        answers = dns.resolver.resolve(f"_mongodb._tcp.{host}", "SRV", lifetime=_DNS_TIMEOUT_SECONDS)
    except (dns.exception.DNSException, OSError):
        raise ConnectionStringError(f"Couldn't find the MongoDB cluster '{host}'. Check the cluster address.")
    parent = host.split(".", 1)[1] if host.count(".") >= 2 else host
    targets = []
    for record in list(answers)[:_MAX_SRV_TARGETS]:
        target = str(record.target).rstrip(".").lower()
        if not target.endswith(f".{parent}") and target != parent:
            raise ConnectionStringError("The cluster's DNS records point outside its own domain.")
        targets.append((target, int(record.port)))
    if not targets:
        raise ConnectionStringError(f"Couldn't find the MongoDB cluster '{host}'. Check the cluster address.")
    return targets


def ensure_public_hosts(parsed: ParsedConnection) -> None:
    if _private_hosts_allowed():
        return
    endpoints = _srv_targets(parsed.hosts[0][0]) if parsed.is_srv else parsed.hosts
    for host, port in endpoints:
        for address in _resolve(host, port):
            if not _is_public(address):
                raise ConnectionStringError(
                    "This connection string points to a private or local network address, which "
                    "isn't allowed. Use your database's public host name."
                )
