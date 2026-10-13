"""Connectivity-aware waiting for dispatcher-owned kanban workers (harness #32).

A laptop that drops Wi-Fi turns every provider call into ``APIConnectionError`` / DNS failure.
The generic retry stack (3 attempts + a 5-cycle ~4 min ladder) is sized for a provider blip, not
for the host being offline, so a worker used to give up mid-turn (rc=75) or burn its stop-loop
nudges against a dead model and exit rc=0 with no board call. Claude Code / Codex CLI treat
connection errors as retryable with exponential backoff and keep the session; this module does
the same for kanban workers: probe connectivity with bounded backoff, resume the SAME turn when
the network is back, and report ``False`` (caller records an explicit ``network_unavailable``
block) when the budget is spent.

Knobs: ``HERMES_KANBAN_NETWORK_WAIT_SECONDS`` (total budget, default 1800, 0 disables) and
``HERMES_NETWORK_PROBE_HOSTS`` (``host:port,...``).
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("agent.conversation_loop")

NETWORK_UNAVAILABLE = "network_unavailable"

_DEFAULT_BUDGET_S = 1800.0
_BASE_DELAY_S = 5.0
_CAP_DELAY_S = 60.0
_DEFAULT_PROBES = "api.githubcopilot.com:443,api.github.com:443,1.1.1.1:53"

# Error text / types that mean "this host cannot reach the network", not "the provider is slow".
_NETWORK_PATTERNS = (
    "can't reach the model provider", "you may be offline", "network is unreachable",
    "network unreachable", "nodename nor servname provided", "name or service not known",
    "temporary failure in name resolution", "getaddrinfo", "no route to host",
    "connection refused", "econnrefused", "eai_again", "enotfound", "connection error",
    "fetch failed", "failed to fetch", "internet connection",
)
_NETWORK_TYPES = frozenset({"APIConnectionError", "ConnectError", "ConnectionError", "gaierror"})


def network_wait_budget() -> float:
    raw = (os.environ.get("HERMES_KANBAN_NETWORK_WAIT_SECONDS") or "").strip()
    try:
        return max(float(raw), 0.0) if raw else _DEFAULT_BUDGET_S
    except ValueError:
        return _DEFAULT_BUDGET_S


def is_kanban_worker() -> bool:
    try:
        from agent.delegation_context import owned_kanban_task
        return bool(owned_kanban_task())
    except Exception:
        return False


def is_network_error(error: Any) -> bool:
    """True for connect/DNS/offline failures (not read timeouts on a live connection)."""
    if error is None:
        return False
    if type(error).__name__ in _NETWORK_TYPES or isinstance(error, socket.gaierror):
        return True
    text = str(error).lower()
    cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    if cause is not None and cause is not error:
        text += " " + str(cause).lower()
        if type(cause).__name__ in _NETWORK_TYPES or isinstance(cause, socket.gaierror):
            return True
    return any(p in text for p in _NETWORK_PATTERNS)


def _probe_targets() -> list[tuple[str, int]]:
    out = []
    for item in (os.environ.get("HERMES_NETWORK_PROBE_HOSTS") or _DEFAULT_PROBES).split(","):
        host, _, port = item.strip().rpartition(":")
        if host and port.isdigit():
            out.append((host, int(port)))
    return out


def network_available(timeout: float = 3.0, targets: Optional[Iterable[tuple[str, int]]] = None) -> bool:
    """One TCP connect to any probe target succeeds (DNS included)."""
    for host, port in (targets if targets is not None else _probe_targets()):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_for_network(
    *, budget_s: Optional[float] = None, probe: Optional[Callable[[], bool]] = None,
    sleep: Optional[Callable[[float], None]] = None, should_abort: Callable[[], bool] = lambda: False,
    on_wait: Callable[[str], None] = lambda s: None, clock: Optional[Callable[[], float]] = None,
) -> bool:
    """Block until ``probe()`` is True (→ True) or ``budget_s`` elapses / abort (→ False).
    Exponential backoff 5,10,20,40,60,60… s. Never raises."""
    probe = probe or network_available
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    budget = network_wait_budget() if budget_s is None else budget_s
    start = clock()
    delay = _BASE_DELAY_S
    attempt = 0
    while True:
        try:
            if probe():
                if attempt:
                    logger.info("network back after %.0fs offline wait (%d probes)", clock() - start, attempt)
                return True
        except Exception:
            logger.debug("network probe failed", exc_info=True)
        elapsed = clock() - start
        if elapsed >= budget or should_abort():
            logger.warning("network still unavailable after %.0fs — giving up", elapsed)
            return False
        attempt += 1
        wait = min(delay, _CAP_DELAY_S, max(budget - elapsed, 0.0))
        on_wait(f"🌐 Network unavailable — waiting {wait:.0f}s for connectivity "
                f"({elapsed:.0f}/{budget:.0f}s budget)")
        sleep(wait)
        delay = min(delay * 2, _CAP_DELAY_S)


__all__ = [
    "NETWORK_UNAVAILABLE", "is_kanban_worker", "is_network_error", "network_available",
    "network_wait_budget", "wait_for_network",
]
