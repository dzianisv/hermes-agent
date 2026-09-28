"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel at ``$HERMES_HOME/ESTOP``; ``hermes resume``
removes it. While it exists the cron scheduler, kanban dispatcher and new gateway
turns skip work; in-flight work is never killed. The check is one or two uncached
``os.stat`` calls (process home + fleet root when they differ). The body is optional
JSON ``{"reason", "engaged_at", "expires_at"}``; a corrupt/empty file still counts as engaged
(fail safe, e.g. ``touch ~/.hermes/ESTOP``). Only a well-formed body whose ``expires_at`` (or, without
one, ``engaged_at`` + ``estop.default_max_seconds``) is in the past lapses: the sentinel is removed
and a WARNING logged once. Ported from gastownhall/gastown estop.go (MIT).
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Same profile-aware / fleet-root resolvers the file-safety guards use (fail-open to ~/.hermes).
from agent.file_safety import _hermes_home_path as _hermes_home, _hermes_root_path as _canonical_root

SENTINEL_NAME = "ESTOP"

logger = logging.getLogger(__name__)

# Per-component "logged already for this engagement" flags: log once per engagement, not per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()
# Lapsed-sentinel bodies already warned about (in case the unlink itself fails).
_lapse_warned: set[tuple] = set()


def sentinel_path() -> Path:
    """Path of the ESTOP sentinel this process would write on `hermes pause`."""
    return _hermes_home() / SENTINEL_NAME


def _candidate_sentinel_paths() -> list:
    """Profile home first, then the fleet root if it is a different directory: a profile
    gateway (HERMES_HOME=~/.hermes/profiles/<n>) must still honor an operator's ~/.hermes/ESTOP."""
    primary = sentinel_path()
    try:
        root = _canonical_root() / SENTINEL_NAME
    except Exception:
        return [primary]
    try:
        distinct = root.resolve() != primary.resolve()
    except Exception:
        # Non-Path test doubles fail .resolve(); plain equality still dedupes.
        distinct = root != primary
    return [primary, root] if distinct else [primary]


def default_max_seconds() -> int:
    """``estop.default_max_seconds`` from the active home's config.yaml (0 = never lapse)."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    fallback = int(DEFAULT_CONFIG["estop"]["default_max_seconds"])
    try:
        from hermes_cli.config_effective import load_user_config_effective
        section = load_user_config_effective(_hermes_home() / "config.yaml").get("estop")
    except Exception:
        logger.debug("estop: could not read config.yaml; using default_max_seconds=%s", fallback, exc_info=True)
        return fallback
    raw = section.get("default_max_seconds", fallback) if isinstance(section, dict) else fallback
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning("estop.default_max_seconds=%r is not an integer; using %s", raw, fallback)
        return fallback


def _parse_ts(value) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _effective_expiry(body: dict) -> Optional[datetime]:
    """When a well-formed sentinel body lifts: explicit ``expires_at``, else ``engaged_at`` plus
    ``estop.default_max_seconds``; None = held until `hermes resume`."""
    explicit = _parse_ts(body.get("expires_at"))
    if explicit is not None:
        return explicit
    engaged_at = _parse_ts(body.get("engaged_at"))
    max_seconds = default_max_seconds() if engaged_at is not None else 0
    return engaged_at + timedelta(seconds=max_seconds) if max_seconds > 0 else None


def _read_body(path) -> tuple[Optional[str], Optional[dict]]:
    """``(text, body)``; body is None for an unreadable/corrupt/non-object sentinel."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, AttributeError, ValueError):
        return None, None
    try:
        body = json.loads(text)
    except ValueError:
        return text, None
    return text, body if isinstance(body, dict) else None


def _lapsed(path, text: Optional[str], body: Optional[dict]) -> bool:
    """True when *body* is well-formed and past its expiry; removes the sentinel (only if its bytes
    are unchanged, so a concurrent re-engage survives) and warns once."""
    if body is None:
        return False
    expiry = _effective_expiry(body)
    if expiry is None or expiry > datetime.now(timezone.utc):
        return False
    key = (str(path), text)
    with _log_lock:
        first = key not in _lapse_warned
        _lapse_warned.add(key)
    if first:
        logger.warning(
            "Global emergency stop lapsed (engaged_at=%s, expired %s, reason=%s) — new work resumes; "
            "removing %s", body.get("engaged_at"), expiry.isoformat(), body.get("reason"), path)
    if _read_body(path)[0] == text:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove lapsed ESTOP sentinel %s: %s", path, exc)
    return True


def is_engaged() -> bool:
    """True if ANY candidate sentinel exists and has not lapsed; fail SAFE (True) on stat errors
    and on unreadable/corrupt bodies."""
    saw_stat_error = False
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            saw_stat_error = True
            continue
        if not _lapsed(path, *_read_body(path)):
            return True
    return saw_stat_error


def engage(reason: Optional[str] = None, *, expires_in: Optional[float] = None) -> Path:
    """Create the ESTOP sentinel. Idempotent; re-engaging updates the file. ``expires_in`` seconds
    sets an explicit ``expires_at``; without it ``estop.default_max_seconds`` applies at read time."""
    path = sentinel_path()
    now = datetime.now(timezone.utc)
    payload = {"engaged_at": now.isoformat(), "reason": reason or None}
    if expires_in is not None:
        payload["expires_at"] = (now + timedelta(seconds=float(expires_in))).isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        with suppress(OSError):  # Best effort: an empty/partial sentinel still pauses (fail safe).
            path.touch(exist_ok=True)
    return path


def disengage() -> bool:
    """Remove every visible sentinel (process-local and fleet-root)."""
    lifted = False
    for path in _candidate_sentinel_paths():
        try:
            path.unlink()
            lifted = True
        except (OSError, AttributeError):
            continue
    return lifted


def get_state() -> Optional[dict]:
    """Return ``{"reason", "engaged_at", "expires_at"}`` or None when not engaged. ``expires_at`` is
    the effective lift time (ISO, UTC) or None when the pause holds until resumed; an
    unreadable/corrupt body still reports engaged with every field None."""
    if not is_engaged():
        return None
    blank = {"reason": None, "engaged_at": None, "expires_at": None}
    live: list[dict] = []
    held_forever = False
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            return dict(blank)
        except AttributeError:
            continue
        text, body = _read_body(path)
        if body is None:
            held_forever = True
            live.append(dict(blank))
            continue
        if _lapsed(path, text, body):
            continue
        expiry = _effective_expiry(body)
        held_forever = held_forever or expiry is None
        live.append({"reason": body.get("reason") or None, "engaged_at": body.get("engaged_at") or None,
                     "expires_at": expiry.astimezone(timezone.utc).isoformat() if expiry else None})
    if not live:
        return None
    state = next((s for s in live if s["engaged_at"] or s["reason"]), live[0])
    # The pause lifts when the LAST live sentinel does.
    state["expires_at"] = None if held_forever else max(s["expires_at"] for s in live)
    return state


def remaining_seconds(state: Optional[dict]) -> Optional[int]:
    """Seconds until *state* lifts on its own, or None when it holds until `hermes resume`."""
    expiry = _parse_ts((state or {}).get("expires_at"))
    if expiry is None:
        return None
    return max(0, int((expiry - datetime.now(timezone.utc)).total_seconds()))


def format_remaining(seconds: int) -> str:
    """``5400`` → ``1h 30m``; under a minute → ``<1m``."""
    hours, minutes = divmod(int(seconds) // 60, 60)
    if not hours and not minutes:
        return "<1m"
    return " ".join(part for part in (f"{hours}h" if hours else "", f"{minutes}m" if minutes else "") if part)


def lifts_phrase(state: Optional[dict]) -> Optional[str]:
    """``"lifts automatically at 18:00 UTC (in 2h 5m)"`` or None when not time-bound."""
    remaining = remaining_seconds(state)
    if remaining is None:
        return None
    expiry = _parse_ts(state["expires_at"]).astimezone(timezone.utc)
    return f"lifts automatically at {expiry:%Y-%m-%d %H:%M} UTC (in {format_remaining(remaining)})"


def paused_reply(state: Optional[dict] = None) -> Optional[str]:
    """Short user-facing notice for new gateway turns, or None if not paused. Pass a *state* already
    fetched with :func:`get_state` to avoid a second read."""
    if state is None:
        state = get_state()
    if state is None:
        return None
    tag = f" ({state['reason']})" if state.get("reason") else ""
    lifts = lifts_phrase(state)
    if lifts:
        return f"⏸️ Hermes is paused{tag} — {lifts}. New work is on hold; run `hermes resume` to lift it sooner."
    return f"⏸️ Hermes is paused{tag}. New work is on hold; run `hermes resume` to pick things back up."


def check_paused(component: str, logger: logging.Logger) -> bool:
    """Return True when engaged, logging once per engagement per component (re-armed after a resume)."""
    if not is_engaged():
        with _log_lock:
            _logged_components.discard(component)
        return False
    with _log_lock:
        first = component not in _logged_components
        _logged_components.add(component)
    if first:
        reason = (get_state() or {}).get("reason")
        suffix = f" (reason: {reason})" if reason else ""
        logger.info(
            "%s dispatch paused by global emergency stop%s — remove with `hermes resume` (%s)",
            component, suffix, sentinel_path(),
        )
    return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
