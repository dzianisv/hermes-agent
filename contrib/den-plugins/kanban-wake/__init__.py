"""kanban-wake — wake the supervisor's *current chat session* when a kanban card
blocks or completes.

Two halves, one plugin (both gateways load it):

1. Emitter (runs wherever kanban lifecycle hooks fire — dispatcher/worker
   gateway): `kanban_task_blocked` / `kanban_task_completed` -> HMAC-signed POST
   to the default gateway's webhook route `kanban-wake`.

2. Redirector (default gateway): `pre_gateway_dispatch` intercepts the webhook turn and
   delivers it ONLY to the conversation the card was created in, read from the card's
   `kanban_notify_subs` rows (written at creation: the chat session, inherited from the
   creating card, or the creating cron job's deliver target). Wake-mode origins are
   already woken by core's kanban notifier, so the redirector skips them. A card with no
   stored origin (plain CLI) gets nothing — never "the user's most recent session".

Config (config.yaml of the default profile):
  kanban_wake:
    url:    http://127.0.0.1:8644/webhooks/kanban-wake
    secret_file: ~/.secrets/kanban-wake-secret
    debounce_seconds: 60
  plugins.entries.kanban-wake.allow_gateway_injection: true

"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.request
import uuid

logger = logging.getLogger(__name__)

_ROUTE = "kanban-wake"
_last: dict[str, float] = {}


def _cfg() -> dict:
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get("kanban_wake", {}) or {}
    except Exception:
        return {}


def _secret() -> str:
    p = os.path.expanduser(_cfg().get("secret_file", "~/.secrets/kanban-wake-secret"))
    try:
        return open(p).read().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- emitter ---

def _post(text: str, task_id: str = "", event: str = "") -> None:
    url = _cfg().get("url", f"http://127.0.0.1:8644/webhooks/{_ROUTE}")
    body = json.dumps({"text": text, "task_id": task_id, "event": event}).encode()
    ts = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
    }
    sec = _secret()
    if sec:
        # The gateway verifier (gateway/platforms/webhook.py, `v2_sig`) compares
        # this header against a BARE lower-case hex digest of "<timestamp>.<body>".
        # A "sha256=" prefix (GitHub's scheme, not this one) 401s every request.
        sig = hmac.new(sec.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
        headers.update({"X-Webhook-Timestamp": ts, "X-Webhook-Signature-V2": sig})
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()


def _emit(event: str, task_id: str | None, **fields) -> None:
    try:
        if not task_id:
            return
        now = time.time()
        key = f"{event}:{task_id}"
        if now - _last.get(key, 0) < float(_cfg().get("debounce_seconds", 60)):
            return
        title = fields.get("title") or ""
        assignee = fields.get("assignee") or ""
        reason = fields.get("reason") or fields.get("blocked_reason") or fields.get("summary") or ""
        if not (title and reason):
            try:
                from hermes_cli import kanban_db as kb
                with kb.connect() as conn:
                    t = kb.get_task(conn, task_id)
                    if t:
                        title = title or (t.title or "")
                        assignee = assignee or (t.assignee or "")
                        if not reason:
                            cs = kb.list_comments(conn, task_id)
                            if cs:
                                reason = cs[-1].body or ""
            except Exception:
                pass
        head = "BLOCKED" if event == "blocked" else "DONE"
        text = f"KANBAN {head} {task_id} [{assignee}] {title}\n{reason[:1500]}"
        _post(text, task_id, event)
        # Only a DELIVERED wake is debounced. Recording this before _post meant a
        # failed delivery suppressed the next attempt for the whole window.
        _last[key] = time.time()
        logger.info("[kanban-wake] emitted %s for %s", event, task_id)
    except Exception as e:  # never break dispatch
        logger.warning("[kanban-wake] emit failed: %s", e)


def _on_blocked(task_id=None, **f):
    _emit("blocked", task_id, **f)


def _on_completed(task_id=None, **f):
    _emit("completed", task_id, **f)


# ------------------------------------------------------------- redirector ---

def _origin_subs(task_id: str) -> list[dict]:
    """The card's stored origin(s): ``kanban_notify_subs`` rows written at creation
    (chat session, inherited from the creating card, or the creating cron job's deliver target)."""
    if not task_id:
        return []
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_notify as kbn
    with kb.connect() as conn:
        return kbn.list_notify_subs(conn, task_id)


def _wakes(sub: dict) -> bool:
    return (sub.get("delivery_mode") or "notify") in ("notify+wake", "wake")


def _origin_session_key(gateway, sub: dict) -> str | None:
    """Session key of the conversation the card was created in, rebuilt from its stored origin."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    md = sub.get("delivery_metadata") or {}
    source = SessionSource(
        platform=Platform(sub["platform"]), chat_id=str(sub["chat_id"]),
        chat_type=str(sub.get("chat_type") or md.get("chat_type") or "dm"),
        thread_id=sub.get("thread_id") or None, user_id=sub.get("user_id"),
        user_id_alt=sub.get("user_id_alt"), parent_chat_id=md.get("parent_chat_id"),
        scope_id=md.get("scope_id"))
    return gateway._session_key_for_source(source)


def _pre_gateway_dispatch(event=None, gateway=None, **_):
    """Intercept the kanban-wake webhook turn; deliver it ONLY to the card's stored origin."""
    src = getattr(event, "source", None)
    if src is None or not str(getattr(src, "platform", "")).lower().endswith("webhook"):
        return None
    if getattr(src, "user_id", "") != f"webhook:{_ROUTE}":
        return None
    try:
        payload = getattr(event, "raw_message", None) or {}
        task_id = str(payload.get("task_id") or "") if isinstance(payload, dict) else ""
        subs = _origin_subs(task_id)
        if not subs:
            logger.info("[kanban-wake] %s has no origin conversation; nothing sent", task_id or "event")
            return {"action": "skip", "reason": "kanban-wake: card has no origin"}
        if any(_wakes(s) for s in subs):
            # Core's kanban notifier already wakes these origins; a second wake would double-post.
            return {"action": "skip", "reason": "kanban-wake: core notifier owns this origin"}
        sched = getattr(gateway, "_schedule_plugin_message_injection", None)
        content = ("[kanban-wake — event from the board you created this card from, not the user.]\n"
                   + (getattr(event, "text", "") or ""))
        for sub in subs:
            key = _origin_session_key(gateway, sub)
            ok = bool(key and callable(sched) and sched(session_key=key, content=content, plugin_id=_ROUTE))
            logger.log(logging.INFO if ok else logging.ERROR,
                       "[kanban-wake] %s -> origin %s %s", task_id, key, "injected" if ok else "REFUSED")
        return {"action": "skip", "reason": "kanban-wake delivered to card origin"}
    except Exception as e:
        logger.error("[kanban-wake] origin delivery failed: %s", e)
        return {"action": "skip", "reason": "kanban-wake: origin delivery failed"}


def register(ctx) -> None:
    ctx.register_hook("kanban_task_blocked", _on_blocked)
    ctx.register_hook("kanban_task_completed", _on_completed)
    ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch)
