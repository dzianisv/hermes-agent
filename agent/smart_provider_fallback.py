"""Opt-in free-first routing, with credential-scoped recovery between turns.

No chat state is sent to recovery probes. Catalog availability is not evidence of
inference recovery, and synthetic/curated Codex slugs are not entitlements.

Opt in only in product-lead-agentpod's config.yaml with::

    openrouter_free_fallback:
      enabled: true
      smart_provider_fallback: true

No live configuration is changed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from pathlib import Path

from utils import atomic_write_text

_lock = threading.Lock()
_holder = threading.local()
_codex_cache = {}
_PROBE_URL = "https://openrouter.ai/api/v1/chat/completions"


def enabled():
    from hermes_constants import get_hermes_home
    from agent.openrouter_free_fallback import _read_config
    return (get_hermes_home().name == "product-lead-agentpod"
            and _read_config().get("smart_provider_fallback") is True)


def _key(agent=None):
    if agent is not None:
        if getattr(agent, "provider", "") == "openrouter":
            key = getattr(agent, "api_key", None)
        else:
            primary = getattr(agent, "_primary_runtime", {}) or {}
            key = primary.get("api_key") if primary.get("provider") == "openrouter" else None
        if isinstance(key, str) and key:
            return key
    from agent.secret_scope import get_secret
    return get_secret("OPENROUTER_API_KEY") or ""


def _path(key):
    from hermes_constants import get_hermes_home
    digest = hashlib.sha256(key.encode()).hexdigest()
    return Path(get_hermes_home()) / "rate_limits" / f"openrouter-free-{digest}.json"


def _read(path):
    try:
        state = json.loads(path.read_text())
        if (isinstance(state, dict) and isinstance(state.get("blocked", False), bool)
                and isinstance(state.get("next_probe", 0), (int, float))
                and math.isfinite(state.get("next_probe", 0))
                and isinstance(state.get("failures", 0), int)
                and state.get("failures", 0) >= 0):
            return state
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        pass
    # Corruption is not proof of recovery; one serialized inference can repair it.
    return {"blocked": True, "next_probe": 0, "failures": 0}


def account_quota_error(error):
    """Only OpenRouter's explicit account free-day limit trips the shared breaker.

    A provider's daily model quota/capacity limit must not poison other models.
    """
    if getattr(error, "status_code", 429) != 429:
        return False
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if not isinstance(details, dict):
            return False
        # metadata.raw describes an upstream provider, not OpenRouter's account.
        text = str(details.get("message") or "").lower()
        metadata = details.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("provider_name"):
            return False
    else:
        text = str(error).lower()
    return bool(re.search(r"free[- ](?:model|tier|models).*?(?:per[- ]day|daily)|(?:per[- ]day|daily).*?free[- ](?:model|tier|models)", text))


def _reset_at(error, context):
    from agent.retry_utils import parse_retry_after_seconds
    now = time.time()
    from agent.credential_pool import _parse_absolute_timestamp
    until = _parse_absolute_timestamp((context or {}).get("reset_at")) or 0
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    delay = parse_retry_after_seconds(headers.get("retry-after"))
    if delay and math.isfinite(delay):
        until = max(until, now + delay)
    # OpenRouter's reset header is Unix milliseconds.
    try:
        hint = float(headers.get("x-ratelimit-reset", 0))
        if hint > 1e11:
            hint /= 1000
        if math.isfinite(hint):
            until = max(until, hint)
    except (TypeError, ValueError):
        pass
    return max(now + 300, until)


def record_error(agent, error, context=None):
    if not enabled() or getattr(agent, "provider", "") != "openrouter":
        return
    if not account_quota_error(error):
        return
    block_free(agent, _reset_at(error, context))
    return True


def block_free(agent, until):
    """Also gate restoration after all free candidates failed, not just quota."""
    agent._smart_free_available = False
    key = _key(agent)
    if not key:
        return
    from hermes_cli.auth import _file_lock
    path = _path(key)
    try:
        with _lock, _file_lock(path.with_suffix(".lock"), _holder, 1, "Free quota state busy"):
            state = _read(path)
            state["blocked"] = True
            state["next_probe"] = max(state.get("next_probe", 0), until)
            atomic_write_text(str(path), json.dumps(state))
    except (OSError, TimeoutError):
        # The current request must still skip the known-empty account.
        agent._smart_free_available = False


def codex_models():
    """Live account catalog only; no default, cached local or synthetic guesses."""
    from hermes_cli.auth_codex import resolve_codex_runtime_credentials
    from hermes_cli.codex_models import _ranked_slugs
    from agent.codex_headers import codex_account_headers
    from agent.model_metadata import fetch_codex_catalog_entries, _codex_catalog_probe_allowed
    import httpx
    try:
        creds = resolve_codex_runtime_credentials()
        token, base = creds.get("api_key"), creds.get("base_url")
        if not token or not base or not _codex_catalog_probe_allowed(token, base):
            return []
        identity = hashlib.sha256((base + token).encode()).hexdigest()
        cached = _codex_cache.get(identity)
        if cached and time.monotonic() - cached[0] < 300:
            return list(cached[1])
        # Reserve a negative cache slot before I/O so concurrent chats do not
        # fan out catalog failures. Empty discovery remains fail-closed.
        _codex_cache[identity] = (time.monotonic(), [])
        headers = {"Authorization": f"Bearer {token}", **codex_account_headers(token)}
        entries, status = fetch_codex_catalog_entries(
            lambda url: httpx.get(url, headers=headers, timeout=5), base_url=base)
        models = [m for m in _ranked_slugs(entries) if eligible_codex(m)] if status == 200 else []
        _codex_cache[identity] = (time.monotonic(), models)
        return models
    except Exception:
        return []


def eligible_codex(model):
    match = re.fullmatch(r"gpt-(\d+)(?:[.-][a-zA-Z0-9]+)*", model)
    return bool(match and int(match[1]) >= 6)


def _probe(key, model):
    import httpx
    # No SDK retries, redirects, conversation, tools or paid model substitution.
    with httpx.Client(timeout=httpx.Timeout(5), follow_redirects=False) as client:
        response = client.post(_PROBE_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "Say OK"}],
                  "max_tokens": 8, "stream": False,
                  "provider": {"max_price": {"prompt": 0, "completion": 0}}})
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        return bool(not body.get("error") and choices and
                    choices[0].get("message", {}).get("content") and
                    choices[0].get("finish_reason") in {"stop", "length"})


def free_available(key, free):
    """Serialize probes across chats/processes; reserve next slot before network I/O."""
    if not key:
        return False
    path = _path(key)
    state = _read(path)
    if not state.get("blocked"):
        return True
    if state.get("next_probe", 0) > time.time() or not free:
        return False
    if not _lock.acquire(blocking=False):
        return False
    try:
        from hermes_cli.auth import _file_lock
        with _file_lock(path.with_suffix(".lock"), _holder, 1, "Free quota probe busy"):
            state = _read(path)
            if not state.get("blocked"):
                return True
            if state.get("next_probe", 0) > time.time():
                return False
            failures = min(int(state.get("failures", 0)) + 1, 8)
            state.update(failures=failures,
                         next_probe=time.time() + min(3600, 300 * 2 ** (failures - 1)) * random.uniform(1, 1.2))
            atomic_write_text(str(path), json.dumps(state))
            try:
                success = _probe(key, free[0]["model"])
            except Exception as error:
                state["next_probe"] = max(state["next_probe"], _reset_at(error, {}))
                success = False
            from agent.openrouter_free_fallback import record_model_success, record_model_failure
            if success:
                record_model_success(free[0]["model"])
                state = {"blocked": False, "failures": 0, "next_probe": 0}
            else:
                record_model_failure(free[0]["model"])
            atomic_write_text(str(path), json.dumps(state))
            return success
    except (OSError, TimeoutError, ValueError, TypeError):
        return False
    finally:
        _lock.release()


def prepare_turn(agent):
    """Called only between user requests; reuse the existing runtime swap/restore."""
    if not enabled():
        return
    from agent.openrouter_free_fallback import fetch_free_openrouter_models
    free = fetch_free_openrouter_models(max_entries=6)
    available = bool(free) and free_available(_key(agent), free)
    agent._smart_free_models = {entry["model"] for entry in free}
    agent._smart_free_available = available
    codex = [{"provider": "openai-codex", "model": m} for m in codex_models()]
    agent._smart_codex_models = {entry["model"] for entry in codex}
    agent._fallback_chain = (free if available else []) + codex
    agent._fallback_index = 0
    primary = getattr(agent, "_primary_runtime", {}) or {}
    if primary.get("provider") == "openrouter":
        agent._rate_limited_until = 0 if available else float("inf")
    provider, model = getattr(agent, "provider", ""), getattr(agent, "model", "")
    valid = ((provider == "openrouter" and available and model in agent._smart_free_models)
             or (not available and provider == "openai-codex" and model in agent._smart_codex_models))
    if not valid and not agent._try_activate_fallback():
        raise RuntimeError("No eligible zero-priced OpenRouter or authenticated GPT-6+ Codex route")


def constrain_request(agent, kwargs):
    """A catalog TTL must never turn a zero-cost policy into a paid request."""
    if enabled() and getattr(agent, "provider", "") == "openrouter":
        if skip_candidate(agent, "openrouter", agent.model):
            raise RuntimeError("OpenRouter route is not an eligible available free model")
        kwargs["model"] = agent.model
        extra = dict(kwargs.get("extra_body") or {})
        extra.pop("model", None)
        for key in ("models", "route", "plugins"):
            extra.pop(key, None)
            kwargs.pop(key, None)
        provider = dict(extra.get("provider") or {})
        provider["max_price"] = {"prompt": 0, "completion": 0}
        extra["provider"] = provider
        kwargs["extra_body"] = extra
    return kwargs


def skip_candidate(agent, provider, model):
    if not enabled():
        return False
    if provider == "openrouter":
        key = _key(agent)
        return (not key or not getattr(agent, "_smart_free_available", False)
                or _read(_path(key)).get("blocked", False)
                or model not in getattr(agent, "_smart_free_models", set()))
    return provider != "openai-codex" or model not in getattr(agent, "_smart_codex_models", set())
