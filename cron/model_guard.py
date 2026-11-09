"""Catalog-driven guard for cron model pins and dispatch.

Why: a cron job can hold a provider/model pair that looks syntactically valid but is no longer
served by that provider. Without a pre-dispatch guard, the scheduler burns a full agent turn and
only then learns from the provider that the slug is gone. The measured incident was a pinned
OpenRouter slug whose retired ``:free`` variant survived in the stored job long after the main
agent moved away.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Optional

from hermes_cli.models import fetch_api_models, normalize_provider, provider_label
from hermes_cli.models_validate import _is_self_hosted_provider, validate_requested_model
from hermes_cli.providers import get_provider

logger = logging.getLogger(__name__)

_GUARD_OFF_VALUES = {"0", "false", "off"}
_OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/auth/key"


def _guard_enabled() -> bool:
    raw = str(os.getenv("HERMES_CRON_MODEL_GUARD") or "").strip().lower()
    return raw not in _GUARD_OFF_VALUES


def _normalize_provider_or_blank(provider: Any) -> str:
    raw = str(provider or "").strip()
    return normalize_provider(raw) if raw else ""


def _provider_base_url(provider: str) -> str:
    if not provider:
        return ""
    try:
        pdef = get_provider(provider, allow_network=False)
    except Exception:
        return ""
    return str(getattr(pdef, "base_url", "") or "").strip().rstrip("/")


def _catalog_alternatives(
    model: str,
    provider: str,
    *,
    base_url: str | None = None,
) -> tuple[list[str], int, str] | None:
    resolved_base_url = str(base_url or "").strip().rstrip("/")
    if not resolved_base_url:
        resolved_base_url = _provider_base_url(provider)
    if not resolved_base_url:
        return None
    catalog = fetch_api_models(None, resolved_base_url)
    if not catalog:
        return None

    seen: set[str] = set()
    deduped: list[str] = []
    for model_id in catalog:
        text = str(model_id or "").strip()
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    if not deduped:
        return None

    vendor_prefix = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    same_vendor = [mid for mid in deduped if vendor_prefix and mid.lower().startswith(f"{vendor_prefix}/")]
    choices = same_vendor or deduped
    shown = choices[:8]
    more = max(len(choices) - len(shown), 0)
    scope = f"same `{vendor_prefix}/*` vendor" if same_vendor else "provider catalog"
    return shown, more, scope


def _openrouter_key_limit() -> Optional[float]:
    headers = {"Accept": "application/json"}
    api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(_OPENROUTER_KEY_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310 - fixed HTTPS URL
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.debug("cron model guard: OpenRouter auth/key probe failed", exc_info=True)
        return None

    for candidate in (payload, payload.get("data") if isinstance(payload, dict) else None):
        if isinstance(candidate, dict) and "limit" in candidate:
            try:
                return float(candidate["limit"])
            except (TypeError, ValueError):
                return None
    return None


def _job_target(job_id: str, job_name: str) -> str:
    if job_name and job_id:
        return f"cron job '{job_name}' ({job_id})"
    if job_name:
        return f"cron job '{job_name}'"
    if job_id:
        return f"cron job {job_id}"
    return "cron job"


def guard_provider(job: dict, cfg: dict | None = None) -> str:
    """Resolve the provider that would serve *job*'s model pin or fallback default."""
    if not _guard_enabled():
        return ""
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = str((job or {}).get("provider") or "").strip()
    if not raw:
        cron_cfg = cfg.get("cron") if isinstance(cfg.get("cron"), dict) else {}
        raw = str(cron_cfg.get("model_provider") or "").strip()
    if not raw:
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        raw = str(model_cfg.get("provider") or "").strip()
    return _normalize_provider_or_blank(raw)


def unserved_model_diagnosis(
    model: str,
    provider: str,
    *,
    job_id: str = "",
    job_name: str = "",
    cfg: dict | None = None,
) -> str | None:
    """Explain why a cron model pin is unservable, or ``None`` when the check cannot decide.

    The guard is intentionally fail-open: blank inputs, custom/self-hosted endpoints, validator
    errors, and catalog fetch failures all return ``None`` so scheduled work keeps running unless
    Hermes has positive proof that the slug is absent from the provider's own catalog.
    """
    del cfg  # reserved for future provider-specific context; signature is part of the public API
    if not _guard_enabled():
        return None

    requested_model = str(model or "").strip()
    normalized_provider = _normalize_provider_or_blank(provider)
    if not requested_model or not normalized_provider:
        return None
    if _is_self_hosted_provider(normalized_provider) or normalized_provider.startswith("custom:"):
        return None

    base_url = _provider_base_url(normalized_provider) or None

    try:
        verdict = validate_requested_model(
            requested_model,
            normalized_provider,
            base_url=base_url or None,
        )
    except Exception:
        logger.warning(
            "cron model guard: validator errored for model=%r provider=%r; failing open",
            requested_model,
            normalized_provider,
            exc_info=True,
        )
        return None
    if verdict.get("accepted"):
        return None

    label = provider_label(normalized_provider)
    target = _job_target(job_id, job_name)
    message = str(verdict.get("message") or "").strip()
    lines = [
        f"Refusing {target}: model `{requested_model}` is not served by {label}.",
    ]
    if message:
        lines.append(f"Validator: {message}")

    try:
        alternatives = _catalog_alternatives(
            requested_model,
            normalized_provider,
            base_url=base_url,
        )
    except Exception:
        logger.debug(
            "cron model guard: catalog suggestion lookup failed for model=%r provider=%r",
            requested_model,
            normalized_provider,
            exc_info=True,
        )
        alternatives = None
    if alternatives is not None:
        shown, more, scope = alternatives
        suffix = f" (+{more} more)" if more else ""
        lines.append(
            f"Serving alternatives from the live {label} {scope}: "
            + ", ".join(f"`{model_id}`" for model_id in shown)
            + suffix
        )

    if normalized_provider == "openrouter":
        limit = _openrouter_key_limit()
        if limit == 0:
            lines.append(
                "OpenRouter key status: the key's total spend limit is $0.00, so paid slugs are "
                "unreachable on this key and a retirement 404's 'use the paid slug' advice is a trap."
            )

    edit_job_id = job_id or "<job_id>"
    lines.append(
        "Operator action: re-pin the job to a serving slug with "
        f"`hermes cron edit {edit_job_id} --model <slug>`, or unpin it with `--pinned false`."
    )
    return "\n".join(lines)


def guard_dispatch(
    model: str,
    provider: str,
    *,
    job_id: str = "",
    job_name: str = "",
    cfg: dict | None = None,
) -> None:
    """Raise before dispatch when a provider definitively does not serve the selected model."""
    diagnosis = unserved_model_diagnosis(model, provider, job_id=job_id, job_name=job_name, cfg=cfg)
    if diagnosis is not None:
        raise RuntimeError(diagnosis)


def guard_pin_write(
    model: str,
    provider: str,
    *,
    job_id: str = "",
    job_name: str = "",
    cfg: dict | None = None,
) -> None:
    """Reject a stored pin that the provider's catalog says cannot be served."""
    diagnosis = unserved_model_diagnosis(model, provider, job_id=job_id, job_name=job_name, cfg=cfg)
    if diagnosis is not None:
        raise ValueError(diagnosis)
