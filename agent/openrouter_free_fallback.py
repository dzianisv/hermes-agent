"""Automated OpenRouter free-model fallback with dynamic, health-aware ranking.

Discovers free models from OpenRouter's live ``GET /api/v1/models`` catalog,
ranks them by a composite popularity score (benchmarks + context length +
creation recency) adjusted by observed runtime health (success/failure
counts), pins ``nvidia/nemotron-3.5-super-vl:free`` as the priority fallback,
and injects them as dynamic fallback candidates when the static chain is
exhausted. This lets Hermes keep serving turns on zero-cost OpenRouter
models without manually maintaining ``fallback_model`` entries in config.yaml.

Design
------
- ``fetch_free_openrouter_models()`` hits OpenRouter's live catalog, filters for
  zero-priced (free) models that advertise tool-calling support, ranks them by
  popularity score adjusted by observed health, and returns
  ``{provider: "openrouter", model: "<id>"}`` entries pinned so the configured
  priority model is first. Results are cached per-process with an adaptive TTL
  (shorter when models are failing, longer when all is well).
- ``build_dynamic_fallback_chain()`` merges the static config chain with
  ranked free models: static entries first, ranked free models after, deduped
  against each other and against the primary model. Models with too many
  consecutive failures are temporarily suppressed.
- ``prime_agent_free_fallback()`` injects ranked free models into an agent's
  ``_fallback_chain``.
- ``record_model_success(model_id)`` / ``record_model_failure(model_id)`` let
  the fallback path feed observed outcomes back into the ranking so models
  that disappear or start failing get deprioritized automatically.

Configuration (config.yaml, top level)::

    openrouter_free_fallback:
      enabled: true
      max_entries: 6          # how many free models to append to the chain
      cache_ttl_seconds: 300  # base cache TTL (seconds)
      priority_model: nvidia/nemotron-3.5-super-vl:free
      suppress_after_failures: 3   # consecutive failures before suppressing a model
      health_ttl_seconds: 600      # how long to remember a model's health record

Set ``enabled: false`` to opt out. When disabled the static chain is unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- knobs ------------------------------------------------------------------

_DEFAULT_MAX_ENTRIES = 6
_DEFAULT_CACHE_TTL_S = 300.0
_DEFAULT_PRIORITY_MODEL = "nvidia/nemotron-3.5-super-vl:free"
_DEFAULT_SUPPRESS_AFTER_FAILURES = 3
_DEFAULT_HEALTH_TTL_S = 600.0

# In-memory cache: (timestamp, entries)
_free_model_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])

# Per-model health registry: model_id -> {"successes": int, "failures": int,
# "consecutive_failures": int, "last_failure_ts": float, "last_success_ts": float}
_model_health: Dict[str, Dict[str, Any]] = {}

# Never offer these as free fallback (aggregator router, known-problematic).
_RESERVED_PREFIXES = ("openrouter/pareto",)

# Weights for the composite popularity score.
_WEIGHT_INTELLIGENCE = 0.5   # artificial_analysis.intelligence_index (0-100)
_WEIGHT_ELO = 0.3            # average design_arena ELO
_WEIGHT_CONTEXT = 0.1        # log-scaled context length
_WEIGHT_RECENCY = 0.1        # log-scaled creation recency

# Health adjustment: each consecutive failure subtracts this many points.
_HEALTH_FAILURE_PENALTY = 25.0
_HEALTH_SUCCESS_BONUS = 2.0


# --- config reading ---------------------------------------------------------


def _read_config() -> Dict[str, Any]:
    """Read the ``openrouter_free_fallback`` block from the active config.

    Fail-open: any read/parse error returns an empty dict (feature off).
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        return {}
    block = cfg.get("openrouter_free_fallback")
    return block if isinstance(block, dict) else {}


def _config_enabled() -> bool:
    return bool(_read_config().get("enabled", True))


def _config_max_entries() -> int:
    try:
        return int(_read_config().get("max_entries", _DEFAULT_MAX_ENTRIES))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ENTRIES


def _config_cache_ttl() -> float:
    try:
        return float(_read_config().get("cache_ttl_seconds", _DEFAULT_CACHE_TTL_S))
    except (TypeError, ValueError):
        return _DEFAULT_CACHE_TTL_S


def _config_priority_model() -> str:
    return str(_read_config().get("priority_model", _DEFAULT_PRIORITY_MODEL)).strip()


def _config_suppress_after_failures() -> int:
    try:
        return int(_read_config().get("suppress_after_failures", _DEFAULT_SUPPRESS_AFTER_FAILURES))
    except (TypeError, ValueError):
        return _DEFAULT_SUPPRESS_AFTER_FAILURES


def _config_health_ttl() -> float:
    try:
        return float(_read_config().get("health_ttl_seconds", _DEFAULT_HEALTH_TTL_S))
    except (TypeError, ValueError):
        return _DEFAULT_HEALTH_TTL_S


# --- runtime health tracking ------------------------------------------------

def _prune_stale_health(now: Optional[float] = None) -> None:
    """Drop health records older than ``health_ttl_seconds`` so a model that
    was failing long ago but has since recovered isn't penalized forever."""
    now = now if now is not None else time.time()
    ttl = _config_health_ttl()
    stale = [
        mid for mid, rec in _model_health.items()
        if (now - max(rec.get("last_success_ts", 0), rec.get("last_failure_ts", 0))) > ttl
    ]
    for mid in stale:
        _model_health.pop(mid, None)


def record_model_success(model_id: str) -> None:
    """Record a successful call to *model_id*. Boosts its health score."""
    now = time.time()
    rec = _model_health.setdefault(model_id, {"successes": 0, "failures": 0,
                                               "consecutive_failures": 0,
                                               "last_failure_ts": 0.0,
                                               "last_success_ts": 0.0})
    rec["successes"] += 1
    rec["consecutive_failures"] = 0
    rec["last_success_ts"] = now
    _prune_stale_health(now)


def record_model_failure(model_id: str) -> None:
    """Record a failed call to *model_id*. Penalizes its health score and may
    suppress it after ``suppress_after_failures`` consecutive failures."""
    now = time.time()
    rec = _model_health.setdefault(model_id, {"successes": 0, "failures": 0,
                                               "consecutive_failures": 0,
                                               "last_failure_ts": 0.0,
                                               "last_success_ts": 0.0})
    rec["failures"] += 1
    rec["consecutive_failures"] += 1
    rec["last_failure_ts"] = now
    _prune_stale_health(now)


def is_model_suppressed(model_id: str) -> bool:
    """True when *model_id* has hit the consecutive-failure suppression threshold."""
    rec = _model_health.get(model_id)
    if rec is None:
        return False
    return rec.get("consecutive_failures", 0) >= _config_suppress_after_failures()


def health_adjustment(model_id: str) -> float:
    """Return a point adjustment for *model_id* based on observed health.

    Positive = more preferred (recent successes), negative = less preferred
    (recent consecutive failures).
    """
    rec = _model_health.get(model_id)
    if rec is None:
        return 0.0
    consecutive = rec.get("consecutive_failures", 0)
    successes = rec.get("successes", 0)
    return _HEALTH_SUCCESS_BONUS * successes - _HEALTH_FAILURE_PENALTY * consecutive


# --- ranking ----------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _model_popularity_score(model: dict) -> float:
    """Composite score for ranking free models by automated popularity signals.

    Signals used (all from OpenRouter's ``/api/v1/models`` response):
      * ``benchmarks.artificial_analysis.intelligence_index`` — overall
        intelligence (0-100 scale).
      * ``benchmarks.design_arena[].elo`` — average ELO across arena
        categories (coding, design, gamedev, website, ui/component).
      * ``context_length`` — context window size (log-scaled).
      * ``created`` — model creation timestamp (log-scaled recency).

    Returns a float where higher = more popular/capable.
    """
    benchmarks = model.get("benchmarks") or {}
    aa = benchmarks.get("artificial_analysis") or {}
    intelligence = _safe_float(aa.get("intelligence_index"), 0.0)

    design_arena = benchmarks.get("design_arena") or []
    elos = [
        _safe_float(entry.get("elo"), 0.0)
        for entry in design_arena
        if isinstance(entry, dict)
    ]
    avg_elo = sum(elos) / len(elos) if elos else 0.0

    context_length = _safe_float(model.get("context_length"), 0.0)
    context_score = math.log1p(context_length)

    created = _safe_float(model.get("created"), 0.0)
    now_ts = time.time()
    age_seconds = max(now_ts - created, 1.0)
    recency_score = math.log1p(now_ts) - math.log1p(age_seconds)

    return (
        _WEIGHT_INTELLIGENCE * intelligence
        + _WEIGHT_ELO * avg_elo / 100.0
        + _WEIGHT_CONTEXT * context_score
        + _WEIGHT_RECENCY * recency_score
    )


def _model_supports_tools(model: dict) -> bool:
    """True when the model advertises tool-calling support."""
    params = model.get("supported_parameters") if isinstance(model, dict) else None
    return "tools" in params if isinstance(params, list) else True


def _model_is_free(model: dict) -> bool:
    """True when both prompt and completion pricing are zero."""
    pricing = model.get("pricing") or {}
    if not isinstance(pricing, dict):
        return False
    return (
        _safe_float(pricing.get("prompt", "0"), 0.0) == 0.0
        and _safe_float(pricing.get("completion", "0"), 0.0) == 0.0
    )


# --- discovery --------------------------------------------------------------


def _adaptive_cache_ttl(base_ttl: float, now: float) -> float:
    """Shorten the cache TTL when models are failing so we re-discover
    healthy alternatives faster; lengthen it when all is well."""
    failing = sum(1 for rec in _model_health.values()
                  if rec.get("consecutive_failures", 0) > 0)
    if failing == 0:
        return base_ttl  # all healthy — cache longer
    if failing >= 3:
        return max(30.0, base_ttl / 4)  # many failing — refresh aggressively
    return max(60.0, base_ttl / 2)  # some failing — refresh more often


def fetch_free_openrouter_models(
    timeout: float = 8.0, *, force_refresh: bool = False, max_entries: int = _DEFAULT_MAX_ENTRIES
) -> List[Dict[str, Any]]:
    """Return ranked free OpenRouter models as fallback entries.

    Each entry: ``{"provider": "openrouter", "model": "<model-id>"}``.

    Fetches the full OpenRouter catalog (with benchmarks), filters for
    zero-priced tool-calling models, ranks by popularity score adjusted by
    observed health, and pins the configured priority model first. Results are
    cached per-process with an adaptive TTL.
    """
    global _free_model_cache
    if not _config_enabled():
        return []

    now = time.monotonic()
    cached_at, cached = _free_model_cache
    base_ttl = _config_cache_ttl()
    ttl = _adaptive_cache_ttl(base_ttl, now)
    if not force_refresh and cached and (now - cached_at) < ttl:
        return cached[:max_entries]

    priority_model = _config_priority_model()

    # Fetch the full live catalog (model objects with benchmarks), falling back
    # to the curated in-repo snapshot when the network is unavailable.
    try:
        from hermes_cli.models import _fetch_live_catalog_index, _OPENROUTER_CATALOG_URL
        from hermes_cli.models import _urlopen_model_catalog_request
        live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    except Exception:
        live = None

    models: List[dict] = []
    if live is not None:
        _live_items, live_by_id = live
        models = list(live_by_id.values())

    free_models: List[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        mid = str(model.get("id") or "").strip()
        if not mid:
            continue
        if not mid.endswith(":free"):
            continue
        if any(mid.startswith(p) for p in _RESERVED_PREFIXES):
            continue
        if not _model_is_free(model):
            continue
        if not _model_supports_tools(model):
            continue
        if is_model_suppressed(mid):
            continue
        if mid in free_models:
            continue
        free_models.append(mid)

    # Rank by popularity score adjusted by observed health; the priority model
    # always goes first.
    def sort_key(mid: str) -> Tuple[int, float]:
        if mid == priority_model:
            return (0, 0.0)  # always first
        model_obj = live_by_id.get(mid, {}) if live is not None else {}
        score = _model_popularity_score(model_obj) if isinstance(model_obj, dict) else 0.0
        score += health_adjustment(mid)
        return (1, -score)  # higher score = sorts earlier (negated)

    free_models.sort(key=sort_key)

    entries: List[Dict[str, Any]] = [
        {"provider": "openrouter", "model": mid} for mid in free_models[:max_entries]
    ]

    _free_model_cache = (now, entries)
    logger.debug(
        "Discovered and ranked %d free OpenRouter fallback models (priority=%s, suppressed=%d)",
        len(entries), priority_model,
        sum(1 for mid in free_models if is_model_suppressed(mid)),
    )
    return entries


def build_dynamic_fallback_chain(
    primary_model: str,
    primary_provider: str,
    static_chain: Optional[List[Dict[str, Any]]] = None,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> List[Dict[str, Any]]:
    """Merge static fallback chain with ranked free OpenRouter models.

    Static chain entries take priority (they appear first). Ranked free models
    are appended after, deduped against both the static chain and the primary
    model itself. The priority model (``nvidia/nemotron-3.5-super-vl:free``)
    is first among free models. Suppressed models are excluded.
    """
    chain: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    primary_key = (
        str(primary_provider or "").strip().lower(),
        str(primary_model or "").strip().lower(),
    )

    # 1. Static chain first (user-configured, highest priority).
    for entry in (static_chain or []):
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue
        key = (provider, model.lower())
        if key == primary_key or key in seen:
            continue
        seen.add(key)
        chain.append({"provider": provider, "model": model})
        if len(chain) >= max_entries:
            return chain

    # 2. Ranked free OpenRouter models after static entries.
    free_entries = fetch_free_openrouter_models(max_entries=max_entries * 2)
    for entry in free_entries:
        provider = str(entry.get("provider") or "").strip().lower()
        model = str(entry.get("model") or "").strip()
        key = (provider, model.lower())
        if key == primary_key or key in seen:
            continue
        seen.add(key)
        chain.append({"provider": provider, "model": model})
        if len(chain) >= max_entries:
            break

    return chain


def prime_agent_free_fallback(
    agent,
    primary_model: str,
    primary_provider: str,
    static_chain: Optional[List[Dict[str, Any]]] = None,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> int:
    """Inject ranked free OpenRouter models into an agent's fallback chain.

    Appends to the existing ``_fallback_chain`` (deduped), resets
    ``_fallback_index`` so the expanded chain is walked from the start, and
    returns the number of entries added. Safe to call multiple times —
    re-fetches honor the cache TTL.
    """
    dynamic = build_dynamic_fallback_chain(
        primary_model, primary_provider,
        static_chain or getattr(agent, "_fallback_chain", None) or [],
        max_entries=max_entries,
    )
    if not dynamic:
        return 0

    existing = list(getattr(agent, "_fallback_chain", None) or [])
    seen = {
        (str(e.get("provider", "")).strip().lower(), str(e.get("model", "")).strip().lower())
        for e in existing if isinstance(e, dict)
    }

    added = 0
    for entry in dynamic:
        key = (
            str(entry.get("provider", "")).strip().lower(),
            str(entry.get("model", "")).strip().lower(),
        )
        if key in seen:
            continue
        existing.append(entry)
        seen.add(key)
        added += 1

    agent._fallback_chain = existing
    agent._fallback_index = 0
    if added:
        logger.info(
            "Primed agent with %d ranked free OpenRouter fallback models "
            "(primary=%s/%s)", added, primary_provider, primary_model,
        )
    return added
