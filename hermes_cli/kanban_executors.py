"""Native coding-agent CLI executors for kanban tasks.

A card may name a *native executor* — a coding-agent CLI (``pi``, ``claude``,
``codex``, ``copilot``) that the dispatcher runs directly in the task's
workspace with the card's title + body as the prompt. No Hermes LLM turn sits
in between: the CLI is the worker.

This module owns only the declarative part of that path:

* the registry of known executors (binary name + argv template + how the
  prompt is passed),
* binary resolution via ``shutil.which`` at spawn time (a missing binary is a
  spawn failure with the binary named, never a doomed process),
* provider resolution for ``pi`` (explicit override, else runtime readiness
  probing over the providers the CLI itself reports).

Deliberately free of hardcoded model ids, provider constants and binary paths:
everything either comes from the card / config, or is probed from the CLI at
spawn time.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

__all__ = [
    "ExecutorSpec",
    "EXECUTORS",
    "executor_names",
    "normalize_executor",
    "get_executor",
    "resolve_executor_binary",
    "build_executor_argv",
    "resolve_pi_provider",
    "UnknownExecutorError",
    "ExecutorBinaryMissing",
    "ExecutorProviderError",
]

# Placeholder tokens understood by an ``argv_template`` entry.
PROMPT_TOKEN = "{prompt}"
PROVIDER_TOKEN = "{provider}"

# Probe timeouts, seconds. Kept small: these run inside a dispatcher tick.
_PROBE_TIMEOUT = 30


class UnknownExecutorError(ValueError):
    """Raised when a card names an executor that is not in the registry."""


class ExecutorBinaryMissing(RuntimeError):
    """Raised when the executor's CLI is not on PATH at spawn time."""


class ExecutorProviderError(RuntimeError):
    """Raised when no usable provider could be resolved for an executor."""


@dataclass(frozen=True)
class ExecutorSpec:
    """One native coding-agent CLI.

    ``argv_template`` are the arguments after the binary for a
    non-interactive, single-shot run. ``{prompt}`` is replaced by the card's
    prompt (title + body) and ``{provider}`` — when the spec declares
    ``needs_provider`` — by the resolved provider name.
    """

    name: str
    binary: str
    argv_template: tuple[str, ...]
    needs_provider: bool = False
    # Human-readable note, surfaced in error/--help text.
    description: str = ""
    # Extra environment for the CLI child (never secrets).
    env: dict = field(default_factory=dict)

    def prompt_style(self) -> str:
        """How the prompt reaches the CLI: ``flag`` or ``positional``."""
        idx = self.argv_template.index(PROMPT_TOKEN)
        if idx > 0 and self.argv_template[idx - 1].startswith("-"):
            return "flag"
        return "positional"


EXECUTORS: dict[str, ExecutorSpec] = {
    "pi": ExecutorSpec(
        name="pi",
        binary="pi",
        # `pi --print --provider <provider> <prompt>`
        argv_template=("--print", "--provider", PROVIDER_TOKEN, PROMPT_TOKEN),
        needs_provider=True,
        description="pi coding agent (non-interactive --print run)",
    ),
    "claude": ExecutorSpec(
        name="claude",
        binary="claude",
        argv_template=("--print", "--permission-mode", "acceptEdits", PROMPT_TOKEN),
        description="Claude Code (-p print mode, edits auto-accepted)",
    ),
    "codex": ExecutorSpec(
        name="codex",
        binary="codex",
        argv_template=("exec", "--full-auto", PROMPT_TOKEN),
        description="OpenAI Codex CLI (`codex exec`)",
    ),
    "copilot": ExecutorSpec(
        name="copilot",
        binary="copilot",
        argv_template=("--allow-all-tools", "--prompt", PROMPT_TOKEN),
        description="GitHub Copilot CLI (-p non-interactive prompt)",
    ),
}


def executor_names() -> list[str]:
    """Valid executor names, sorted — used in error messages and --help."""
    return sorted(EXECUTORS)


def normalize_executor(value: Optional[str]) -> Optional[str]:
    """Validate + normalize an executor name. ``None``/empty → ``None``.

    Raises :class:`UnknownExecutorError` (a ``ValueError``) naming every valid
    executor when the value isn't registered — failing loudly at create time
    beats a card that can never be dispatched.
    """
    name = (value or "").strip().lower()
    if not name:
        return None
    if name not in EXECUTORS:
        raise UnknownExecutorError(
            f"unknown executor {value!r}; valid executors: "
            f"{', '.join(executor_names())}"
        )
    return name


def get_executor(name: str) -> ExecutorSpec:
    """Return the spec for ``name`` (validated)."""
    return EXECUTORS[normalize_executor(name)]  # type: ignore[index]


def resolve_executor_binary(
    spec: ExecutorSpec,
    *,
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Resolve the CLI's absolute path, or fail naming the binary.

    Resolution happens at spawn time (PATH differs between the dispatcher's
    shell and a systemd/launchd service). A missing binary raises instead of
    starting a process that is guaranteed to die.
    """
    # Resolved lazily (not as a default argument) so tests and callers can
    # monkeypatch ``shutil.which`` itself.
    lookup = which or shutil.which
    path = lookup(spec.binary)
    if not path:
        raise ExecutorBinaryMissing(
            f"executor {spec.name!r}: `{spec.binary}` not found on PATH — "
            f"install it (or add it to the dispatcher's PATH) before "
            f"dispatching tasks with --executor {spec.name}"
        )
    return path


def _probe(argv: Sequence[str], *, timeout: int = _PROBE_TIMEOUT) -> tuple[int, str]:
    """Run a short read-only probe command, returning (rc, stdout+stderr)."""
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv built from the registry
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, str(exc))
    return (proc.returncode, f"{proc.stdout}\n{proc.stderr}")


def _pi_reported_providers(binary: str, runner=None) -> list[str]:
    """Providers the ``pi`` CLI itself reports, in the order it lists them.

    Derived from ``pi --list-models`` (first column), so the set follows the
    CLI's own catalog — nothing about providers or models is hardcoded here.
    """
    runner = runner or _probe
    rc, out = runner([binary, "--list-models"])
    if rc != 0:
        return []
    providers: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        candidate = parts[0].strip()
        if not candidate or candidate.lower() == "provider":
            continue
        if candidate not in providers:
            providers.append(candidate)
    return providers


def _pi_provider_ready(binary: str, provider: str, runner=None) -> bool:
    """True when ``pi auth check --provider <p>`` reports readiness."""
    runner = runner or _probe
    rc, out = runner([binary, "auth", "check", "--provider", provider])
    if rc != 0:
        return False
    text = out.strip().lower()
    if not text:
        return False
    # Plain mode prints `ready` / `not_ready`; --json prints {"status": ...}.
    # `not_ready` must never match `ready`, so check line-wise and exactly.
    for line in text.splitlines():
        token = line.strip().strip('"')
        if token == "ready":
            return True
        if '"status"' in token and '"ready"' in token:
            return True
    return False


def resolve_pi_provider(
    binary: str,
    *,
    override: Optional[str] = None,
    runner=None,
) -> str:
    """Resolve which provider a ``pi`` run should use.

    ``override`` (card ``provider_override`` or ``kanban.executors.pi.provider``
    in config.yaml) wins outright and is used as-is. Otherwise every provider
    the CLI reports is probed with ``pi auth check --provider <p>`` and the
    first one that reports ``ready`` is used. No provider constant is pinned.
    """
    chosen = (override or "").strip()
    if chosen:
        return chosen
    providers = _pi_reported_providers(binary, runner=runner)
    if not providers:
        raise ExecutorProviderError(
            "executor 'pi': could not read any provider from "
            "`pi --list-models`; set a provider explicitly "
            "(card --provider, or kanban.executors.pi.provider in config.yaml)"
        )
    for provider in providers:
        if _pi_provider_ready(binary, provider, runner=runner):
            return provider
    raise ExecutorProviderError(
        "executor 'pi': no provider is ready (`pi auth check` reported none of "
        f"{', '.join(providers)} as ready) — authenticate one, or set a "
        "provider explicitly (card --provider, or "
        "kanban.executors.pi.provider in config.yaml)"
    )


def build_executor_argv(
    executor: str,
    prompt: str,
    *,
    provider_override: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    runner=None,
) -> list[str]:
    """Build the full argv for a single-shot native executor run.

    Raises :class:`UnknownExecutorError` for an unregistered name,
    :class:`ExecutorBinaryMissing` when the CLI is absent, and
    :class:`ExecutorProviderError` when a provider-needing CLI has none ready.
    """
    spec = get_executor(executor)
    binary = resolve_executor_binary(spec, which=which)
    provider: Optional[str] = None
    if spec.needs_provider:
        # Only ``pi`` declares a provider today, and its readiness protocol is
        # pi-specific; keep the resolver keyed on the spec rather than growing
        # a generic mechanism with a single consumer.
        provider = resolve_pi_provider(
            binary, override=provider_override, runner=runner
        )

    argv = [binary]
    for token in spec.argv_template:
        if token == PROMPT_TOKEN:
            argv.append(prompt)
        elif token == PROVIDER_TOKEN:
            if not provider:
                raise ExecutorProviderError(
                    f"executor {spec.name!r}: no provider resolved"
                )
            argv.append(provider)
        else:
            argv.append(token)
    return argv
