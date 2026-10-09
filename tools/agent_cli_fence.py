"""agent-cli-fence: stop a non-worker session (e.g. the EM/lead profile) from
launching coding-agent CLIs through the terminal tool.

Kanban workers carry ``HERMES_KANBAN_TASK`` (set by the dispatcher); they are
never fenced. Opt-in per profile via config.yaml::

    agent_cli_fence:
      enabled: true
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any, Dict, List, Optional

KANBAN_TASK_ENV = "HERMES_KANBAN_TASK"
CONFIG_KEY = "agent_cli_fence"

# Wrappers whose own options precede the real command.
_WRAPPERS = {"env", "timeout", "gtimeout", "nohup", "sudo", "exec", "command", "nice", "time", "stdbuf", "xargs"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_NPX_PKGS = {"@openai/codex": "codex", "@anthropic-ai/claude-code": "claude", "opencode-ai": "opencode"}


def _base(tok: str) -> str:
    return os.path.basename(tok)


def _segments(command: str) -> List[List[str]]:
    segs: List[List[str]] = []
    for line in command.splitlines() or [""]:
        segs.extend(_line_segments(line))
    return segs


def _line_segments(command: str) -> List[List[str]]:
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lex.whitespace_split = True
        lex.commenters = "#"
        toks = list(lex)
    except ValueError:
        toks = command.split()
    segs: List[List[str]] = [[]]
    for t in toks:
        if t and set(t) <= set(";&|"):
            segs.append([])
        elif t == "\n":
            segs.append([])
        else:
            segs[-1].append(t)
    out: List[List[str]] = []
    for s in segs:
        # strip subshell / group punctuation
        s = [t.lstrip("(").rstrip(")") if t not in ("(", ")") else "" for t in s]
        s = [t for t in s if t and t not in ("{", "}", "!", "then", "do", "else", "if", "while")]
        if s:
            out.append(s)
    return out


def _strip_prefixes(toks: List[str]) -> List[str]:
    i = 0
    while i < len(toks):
        t = toks[i]
        if _ASSIGN_RE.match(t):
            i += 1
            continue
        b = _base(t)
        if b in _WRAPPERS:
            i += 1
            # skip wrapper options; timeout/gtimeout take a duration positional
            while i < len(toks) and toks[i].startswith("-"):
                opt = toks[i]
                i += 1
                if b == "sudo" and opt in ("-u", "-g", "-C", "-D", "-h", "-p", "-U") and i < len(toks):
                    i += 1
                elif b in ("timeout", "gtimeout") and opt in ("-s", "-k", "--signal", "--kill-after") and i < len(toks):
                    i += 1
                elif b == "env" and opt in ("-u", "-C", "-S") and i < len(toks):
                    i += 1
                elif b == "nice" and opt == "-n" and i < len(toks):
                    i += 1
            if b in ("timeout", "gtimeout") and i < len(toks):
                i += 1  # duration
            continue
        break
    return toks[i:]


def _match(toks: List[str]) -> Optional[str]:
    if not toks:
        return None
    b = _base(toks[0])
    rest = toks[1:]
    if b in ("pi", "pi-acp"):
        return b
    if b == "opencode" and "run" in rest:
        return "opencode run"
    if b == "codex" and "exec" in rest:
        return "codex exec"
    if b == "claude" and any(a in ("-p", "--print") or a.startswith("--print=") for a in rest):
        return "claude -p"
    if b in ("npx", "bunx", "pnpx") and rest:
        inner = [t for t in rest if not t.startswith("-")]
        if inner:
            pkg = inner[0]
            at = pkg.find("@", 1)
            if at > 0:
                pkg = pkg[:at]  # drop @version
            name = _NPX_PKGS.get(pkg, _base(pkg))
            return _match([name] + inner[1:])
    return None


def find_agent_cli(command: str, _depth: int = 0) -> Optional[str]:
    """Return the agent-CLI label launched by ``command``, or None."""
    if not command or _depth > 4:
        return None
    for seg in _segments(command):
        toks = _strip_prefixes(seg)
        if not toks:
            continue
        if _base(toks[0]) in _SHELLS:
            for j, t in enumerate(toks[1:], 1):
                if t.startswith("-") and "c" in t.lstrip("-") and not t.startswith("--") and j + 1 < len(toks):
                    hit = find_agent_cli(toks[j + 1], _depth + 1)
                    if hit:
                        return hit
                    break
            continue
        if _base(toks[0]) == "eval":
            hit = find_agent_cli(" ".join(toks[1:]), _depth + 1)
            if hit:
                return hit
            continue
        hit = _match(toks)
        if hit:
            return hit
    return None


def denial_message(binary: str) -> str:
    return (
        f"agent-cli-fence: this session is not a kanban worker (no {KANBAN_TASK_ENV}), so launching "
        f"`{binary}` from the terminal is denied. Coding agents run only as kanban workers: create or "
        f"update a kanban card and let the dispatcher spawn the worker."
    )


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            return False
    sect = (config or {}).get(CONFIG_KEY)
    if isinstance(sect, bool):
        return sect
    return bool(isinstance(sect, dict) and sect.get("enabled"))


def check(tool_name: str, args: Dict[str, Any], *, env: Optional[Dict[str, str]] = None,
          config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return a denial message if the call must be blocked, else None."""
    if tool_name != "terminal" or not isinstance(args, dict):
        return None
    env = os.environ if env is None else env
    if env.get(KANBAN_TASK_ENV):
        return None
    command = args.get("command")
    if not isinstance(command, str):
        return None
    hit = find_agent_cli(command)
    if not hit:
        return None
    if not is_enabled(config):
        return None
    return denial_message(hit)
