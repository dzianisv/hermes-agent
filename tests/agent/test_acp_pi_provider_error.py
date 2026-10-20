"""Recover pi-acp's silent terminal errors without relabeling empty model replies."""

import json
import sys
from pathlib import Path

import pytest

from agent.copilot_acp_client import CopilotACPClient
from agent.error_classifier import FailoverReason, classify_api_error


ERROR = "Codex error: The usage limit has been reached"


@pytest.fixture
def pi_client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(home))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
    monkeypatch.setattr(Path, "home", lambda: home)
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    server = tmp_path / "adapter.py"
    server.write_text('''import json, os, re, sys, time
from pathlib import Path

scenario = sys.argv[1]
for line in sys.stdin:
    req = json.loads(line)
    method = req["method"]
    result = {}
    if method == "session/new":
        cwd = req["params"]["cwd"]
        result = {"sessionId": "acp-session"}
    elif method == "session/prompt":
        slug = "--" + re.sub(r"[/\\\\:]", "-", cwd.lstrip("/\\\\")) + "--"
        directory = Path(os.environ["HOME"]) / ".pi/agent/sessions" / slug
        directory.mkdir(parents=True)
        prompt = req["params"]["prompt"]
        records = [
            {"type": "session", "id": "pi-session", "cwd": cwd},
            {"type": "message", "message": {"role": "user", "content": prompt}},
            {"type": "message", "message": {"role": "assistant", "content": [],
                "timestamp": time.time() * 1000,
                "stopReason": "stop" if scenario == "benign" else "error",
                "errorMessage": "Codex error: The usage limit has been reached"}},
        ]
        if scenario == "other-prompt":
            records[1]["message"]["content"] = [{"type": "text", "text": "another task"}]
        if scenario == "wrong-cwd":
            records[0]["cwd"] = cwd + "/other"
        if scenario == "touched-old-error":
            records[-1]["message"]["timestamp"] = 1000
        if scenario == "old-error":
            records.append({"type": "message", "message": {
                "role": "assistant", "content": [], "stopReason": "stop"}})
        for name in (["one", "two"] if scenario == "ambiguous" else ["one"]):
            path = directory / (name + ".jsonl")
            path.write_text("".join(json.dumps(r) + "\\n" for r in records))
            if scenario == "stale":
                os.utime(path, (1, 1))
            if scenario == "malformed":
                path.write_text("not json\\n")
        result = {"stopReason": "end_turn"}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''', encoding="utf-8")

    def make(scenario, base_url="acp://pi"):
        return CopilotACPClient(base_url=base_url, command=sys.executable,
                                args=[str(server), scenario], acp_cwd=str(cwd))
    return make


def test_silent_pi_failure_preserves_provider_cause(pi_client):
    with pytest.raises(ConnectionError, match=ERROR) as caught:
        pi_client("error").chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    classified = classify_api_error(caught.value)
    # Quota wording is more specific than the ConnectionError transport fallback.
    assert classified.reason is FailoverReason.billing
    assert ERROR in classified.message
    assert classified.should_fallback


@pytest.mark.parametrize("scenario,base_url", [
    ("benign", "acp://pi"),
    ("stale", "acp://pi"),
    ("other-prompt", "acp://pi"),
    ("wrong-cwd", "acp://pi"),
    ("old-error", "acp://pi"),
    ("touched-old-error", "acp://pi"),
    ("ambiguous", "acp://pi"),
    ("malformed", "acp://pi"),
    ("error", "acp://copilot"),
])
def test_empty_or_uncorrelated_turn_stays_empty(pi_client, scenario, base_url):
    completion = pi_client(scenario, base_url).chat.completions.create(
        messages=[{"role": "user", "content": "hi"}])
    choice = completion.choices[0]
    assert not choice.message.content
    assert not choice.message.tool_calls
    assert choice.finish_reason == "stop"
    assert completion.usage.total_tokens == 0
