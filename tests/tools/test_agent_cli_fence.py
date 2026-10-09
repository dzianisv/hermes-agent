import pytest

from tools import agent_cli_fence as f

ON = {"agent_cli_fence": {"enabled": True}}
OFF = {}
WORKER = {"HERMES_KANBAN_TASK": "t_123"}

DENY = [
    ("pi 'fix bug'", "pi"),
    ("pi-acp --stdio", "pi-acp"),
    ("opencode run 'do it'", "opencode run"),
    ("codex exec --full-auto 'x'", "codex exec"),
    ("claude -p 'hello'", "claude -p"),
    ("claude --print hi", "claude -p"),
    ("/usr/local/bin/codex exec x", "codex exec"),
    ("FOO=1 BAR=2 codex exec x", "codex exec"),
    ("env -u X FOO=1 claude -p hi", "claude -p"),
    ("timeout 600 opencode run x", "opencode run"),
    ("gtimeout -k 5 300 pi x", "pi"),
    ("nohup codex exec x > log 2>&1 &", "codex exec"),
    ("sudo -u bob claude -p hi", "claude -p"),
    ("nohup env A=1 timeout 60 pi x", "pi"),
    ("cd /repo && codex exec x", "codex exec"),
    ("echo hi | claude -p", "claude -p"),
    ("ls; pi go", "pi"),
    ("false || opencode run x", "opencode run"),
    ("bash -c 'cd /r && codex exec x'", "codex exec"),
    ("bash -lc \"timeout 60 claude -p hi\"", "claude -p"),
    ("sh -c 'sh -c \"pi x\"'", "pi"),
    ("(cd r && pi x)", "pi"),
    ("npx -y @openai/codex@latest exec x", "codex exec"),
]

ALLOW = [
    "ls -la",
    "git commit -m 'pi run codex exec'",
    "echo 'claude -p hi'",
    "grep -r 'opencode run' .",
    "claude --version",
    "codex --help",
    "opencode serve",
    "pip install foo",
    "python3 -c 'print(1)'",
    "bash -c 'echo codex exec'",
    "cat pi.txt",
    "hermes kanban list",
]


@pytest.mark.parametrize("cmd,label", DENY)
def test_deny_non_worker(cmd, label):
    msg = f.check("terminal", {"command": cmd}, env={}, config=ON)
    assert msg is not None, cmd
    assert msg.startswith(
        "agent-cli-fence: this session is not a kanban worker (no HERMES_KANBAN_TASK), so launching "
        f"`{label}`"
    )


@pytest.mark.parametrize("cmd,label", DENY)
def test_worker_allowed(cmd, label):
    assert f.check("terminal", {"command": cmd}, env=WORKER, config=ON) is None


@pytest.mark.parametrize("cmd,label", DENY)
def test_disabled_by_default(cmd, label):
    assert f.check("terminal", {"command": cmd}, env={}, config=OFF) is None


@pytest.mark.parametrize("cmd", ALLOW)
def test_allow_benign(cmd):
    assert f.check("terminal", {"command": cmd}, env={}, config=ON) is None


def test_other_tools_ignored():
    assert f.check("execute_code", {"command": "codex exec x"}, env={}, config=ON) is None


def test_bool_config_and_disabled_dict():
    assert f.is_enabled({"agent_cli_fence": True})
    assert not f.is_enabled({"agent_cli_fence": {"enabled": False}})
