"""Behavior contracts for opt-in free-first provider routing."""
import json
import threading
from types import SimpleNamespace

import pytest

from agent import smart_provider_fallback as smart
from agent.openrouter_free_fallback import _model_is_free, _model_supports_tools


@pytest.mark.parametrize("message,account", [
    ("Rate limit exceeded: free-models-per-day", True),
    ("Daily quota exceeded for free tier", True),
    ("Provider daily quota exceeded for model", False),
    ("Provider returned 429: capacity exceeded", False),
])
def test_quota_scope(message, account):
    assert smart.account_quota_error(Exception(message)) is account
    upstream = Exception("upstream")
    upstream.body = {"error": {"message": "Provider returned error", "metadata": {
        "raw": "Daily quota exceeded for free tier", "provider_name": "upstream"}}}
    assert not smart.account_quota_error(upstream)


@pytest.mark.parametrize("model,eligible", [
    ("gpt-6-sol", True), ("gpt-6.1", True), ("gpt-12-codex", True),
    ("gpt-5.9-codex", False), ("gpt-60evil", False), ("gpt-6-sol/paid", False),
])
def test_codex_generation(model, eligible):
    assert smart.eligible_codex(model) is eligible


def test_catalog_free_fail_closed():
    for price in ({}, {"prompt": "0"}, {"prompt": "bad", "completion": "0"},
                  {"prompt": "0", "completion": "0.001"}):
        assert not _model_is_free({"pricing": price})
    assert _model_is_free({"pricing": {"prompt": "0", "completion": "0"}})
    assert not _model_supports_tools({})
    assert _model_supports_tools({"supported_parameters": ["tools"]})


def test_request_policy_preserves_prompt_messages_and_tools(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import rewrite_prompt_model_identity
    monkeypatch.setattr(smart, "enabled", lambda: True)
    monkeypatch.setattr(smart, "_path", lambda key: tmp_path / "quota.json")
    agent = SimpleNamespace(provider="openrouter", model="a:free", api_key="test-account",
                            _smart_free_available=True, _smart_free_models={"a:free"},
                            _cached_system_prompt="Model: primary\nProvider: openrouter")
    prompt = agent._cached_system_prompt
    history = [{"role": "assistant", "tool_calls": [{"id": "t", "type": "function",
                "function": {"name": "test", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "t", "content": "result"}]
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    kwargs = {"model": "paid-model", "messages": history, "tools": tools,
              "extra_body": {"models": ["paid-model"], "provider": {"max_price": {"prompt": 99}}}}
    result = smart.constrain_request(agent, kwargs)
    assert result["messages"] is history and result["tools"] is tools
    assert result["model"] == "a:free"
    assert "models" not in result["extra_body"]
    assert result["extra_body"]["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    rewrite_prompt_model_identity(agent, "gpt-6-sol", "openai-codex")
    assert agent._cached_system_prompt == prompt
    agent._smart_free_available = False
    with pytest.raises(RuntimeError):
        smart.constrain_request(agent, kwargs)


def test_shared_probe_throttle_concurrency_and_recovery(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"blocked": True, "next_probe": 0}))
    monkeypatch.setattr(smart, "_path", lambda key: path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def probe(key, model):
        calls.append((key, model))
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(smart, "_probe", probe)
    free = [{"provider": "openrouter", "model": "a:free"}]
    results = []
    thread = threading.Thread(target=lambda: results.append(smart.free_available("key", free)))
    thread.start()
    try:
        assert entered.wait(5)
        assert not smart.free_available("key", free)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert results == [True]
    assert smart.free_available("key", free)
    assert len(calls) == 1
    path.write_text(json.dumps({"blocked": True, "next_probe": 0}))
    monkeypatch.setattr(smart, "_probe", lambda *args: False)
    assert not smart.free_available("key", free)
    state = json.loads(path.read_text())
    assert state["blocked"] and state["next_probe"] > smart.time.time()
    assert not smart.free_available("key", free)


def test_profile_and_credential_isolation(tmp_path, monkeypatch):
    import hermes_constants
    import agent.openrouter_free_fallback as free
    home = tmp_path / "product-lead-agentpod"
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(free, "_read_config", lambda: {"smart_provider_fallback": True})
    assert smart.enabled()
    a = smart._path("account-a")
    assert a != smart._path("account-b")
    home = tmp_path / "other"
    assert not smart.enabled()
    assert smart._path("account-a") != a
    home = tmp_path / "product-lead-agentpod"
    assert smart._path("account-a") == a


@pytest.mark.parametrize("status", [200, 401, 403, 429, 503])
def test_catalog_unavailable_never_guesses(monkeypatch, status):
    import hermes_cli.auth_codex as auth
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", lambda: {})
    assert smart.codex_models() == []
    import agent.model_metadata as metadata
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", lambda: {
        "api_key": "unavailable-catalog-test", "base_url": "https://example.invalid"})
    monkeypatch.setattr(metadata, "_codex_catalog_probe_allowed", lambda *a: True)
    calls = []
    monkeypatch.setattr(metadata, "fetch_codex_catalog_entries", lambda *a, **kw: (
        calls.append(True) or [], status))
    smart._codex_cache.clear()
    assert smart.codex_models() == []
    assert smart.codex_models() == []
    assert len(calls) == 1  # No curated/forward-compat guesses or catalog retry storm.


def test_prepare_keeps_healthy_codex_on_failed_probe(tmp_path, monkeypatch):
    import agent.openrouter_free_fallback as free
    monkeypatch.setattr(smart, "enabled", lambda: True)
    monkeypatch.setattr(smart, "_key", lambda agent=None: "key")
    monkeypatch.setattr(smart, "_path", lambda key: tmp_path / "state.json")
    smart._path("key").write_text(json.dumps({"blocked": True, "next_probe": 0}))
    monkeypatch.setattr(free, "fetch_free_openrouter_models", lambda **kw: [{"provider": "openrouter", "model": "a:free"}])
    monkeypatch.setattr(smart, "codex_models", lambda: ["gpt-6-sol"])
    monkeypatch.setattr(smart, "_probe", lambda *a: (_ for _ in ()).throw(TimeoutError()))
    agent = SimpleNamespace(provider="openai-codex", model="gpt-6-sol",
                            _primary_runtime={"provider": "openrouter", "model": "a:free"})
    smart.prepare_turn(agent)
    assert agent._fallback_chain == [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    assert smart.skip_candidate(agent, "openrouter", "a:free")
    monkeypatch.setattr(smart, "_probe", lambda *a: True)
    smart._path("key").write_text(json.dumps({"blocked": True, "next_probe": 0}))
    switches = []
    agent._try_activate_fallback = lambda: switches.append(agent._fallback_chain[0]) or True
    smart.prepare_turn(agent)
    assert switches == [{"provider": "openrouter", "model": "a:free"}]
    assert agent._rate_limited_until == 0
    monkeypatch.setattr(free, "fetch_free_openrouter_models", lambda **kw: [])
    smart.prepare_turn(agent)
    assert len(switches) == 1  # A failed free catalog cannot interrupt healthy Codex.


def test_discovery_recovers_stale_codex_cooldown(tmp_path, monkeypatch):
    import hermes_cli.auth as auth
    import agent.model_metadata as metadata
    store = {"version": 1, "providers": {}, "credential_pool": {"openai-codex": [{
        "id": "test-quota", "label": "test", "auth_type": "oauth", "priority": 0,
        "source": "device_code", "access_token": "test-token", "last_status": "exhausted",
        "last_status_at": smart.time.time(), "last_error_code": 429,
        "last_error_reason": "usage_limit_reached", "last_error_message": "usage limit reached",
        "last_error_reset_at": smart.time.time() + 86400,
    }]}}
    (tmp_path / "auth.json").write_text(json.dumps(store))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    probes = []
    import hermes_cli.auth_codex as auth_codex
    probe = lambda token, **kw: probes.append(token) or True
    monkeypatch.setattr(auth, "_probe_codex_quota_restored", probe)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", probe)
    monkeypatch.setattr(metadata, "_codex_catalog_probe_allowed", lambda *a: True)
    monkeypatch.setattr(metadata, "fetch_codex_catalog_entries", lambda *a, **kw: ([
        {"slug": "gpt-5-codex"}, {"slug": "gpt-7-codex"}], 200))
    smart._codex_cache.clear()
    from hermes_cli.auth_codex import resolve_codex_runtime_credentials
    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "test-token"
    assert smart.codex_models() == ["gpt-7-codex"]
    assert probes
    persisted = json.loads((tmp_path / "auth.json").read_text())
    assert persisted["credential_pool"]["openai-codex"][0]["last_error_reset_at"] is None


def test_real_http_catalog_and_inference_probe(tmp_path, monkeypatch):
    """Exercise real HTTP/JSON and persistence, not a mocked route selector."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import hermes_cli.auth_codex as auth
    import hermes_constants as constants
    from agent import secret_scope

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = {"models": [
                {"slug": "gpt-5.9-codex"}, {"slug": "gpt-6-test"},
                {"slug": "gpt-7-hidden", "visibility": "hidden"},
                {"slug": "gpt-8-codex", "priority": 1},
            ]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(data)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", lambda: {"api_key": "test-account", "base_url": base})
    monkeypatch.setattr(smart, "_PROBE_URL", base + "/chat/completions")
    smart._codex_cache.clear()
    home = tmp_path / "product-lead-agentpod"
    home.mkdir()
    (home / "config.yaml").write_text("openrouter_free_fallback:\n  enabled: true\n  smart_provider_fallback: true\n")
    token = constants.set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({"OPENROUTER_API_KEY": "test-account"})
    secret_scope.set_multiplex_active(True)
    try:
        assert smart.enabled()
        assert smart.codex_models() == ["gpt-8-codex", "gpt-6-test"]
        agent = SimpleNamespace(provider="openrouter", api_key="test-account")
        future = smart.time.time() + 1200
        assert smart.record_error(agent, Exception("free-models-per-day exceeded"), {"reset_at": future})
        path = smart._path("test-account")
        assert json.loads(path.read_text())["next_probe"] >= future
        free = [{"provider": "openrouter", "model": "test:free"}]
        assert not smart.free_available("test-account", free)
        assert requests == []
        path.write_text(json.dumps({"blocked": True, "next_probe": 0}))
        assert smart.free_available("test-account", free)
        assert requests[0]["model"] == "test:free"
        assert requests[0]["max_tokens"] <= 8
        assert requests[0]["messages"] == [{"role": "user", "content": "Say OK"}]
        assert not json.loads(path.read_text())["blocked"]
        other = tmp_path / "other"
        other.mkdir()
        other_token = constants.set_hermes_home_override(other)
        other_secret = secret_scope.set_secret_scope({"OPENROUTER_API_KEY": "other-account"})
        try:
            assert not smart.enabled()
            assert smart._key() == "other-account"
            assert smart._path("other-account") != path
        finally:
            secret_scope.reset_secret_scope(other_secret)
            constants.reset_hermes_home_override(other_token)
        assert smart.enabled() and smart._key() == "test-account"
    finally:
        secret_scope.reset_secret_scope(secret_token)
        secret_scope.set_multiplex_active(False)
        constants.reset_hermes_home_override(token)
        server.shutdown()
        server.server_close()
        thread.join(5)
