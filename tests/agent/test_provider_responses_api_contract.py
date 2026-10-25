"""Routing callers may supply credentials without changing non-Copilot decisions."""
import pytest

from run_agent import AIAgent


@pytest.mark.parametrize("api_key", [None, "test-routing-key"])
@pytest.mark.parametrize("model, provider, expected", [
    ("gpt-5", "openrouter", True),
    ("gpt-4o", "openrouter", False),
    ("gpt-5", "custom", False),
])
def test_responses_routing_accepts_api_key_keyword(model, provider, expected, api_key):
    route = AIAgent._provider_model_requires_responses_api
    assert route(model, provider=provider, api_key=api_key) is expected
    assert route(model, provider=provider) is expected
