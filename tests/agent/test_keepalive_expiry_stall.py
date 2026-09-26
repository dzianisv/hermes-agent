"""Regression guard for the 90s stall fix (t_4e6e7743), scope cut by EM decision.

A reused httpx connection can already be dead when a middlebox drops it without
FIN/RST. httpx will still hand that socket out while it is inside
``keepalive_expiry``. 20s was above the ~15s idle-kill floor some NATs/LBs use;
10s is under it. This guards only the expiry value — the first-byte liveness
probe and pooled-connection retry from the original PR were dropped from this
branch (reviewer found they killed slow-but-alive replies and caused duplicate
LLM calls; Den accepted the narrower fix as non-critical).
"""

from __future__ import annotations

from agent.process_bootstrap import build_keepalive_http_client, close_shared_transports


def test_keepalive_expiry_is_under_middlebox_idle_kill():
    """10s expiry reaps idle sockets before a ~15s NAT/LB silently drops them."""
    close_shared_transports()
    client = build_keepalive_http_client("http://genai.example")
    assert client is not None
    try:
        expiries = {
            mount._pool._keepalive_expiry
            for mount in client._mounts.values()
            if mount is not None and getattr(mount, "_pool", None) is not None
        }
        assert expiries == {10.0}
        assert client.timeout.read is None  # SSE must stay unbounded at the client
    finally:
        client.close()
        close_shared_transports()
