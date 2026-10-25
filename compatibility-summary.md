# Live retry compatibility

## Outcome

- Added routing-keyword regression coverage on installed, whose routing predicate already accepts `api_key`.
- Made the TUI status callback's `(kind, text=None)` contract explicit. Both text-only and two-argument producers still reach the existing presentation sink before marker-to-string conversion.
- Added behavior tests for real `status.update` frames, provider-retry suppression, broad warning suppression, unrelated warnings, and identical unmarked requested text.
- `compatibility.patch` changes only the live routing signature and callback wiring, and adds the two focused test files. The live routing method body is preserved unchanged.

## Verification

Executed from `/Users/engineer/workspace/hermes-silent-rate-limit`:

```sh
HERMES_PYTHON=/Users/engineer/.hermes/hermes-agent/.venv/bin/python scripts/run_tests.sh tests/agent/test_provider_responses_api_contract.py tests/tui_gateway/test_status_callback_contract.py -j 2 -v --tb=short
```

Result: **2 files, 12 tests passed, 0 failed**, runner wall time 4.9 seconds. No full-suite run.

```sh
git -C /Users/engineer/.hermes/hermes-agent apply --check /Users/engineer/workspace/hermes-silent-rate-limit/compatibility.patch
```

Result: exit 0 against the current already-patched live checkout. The patch was not applied.

## Exact deployment paths

- Incremental artifact: `/Users/engineer/workspace/hermes-silent-rate-limit/compatibility.patch`
- Summary: `/Users/engineer/workspace/hermes-silent-rate-limit/compatibility-summary.md`
- Target checkout: `/Users/engineer/.hermes/hermes-agent`
- Target files: `run_agent.py`, `tui_gateway/agent_callbacks.py`, `tests/agent/test_provider_responses_api_contract.py`, `tests/tui_gateway/test_status_callback_contract.py`.
- Coordinator's isolated tree from `/tmp/hermes-silent-retry-live-path`: `/tmp/hermes-silent-retry-lean-y689ru_9` (read-only; not changed).

## Observations / remaining work

The supplied `reports/silent-retry-20261025/live-tests.log` was absent in this worktree. Direct inspection confirmed the live routing signature mismatch. Both current live and isolated callback lambdas already accepted optional text, unlike the reported earlier one-argument shape; the incremental patch preserves that compatibility explicitly rather than replacing the live callback module. Installed `run_agent.py` needed no source edit.

No live source, isolated source, active configuration, credentials, services, or dependencies were changed. The coordinator owns deployment and the live retry-test rerun.
