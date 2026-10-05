"""pi coding agent as an ACP provider (npm ``pi-acp`` adapter over stdio).

Reuses the generic ACP stdio client from copilot-acp: ``pi-acp`` speaks ACP v1
(``initialize`` / ``session/new`` with a ``model`` config option / ``session/prompt``),
so only the launch command differs. pi owns model auth (``~/.pi``).
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class PiACPProfile(ProviderProfile):
    """pi via pi-acp — external process, models advertised by ``session/new``."""

    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.copilot_acp_client import CopilotACPClient

        return CopilotACPClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 15.0
    ) -> list[str] | None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        try:
            creds = resolve_external_process_provider_credentials(self.name)
            client = self.create_client(
                api_key=creds.get("api_key"), base_url=creds.get("base_url"),
                command=creds.get("command"), args=creds.get("args"))
            return client.list_models(timeout_seconds=timeout) or None
        except Exception:
            return None


pi_acp = PiACPProfile(
    name="pi-acp", aliases=("pi",),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://pi",
    auth_type="external_process",
    process_command="pi-acp",
    process_args=(),
    process_command_env_vars=("HERMES_PI_ACP_COMMAND",),
    process_args_env_var="HERMES_PI_ACP_ARGS",
)

register_provider(pi_acp)
