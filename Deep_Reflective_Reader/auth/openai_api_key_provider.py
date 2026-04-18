import os

from auth.api_key_provider import APIKeyProvider


class OpenAIAPIKeyProvider(APIKeyProvider):
    """Load OpenAI API key from environment and expose it to providers."""
    api_key: str = ""
    def __init__(self, env_var: str = "OPENAI_API_KEY"):
        # Resolve API key from environment once and expose it via `get()`.
        """Initialize object state and injected dependencies.

Args:
    env_var: Environment variable name storing API key.
"""
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(
                f"{env_var} environment variable not found."
            )
        self.api_key = api_key.strip()

    def get(self) -> str:
        """Return configured value from provider.

Returns:
    OpenAI API key string resolved during initialization."""
        return self.api_key
