import os

class APIKeyProvider:
    api_key: str = ""
    def __init__(self, env_var: str = "OPENAI_API_KEY"):
        # Resolve API key from environment once and expose it via `get()`.
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(
                f"{env_var} environment variable not found."
            )
        self.api_key = api_key.strip()

    def get(self) -> str:
        return self.api_key
