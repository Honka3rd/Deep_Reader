from llama_index.llms.openai import OpenAI

from api_key_provider import APIKeyProvider
from llm_provider import LLMProvider


class OpenAILLMProvider(LLMProvider):
    llm: OpenAI

    def __init__(self, api_key_provider: APIKeyProvider, model: str = "gpt-4.1-mini"):
        api_key = api_key_provider.get()

        self.llm = OpenAI(
            model=model,
            api_key=api_key,
        )

    def complete_text(self, prompt: str) -> str:
        print("OpenAI LLM:prompt", prompt)
        response = self.llm.complete(prompt)
        return response.text.strip()