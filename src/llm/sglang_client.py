"""SGLang OpenAI-compatible client for schema-constrained Structured Outputs.

Method & Literature References:
- SGLang Structured Outputs & RadixAttention:
  [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2024, arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)
"""

from __future__ import annotations

import json
import urllib.request
from pydantic import BaseModel

from src.llm.base import BaseStructuredLLMClient


class SGLangStructuredClient(BaseStructuredLLMClient):
    """OpenAI-compatible client for local SGLang (`Qwen3.8-Flash-Next`) structured outputs."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "Qwen3.8-Flash-Next",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Call the local SGLang endpoint with strict JSON-Schema constrained decoding."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "strict": True,
                    "schema": schema_cls.model_json_schema(),
                },
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return schema_cls.model_validate_json(content)
