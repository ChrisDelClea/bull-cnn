"""Google Gemini API Client for schema-constrained Structured Outputs (`BaseStructuredLLMClient`).

Uses IPv4 socket resolution to avoid Linux IPv6 timeouts against `generativelanguage.googleapis.com`
and reads `GEMINI_API_KEY` from `.env` or environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
from typing import Any
import urllib.error
import urllib.request
from pydantic import BaseModel

from src.llm.base import BaseStructuredLLMClient

# Force IPv4 resolution for Google Generative Language API to prevent IPv6 socket timeouts
_ORIG_GETADDRINFO = socket.getaddrinfo


def _ipv4_aware_getaddrinfo(
    host: str, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
) -> Any:
    if isinstance(host, str) and "googleapis.com" in host:
        return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
    return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_aware_getaddrinfo


def load_gemini_api_key(env_path: str | Path | None = None) -> str:
    """Safely load `GEMINI_API_KEY` from environment or `.env` without printing."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    candidates = []
    if env_path is not None:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([repo_root / ".env", Path.cwd() / ".env", Path.home() / ".env"])

    for path in candidates:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("GEMINI_API_KEY="):
                    val = s.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    raise RuntimeError("GEMINI_API_KEY not found in environment or .env file.")


def _clean_json_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic JSON Schema (with `$defs` / `additionalProperties`) into Gemini OpenAPI 3.0 subset."""
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_name, {}))
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("$defs", "title", "additionalProperties", "default"):
                    continue
                out[k] = _resolve(v)
            return out
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


class GeminiStructuredClient(BaseStructuredLLMClient):
    """Structured Output client strictly using `gemini-3.8-flash` with High-Effort thinking."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        fallback_models: list[str] | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_output_tokens: int = 65536,
        thinking_level: str = "HIGH",
    ) -> None:
        self.model = model
        # Strictly NO fallback to older/weaker models unless explicitly passed
        self.fallback_models = fallback_models if fallback_models is not None else []
        self.api_key = api_key or load_gemini_api_key()
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.api_base = "https://generativelanguage.googleapis.com/v1beta"

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate a validated Pydantic model instance via `gemini-3.8-flash` `responseSchema`."""
        clean_schema = _clean_json_schema_for_gemini(schema_cls.model_json_schema())
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]

        last_err: Exception | None = None
        for model_name in models_to_try:
            body: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": self.max_output_tokens,
                    "responseMimeType": "application/json",
                    "responseSchema": clean_schema,
                    "thinkingConfig": {"thinkingLevel": self.thinking_level},
                },
            }

            for attempt in range(6):
                req = urllib.request.Request(
                    f"{self.api_base}/models/{model_name}:generateContent",
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError(f"Empty candidates from {model_name}: {resp_data}")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    # Filter out thought parts if returned, keeping only the final JSON text part
                    text_parts = [
                        p.get("text", "")
                        for p in parts
                        if "text" in p and not p.get("thought", False)
                    ]
                    text_content = "".join(text_parts) if text_parts else "".join(
                        p.get("text", "") for p in parts if "text" in p
                    )
                    return schema_cls.model_validate_json(text_content)
                except urllib.error.HTTPError as e:
                    err_raw = e.read().decode("utf-8", errors="replace")
                    last_err = RuntimeError(f"HTTP {e.code} on {model_name}: {err_raw[:400]}")
                    if e.code in (429, 500, 502, 503, 504):
                        if e.code == 429 and ("PerDay" in err_raw or "PerModelPerDay" in err_raw):
                            break
                        if attempt < 5:
                            wait_s = 15.0 + 10.0 * attempt
                            print(
                                f"[{model_name}] Transient HTTP {e.code} (attempt {attempt+1}/6), retrying in {wait_s:.0f}s...",
                                flush=True,
                            )
                            time.sleep(wait_s)
                            continue
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < 3:
                        wait_s = 15.0 + 10.0 * attempt
                        print(
                            f"[{model_name}] Transient error {exc} (attempt {attempt+1}/6), retrying in {wait_s:.0f}s...",
                            flush=True,
                        )
                        time.sleep(wait_s)
                        continue
                    break

        raise RuntimeError(f"Gemini model {self.model} failed. Last error: {last_err}")
