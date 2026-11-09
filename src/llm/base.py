"""Abstract base class for schema-constrained Structured Output LLM clients.

Method & Literature References:
- Constrained Decoding / FSM-Guided Structured Generation:
  [Efficient Guided Generation for Large Language Models (Willard & Louf, 2023, arXiv:2307.09702)](https://arxiv.org/abs/2307.09702)
- SGLang Compressed Finite State Machine (Jump-Forward Decoding):
  [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2024, arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseStructuredLLMClient(ABC):
    """Abstract interface for LLM clients supporting strict JSON-Schema constrained decoding."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate a validated Pydantic model instance via schema-constrained decoding."""
