"""Thin OpenAI-compatible chat client (works with Ollama, vLLM, OpenAI, Gemini's OpenAI endpoint...)."""
from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings


@dataclass
class LLMReply:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ChatLLM:
    def __init__(self, model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or settings.LLM_MODEL_NAME
        self.temperature = settings.TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or settings.MAX_TOKENS
        self.client = OpenAI(base_url=base_url or settings.LLM_API_BASE,
                             api_key=api_key or settings.LLM_API_KEY, timeout=300, max_retries=1)

    def chat(self, messages: list[dict]) -> LLMReply:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        usage = resp.usage
        return LLMReply(
            content=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.perf_counter() - t0,
        )
