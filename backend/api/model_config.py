from __future__ import annotations

import json
import os
import time
from pathlib import Path

from groq import Groq, RateLimitError, APIStatusError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_file() -> Path:
    return _repo_root() / "output" / "model_benchmark_latest.json"


# ---------------------------------------------------------------------------
#  Fallback model chains (ordered by benchmark composite score)
#  Each role maps to a list: [primary, fallback_1, fallback_2, ...]
#
#  Active Groq Models:
#    - SLMs: openai/gpt-oss-20b, allam-2-7b
#    - LLMs: openai/gpt-oss-120b, qwen/qwen3.8-27b
#
#  Deprecated / Inactive models removed:
#    - llama-3.1-8b-instant & llama-3.3-70b-versatile (shut down)
#    - groq/compound-mini & groq/compound (unavailable)
#    - qwen/qwen3.6-27b (superseded by qwen/qwen3.8-27b)
# ---------------------------------------------------------------------------
MODEL_FALLBACKS: dict[str, list[str]] = {
    # SLM — intent recognition  (Stage 1)
    "slm_intent": [
        "openai/gpt-oss-20b",    # composite=0.628, 131k context, fast structured output
        "allam-2-7b",            # composite=0.614, 4k context, fast fallback
    ],
    # LLM — legal reasoning  (Stage 2)
    "llm_reasoning": [
        "qwen/qwen3.8-27b",      # composite=0.731, F1=0.603, 100% JSON valid, 0.44s latency
        "openai/gpt-oss-120b",   # composite=0.610, 120B high-capacity fallback
    ],
    # Summarisation / Stage-2 helpers
    "summarisation": [
        "openai/gpt-oss-20b",    # reliable fast SLM
        "qwen/qwen3.8-27b",      # LLM quality fallback
    ],
    # Q&A — PrecedentQA
    "qa": [
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ],
}


def get_fallback_chain(role: str) -> list[str]:
    """Return the ordered fallback chain for a given role."""
    return list(MODEL_FALLBACKS.get(role, ["openai/gpt-oss-20b"]))


# ---------------------------------------------------------------------------
#  Retry-with-fallback for raw Groq SDK calls
# ---------------------------------------------------------------------------
_RETRYABLE_CODES = {429, 503, 529}  # rate-limit, overloaded, over-capacity


def groq_chat_with_fallback(
    client: Groq,
    *,
    role: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 300,
    max_retries: int = 1,
) -> str:
    """
    Call Groq chat completions, automatically falling back to the next model
    in the chain on rate-limit / over-capacity / out-of-tokens errors.

    Returns the assistant content string.
    Raises the last exception if ALL models in the chain fail.
    """
    chain = get_fallback_chain(role)
    last_exc: Exception | None = None

    for model in chain:
        for attempt in range(1, max_retries + 2):  # 1 try + max_retries
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                last_exc = e
                wait = min(2 ** attempt, 8)
                print(f"[Fallback] {model} rate-limited, retry {attempt} in {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                last_exc = e
                if e.status_code in _RETRYABLE_CODES:
                    print(f"[Fallback] {model} returned {e.status_code}, trying next model...")
                    break  # skip to next model
                raise  # non-retryable status → propagate immediately
            except Exception as e:
                # Catch-all: check if message contains capacity/token keywords
                msg = str(e).lower()
                if any(kw in msg for kw in ("rate_limit", "capacity", "overloaded", "tokens per")):
                    last_exc = e
                    print(f"[Fallback] {model} error ({e}), trying next model...")
                    break
                raise

    # All models exhausted
    raise last_exc or RuntimeError("All fallback models failed")


# ---------------------------------------------------------------------------
#  Legacy helper (backward compat)
# ---------------------------------------------------------------------------
def get_preferred_groq_model(default_model: str) -> str:
    """
    Resolve Groq model in this priority order:
    1) `GROQ_MODEL` env var
    2) `output/model_benchmark_latest.json` -> `best_model`
    3) fallback `default_model`
    """
    env_model = os.environ.get("GROQ_MODEL", "").strip()
    if env_model:
        return env_model

    benchmark_path = _benchmark_file()
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = str(data.get("best_model", "")).strip()
            if model:
                return model
        except Exception:
            pass

    return default_model
