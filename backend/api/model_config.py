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
#  Fallback model chains (ordered by August 2026 benchmark composite score)
#  Each role maps to a list: [primary, fallback_1, fallback_2, ...]
#
#  Re-benchmarked: 2026-08-11 on 8 live models (4 SLM, 4 LLM).
#  Judge: llama-3.3-70b-versatile (not under test → no TPM conflict).
#  Composite = (ROUGE1 + BLEU + METEOR + Faithfulness + (1-Hallucination)) / 5
#
#  SLM ranking (non-deprecated):
#    compound-mini 0.671 > gpt-oss-20b 0.628 > allam-2-7b 0.614
#  LLM ranking (non-deprecated, all 5/5 cases):
#    gpt-oss-120b 0.548 > qwen3.6-27b 0.522
#    (groq/compound 0.628 but only 4/5 — uses gpt-oss-120b internally,
#     same TPM bucket; kept as last-resort only)
#
#  Deprecated models removed (shutdown Aug 16 2026):
#    llama-3.1-8b-instant (was best SLM at 0.739 — no longer available)
#    llama-3.3-70b-versatile (was #2 LLM at 0.610 — no longer available)
# ---------------------------------------------------------------------------
MODEL_FALLBACKS: dict[str, list[str]] = {
    # SLM — intent recognition  (Stage 1)
    # Benchmark Aug 11 2026: compound-mini > gpt-oss-20b > allam-2-7b
    "slm_intent": [
        "groq/compound-mini",    # composite=0.671, R1=0.589, Faith=0.88, free-tier
        "openai/gpt-oss-20b",    # composite=0.628, R1=0.571, Faith=0.94
        "allam-2-7b",            # composite=0.614, R1=0.584, Faith=0.78, fastest
    ],
    # LLM — legal reasoning  (Stage 2)
    # Benchmark Aug 11 2026: gpt-oss-120b > qwen3.6-27b (both 5/5 complete)
    # groq/compound uses gpt-oss-120b internally → same TPM; kept as last-resort
    "llm_reasoning": [
        "openai/gpt-oss-120b",   # composite=0.548, R1=0.480, Faith=0.76  (5/5)
        "qwen/qwen3.6-27b",      # composite=0.522, R1=0.415, Faith=0.88  (5/5)
        "groq/compound",         # composite=0.628 (4/5 cases only) — last-resort
    ],
    # Summarisation / Stage-2 helpers
    # Prefer fast SLM; escalate to LLM if quality demands it
    "summarisation": [
        "groq/compound-mini",    # best SLM, free-tier
        "openai/gpt-oss-20b",    # reliable paid SLM fallback
        "qwen/qwen3.6-27b",      # LLM quality fallback
    ],
    # Q&A — PrecedentQA
    "qa": [
        "groq/compound-mini",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
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
