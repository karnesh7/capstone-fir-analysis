#!/usr/bin/env python3
"""
auto_benchmark_models.py — Automated Groq Model Discovery, Benchmarking & Configuration
========================================================================================

Workflow:
  1. Queries live Groq models via `client.models.list()`.
  2. Filters out non-chat models (Whisper audio, Guard/Safeguard moderation, Orpheus TTS).
  3. Categorizes candidate models into SLM (<= 20B) and LLM (> 20B).
  4. Evaluates each candidate model across standard legal FIR test cases:
     - Stage 1: Intent Classification Accuracy
     - Stage 2: IPC Section Prediction (F1 Score)
     - JSON parse reliability & response latency
  5. Computes composite scores and automatically ranks models for each role:
     - slm_intent    (Primary SLM -> Fallback SLM)
     - llm_reasoning (Primary LLM -> Fallback LLM)
     - summarisation (Primary SLM -> Fallback LLM)
     - qa            (Primary SLM -> Fallback LLM)
  6. Displays a clean recommendation table and proposed `MODEL_FALLBACKS`.
  7. If run with `--apply`, automatically updates `backend/api/model_config.py`
     and `output/model_benchmark_latest.json`.

Usage:
    # Benchmark and view recommendations:
    python backend/evaluation/auto_benchmark_models.py

    # Quick mode (evaluates on first 2 test cases):
    python backend/evaluation/auto_benchmark_models.py --quick

    # Benchmark and automatically update backend/api/model_config.py:
    python backend/evaluation/auto_benchmark_models.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"
MODEL_CONFIG_PATH = REPO_ROOT / "backend" / "api" / "model_config.py"
BENCHMARK_LATEST_PATH = OUTPUT_DIR / "model_benchmark_latest.json"

# Load environment
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "backend" / ".env")
load_dotenv()


# ---------------------------------------------------------------------------
# Test Cases (Aligned with LexIR Pipeline)
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    case_id: str
    incident: str
    expected_intent: str
    expected_sections: list[str]


TEST_CASES: list[TestCase] = [
    TestCase(
        case_id="c1_armed_robbery",
        incident="Two men threatened a woman with a knife, snatched her gold chain and phone, causing cuts and bruises.",
        expected_intent="Robbery",
        expected_sections=["356", "394", "398", "324"],
    ),
    TestCase(
        case_id="c2_house_theft",
        incident="Unknown person entered a locked house at night and stole cash and jewellery without confronting anyone.",
        expected_intent="Theft",
        expected_sections=["380", "454"],
    ),
    TestCase(
        case_id="c3_dowry_cruelty",
        incident="Husband and in-laws repeatedly demanded dowry, threatened wife, and subjected her to mental and physical cruelty.",
        expected_intent="Cruelty",
        expected_sections=["498A", "506"],
    ),
    TestCase(
        case_id="c4_cheating_trust",
        incident="Business partner took money promising to deliver goods, forged documents, and dishonestly misappropriated funds.",
        expected_intent="Cheating",
        expected_sections=["406", "420", "467", "468"],
    ),
    TestCase(
        case_id="c5_kidnap_assault",
        incident="Victim was forcibly taken in a car, beaten, and threatened with death if she informed police.",
        expected_intent="Kidnapping",
        expected_sections=["365", "323", "506"],
    ),
]

EXCLUDED_PATTERNS = [
    r"^whisper",
    r"guard",
    r"safeguard",
    r"orpheus",
]


def should_exclude(model_id: str) -> bool:
    m = model_id.lower()
    return any(re.search(p, m) for p in EXCLUDED_PATTERNS)


def classify_size_tier(model_id: str) -> str:
    m = model_id.lower()
    if re.search(r"(\d+)m", m):
        return "SLM"
    bs = [int(x) for x in re.findall(r"(\d+)b", m)]
    if bs:
        return "SLM" if max(bs) <= 20 else "LLM"
    if "compound-mini" in m:
        return "SLM"
    return "LLM"


def extract_sections(text: str) -> list[str]:
    nums = re.findall(r"(?:section\s*)?(\d{2,4}[a-zA-Z]?)", text.lower())
    clean = []
    for n in nums:
        n = n.upper()
        if n not in clean:
            clean.append(n)
    return clean


def section_f1(pred: list[str], gold: list[str]) -> float:
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    tp = len(ps & gs)
    precision = tp / len(ps)
    recall = tp / len(gs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def intent_prompt(case: TestCase) -> str:
    return (
        "Classify the FIR incident into ONE primary legal intent from this list only: "
        "Robbery, Theft, Assault, Kidnapping, Murder, Cheating, Cruelty, Fraud, Sexual Offence, Threat. "
        "Return ONLY valid JSON: {\"primary_intent\": \"...\", \"confidence\": 0.0}.\n\n"
        f"Incident: {case.incident}"
    )


def reasoning_prompt(case: TestCase) -> str:
    return (
        "Given the FIR incident, return likely IPC sections relevant to prosecution. "
        "Return ONLY valid JSON: {\"ipc_sections\": [\"356\", \"394\"]}.\n\n"
        f"Incident: {case.incident}"
    )


# ---------------------------------------------------------------------------
# Resilient API Call with Rate-Limit Backoff & Reasoning Token Budget
# ---------------------------------------------------------------------------
def call_chat_json(client: Groq, model: str, prompt: str, max_tokens: int = 500, max_retries: int = 3) -> tuple[dict, float, bool]:
    """
    Call Groq chat completions with support for reasoning models (e.g. gpt-oss-20b/120b),
    which use reasoning_tokens before emitting content. Requires adequate max_tokens (>= 500).
    """
    for attempt in range(1, max_retries + 2):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON generator for Indian legal NLP. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                parsed = json.loads(raw)
                return parsed, elapsed, True
            except Exception:
                # Fallback: extract {...} substring
                match = re.search(r"(\{.*\})", raw, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1)), elapsed, True
                    except Exception:
                        pass
                return {"_raw": raw}, elapsed, False
        except RateLimitError as e:
            if attempt <= max_retries:
                wait = min(2 ** attempt, 8)
                print(f"      [429 RateLimit] {model} waiting {wait}s before retry {attempt}...")
                time.sleep(wait)
            else:
                return {"_error": str(e)}, 0.0, False
        except APIStatusError as e:
            if e.status_code in {500, 502, 503, 504} and attempt <= max_retries:
                time.sleep(2)
            else:
                return {"_error": str(e)}, 0.0, False
        except Exception as e:
            return {"_error": str(e)}, 0.0, False
    return {"_error": "Exhausted retries"}, 0.0, False


# ---------------------------------------------------------------------------
# Auto-Update model_config.py
# ---------------------------------------------------------------------------
def apply_to_model_config(chains: dict[str, list[str]], best_model_overall: str) -> bool:
    if not MODEL_CONFIG_PATH.exists():
        print(f"[Error] model_config.py not found at {MODEL_CONFIG_PATH}")
        return False

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    chains_repr = "MODEL_FALLBACKS: dict[str, list[str]] = {\n"
    for role, models in chains.items():
        chains_repr += f'    "{role}": [\n'
        for m in models:
            chains_repr += f'        "{m}",\n'
        chains_repr += "    ],\n"
    chains_repr += "}"

    pattern = r"MODEL_FALLBACKS:\s*dict\[str,\s*list\[str\]\]\s*=\s*\{.*?\n\}"
    new_content, count = re.subn(pattern, chains_repr, content, flags=re.DOTALL)
    if count == 0:
        print("[Warning] Could not match MODEL_FALLBACKS block in model_config.py")
        return False

    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    # Also update model_benchmark_latest.json
    try:
        if BENCHMARK_LATEST_PATH.exists():
            with open(BENCHMARK_LATEST_PATH, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_data["best_model"] = best_model_overall
            with open(BENCHMARK_LATEST_PATH, "w", encoding="utf-8") as f:
                json.dump(b_data, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to update {BENCHMARK_LATEST_PATH}: {e}")

    return True


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-benchmark Groq models and configure primary/fallbacks.")
    parser.add_argument("--apply", action="store_true", help="Automatically write top models into backend/api/model_config.py")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark on first 2 test cases only")
    parser.add_argument("--output", type=str, default="", help="Custom output JSON path")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[Error] GROQ_API_KEY is not set in environment or .env file.")
        return 1

    client = Groq(api_key=api_key)

    print("======================================================================")
    print("  Groq Live Model Discovery & Benchmark")
    print("======================================================================")

    # 1. Discover models
    try:
        live_models = client.models.list().data
        all_ids = sorted([m.id for m in live_models])
    except Exception as e:
        print(f"[Error] Failed to fetch models from Groq API: {e}")
        return 1

    candidates = [m for m in all_ids if not should_exclude(m)]
    slm_candidates = [m for m in candidates if classify_size_tier(m) == "SLM"]
    llm_candidates = [m for m in candidates if classify_size_tier(m) == "LLM"]

    print(f"Total Groq models discovered: {len(all_ids)}")
    print(f"Chat candidates: {len(candidates)} ({len(slm_candidates)} SLMs, {len(llm_candidates)} LLMs)")
    print(f"  SLMs: {slm_candidates}")
    print(f"  LLMs: {llm_candidates}\n")

    cases_to_run = TEST_CASES[:2] if args.quick else TEST_CASES
    print(f"Running benchmark on {len(cases_to_run)} test case(s)...")

    results = []
    for model in candidates:
        tier = classify_size_tier(model)
        print(f"\n--> Testing [{tier}] {model}...")

        intent_correct = 0
        parse_ok = 0
        latencies: list[float] = []
        f1_scores: list[float] = []
        total_calls = 0

        for case in cases_to_run:
            # Stage 1 Intent
            total_calls += 1
            out_i, lat_i, ok_i = call_chat_json(client, model, intent_prompt(case), max_tokens=400)
            latencies.append(lat_i)
            if ok_i:
                parse_ok += 1
                pred_intent = str(out_i.get("primary_intent", "")).strip().lower()
                if pred_intent == case.expected_intent.lower():
                    intent_correct += 1

            time.sleep(0.3)  # Gentle delay for rate limiting

            # Stage 2 Legal Reasoning
            total_calls += 1
            out_r, lat_r, ok_r = call_chat_json(client, model, reasoning_prompt(case), max_tokens=450)
            latencies.append(lat_r)
            if ok_r:
                parse_ok += 1
                pred_sections = extract_sections(json.dumps(out_r, ensure_ascii=False))
                f1_scores.append(section_f1(pred_sections, case.expected_sections))
            else:
                f1_scores.append(0.0)

            time.sleep(0.3)

        intent_acc = intent_correct / len(cases_to_run)
        reasoning_f1 = statistics.mean(f1_scores) if f1_scores else 0.0
        parse_rate = parse_ok / max(1, total_calls)
        avg_lat = statistics.mean(latencies) if latencies else 999.0
        composite = 0.45 * intent_acc + 0.45 * reasoning_f1 + 0.10 * parse_rate

        results.append({
            "model": model,
            "tier": tier,
            "intent_accuracy": round(intent_acc, 3),
            "reasoning_f1": round(reasoning_f1, 3),
            "parse_rate": round(parse_rate, 3),
            "avg_latency_sec": round(avg_lat, 2),
            "composite_score": round(composite, 3),
        })

    # Sort rankings
    ranked_slms = sorted([r for r in results if r["tier"] == "SLM"], key=lambda x: (x["composite_score"], -x["avg_latency_sec"]), reverse=True)
    ranked_llms = sorted([r for r in results if r["tier"] == "LLM"], key=lambda x: (x["composite_score"], -x["avg_latency_sec"]), reverse=True)

    print("\n" + "=" * 78)
    print(f"{'MODEL':<32} {'TIER':<5} {'INTENT':<8} {'SEC_F1':<8} {'JSON':<6} {'LATENCY':<8} {'COMPOSITE'}")
    print("=" * 78)
    for r in sorted(results, key=lambda x: x["composite_score"], reverse=True):
        print(f"{r['model']:<32} {r['tier']:<5} {r['intent_accuracy']:<8.2f} {r['reasoning_f1']:<8.2f} {r['parse_rate']:<6.2f} {r['avg_latency_sec']:<7.2f}s {r['composite_score']:<8.3f}")
    print("=" * 78)

    # Derive role assignments
    slm_chain = [r["model"] for r in ranked_slms]
    llm_chain = [r["model"] for r in ranked_llms]

    best_slm = slm_chain[0] if slm_chain else "openai/gpt-oss-20b"
    best_llm = llm_chain[0] if llm_chain else "openai/gpt-oss-120b"

    recommended_chains = {
        "slm_intent": slm_chain if slm_chain else [best_slm],
        "llm_reasoning": llm_chain if llm_chain else [best_llm],
        "summarisation": [best_slm] + [m for m in llm_chain if m != best_slm][:1],
        "qa": [best_slm] + [m for m in llm_chain if m != best_slm][:1],
    }

    print("\n[RECOMMENDED MODEL ASSIGNMENTS]")
    for role, chain in recommended_chains.items():
        print(f"  {role:<15} -> Primary: {chain[0]:<22} | Fallback(s): {chain[1:]}")

    # Save output JSON
    out_file = Path(args.output) if args.output else (OUTPUT_DIR / "auto_benchmark_latest.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_models_evaluated": len(results),
        "results": results,
        "recommended_chains": recommended_chains,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n[OK] Benchmark results saved to: {out_file}")

    # Auto-apply if requested
    if args.apply:
        print("\n--> Applying recommended fallback chains to backend/api/model_config.py...")
        success = apply_to_model_config(recommended_chains, best_slm)
        if success:
            print("[SUCCESS] backend/api/model_config.py has been updated with live models!")
        else:
            print("[FAILED] Could not auto-apply configurations.")
    else:
        print("\n(Tip: Run with '--apply' to automatically update backend/api/model_config.py)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
