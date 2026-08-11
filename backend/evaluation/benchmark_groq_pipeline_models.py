#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


@dataclass
class EvalCase:
    case_id: str
    incident: str
    expected_intent: str
    expected_sections: list[str]


TEST_CASES: list[EvalCase] = [
    EvalCase(
        case_id="c1_armed_robbery",
        incident="Two men threatened a woman with a knife and snatched her gold chain and phone while causing cuts and bruises.",
        expected_intent="Robbery",
        expected_sections=["356", "394", "398", "324"],
    ),
    EvalCase(
        case_id="c2_house_theft",
        incident="Unknown person entered a locked house at night and stole cash and jewellery without confronting anyone.",
        expected_intent="Theft",
        expected_sections=["380", "454"],
    ),
    EvalCase(
        case_id="c3_dowry_cruelty",
        incident="Husband and in-laws repeatedly demanded dowry, threatened wife, and subjected her to mental and physical cruelty.",
        expected_intent="Cruelty",
        expected_sections=["498A", "506"],
    ),
    EvalCase(
        case_id="c4_cheating_trust",
        incident="Business partner took money promising to deliver goods, forged documents, and dishonestly misappropriated funds.",
        expected_intent="Cheating",
        expected_sections=["406", "420", "467", "468"],
    ),
    EvalCase(
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def extract_sections(text: str) -> list[str]:
    nums = re.findall(r"(?:section\s*)?(\d{2,4}[a-zA-Z]?)", text.lower())
    clean = []
    for n in nums:
        n = n.upper()
        if n not in clean:
            clean.append(n)
    return clean


def intent_prompt(case: EvalCase) -> str:
    return (
        "Classify the FIR incident into ONE primary legal intent from this list only: "
        "Robbery, Theft, Assault, Kidnapping, Murder, Cheating, Cruelty, Fraud, Sexual Offence, Threat. "
        "Return only valid JSON: {\"primary_intent\": \"...\", \"confidence\": 0.0}.\n\n"
        f"Incident: {case.incident}"
    )


def reasoning_prompt(case: EvalCase) -> str:
    return (
        "Given the FIR incident, return likely IPC sections relevant to prosecution. "
        "Return only valid JSON: {\"ipc_sections\": [\"356\", \"394\"]}.\n\n"
        f"Incident: {case.incident}"
    )


def chat_json(client: Groq, model: str, prompt: str, max_tokens: int = 220) -> tuple[dict, float, bool]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict JSON generator for legal NLP tasks."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw), elapsed, True
    except Exception:
        return {"_raw": raw}, elapsed, False


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


def run() -> int:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")

    client = Groq(api_key=api_key)
    model_ids = sorted([m.id for m in client.models.list().data])
    candidates = [m for m in model_ids if not should_exclude(m)]

    per_model = []
    for model in candidates:
        print(f"[MODEL] {model}")
        intent_correct = 0
        parse_ok = 0
        latencies = []
        f1_scores = []
        total_calls = 0

        for case in TEST_CASES:
            total_calls += 1
            try:
                out_i, lat_i, ok_i = chat_json(client, model, intent_prompt(case), max_tokens=120)
                latencies.append(lat_i)
                parse_ok += 1 if ok_i else 0
                pred_intent = str(out_i.get("primary_intent", "")).strip().lower()
                if pred_intent == case.expected_intent.lower():
                    intent_correct += 1
            except Exception:
                pass

            total_calls += 1
            try:
                out_r, lat_r, ok_r = chat_json(client, model, reasoning_prompt(case), max_tokens=180)
                latencies.append(lat_r)
                parse_ok += 1 if ok_r else 0
                pred_sections = extract_sections(json.dumps(out_r, ensure_ascii=False))
                f1_scores.append(section_f1(pred_sections, case.expected_sections))
            except Exception:
                f1_scores.append(0.0)

        intent_acc = intent_correct / len(TEST_CASES)
        reasoning_f1 = statistics.mean(f1_scores) if f1_scores else 0.0
        parse_rate = parse_ok / max(1, total_calls)
        avg_latency = statistics.mean(latencies) if latencies else 999.0

        composite = 0.45 * intent_acc + 0.45 * reasoning_f1 + 0.10 * parse_rate
        per_model.append(
            {
                "model": model,
                "tier": classify_size_tier(model),
                "intent_accuracy": intent_acc,
                "reasoning_section_f1": reasoning_f1,
                "json_parse_rate": parse_rate,
                "avg_latency_sec": avg_latency,
                "composite_score": composite,
            }
        )

    ranked = sorted(per_model, key=lambda r: (r["composite_score"], -r["avg_latency_sec"]), reverse=True)

    best_slm = next((r for r in ranked if r["tier"] == "SLM"), None)
    best_llm = next((r for r in ranked if r["tier"] == "LLM"), None)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [c.__dict__ for c in TEST_CASES],
        "models_evaluated": [r["model"] for r in ranked],
        "ranking": ranked,
        "best_slm": best_slm,
        "best_llm": best_llm,
        "metric_definitions": {
            "intent_accuracy": "correct_primary_intent_predictions / total_cases",
            "reasoning_section_f1": "Mean per-case F1 over predicted IPC section set vs expected set",
            "json_parse_rate": "valid_json_outputs / total_model_calls",
            "composite_score": "0.45*intent_accuracy + 0.45*reasoning_section_f1 + 0.10*json_parse_rate",
        },
    }

    out = repo_root() / "output" / "groq_pipeline_model_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== TOP MODELS ===")
    for row in ranked[:10]:
        print(
            f"{row['model']:<45} {row['tier']:<4} "
            f"intent={row['intent_accuracy']:.3f} f1={row['reasoning_section_f1']:.3f} "
            f"parse={row['json_parse_rate']:.3f} score={row['composite_score']:.3f} "
            f"lat={row['avg_latency_sec']:.2f}s"
        )

    print(f"\n[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
