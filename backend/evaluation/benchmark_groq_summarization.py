#!/usr/bin/env python3
"""
Benchmark Groq free-tier models for legal summarization.

Objectives:
- Compare 3 available Groq models on time + estimated list-price cost
- Evaluate extractive summaries using ROUGE, BLEU, METEOR
- Evaluate abstractive summaries using automated faithfulness + hallucination
- Select the best model and write it to output/model_benchmark_latest.json

Usage:
  python backend/evaluation/benchmark_groq_summarization.py
  python backend/evaluation/benchmark_groq_summarization.py --samples 4
  python backend/evaluation/benchmark_groq_summarization.py --models llama-3.1-8b-instant llama-3.3-70b-versatile openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# Updated August 2026 — reflects currently live Groq models after mass deprecation
DEFAULT_CANDIDATE_MODELS = [
    "groq/compound-mini",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "groq/compound",
]

# Approximate list-price defaults (USD per 1M tokens) for comparative analysis.
DEFAULT_PRICE_PER_1M = {
    "groq/compound-mini":  {"input": 0.00, "output": 0.00},
    "openai/gpt-oss-20b":  {"input": 0.11, "output": 0.33},
    "allam-2-7b":          {"input": 0.05, "output": 0.08},
    "qwen/qwen3.6-27b":    {"input": 0.29, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 1.20, "output": 3.60},
    "groq/compound":       {"input": 0.00, "output": 0.00},
}


@dataclass
class SampleItem:
    sample_id: str
    source_text: str
    reference_extractive: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def build_reference_extractive(text: str, num_sentences: int = 3) -> str:
    sents = split_sentences(text)
    return " ".join(sents[:num_sentences]) if sents else normalize_whitespace(text)[:300]


def load_samples(max_samples: int) -> list[SampleItem]:
    root = repo_root()
    items: list[SampleItem] = []

    fir_path = root / "src_dataset_files" / "fir_sample.json"
    if fir_path.exists():
        with open(fir_path, "r", encoding="utf-8") as f:
            fir = json.load(f)
        fields = [
            ("fir_incident", fir.get("incident_description", "")),
            ("fir_impact", fir.get("victim_impact", "")),
            ("fir_evidence", fir.get("evidence", "")),
        ]
        for sid, text in fields:
            text = normalize_whitespace(text)
            if len(text) >= 80:
                items.append(
                    SampleItem(
                        sample_id=sid,
                        source_text=text,
                        reference_extractive=build_reference_extractive(text),
                    )
                )

    analysis_path = root / "output" / "fir_analysis_result.json"
    if analysis_path.exists():
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        statutes = analysis.get("applicable_statutes", [])
        for idx, st in enumerate(statutes[:6], 1):
            p = st.get("primary", {})
            text = normalize_whitespace(
                " ".join(
                    [
                        p.get("title", ""),
                        p.get("extract", ""),
                        p.get("reasoning", ""),
                    ]
                )
            )
            if len(text) >= 80:
                items.append(
                    SampleItem(
                        sample_id=f"statute_{idx}",
                        source_text=text,
                        reference_extractive=build_reference_extractive(text),
                    )
                )

    dedup: list[SampleItem] = []
    seen = set()
    for item in items:
        key = item.source_text[:180]
        if key not in seen:
            dedup.append(item)
            seen.add(key)

    return dedup[:max_samples]


def safe_price_key(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", model).upper()


def get_price_config(model: str) -> dict[str, float]:
    base = DEFAULT_PRICE_PER_1M.get(model, {"input": 0.0, "output": 0.0}).copy()
    env_key = safe_price_key(model)
    inp = os.environ.get(f"PRICE_INPUT_{env_key}")
    out = os.environ.get(f"PRICE_OUTPUT_{env_key}")
    if inp is not None:
        try:
            base["input"] = float(inp)
        except ValueError:
            pass
    if out is not None:
        try:
            base["output"] = float(out)
        except ValueError:
            pass
    return base


def call_groq(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.2,
) -> dict[str, Any]:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start

    output = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    return {
        "text": output,
        "latency_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def metric_extractive(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    ref_tokens = simple_tokens(reference)
    pred_tokens = simple_tokens(prediction)

    if not ref_tokens or not pred_tokens:
        bleu = 0.0
        meteor = 0.0
    else:
        bleu = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
        try:
            meteor = meteor_score([ref_tokens], pred_tokens)
        except Exception:
            meteor = 0.0

    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
        "bleu": float(bleu),
        "meteor": float(meteor),
    }


def metric_abstractive(summary: str, source: str) -> dict[str, float]:
    summary_sents = split_sentences(summary)
    source_sents = split_sentences(source)

    if not summary_sents or not source_sents:
        return {"faithfulness": 0.0, "hallucination": 1.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    supports: list[float] = []
    for ss in summary_sents:
        best = 0.0
        for src in source_sents:
            sc = scorer.score(src, ss)["rougeL"].fmeasure
            if sc > best:
                best = sc
        supports.append(float(best))

    faithfulness = float(sum(supports) / len(supports))
    unsupported = [s for s in supports if s < 0.20]
    hallucination = float(len(unsupported) / len(supports))
    return {"faithfulness": faithfulness, "hallucination": hallucination}


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def minmax_inverse(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    norm = (value - low) / (high - low)
    return max(0.0, min(1.0, 1.0 - norm))


def select_best_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    latencies = [r["avg_latency_sec"] for r in rows]
    costs = [r["estimated_cost_usd"] for r in rows]
    min_lat, max_lat = min(latencies), max(latencies)
    min_cost, max_cost = min(costs), max(costs)

    for r in rows:
        extractive_quality = (r["avg_rougeL"] + r["avg_bleu"] + r["avg_meteor"]) / 3.0
        faithful_quality = (r["avg_faithfulness"] + (1.0 - r["avg_hallucination"])) / 2.0

        latency_score = minmax_inverse(r["avg_latency_sec"], min_lat, max_lat)
        cost_score = minmax_inverse(r["estimated_cost_usd"], min_cost, max_cost)

        final_score = (
            0.40 * extractive_quality
            + 0.35 * faithful_quality
            + 0.15 * latency_score
            + 0.10 * cost_score
        )

        r["composite_score"] = float(final_score)

    return sorted(rows, key=lambda x: x["composite_score"], reverse=True)


def benchmark(models: list[str], samples: list[SampleItem]) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env before running benchmark.")

    client = Groq(api_key=api_key)
    available_models = []

    probe_prompt = "Reply with exactly: READY"
    for model in models:
        try:
            call_groq(
                client=client,
                model=model,
                system_prompt="You are a concise assistant.",
                user_prompt=probe_prompt,
                max_tokens=8,
                temperature=0,
            )
            available_models.append(model)
        except Exception as e:
            print(f"[SKIP] Model unavailable: {model} ({e})")

        if len(available_models) >= 3:
            break

    if len(available_models) < 3:
        raise RuntimeError(
            f"Need 3 available Groq models, found only {len(available_models)}: {available_models}"
        )

    active_models = available_models[:3]
    print(f"[INFO] Active models: {active_models}")

    per_model_raw: dict[str, list[dict[str, Any]]] = {m: [] for m in active_models}

    for model in active_models:
        price = get_price_config(model)
        print(f"\n[MODEL] {model}")
        for sample in samples:
            source = sample.source_text

            extractive_user = (
                "Create an EXTRACTIVE legal summary in 3-5 sentences. "
                "Use only exact phrases from the source text and keep chronology.\n\n"
                f"SOURCE:\n{source[:3500]}"
            )
            extractive = call_groq(
                client,
                model,
                "You are a legal summarizer. Output plain text only.",
                extractive_user,
                max_tokens=220,
                temperature=0.1,
            )

            abstractive_user = (
                "Create an ABSTRACTIVE legal summary in 3-5 sentences. "
                "Capture allegations, key legal context, and outcome-relevant facts.\n\n"
                f"SOURCE:\n{source[:3500]}"
            )
            abstractive = call_groq(
                client,
                model,
                "You are a legal analyst. Be factual and concise.",
                abstractive_user,
                max_tokens=220,
                temperature=0.2,
            )

            ext_m = metric_extractive(extractive["text"], sample.reference_extractive)
            abs_m = metric_abstractive(abstractive["text"], source)

            tokens_in = (
                extractive["prompt_tokens"]
                + abstractive["prompt_tokens"]
            )
            tokens_out = (
                extractive["completion_tokens"]
                + abstractive["completion_tokens"]
            )
            est_cost = (tokens_in / 1_000_000.0) * price["input"] + (tokens_out / 1_000_000.0) * price["output"]

            per_model_raw[model].append(
                {
                    "sample_id": sample.sample_id,
                    "extractive_summary": extractive["text"],
                    "abstractive_summary": abstractive["text"],
                    "latency_sec": extractive["latency_sec"] + abstractive["latency_sec"],
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "estimated_cost_usd": est_cost,
                    **ext_m,
                    **abs_m,
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    for model, rows in per_model_raw.items():
        aggregate_rows.append(
            {
                "model": model,
                "samples": len(rows),
                "avg_latency_sec": mean_or_zero([r["latency_sec"] for r in rows]),
                "avg_prompt_tokens": mean_or_zero([r["prompt_tokens"] for r in rows]),
                "avg_completion_tokens": mean_or_zero([r["completion_tokens"] for r in rows]),
                "avg_rouge1": mean_or_zero([r["rouge1"] for r in rows]),
                "avg_rougeL": mean_or_zero([r["rougeL"] for r in rows]),
                "avg_bleu": mean_or_zero([r["bleu"] for r in rows]),
                "avg_meteor": mean_or_zero([r["meteor"] for r in rows]),
                "avg_faithfulness": mean_or_zero([r["faithfulness"] for r in rows]),
                "avg_hallucination": mean_or_zero([r["hallucination"] for r in rows]),
                "estimated_cost_usd": float(sum(r["estimated_cost_usd"] for r in rows)),
                "free_tier_effective_cost_usd": 0.0,
            }
        )

    ranked = select_best_model(aggregate_rows)
    best_model = ranked[0]["model"] if ranked else ""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "groq_legal_summarization_benchmark",
        "samples_used": [s.sample_id for s in samples],
        "models_tested": [r["model"] for r in ranked],
        "best_model": best_model,
        "ranking": ranked,
        "details": per_model_raw,
        "notes": {
            "extractive_metrics": ["rouge1", "rougeL", "bleu", "meteor"],
            "abstractive_metrics": ["faithfulness", "hallucination"],
            "cost_interpretation": "Estimated list-price cost is shown for comparison; effective free-tier run cost is reported as 0.",
        },
    }


def print_ranking(result: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("GROQ MODEL BENCHMARK RESULTS")
    print("=" * 90)
    print(f"Best model: {result.get('best_model', 'N/A')}")
    print("\nRanked comparison:")
    header = (
        f"{'Model':35} {'R-L':>6} {'BLEU':>6} {'MET':>6} {'Faith':>7} {'Hallu':>7} {'Time(s)':>8} {'Cost($)':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in result.get("ranking", []):
        print(
            f"{row['model'][:35]:35} "
            f"{row['avg_rougeL']:.3f} "
            f"{row['avg_bleu']:.3f} "
            f"{row['avg_meteor']:.3f} "
            f"{row['avg_faithfulness']:.3f} "
            f"{row['avg_hallucination']:.3f} "
            f"{row['avg_latency_sec']:.2f} "
            f"{row['estimated_cost_usd']:.4f}"
        )
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Groq models for legal summarization")
    parser.add_argument("--samples", type=int, default=5, help="Max number of local benchmark samples")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_CANDIDATE_MODELS,
        help="Candidate Groq models (script picks first 3 available)",
    )
    args = parser.parse_args()

    samples = load_samples(max_samples=max(1, args.samples))
    if not samples:
        raise RuntimeError("No benchmark samples available from local project files.")

    print(f"[INFO] Loaded {len(samples)} samples: {[s.sample_id for s in samples]}")
    result = benchmark(models=args.models, samples=samples)

    out_path = repo_root() / "output" / "model_benchmark_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print_ranking(result)
    print(f"\n[OK] Benchmark saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
