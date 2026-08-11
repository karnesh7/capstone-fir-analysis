#!/usr/bin/env python3
"""
Groq Model Benchmark — Legal FIR Pipeline (Metric-Aligned)
===========================================================
Metrics (from Additional Change.txt):
  Extractive : ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR
  Abstractive: Faithfulness, Hallucination  (LLM-as-judge)
  Comparative: Latency (sec/call), Estimated Cost ($/call)

Benchmark A — SLMs  → Intent Recognition  (Stage 1 of pipeline)
Benchmark B — LLMs  → Legal Reasoning     (Stage 2 of pipeline)

Prompts mirror rag_llm_chain_prompting.py exactly.
Test cases use full FIR fields aligned with project functionality.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv
from groq import Groq

# ----- metric imports ------------------------------------------------
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

# =====================================================================
#  CONFIGURATION
# =====================================================================
# NOTE: Using llama-3.3-70b-versatile as judge until its Aug 16 2026 shutdown.
# It is NOT a candidate under test, so no TPM conflict with the benchmarked models.
# After Aug 16, switch to: JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "llama-3.3-70b-versatile"

# Approximate Groq pricing USD / 1 M tokens  (input, output)
# Updated August 2026 — reflects currently live models after mass deprecation
PRICING: dict[str, tuple[float, float]] = {
    # --- Active SLMs ---
    "allam-2-7b":              (0.05, 0.08),
    "openai/gpt-oss-20b":      (0.11, 0.33),   # recommended replacement for llama-3.1-8b-instant
    "groq/compound-mini":      (0.00, 0.00),   # compound-mini (renamed)
    # --- Active LLMs ---
    "openai/gpt-oss-120b":     (1.20, 3.60),   # recommended replacement for 70B+ models
    "qwen/qwen3.6-27b":        (0.29, 0.79),   # recommended replacement for qwen3-32b / llama-4-scout
    "groq/compound":           (0.00, 0.00),   # compound (renamed)
    # --- Deprecated: shutting down Aug 16 2026 (keep until final cutover) ---
    "llama-3.1-8b-instant":    (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}

EXCLUDED_PATTERNS = [r"^whisper", r"guard", r"safeguard", r"orpheus"]

# =====================================================================
#  TEST CASES  (aligned with project FIR schema & pipeline prompts)
# =====================================================================

@dataclass
class FIRTestCase:
    case_id: str
    complainant: str
    accused: str
    victim: str
    incident: str
    victim_impact: str
    evidence: str
    statute_context: str          # simulated Pinecone retrieval for Stage 2
    gold_primary_intent: str
    gold_intent_text: str         # flattened reference for extractive metrics
    gold_reasoning_text: str      # flattened reference for extractive metrics


TEST_CASES: list[FIRTestCase] = [
    # ---- Case 1: Armed Robbery ----
    FIRTestCase(
        case_id="c1_armed_robbery",
        complainant="Rajesh Kumar",
        accused="Amit Singh, Vikram Patel",
        victim="Priya Sharma",
        incident=(
            "On 2024-01-09 at approximately 11:30 PM, the victim was returning home from "
            "her office when she was approached by two armed men. They forcefully grabbed "
            "her, threatened her with a knife, and stole her mobile phone and gold jewelry "
            "worth approximately Rs. 50,000. The victim suffered multiple cuts and bruises. "
            "The perpetrators fled the scene on a motorcycle."
        ),
        victim_impact=(
            "Victim suffered physical injuries including multiple cuts and contusions on "
            "arms and face. Psychological trauma and fear. Loss of valuables. Unable to "
            "work for 3 days."
        ),
        evidence=(
            "CCTV footage showing incident clearly. Medical report documenting injuries. "
            "FIR witness: street vendor present at scene. Torn dupatta found at scene."
        ),
        statute_context=(
            "[IPC 356] Whoever assaults or uses criminal force to any person, in attempting "
            "to commit theft of any property which that person is then wearing or carrying, "
            "shall be punished with imprisonment up to two years, or with fine, or both.\n"
            "[IPC 390] In all robbery there is either theft or extortion. When theft is "
            "robbery: theft is robbery if the offender causes or attempts to cause death, "
            "hurt, or wrongful restraint.\n"
            "[IPC 394] If any person, in committing or attempting to commit robbery, "
            "voluntarily causes hurt, such person and any other person jointly concerned "
            "shall be punished with imprisonment for life, or with rigorous imprisonment "
            "for a term which may extend to ten years, and shall also be liable to fine.\n"
            "[IPC 324] Whoever voluntarily causes hurt by means of any instrument for "
            "shooting, stabbing or cutting, or any instrument which, used as a weapon of "
            "offence, is likely to cause death, shall be punished."
        ),
        gold_primary_intent="Robbery",
        gold_intent_text=(
            "Primary intent: Robbery. Confidence: 0.95. "
            "Secondary intents: Assault, Theft."
        ),
        gold_reasoning_text=(
            "Applicable statutes: IPC Section 356 — Assault or criminal force in attempt "
            "to commit theft of property carried by a person — The accused used criminal "
            "force while attempting to steal the victim's belongings. "
            "IPC Section 390 — Robbery — The accused committed theft accompanied by force "
            "and threat of force using a knife, meeting the statutory definition of robbery. "
            "IPC Section 394 — Voluntarily causing hurt in committing robbery — The victim "
            "sustained cuts and bruises during the robbery, invoking the aggravated provision. "
            "IPC Section 324 — Voluntarily causing hurt by dangerous weapon — A knife was "
            "used causing cuts, satisfying the weapon element. "
            "Legal basis: The facts demonstrate theft accompanied by criminal force, use of "
            "a dangerous weapon, and resulting physical injury, constituting robbery with "
            "hurt under IPC. Severity assessment: high. Confidence: 0.93."
        ),
    ),
    # ---- Case 2: House Break-in Theft ----
    FIRTestCase(
        case_id="c2_house_theft",
        complainant="Meera Devi",
        accused="Unknown",
        victim="Meera Devi",
        incident=(
            "On 2024-02-15 between 10:00 AM and 4:00 PM, while the complainant was away "
            "at work, an unknown person broke the lock of the main door and entered the "
            "locked house. Cash of Rs. 80,000, gold ornaments, and a laptop were stolen. "
            "No one was present in the house at the time."
        ),
        victim_impact=(
            "Loss of cash Rs. 80,000, gold ornaments valued at Rs. 2,00,000, and laptop "
            "worth Rs. 60,000. Emotional distress and feeling of insecurity."
        ),
        evidence=(
            "Broken lock recovered from scene. Neighbor's CCTV showing a suspicious person "
            "near the house at 1:15 PM. Fingerprints lifted from almirah."
        ),
        statute_context=(
            "[IPC 380] Whoever commits theft in any building, tent, or vessel used as a "
            "human dwelling or for the custody of property, shall be punished with "
            "imprisonment which may extend to seven years, and shall also be liable to fine.\n"
            "[IPC 454] Whoever commits lurking house-trespass or house-breaking in order "
            "to commit any offence punishable with imprisonment, shall be punished with "
            "imprisonment which may extend to three years, and shall also be liable to fine.\n"
            "[IPC 457] Whoever commits lurking house-trespass by night, or house-breaking "
            "by night, in order to commit any offence punishable with imprisonment, shall "
            "be punished with imprisonment which may extend to five years."
        ),
        gold_primary_intent="Theft",
        gold_intent_text=(
            "Primary intent: Theft. Confidence: 0.93. "
            "Secondary intents: Trespass, Burglary."
        ),
        gold_reasoning_text=(
            "Applicable statutes: IPC Section 380 — Theft in dwelling house — The theft "
            "occurred in a residential building used as human dwelling, meeting the "
            "elements of theft in a dwelling house. "
            "IPC Section 454 — Lurking house-trespass or house-breaking to commit offence "
            "— The accused broke the lock and entered the house to commit theft, "
            "constituting house-breaking. "
            "IPC Section 457 — Lurking house-trespass by night or house-breaking by night "
            "— The break-in occurred during daytime; however this section may apply if "
            "evidence shows nighttime entry. "
            "Legal basis: The accused broke into a locked residential dwelling and committed "
            "theft of valuables, satisfying the elements of theft in dwelling under IPC 380 "
            "and house-breaking under IPC 454. Severity assessment: medium. Confidence: 0.90."
        ),
    ),
    # ---- Case 3: Dowry Cruelty ----
    FIRTestCase(
        case_id="c3_dowry_cruelty",
        complainant="Anita Kumari",
        accused="Rakesh Gupta, Sunita Gupta, Mahesh Gupta",
        victim="Anita Kumari",
        incident=(
            "Since marriage in 2023, the husband Rakesh Gupta and in-laws Sunita Gupta "
            "and Mahesh Gupta have been demanding Rs. 10,00,000 as additional dowry. "
            "The victim has been subjected to daily verbal abuse, physical beating, "
            "deprivation of food, and threats of being thrown out. On 2024-03-01, the "
            "husband slapped and kicked the victim and threatened to kill her if dowry "
            "is not paid within one month."
        ),
        victim_impact=(
            "Physical injuries from repeated beatings. Severe mental trauma and "
            "depression. Malnutrition from deprivation of food. Fear for life due to "
            "death threats."
        ),
        evidence=(
            "Medical reports showing injuries on multiple dates. WhatsApp messages from "
            "husband demanding money. Testimony of neighbor who heard screaming. "
            "Victim's parents confirming dowry demand."
        ),
        statute_context=(
            "[IPC 498A] Whoever, being the husband or the relative of the husband of a "
            "woman, subjects such woman to cruelty shall be punished with imprisonment "
            "for a term which may extend to three years and shall also be liable to fine. "
            "Cruelty means any wilful conduct likely to drive the woman to commit suicide "
            "or cause grave injury, or harassment with a view to coercing her or any "
            "related person to meet any unlawful demand for property.\n"
            "[IPC 506] Whoever commits the offence of criminal intimidation shall be "
            "punished with imprisonment which may extend to two years, or with fine, or "
            "both; if threat be to cause death or grievous hurt — imprisonment up to "
            "seven years.\n"
            "[IPC 34] When a criminal act is done by several persons in furtherance of "
            "the common intention of all, each of such persons is liable for that act in "
            "the same manner as if it were done by him alone."
        ),
        gold_primary_intent="Cruelty",
        gold_intent_text=(
            "Primary intent: Cruelty. Confidence: 0.96. "
            "Secondary intents: Dowry Harassment, Criminal Intimidation."
        ),
        gold_reasoning_text=(
            "Applicable statutes: IPC Section 498A — Cruelty by husband or relatives of "
            "husband — The husband and in-laws subjected the wife to physical and mental "
            "cruelty including beating, deprivation of food, and dowry demands, satisfying "
            "all elements of Section 498A. "
            "IPC Section 506 — Criminal intimidation — The husband threatened to kill the "
            "victim if dowry was not paid, constituting criminal intimidation with threat "
            "of death. "
            "IPC Section 34 — Acts done by several persons in furtherance of common "
            "intention — All three accused acted together in demanding dowry and "
            "perpetrating cruelty, establishing common intention. "
            "Legal basis: The sustained pattern of physical abuse, mental cruelty, dowry "
            "demands, and death threats by the husband and in-laws constitutes cruelty "
            "under IPC 498A, aggravated by criminal intimidation under 506. "
            "Severity assessment: high. Confidence: 0.95."
        ),
    ),
    # ---- Case 4: Cheating / Fraud ----
    FIRTestCase(
        case_id="c4_cheating_fraud",
        complainant="Suresh Mehta",
        accused="Dinesh Agarwal",
        victim="Suresh Mehta",
        incident=(
            "The accused Dinesh Agarwal, who was a business partner of the complainant, "
            "collected Rs. 25,00,000 from the complainant promising to deliver goods. "
            "The accused forged invoices and delivery receipts to show goods were dispatched. "
            "Investigation revealed no goods were ever procured. The accused dishonestly "
            "misappropriated the entire amount."
        ),
        victim_impact=(
            "Financial loss of Rs. 25,00,000. Business operations disrupted. "
            "Reputation damage with downstream clients due to non-delivery of goods."
        ),
        evidence=(
            "Forged invoices recovered. Bank transaction records showing fund transfer. "
            "Warehouse inspection confirming no goods stored. Forensic analysis of "
            "forged signatures."
        ),
        statute_context=(
            "[IPC 420] Whoever cheats and thereby dishonestly induces the person deceived "
            "to deliver any property, shall be punished with imprisonment which may extend "
            "to seven years, and shall also be liable to fine.\n"
            "[IPC 406] Whoever commits criminal breach of trust shall be punished with "
            "imprisonment which may extend to three years, or with fine, or both.\n"
            "[IPC 467] Whoever forges a document which purports to be a valuable security "
            "or a will, shall be punished with imprisonment for life or imprisonment "
            "up to ten years, and shall also be liable to fine.\n"
            "[IPC 468] Whoever commits forgery intending that the document forged shall "
            "be used for the purpose of cheating, shall be punished with imprisonment "
            "which may extend to seven years, and shall also be liable to fine."
        ),
        gold_primary_intent="Cheating",
        gold_intent_text=(
            "Primary intent: Cheating. Confidence: 0.94. "
            "Secondary intents: Fraud, Forgery, Criminal Breach of Trust."
        ),
        gold_reasoning_text=(
            "Applicable statutes: IPC Section 420 — Cheating and dishonestly inducing "
            "delivery of property — The accused cheated the complainant by falsely "
            "promising delivery of goods and induced delivery of Rs. 25 lakhs. "
            "IPC Section 406 — Criminal breach of trust — The accused was entrusted "
            "with funds for a specific purpose and dishonestly misappropriated the "
            "entire amount. "
            "IPC Section 467 — Forgery of valuable security — The accused forged "
            "invoices and delivery receipts which constitute commercial documents. "
            "IPC Section 468 — Forgery for purpose of cheating — The forged documents "
            "were created specifically to deceive the complainant into believing goods "
            "were dispatched. "
            "Legal basis: The accused cheated the complainant through false promises, "
            "misappropriated entrusted funds, and created forged documents to conceal "
            "the fraud, constituting offences under IPC 420, 406, 467, and 468. "
            "Severity assessment: high. Confidence: 0.93."
        ),
    ),
    # ---- Case 5: Kidnapping / Assault ----
    FIRTestCase(
        case_id="c5_kidnap_assault",
        complainant="Kavita Singh",
        accused="Ravi Yadav, Sunil Yadav",
        victim="Kavita Singh",
        incident=(
            "On 2024-04-20 at around 8:00 PM, the victim was walking near her residence "
            "when two men (Ravi Yadav and Sunil Yadav) forcibly pulled her into a car. "
            "She was taken to an isolated location where she was beaten with fists and "
            "a rod. The accused threatened to kill her if she informed the police. "
            "She was released after 6 hours."
        ),
        victim_impact=(
            "Multiple injuries including bruises, swelling on arms and back. "
            "Severe psychological trauma and fear. Unable to leave home unaccompanied."
        ),
        evidence=(
            "CCTV footage near abduction point showing victim being pulled into car. "
            "Medical report documenting injuries. Victim's torn clothing recovered. "
            "Victim identified both accused in photo lineup."
        ),
        statute_context=(
            "[IPC 365] Whoever kidnaps or abducts any person with intent to cause that "
            "person to be secretly and wrongfully confined, shall be punished with "
            "imprisonment which may extend to seven years, and shall also be liable to fine.\n"
            "[IPC 323] Whoever voluntarily causes hurt shall be punished with "
            "imprisonment which may extend to one year, or with fine which may extend "
            "to one thousand rupees, or with both.\n"
            "[IPC 506] Whoever commits the offence of criminal intimidation shall be "
            "punished with imprisonment which may extend to two years, or with fine, "
            "or both; if threat be to cause death or grievous hurt — up to seven years."
        ),
        gold_primary_intent="Kidnapping",
        gold_intent_text=(
            "Primary intent: Kidnapping. Confidence: 0.94. "
            "Secondary intents: Assault, Criminal Intimidation."
        ),
        gold_reasoning_text=(
            "Applicable statutes: IPC Section 365 — Kidnapping or abducting with intent "
            "to secretly and wrongfully confine — The accused forcibly abducted the victim "
            "and confined her at an isolated location for 6 hours. "
            "IPC Section 323 — Voluntarily causing hurt — The accused beat the victim "
            "with fists and a rod causing injuries, constituting voluntarily causing hurt. "
            "IPC Section 506 — Criminal intimidation — The accused threatened to kill "
            "the victim if she informed police, constituting criminal intimidation with "
            "threat of death. "
            "Legal basis: The forcible abduction, unlawful confinement, physical assault, "
            "and death threats by the accused constitute kidnapping, hurt, and criminal "
            "intimidation under IPC. Severity assessment: high. Confidence: 0.94."
        ),
    ),
]

# =====================================================================
#  PROMPT TEMPLATES  (mirror rag_llm_chain_prompting.py exactly)
# =====================================================================

def intent_prompt(case: FIRTestCase) -> str:
    """Stage 1 prompt — same template as StatuteRAGChainSystem.intent_prompt"""
    return dedent(f"""\
        You are a legal analyst. Analyze the FIR case details and identify the PRIMARY crime/intent.

        CASE DETAILS:
        Complainant: {case.complainant}
        Accused: {case.accused}
        Incident: {case.incident}
        Victim Impact: {case.victim_impact}
        Evidence: {case.evidence}

        Based on the facts presented, identify:
        1. The PRIMARY intent/crime type
        2. Confidence level (0-1)
        3. Any secondary intents

        Return ONLY valid JSON (no markdown, no code blocks):
        {{"primary_intent": "string", "confidence": 0.95, "secondary_intents": ["string"]}}""")


def reasoning_prompt(case: FIRTestCase) -> str:
    """Stage 2 prompt — same template as StatuteRAGChainSystem.reasoning_prompt"""
    return dedent(f"""\
        You are a legal expert analyzing applicable criminal statutes.

        CASE FACTS:
        Incident: {case.incident}
        Accused: {case.accused}
        Victim: {case.victim}
        Identified Intent: {case.gold_primary_intent}

        RELEVANT STATUTE SECTIONS:
        {case.statute_context}

        Determine which statutes are APPLICABLE to this case:
        - Consider each statute's elements and how they relate to the case facts
        - Explain WHY each statute applies
        - Assess overall severity (high/medium/low)
        - Provide confidence in your analysis

        Return ONLY valid JSON (no markdown, no code blocks):
        {{"applicable_statutes": [{{"section": "string", "law": "string", "reasoning": "string"}}], "legal_basis": "string", "severity_assessment": "string", "confidence": 0.95}}""")


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    if "compound-mini" in m or "compound-beta-mini" in m:
        return "SLM"
    return "LLM"


def call_groq(client: Groq, model: str, prompt: str,
              max_tokens: int = 400) -> tuple[str, float, int, int]:
    """Call Groq API and return (raw_text, latency_sec, input_tokens, output_tokens)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a legal analysis AI. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    raw = (response.choices[0].message.content or "").strip()
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    usage = response.usage
    return raw, elapsed, usage.prompt_tokens, usage.completion_tokens


def flatten_intent_json(raw: str) -> str:
    """Convert intent JSON output to flattened text for ROUGE/BLEU/METEOR comparison."""
    try:
        obj = json.loads(raw)
        pi = obj.get("primary_intent", "")
        conf = obj.get("confidence", "")
        sec = ", ".join(obj.get("secondary_intents", []))
        return f"Primary intent: {pi}. Confidence: {conf}. Secondary intents: {sec}."
    except Exception:
        return raw  # fall back to raw text


def flatten_reasoning_json(raw: str) -> str:
    """Convert reasoning JSON output to flattened text for ROUGE/BLEU/METEOR comparison."""
    try:
        obj = json.loads(raw)
        parts = []
        for s in obj.get("applicable_statutes", []):
            sec = s.get("section", "")
            law = s.get("law", "")
            reason = s.get("reasoning", "")
            parts.append(f"{sec} — {law} — {reason}")
        statutes_text = "Applicable statutes: " + " ".join(parts)
        basis = obj.get("legal_basis", "")
        sev = obj.get("severity_assessment", "")
        conf = obj.get("confidence", "")
        return (
            f"{statutes_text} Legal basis: {basis}. "
            f"Severity assessment: {sev}. Confidence: {conf}."
        )
    except Exception:
        return raw


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a single API call."""
    inp_rate, out_rate = PRICING.get(model, (0.20, 0.60))  # default mid-range
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


# =====================================================================
#  EXTRACTIVE METRICS:  ROUGE, BLEU, METEOR
# =====================================================================

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(candidate: str, reference: str) -> dict[str, float]:
    scores = _rouge.score(reference, candidate)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def compute_bleu(candidate: str, reference: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0
    smoother = SmoothingFunction().method1
    return round(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoother), 4)


def compute_meteor(candidate: str, reference: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0
    return round(_meteor([ref_tokens], cand_tokens), 4)


# =====================================================================
#  ABSTRACTIVE METRICS:  Faithfulness & Hallucination  (LLM-as-judge)
# =====================================================================

JUDGE_PROMPT_TEMPLATE = dedent("""\
    You are an impartial evaluation judge for a legal AI system.

    INPUT CONTEXT (FIR case details given to the AI):
    {input_context}

    AI MODEL OUTPUT:
    {model_output}

    Evaluate the AI output on two dimensions:

    1. **Faithfulness** (0.0 to 1.0):
       How well does EVERY claim in the output stay grounded in the input context?
       1.0 = all claims are directly supported by or logically derivable from the input.
       0.0 = none of the claims are supported.

    2. **Hallucination** (0.0 to 1.0):
       What fraction of the output contains information NOT present in or derivable
       from the input context?
       0.0 = no hallucination at all.
       1.0 = entirely hallucinated.

    Return ONLY valid JSON:
    {{"faithfulness": 0.0, "hallucination": 0.0}}""")


def judge_abstractive(client: Groq, input_context: str,
                      model_output: str) -> tuple[float, float]:
    """Use JUDGE_MODEL to score faithfulness and hallucination."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        input_context=input_context,
        model_output=model_output,
    )
    try:
        raw, _, _, _ = call_groq(client, JUDGE_MODEL, prompt, max_tokens=120)
        obj = json.loads(raw)
        faith = float(obj.get("faithfulness", 0.0))
        halluc = float(obj.get("hallucination", 1.0))
        return (min(max(faith, 0.0), 1.0), min(max(halluc, 0.0), 1.0))
    except Exception:
        return (0.0, 1.0)  # worst-case if judge fails


# =====================================================================
#  MAIN BENCHMARK LOOP
# =====================================================================

def run() -> int:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    # ----- discover models -----
    all_models = sorted(m.id for m in client.models.list().data)
    candidates = [m for m in all_models if not should_exclude(m)]
    slm_models = [m for m in candidates if classify_size_tier(m) == "SLM"]
    llm_models = [m for m in candidates if classify_size_tier(m) == "LLM"]

    print(f"Discovered {len(candidates)} usable models: "
          f"{len(slm_models)} SLMs, {len(llm_models)} LLMs\n")

    # ===== BENCHMARK A: SLMs → Intent Recognition =====
    print("=" * 70)
    print("  BENCHMARK A — SLMs → Intent Recognition (Stage 1)")
    print("=" * 70)

    slm_results: list[dict] = []
    for model in slm_models:
        print(f"\n[SLM] {model}")
        per_case: list[dict] = []

        for case in TEST_CASES:
            prompt_text = intent_prompt(case)
            input_context = (
                f"Complainant: {case.complainant}\nAccused: {case.accused}\n"
                f"Incident: {case.incident}\nVictim Impact: {case.victim_impact}\n"
                f"Evidence: {case.evidence}"
            )
            try:
                raw_out, latency, in_tok, out_tok = call_groq(
                    client, model, prompt_text, max_tokens=250
                )
            except Exception as e:
                print(f"  ✗ {case.case_id}: {e}")
                continue

            flat_out = flatten_intent_json(raw_out)
            ref_text = case.gold_intent_text

            # Extractive metrics
            rouge = compute_rouge(flat_out, ref_text)
            bleu = compute_bleu(flat_out, ref_text)
            meteor = compute_meteor(flat_out, ref_text)

            # Abstractive metrics (judge)
            faith, halluc = judge_abstractive(client, input_context, raw_out)

            cost = estimate_cost(model, in_tok, out_tok)

            per_case.append({
                "case_id": case.case_id,
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bleu": bleu,
                "meteor": meteor,
                "faithfulness": faith,
                "hallucination": halluc,
                "latency_sec": round(latency, 3),
                "cost_usd": round(cost, 6),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })
            print(f"  ✓ {case.case_id}  R1={rouge['rouge1']:.3f}  BLEU={bleu:.3f}  "
                  f"MTR={meteor:.3f}  Faith={faith:.2f}  Halluc={halluc:.2f}  "
                  f"lat={latency:.2f}s  ${cost:.5f}")

            time.sleep(0.3)  # respect rate limits

        if not per_case:
            continue

        avg = lambda key: round(statistics.mean(c[key] for c in per_case), 4)
        summary = {
            "model": model,
            "tier": "SLM",
            "task": "intent_recognition",
            "avg_rouge1": avg("rouge1"),
            "avg_rouge2": avg("rouge2"),
            "avg_rougeL": avg("rougeL"),
            "avg_bleu": avg("bleu"),
            "avg_meteor": avg("meteor"),
            "avg_faithfulness": avg("faithfulness"),
            "avg_hallucination": avg("hallucination"),
            "avg_latency_sec": avg("latency_sec"),
            "total_cost_usd": round(sum(c["cost_usd"] for c in per_case), 6),
            "per_case": per_case,
        }
        # composite: equal weight on the 5 core metrics (higher is better; invert halluc)
        summary["composite_score"] = round(
            (summary["avg_rouge1"]
             + summary["avg_bleu"]
             + summary["avg_meteor"]
             + summary["avg_faithfulness"]
             + (1.0 - summary["avg_hallucination"])) / 5.0,
            4,
        )
        slm_results.append(summary)

    slm_ranked = sorted(slm_results, key=lambda r: r["composite_score"], reverse=True)

    # ===== BENCHMARK B: LLMs → Legal Reasoning =====
    print("\n" + "=" * 70)
    print("  BENCHMARK B — LLMs → Legal Reasoning (Stage 2)")
    print("=" * 70)

    llm_results: list[dict] = []
    for model in llm_models:
        print(f"\n[LLM] {model}")
        per_case: list[dict] = []

        for case in TEST_CASES:
            prompt_text = reasoning_prompt(case)
            input_context = (
                f"Incident: {case.incident}\nAccused: {case.accused}\n"
                f"Victim: {case.victim}\nIntent: {case.gold_primary_intent}\n"
                f"Statute Context: {case.statute_context}"
            )
            try:
                raw_out, latency, in_tok, out_tok = call_groq(
                    client, model, prompt_text, max_tokens=600
                )
            except Exception as e:
                print(f"  ✗ {case.case_id}: {e}")
                continue

            flat_out = flatten_reasoning_json(raw_out)
            ref_text = case.gold_reasoning_text

            # Extractive metrics
            rouge = compute_rouge(flat_out, ref_text)
            bleu = compute_bleu(flat_out, ref_text)
            meteor = compute_meteor(flat_out, ref_text)

            # Abstractive metrics (judge)
            faith, halluc = judge_abstractive(client, input_context, raw_out)

            cost = estimate_cost(model, in_tok, out_tok)

            per_case.append({
                "case_id": case.case_id,
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bleu": bleu,
                "meteor": meteor,
                "faithfulness": faith,
                "hallucination": halluc,
                "latency_sec": round(latency, 3),
                "cost_usd": round(cost, 6),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })
            print(f"  ✓ {case.case_id}  R1={rouge['rouge1']:.3f}  BLEU={bleu:.3f}  "
                  f"MTR={meteor:.3f}  Faith={faith:.2f}  Halluc={halluc:.2f}  "
                  f"lat={latency:.2f}s  ${cost:.5f}")

            time.sleep(0.3)

        if not per_case:
            continue

        avg = lambda key: round(statistics.mean(c[key] for c in per_case), 4)
        summary = {
            "model": model,
            "tier": "LLM",
            "task": "legal_reasoning",
            "avg_rouge1": avg("rouge1"),
            "avg_rouge2": avg("rouge2"),
            "avg_rougeL": avg("rougeL"),
            "avg_bleu": avg("bleu"),
            "avg_meteor": avg("meteor"),
            "avg_faithfulness": avg("faithfulness"),
            "avg_hallucination": avg("hallucination"),
            "avg_latency_sec": avg("latency_sec"),
            "total_cost_usd": round(sum(c["cost_usd"] for c in per_case), 6),
            "per_case": per_case,
        }
        summary["composite_score"] = round(
            (summary["avg_rouge1"]
             + summary["avg_bleu"]
             + summary["avg_meteor"]
             + summary["avg_faithfulness"]
             + (1.0 - summary["avg_hallucination"])) / 5.0,
            4,
        )
        llm_results.append(summary)

    llm_ranked = sorted(llm_results, key=lambda r: r["composite_score"], reverse=True)

    # ===== PRINT RESULTS =====
    def print_table(title: str, ranked: list[dict]):
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
        print(f"{'Rank':<5} {'Model':<45} {'R1':<7} {'BLEU':<7} {'MTR':<7} "
              f"{'Faith':<7} {'Halluc':<7} {'Score':<7} {'Lat(s)':<7} {'Cost$':<8}")
        print("-" * 110)
        for i, r in enumerate(ranked, 1):
            print(f"{i:<5} {r['model']:<45} "
                  f"{r['avg_rouge1']:<7.3f} {r['avg_bleu']:<7.3f} {r['avg_meteor']:<7.3f} "
                  f"{r['avg_faithfulness']:<7.2f} {r['avg_hallucination']:<7.2f} "
                  f"{r['composite_score']:<7.3f} {r['avg_latency_sec']:<7.2f} "
                  f"{r['total_cost_usd']:<8.5f}")

    print_table("SLM RANKING — Intent Recognition (Stage 1)", slm_ranked)
    print_table("LLM RANKING — Legal Reasoning (Stage 2)", llm_ranked)

    # ===== SAVE JSON =====
    best_slm = slm_ranked[0] if slm_ranked else None
    best_llm = llm_ranked[0] if llm_ranked else None

    # Strip per_case from top-level for cleaner summary (keep in full data)
    def strip_cases(r):
        return {k: v for k, v in r.items() if k != "per_case"}

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_used": [
            "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR",
            "Faithfulness (LLM-judge)", "Hallucination (LLM-judge)",
            "Latency (sec)", "Estimated Cost (USD)",
        ],
        "metric_definitions": {
            "rouge1": "Unigram overlap F-measure between model output and gold reference",
            "rouge2": "Bigram overlap F-measure between model output and gold reference",
            "rougeL": "Longest common subsequence F-measure between model output and gold reference",
            "bleu": "Sentence-level BLEU with smoothing (n-gram precision, brevity penalty)",
            "meteor": "METEOR score (unigram matching with stemming and synonyms)",
            "faithfulness": "LLM-judge: fraction of model claims grounded in input (0=none, 1=all)",
            "hallucination": "LLM-judge: fraction of output that is unsupported by input (0=none, 1=all)",
            "latency_sec": "Wall-clock seconds per API call",
            "cost_usd": "Estimated USD cost based on Groq token pricing",
            "composite_score": "(ROUGE1 + BLEU + METEOR + Faithfulness + (1-Hallucination)) / 5",
        },
        "judge_model": JUDGE_MODEL,
        "test_cases": [
            {
                "case_id": c.case_id,
                "gold_primary_intent": c.gold_primary_intent,
                "gold_intent_text": c.gold_intent_text,
                "gold_reasoning_text": c.gold_reasoning_text[:200] + "...",
            }
            for c in TEST_CASES
        ],
        "slm_benchmark": {
            "task": "intent_recognition",
            "models_tested": len(slm_ranked),
            "ranking": [strip_cases(r) for r in slm_ranked],
            "best_model": strip_cases(best_slm) if best_slm else None,
        },
        "llm_benchmark": {
            "task": "legal_reasoning",
            "models_tested": len(llm_ranked),
            "ranking": [strip_cases(r) for r in llm_ranked],
            "best_model": strip_cases(best_llm) if best_llm else None,
        },
        "comparative_analysis": {
            "slm_top3": [
                {
                    "model": r["model"],
                    "composite": r["composite_score"],
                    "latency_sec": r["avg_latency_sec"],
                    "cost_usd_5cases": r["total_cost_usd"],
                }
                for r in slm_ranked[:3]
            ],
            "llm_top3": [
                {
                    "model": r["model"],
                    "composite": r["composite_score"],
                    "latency_sec": r["avg_latency_sec"],
                    "cost_usd_5cases": r["total_cost_usd"],
                }
                for r in llm_ranked[:3]
            ],
        },
        "full_results_slm": slm_ranked,
        "full_results_llm": llm_ranked,
    }

    out_path = repo_root() / "output" / "groq_benchmark_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Full results saved: {out_path}")

    if best_slm:
        print(f"\n★ Best SLM for Intent: {best_slm['model']} "
              f"(score={best_slm['composite_score']:.3f})")
    if best_llm:
        print(f"★ Best LLM for Reasoning: {best_llm['model']} "
              f"(score={best_llm['composite_score']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
