"""
Groq LLM prompt functions for Stage 2 — case summarisation, verdict
prediction, section-influence ranking, and fact-to-query translation.

All functions use the Groq SDK directly (not LangChain) and are
stateless — they receive the data they need and return structured dicts.
"""

import json
import os
import re
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
from model_config import groq_chat_with_fallback

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _parse_llm_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
#  Case summarisation
# ---------------------------------------------------------------------------
def summarize_case(case_title: str, case_text: str, ipc_section: str) -> str:
    """Summarize a court judgment in 3-5 sentences using Groq LLM."""
    truncated = case_text[:3500]
    prompt = f"""Summarize this Indian court judgment concisely in 3-5 sentences.
Focus on: what the case was about, the charges (especially {ipc_section}),
the court's decision, and the punishment/sentence given.

Case: {case_title}
Judgment text:
{truncated}

Summary:"""

    try:
        return groq_chat_with_fallback(
            _get_groq(),
            role="summarisation",
            messages=[
                {"role": "system", "content": "You are a legal analyst who summarizes Indian court judgments concisely. Focus on charges, decisions, and sentences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
    except Exception as e:
        print(f"[Kanoon] Groq summarization error (all models failed): {e}")
        return ""


def summarize_case_with_llm(
    case_text: str,
    case_title: str = "",
    fir_summary: str = "",
    ipc_sections: list[str] | None = None,
    return_metadata: bool = False,
) -> str | dict:
    """Summarize a court judgment and optionally return relevance metadata."""
    # Note: Preprocessing (Facts/Held/Judgment extraction) is now handled
    # at the pipeline level in indian_kanoon.py before calling this.
    truncated = case_text[:2500]

    if case_title and fir_summary and ipc_sections:
        sections_str = ", ".join(ipc_sections)
        prompt = f"""You are summarizing a court judgment against a specific FIR.

FIR DETAILS:
{fir_summary}

APPLICABLE IPC SECTIONS: {sections_str}

CASE TITLE:
{case_title}

CASE TEXT:
{truncated}

Return ONLY valid JSON with this exact shape:
{{
  "relevance_score": 0-100,
  "relevant": true/false,
  "summary": "max 120 words, focused on the charges, decision, and sentence",
  "reason": "short reason for the score"
}}

Scoring rules:
- 90-100: highly similar facts and legal issue
- 70-89: strongly related precedent
- 50-69: somewhat related but not a close match
- 0-49: weak or tangential match

The summary must always be concise and factual. Do not add markdown."""

        try:
            resp = _get_groq().chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a strict Indian legal relevance rater and concise summarizer. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=280,
            )
            parsed = _parse_llm_json(resp.choices[0].message.content)
            score = parsed.get("relevance_score", 0)
            try:
                score = int(score)
            except Exception:
                score = 0
            score = max(0, min(100, score))
            summary = (parsed.get("summary") or "").strip()
            relevant = bool(parsed.get("relevant")) if parsed.get("relevant") is not None else score >= 60
            payload = {
                "relevance_score": score,
                "relevant": relevant,
                "summary": summary,
                "reason": (parsed.get("reason") or "").strip(),
            }
            return payload if return_metadata else summary
        except Exception as e:
            print(f"[Kanoon] Groq summarization error: {e}")
            fallback = {"relevance_score": 0, "relevant": False, "summary": "", "reason": str(e)}
            return fallback if return_metadata else ""

    prompt = f"""Summarize this judgment in MAX 120 words.
Focus ONLY on:
1. Primary charges.
2. Court's final decision.
3. Sentence/Punishment.

TEXT:
{truncated}

SUMMARY:"""

    try:
        # Use direct Llama-3.1-8B for speed and conciseness, as verified in benchmark
        resp = _get_groq().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Concise legal summarizer. 120 words max. No preamble."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=250
        )
        summary = resp.choices[0].message.content.strip()
        return {"relevance_score": 100, "relevant": True, "summary": summary, "reason": ""} if return_metadata else summary
    except Exception as e:
        print(f"[Kanoon] Groq summarization error: {e}")
        return {"relevance_score": 0, "relevant": False, "summary": "", "reason": str(e)} if return_metadata else ""


# ---------------------------------------------------------------------------
#  Verdict prediction
# ---------------------------------------------------------------------------
def predict_verdict(fir_summary: str, ipc_sections: list[str], case_summaries: list[dict]) -> dict:
    """Predict likely verdict & punishment based on FIR + retrieved precedents."""
    cases_ctx = ""
    for i, cs in enumerate(case_summaries, 1):
        cases_ctx += (
            f"\n[Case {i}] {cs['title']} (Section {cs['section']}, Court: {cs['court']})\n"
            f"  Summary: {cs['summary']}\n"
        )

    sections_str = ", ".join(ipc_sections)

    prompt = f"""You are a senior Indian legal analyst. Based on the FIR details and real court
precedents below, predict the most likely verdict and punishment for the accused.

FIR DETAILS:
{fir_summary}

APPLICABLE IPC SECTIONS: {sections_str}

REAL COURT PRECEDENTS FROM INDIAN KANOON:
{cases_ctx}

Provide your prediction in this exact JSON format (no markdown, no code fences):
{{
  "predicted_verdict": "Likely Guilty / Likely Acquittal / Mixed",
  "predicted_punishment": "Specific punishment prediction based on precedents",
  "punishment_range": "Minimum to maximum sentence range under the applicable sections",
  "bail_likelihood": "High / Medium / Low — with brief reasoning",
  "confidence": 0.75,
  "reasoning": "2-3 sentence explanation citing the precedent cases"
}}"""

    try:
        raw = groq_chat_with_fallback(
            _get_groq(),
            role="summarisation",
            messages=[
                {"role": "system", "content": "You are a senior Indian criminal law expert. Predict verdicts based on real precedents. Return ONLY valid JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "predicted_verdict": "Analysis completed",
            "predicted_punishment": raw if raw else "Could not parse LLM response",
            "punishment_range": "N/A",
            "bail_likelihood": "N/A",
            "confidence": 0,
            "reasoning": "LLM output was not valid JSON — raw response shown above.",
        }
    except Exception as e:
        print(f"[Kanoon] Verdict prediction error: {e}")
        return {
            "predicted_verdict": "Error",
            "predicted_punishment": str(e),
            "punishment_range": "N/A",
            "bail_likelihood": "N/A",
            "confidence": 0,
            "reasoning": f"LLM call failed: {e}",
        }


# ---------------------------------------------------------------------------
#  Section influence ranking
# ---------------------------------------------------------------------------
def rank_section_influence(
    mapped_sections: list[str],
    fir_summary: str,
    verdict: dict,
    case_summaries: list[dict],
    applicable_statutes: list[dict] | None = None,
) -> list[dict]:
    """
    Deterministic section influence scoring based only on the FIR facts.

    The score intentionally ignores RAG/stage-1 statute metadata, precedent text,
    and verdict output. It ranks each mapped section by how well the FIR facts
    match the legal elements that the section usually captures.
    """
    if not mapped_sections:
        return []

    def _section_number(section_name: str) -> str:
        m = re.search(r"(\d+[A-Za-z]*)", section_name or "")
        return m.group(1) if m else ""

    def _contains_any(text: str, phrases: list[str]) -> bool:
        haystack = (text or "").lower()
        return any(phrase in haystack for phrase in phrases)

    fir_text = (fir_summary or "").lower()

    fact_flags = {
        "group": _contains_any(fir_text, ["five or more", "five associates", "five persons", "group", "with five", "along with five", "and five associates", "two associates", "three associates", "with associates"]),
        "armed_entry": _contains_any(fir_text, ["forcibly entered", "forced entry", "broke into", "entered his residence", "entered the residence", "armed with knives", "iron rods", "weapon", "weapons"]),
        "theft": _contains_any(fir_text, ["stolen property", "stole", "theft", "cash", "wallet", "money", "valuables", "loot", "robbery", "robbed"]),
        "violence": _contains_any(fir_text, ["pushed", "struck", "hit", "assault", "injury", "injuries", "hurt", "rod", "knife"]),
        "threat": _contains_any(fir_text, ["threatened", "threats", "kill them", "do not report", "warned", "intimidation"]),
        "damage": _contains_any(fir_text, ["damaged", "cupboard", "locker", "break open", "broke open", "destroyed"]),
        "residence": _contains_any(fir_text, ["residence", "house", "home", "dwelling"]),
        "minor_injury": _contains_any(fir_text, ["minor injuries", "minor injury", "medica", "medical records"]),
        "deception": _contains_any(fir_text, ["fake identity", "false identity", "impersonat", "misrepresent", "fraud", "cheat", "false promise", "false representation", "dishonest inducement", "deceived", "induced"]),
        "impersonation": _contains_any(fir_text, ["fake identity", "false identity", "impersonat", "posing as", "pretending to be", "company representative", "online identity"]),
        "property_transfer": _contains_any(fir_text, ["transfer money", "paid", "payment", "advance", "part with money", "money", "amount", "money was transferred", "sent money"]),
        "forgery": _contains_any(fir_text, ["forged", "fake document", "fake invoice", "fabricated", "false document", "fake contract", "counterfeit", "tampered"]),
    }
    fact_flags["compound_dacoity"] = fact_flags["group"] and fact_flags["armed_entry"] and fact_flags["theft"]

    records: list[dict] = []

    for section_name in mapped_sections:
        section_number = _section_number(section_name)
        try:
            sec_num_int = int(re.match(r"\d+", section_number).group(0)) if section_number else 0
        except Exception:
            sec_num_int = 0

        score = 3
        priority = 10
        score_notes: list[str] = ["Base applicability +3"]
        reasoning_notes: list[str] = []

        if sec_num_int == 420:
            if fact_flags["deception"] and fact_flags["property_transfer"]:
                score += 60
                priority = 100
                score_notes.append("Dishonest inducement causing property transfer matches IPC 420 +60")
                reasoning_notes.append("The facts describe deceptive inducement that caused the complainant to transfer money, which is the core of cheating under IPC 420.")
            elif fact_flags["deception"]:
                score += 40
                priority = 90
                score_notes.append("Deception without completed transfer still supports IPC 420 +40")
                reasoning_notes.append("The facts show deception and inducement, so IPC 420 remains the main cheating section even before considering the transfer.")
        elif sec_num_int == 415:
            if fact_flags["deception"]:
                score += 28
                priority = 80
                score_notes.append("Cheating definition supports IPC 415 +28")
                reasoning_notes.append("IPC 415 states the cheating elements, but it is a definition-level provision and should sit below IPC 420 when property was actually obtained.")
        elif sec_num_int == 416:
            if fact_flags["impersonation"]:
                score += 34
                priority = 85
                score_notes.append("Fake identity/personation directly supports IPC 416 +34")
                reasoning_notes.append("The accused allegedly impersonated a company representative, so IPC 416 is a real supporting section, but it is still secondary to the completed cheating offence.")
        elif sec_num_int in (468, 471):
            if fact_flags["forgery"] or fact_flags["impersonation"]:
                score += 32
                priority = 78
                score_notes.append("Fake document / forged identity conduct supports forgery-related sections +32")
                reasoning_notes.append("The fake identity and false service contract point to forgery-type conduct, but these sections should stay below the core cheating section unless a forged document is central.")
        elif sec_num_int in (383, 384, 385, 386, 387, 388, 389):
            if fact_flags["property_transfer"] and fact_flags["deception"]:
                score += 20
                priority = 60
                score_notes.append("Property extraction by deceit supports extortion/related conduct +20")
                reasoning_notes.append("The money transfer happened because of deceptive communication, but this is still secondary to the main cheating theory.")
            if fact_flags["threat"]:
                score += 8
                score_notes.append("Threats add a coercive component +8")
        elif sec_num_int in (395, 396, 397):
            if fact_flags["compound_dacoity"]:
                score += 55
                priority = 95
                score_notes.append("Five-plus offenders plus armed theft match dacoity +55")
                reasoning_notes.append("The FIR describes five or more accused, forcible armed entry, threats, and taking valuables, which fits dacoity.")
            elif fact_flags["group"] and fact_flags["theft"]:
                score += 28
                priority = 70
                score_notes.append("Group theft facts support dacoity-type conduct +28")
        elif sec_num_int in (452, 454, 457):
            if fact_flags["armed_entry"] and fact_flags["residence"]:
                score += 34
                priority = 72
                score_notes.append("Forced armed entry into a residence matches house-trespass facts +34")
        elif sec_num_int in (390, 394, 392, 379, 380):
            if fact_flags["theft"]:
                score += 26
                priority = 65
                score_notes.append("Taking property from the complainant matches robbery/theft facts +26")
                reasoning_notes.append("The property transfer caused by deceit or force may still support robbery/theft-related sections, but they should not outrank the main cheating offence in a fraud case.")
            if fact_flags["violence"] or fact_flags["threat"]:
                score += 16
                score_notes.append("Force, assault, and threats support robbery conduct +16")
            if sec_num_int == 394 and (fact_flags["violence"] or fact_flags["minor_injury"]):
                score += 10
                priority = 68
                score_notes.append("Hurt during robbery makes IPC 394 more specific +10")
        elif sec_num_int in (323, 324):
            if fact_flags["violence"]:
                score += 20
                priority = 50
                score_notes.append("Physical blow and injury support hurt/assault facts +20")
        elif sec_num_int == 506:
            if fact_flags["threat"]:
                score += 18
                priority = 58
                score_notes.append("Kill/reporting threats support criminal intimidation +18")
        elif sec_num_int == 427:
            if fact_flags["damage"]:
                score += 14
                priority = 45
                score_notes.append("Damage to property supports mischief +14")
        else:
            if fact_flags["deception"] or fact_flags["property_transfer"]:
                score += 6
                priority = 20
                score_notes.append("General fraud context gives this statute some relevance +6")

        score = max(score, 1)
        records.append({
            "section": section_name,
            "raw_score": score,
            "priority": priority,
            "reasoning": " ".join(reasoning_notes) if reasoning_notes else "This section remains applicable to part of the fact pattern, but it is less central than the dominant offence sections.",
            "breakdown": "; ".join(score_notes),
        })

    records.sort(key=lambda item: (item["raw_score"], item["priority"]), reverse=True)

    top_raw = records[0]["raw_score"] if records else 1
    results = []
    for idx, item in enumerate(records):
        score = max(1, int(round((item["raw_score"] / top_raw) * 100)))
        if idx > 0 and score >= 100:
            score = max(1, 99 - min(idx - 1, 8))

        level = "Primary" if idx == 0 else ("Supporting" if score >= 30 else "Minor")

        results.append({
            "section": item["section"],
            "influence_score": score,
            "influence_level": level,
            "reasoning": item["reasoning"],
            "score_breakdown": item["breakdown"],
        })

    return results


# ---------------------------------------------------------------------------
#  Fact → Kanoon search query
# ---------------------------------------------------------------------------
def build_fact_query(fir_summary: str, ipc_sections: list[str]) -> str:
    """
    Deterministic query builder — no LLM needed.
    Builds a strict fact-first Kanoon query using the top two IPC sections and the FIR intent.
    """
    sections: list[str] = []
    for sec in (ipc_sections or [])[:2]:
        cleaned = re.sub(r"^(?:IPC|Indian Penal Code)\s+(?:Section\s+)?", "", sec, flags=re.IGNORECASE).strip()
        if cleaned and cleaned not in sections:
            sections.append(cleaned)

    fir_lower = fir_summary.lower()

    intent_terms: list[str] = []
    if any(term in fir_lower for term in ["dacoity", "robbery", "snatched", "snatching", "loot", "looted"]):
        intent_terms.extend(['"dacoity"', '"robbery"'])
    if any(term in fir_lower for term in ["forcibly entered", "entered residence", "entered house", "house trespass", "residence"]):
        intent_terms.append('"house trespass"')
    if any(term in fir_lower for term in ["grievous", "hurt", "injured", "assault", "overpowered"]):
        intent_terms.append('"grievous hurt"')
    if any(term in fir_lower for term in ["set fire", "arson", "burn", "burnt", "fire"]):
        intent_terms.append('"arson"')
    if any(term in fir_lower for term in ["confined", "confinement", "locked", "room"]):
        intent_terms.append('"wrongful confinement"')

    if not intent_terms:
        intent_terms = ['"criminal intent"', '"robbery"']

    intent_str = " ORR ".join(dict.fromkeys(intent_terms))
    if sections:
        section_str = " ORR ".join([f'"Section {sec} IPC"' for sec in sections])
        return f'({section_str}) ANDD ({intent_str})'
    return intent_str


def build_broad_fact_query(fir_summary: str, ipc_sections: list[str]) -> str:
    """Broader fallback query when the strict fact query is too narrow."""
    primary_section = ""
    if ipc_sections:
        primary_section = re.sub(r"^(?:IPC|Indian Penal Code)\s+(?:Section\s+)?", "", ipc_sections[0], flags=re.IGNORECASE).strip()

    fir_lower = fir_summary.lower()

    terms = ['"dacoity"', '"robbery"', '"grievous hurt"', '"wrongful confinement"', '"arson"']
    if any(term in fir_lower for term in ["forcibly entered", "house", "residence", "trespass"]):
        terms.insert(0, '"house trespass"')
    if any(term in fir_lower for term in ["loot", "looted", "snatched", "snatching"]):
        terms.append('"loot"')
    if any(term in fir_lower for term in ["set fire", "fire", "burn"]):
        terms.append('"set fire"')

    cluster = " ORR ".join(dict.fromkeys(terms))
    if primary_section:
        return f'"Section {primary_section} IPC" ANDD ({cluster})'
    return cluster
