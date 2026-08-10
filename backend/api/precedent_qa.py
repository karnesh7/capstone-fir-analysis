#!/usr/bin/env python3
"""
Precedent Q&A (Stage 3)
Interactive question-answering grounded in IndicLegalQA case precedents.
Uses Groq LLM to synthesize answers from retrieved QA pairs.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import textwrap
from groq import Groq
from dotenv import load_dotenv
from model_config import groq_chat_with_fallback

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()


def _format_stage1_json(stage1_result: dict) -> str:
    try:
        return json.dumps(stage1_result, indent=2, ensure_ascii=False)[:12000]
    except Exception:
        return str(stage1_result)[:12000]


def _format_stage2_cases(stage2_result: dict) -> str:
    lines = []
    vp = stage2_result.get("verdict_prediction") or {}
    lines.append(f"Verdict prediction (structured): {json.dumps(vp, ensure_ascii=False)}")
    for i, c in enumerate(stage2_result.get("cases") or [], 1):
        lines.append(
            f"[Case {i}] {c.get('title', '')}\n"
            f"  Court: {c.get('court', '')}\n"
            f"  Summary: {c.get('summary', '')}\n"
        )
    return "\n".join(lines)


def answer_question(question: str, fir: str, stage1_result: dict, stage2_result: dict, callback: callable = None) -> str:
    """
    Context-grounded Q&A using Stage 1 + Stage 2 results only (no new retrieval).
    """
    if callback:
        callback("Reviewing FIR details and identified legal sections from Stage 1...")
        callback("Integrating relevant precedent cases and verdict predictions from Stage 2...")
        callback("Synthesizing a grounded legal response based on the compiled context...")

    system_prompt = f"""You are a legal assistant analyzing a specific FIR case.

FIR TEXT:
{fir}

IDENTIFIED IPC/BNS SECTIONS (Stage 1):
{_format_stage1_json(stage1_result)}

PRECEDENT CASES FOUND (Stage 2):
{_format_stage2_cases(stage2_result)}

VERDICT PREDICTION:
{json.dumps(stage2_result.get('verdict_prediction') or {}, indent=2, ensure_ascii=False)}

Answer the user's question strictly based on the above context.
Do not search for new cases. Do not say "no precedents found".
If the answer is not in the context, say so explicitly."""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        return groq_chat_with_fallback(
            client,
            role="qa",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=700,
        )
    except Exception as e:
        return f"[LLM Error] {e}"


SYSTEM_PROMPT = """You are an Indian legal expert assistant. Answer the user's question 
using ONLY the provided case precedents retrieved from Indian court judgments. 

Rules:
- Cite the case name and date for every claim you make
- If the precedents don't contain relevant info, say so honestly
- Be concise (3-6 sentences)
- Do NOT invent cases or citations not in the provided precedents
- Relate your answer back to the user's FIR context when relevant"""


class PrecedentQA:
    """Synthesize answers from retrieved legal QA precedents using Groq LLM."""

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print(f"[PrecedentQA] Groq LLM ready (role=qa, fallback enabled)")

    def synthesize(self, user_question: str, retrieval_result: dict,
                   fir_summary: str = "", mapped_sections: list = None) -> str:
        """
        Build prompt from retrieved precedents + send to LLM.
        Returns the synthesized answer string.
        """
        status = retrieval_result.get("status", "no_match")
        precedents = retrieval_result.get("precedents", [])

        # If nothing found, return fallback
        if status == "no_match" or not precedents:
            return self._format_no_match(mapped_sections or [])

        # Build context block
        context_parts = []
        if fir_summary:
            context_parts.append(f"FIR Context: {fir_summary}")
        if mapped_sections:
            context_parts.append(f"Mapped Sections: {', '.join(mapped_sections)}")

        precedent_block = ""
        for i, p in enumerate(precedents[:5], 1):
            precedent_block += (
                f"\n[Precedent {i}] Case: {p['case_name']} ({p['date']})\n"
                f"  Q: {p['question']}\n"
                f"  A: {p['answer']}\n"
            )

        user_msg = (
            f"{'  '.join(context_parts)}\n\n"
            f"Retrieved Precedents:{precedent_block}\n\n"
            f"User Question: {user_question}"
        )

        # Call Groq LLM with fallback
        try:
            return groq_chat_with_fallback(
                self.client,
                role="qa",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            return f"[LLM Error] {e}"

    def _format_no_match(self, sections: list) -> str:
        lines = [
            "No relevant precedents found in the case database for this question.",
            "",
        ]
        if sections:
            lines.append("However, based on the mapped sections from Stage 1:")
            for s in sections:
                lines.append(f"  • {s}")
            lines.append("")
            lines.append("For specific case precedents, consult SCC Online or Indian Kanoon.")
        return "\n".join(lines)


def display_qa_answer(answer: str, is_no_match: bool = False, width: int = 70) -> str:
    """Format the QA answer for terminal display."""
    border = "=" * width
    lines = []
    lines.append(f"\n{'':>2}{border}")
    lines.append(f"{'':>2}  PRECEDENT Q&A")
    lines.append(f"{'':>2}{'-' * width}")
    lines.append("")

    if is_no_match:
        lines.append(f"{'':>4}⚠  {answer.splitlines()[0]}")
        for line in answer.splitlines()[1:]:
            lines.append(f"{'':>4}{line}")
    else:
        for para in answer.split("\n"):
            wrapped = textwrap.wrap(para, width=width - 6)
            for wl in wrapped:
                lines.append(f"{'':>4}{wl}")
            if not wrapped:
                lines.append("")

    lines.append("")
    lines.append(f"{'':>2}{border}")
    return "\n".join(lines)
