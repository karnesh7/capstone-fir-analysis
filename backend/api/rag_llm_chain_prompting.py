#!/usr/bin/env python3
"""
Statute-Aware RAG with LangChain Chain Prompting
=================================================
FIR Analysis → Intent Classification → Legal Reasoning → Applicable Statutes

This module owns the StatuteRAGChainSystem class which:
  - connects to Pinecone for vector retrieval
  - uses Groq LLMs for intent classification → legal reasoning (2-chain pipeline)
  - enriches results with IPC/BNS mappings and statute extracts
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from intent_queries import intent_to_retrieval_queries
from model_config import get_fallback_chain

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

NEGATIVE_RULE_THRESHOLD = 0.75


def apply_negative_rule_filtering(
    chunks: list,
    negative_rules: dict,
    offence_map: dict,
    embedding_model,
) -> list:
    """
    offence_map: chunk_id -> offence label (None if chunk matches no offence key in negative_rules).
    negative_rules: offence_label -> list of negative-rule phrases for that offence only.
    """
    if not chunks:
        return []

    kept: list = []
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        offence_label = offence_map.get(cid)
        phrases = negative_rules.get(offence_label) if offence_label else None
        if not phrases:
            kept.append(chunk)
            continue

        text = (chunk.get("full_text") or chunk.get("section_text") or "").strip()
        if not text:
            kept.append(chunk)
            continue

        embeddings = embedding_model.encode([text] + list(phrases))
        chunk_vec = np.asarray(embeddings[0], dtype=np.float64)
        rule_mat = np.asarray(embeddings[1:], dtype=np.float64)
        if rule_mat.size == 0:
            kept.append(chunk)
            continue

        nc = np.linalg.norm(chunk_vec)
        if nc == 0:
            kept.append(chunk)
            continue
        max_sim = 0.0
        for rv in rule_mat:
            nr = np.linalg.norm(rv)
            if nr == 0:
                continue
            sim = float(np.dot(chunk_vec, rv) / (nc * nr))
            if sim > max_sim:
                max_sim = sim

        if max_sim > NEGATIVE_RULE_THRESHOLD:
            continue
        kept.append(chunk)
    return kept


class StatuteRAGChainSystem:
    """Two-chain RAG pipeline: intent identification → legal reasoning."""

    def __init__(self):
        # Groq API key
        self._groq_key = os.environ.get("GROQ_API_KEY")

        # Primary LLMs (first model in each fallback chain)
        self._intent_models = get_fallback_chain("slm_intent")
        self._reasoning_models = get_fallback_chain("llm_reasoning")

        self.llm_intent = ChatGroq(
            model=self._intent_models[0],
            temperature=0.3,
            api_key=self._groq_key,
            request_timeout=60,
        )
        self.llm_reasoning = ChatGroq(
            model=self._reasoning_models[0],
            temperature=0.4,
            api_key=self._groq_key,
            request_timeout=60,
        )
        print(f"✓ Groq LLMs initialized (intent: {self._intent_models[0]}, reasoning: {self._reasoning_models[0]})")
        print(f"  Fallback intent models:    {self._intent_models[1:]}")
        print(f"  Fallback reasoning models: {self._reasoning_models[1:]}")

        # Pinecone
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index("statute-embeddings")
        print("✓ Pinecone vector DB connected")

        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Embedding model loaded")

        # Static data
        self.statute_chunks = self._load_jsonl(REPO_ROOT / "output" / "statute_chunks_complete.jsonl", key="chunk_id")
        self.statute_mappings = self._load_json(REPO_ROOT / "output" / "ipc_bns_mappings_complete.json")
        self.negative_rules = {
            r["offence"]: r
            for r in self._load_json(REPO_ROOT / "output" / "negative_rules_comprehensive.json")
        }
        print(f"✓ Loaded {len(self.statute_chunks)} chunks, {len(self.statute_mappings)} mappings, {len(self.negative_rules)} rules")

        # Prompt templates
        self._init_prompts()

    # ------------------------------------------------------------------
    #  Static data loaders
    # ------------------------------------------------------------------
    @staticmethod
    def _load_jsonl(path: Path, key: str) -> dict:
        data: dict = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                data[obj[key]] = obj
        return data

    @staticmethod
    def _load_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    #  Prompt templates
    # ------------------------------------------------------------------
    def _init_prompts(self):
        self.intent_prompt = PromptTemplate(
            input_variables=["complainant", "accused", "incident", "victim_impact", "evidence"],
            template="""You are a legal analyst. Analyze the FIR case details and identify the PRIMARY crime/intent.

CASE DETAILS:
Complainant: {complainant}
Accused: {accused}
Incident: {incident}
Victim Impact: {victim_impact}
Evidence: {evidence}

Based on the facts presented, identify:
1. The PRIMARY intent/crime type
2. Confidence level (0-1)
3. Any secondary intents

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "primary_intent": "string",
  "confidence": 0.95,
  "secondary_intents": ["string"]
}}""",
        )

        self.reasoning_prompt = PromptTemplate(
            input_variables=["incident", "victim_impact", "evidence", "accused", "victim", "primary_intent", "statute_context"],
            template="""You are a legal expert analyzing applicable criminal statutes.

CASE FACTS:
Incident: {incident}
Victim Impact: {victim_impact}
Evidence: {evidence}
Accused: {accused}
Victim: {victim}
Identified Intent: {primary_intent}

RELEVANT STATUTE SECTIONS:
{statute_context}

Determine which statutes are APPLICABLE to this case:
- First identify any UMBRELLA / COMPOSITE offences (e.g. IPC 498A for matrimonial cruelty,
  Dowry Prohibition Act) whose broad elements are satisfied by the overall pattern of facts.
- Then identify SPECIFIC offences for individual acts described (hurt, theft, breach of trust, etc.)
- Include BOTH umbrella charges AND specific charges — do NOT omit an umbrella section
  just because its constituent acts are already covered by narrower sections.
- Explain WHY each statute applies
- Assess overall severity (high/medium/low)
- Provide confidence in your analysis

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "applicable_statutes": [
    {{"section": "string", "law": "string", "reasoning": "string"}}
  ],
  "legal_basis": "string",
  "severity_assessment": "string",
  "confidence": 0.95
}}""",
        )

    # ------------------------------------------------------------------
    #  Vector retrieval
    # ------------------------------------------------------------------
    def retrieve_relevant_statutes(self, query_text: str, top_k: int = 8) -> list[dict]:
        embedding = self.embedding_model.encode(query_text).tolist()
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)

        chunks = []
        for match in results["matches"]:
            cid = match["id"]
            if cid in self.statute_chunks:
                c = self.statute_chunks[cid]
                chunks.append({
                    "chunk_id": cid,
                    "law": c["law"],
                    "section_id": c["section_id"],
                    "section_text": c["section_text"][:200],
                    "full_text": c["section_text"],
                    "offence_type": c.get("offence_type", "unknown"),
                    "similarity_score": match["score"],
                })
        return chunks

    # ------------------------------------------------------------------
    #  Filtering & enrichment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_chunks(base: list[dict], new: list[dict]) -> list[dict]:
        seen = {c["chunk_id"] for c in base}
        merged = list(base)
        for c in new:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                merged.append(c)
        return merged

    def apply_negative_rules_filter(self, chunks: list[dict], case_facts: dict) -> list[dict]:
        del case_facts  # offence-scoped semantic filter does not use raw FIR string matching
        negative_rules: dict[str, list[str]] = {
            r["offence"]: list(r.get("failure_examples", []))
            for r in self.negative_rules.values()
        }
        labels_sorted = sorted(negative_rules.keys(), key=len, reverse=True)
        offence_map: dict[str, str | None] = {}
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            ft = (chunk.get("full_text") or "").lower()
            matched: str | None = None
            for label in labels_sorted:
                if label.lower() in ft:
                    matched = label
                    break
            offence_map[cid] = matched
        return apply_negative_rule_filtering(
            chunks, negative_rules, offence_map, self.embedding_model,
        )

    def find_corresponding_sections(self, law: str, section_id: str) -> list[dict]:
        corresponding = []
        for m in self.statute_mappings:
            if law == "IPC" and m["ipc"] == section_id:
                corresponding.append({"law": "BNS", "section_id": m["bns"]})
            elif law == "BNS" and m["bns"] == section_id:
                corresponding.append({"law": "IPC", "section_id": m["ipc"]})

        if not corresponding:
            llm_section = self._llm_map_section(law, section_id, "BNS" if law == "IPC" else "IPC")
            if llm_section:
                corresponding.append({"law": "BNS" if law == "IPC" else "IPC", "section_id": llm_section})
        return corresponding

    def _llm_map_section(self, source_law: str, section_id: str, target_law: str) -> str | None:
        try:
            prompt = (
                f"What is the corresponding {target_law} section number for "
                f"{source_law} Section {section_id}? Reply with ONLY the section number. "
                f'If unsure, reply "unknown".'
            )
            result = self.llm_intent.invoke(prompt).content.strip()
            m = re.search(r"\b(\d+[A-Za-z]*)\b", result)
            if m and result.lower() != "unknown":
                mapped = m.group(1)
                print(f"  [LLM Mapping] {source_law} {section_id} → {target_law} {mapped}")
                return mapped
        except Exception as e:
            print(f"  [LLM Mapping] Failed for {source_law} {section_id}: {e}")
        return None

    def get_section_extract(self, law: str, section_id: str) -> str:
        for fmt in [f"{law.lower()}_{section_id}", f"{law}_{section_id}",
                    f"{law.lower()}{section_id}", f"{law}{section_id}"]:
            if fmt in self.statute_chunks:
                return self.statute_chunks[fmt]["section_text"][:150].replace("\n", " ")
        for chunk in self.statute_chunks.values():
            if chunk["section_id"] == section_id and chunk["law"] == law:
                return chunk["section_text"][:150].replace("\n", " ")
        return f"[{law} {section_id} not in extracted database]"

    # ------------------------------------------------------------------
    #  Section parsing from LLM output
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_section_ref(raw: str) -> tuple[str, str]:
        """Parse 'IPC 498A', 'BNS Section 85', etc. into (law, section_id)."""
        m = re.match(r"(?:IPC|BNS)\s+(?:Section\s+)?(.+)", raw.strip(), re.IGNORECASE)
        if m:
            law = "IPC" if raw.strip().upper().startswith("IPC") else "BNS"
            return law, m.group(1).strip()
        parts = raw.split()
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "IPC", parts[-1] if parts else "394"

    # ------------------------------------------------------------------
    #  LLM invoke with fallback
    # ------------------------------------------------------------------
    def _invoke_with_fallback(self, role: str, prompt_template: PromptTemplate,
                              inputs: dict) -> str:
        """Invoke an LLM with automatic fallback on any API/rate-limit error."""
        models = self._intent_models if role == "intent" else self._reasoning_models
        temp = 0.3 if role == "intent" else 0.4
        formatted = prompt_template.format(**inputs)
        last_exc: Exception | None = None

        for model_name in models:
            llm = ChatGroq(model=model_name, temperature=temp, api_key=self._groq_key, request_timeout=60)
            for attempt in range(2):  # 1 initial + 1 retry
                try:
                    return llm.invoke(formatted).content
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    print(f"[Fallback] {role} model '{model_name}' attempt {attempt+1} failed: {e}")
                    if attempt == 0 and any(kw in msg for kw in ("rate_limit", "429", "capacity", "overloaded")):
                        wait = 2
                        print(f"[Fallback] Retrying '{model_name}' in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[Fallback] Skipping '{model_name}', trying next fallback model...")
                        break

        raise last_exc or RuntimeError(f"All {role} fallback models failed")

    # ------------------------------------------------------------------
    #  Main analysis pipeline
    # ------------------------------------------------------------------
    def analyze_fir_with_chains(self, fir_data: dict, callback: callable = None) -> dict:
        print("\n" + "=" * 80)
        print("STATUTE-AWARE RAG WITH LANGCHAIN CHAIN PROMPTING")
        print("=" * 80)

        def _notify(thought: str):
            if callback:
                try:
                    callback(thought)
                except Exception as cb_err:
                    print(f"[Callback Warning] {cb_err}")

        case_facts = {
            "incident": fir_data.get("incident_description", ""),
            "victim_impact": fir_data.get("victim_impact", ""),
            "evidence": fir_data.get("evidence", ""),
        }

        # Step 0  — initial vector retrieval
        _notify("Retrieving initial statute candidates from Pinecone vector database...")
        query_text = f"{case_facts['incident']} {case_facts['victim_impact']}"
        retrieved_chunks = self.retrieve_relevant_statutes(query_text, top_k=20)
        print(f"✓ Retrieved {len(retrieved_chunks)} statute chunks")

        # Step 1  — negative-rules filter
        _notify("Applying negative rule semantic filtering to exclude irrelevant sections...")
        filtered_chunks = self.apply_negative_rules_filter(retrieved_chunks, case_facts)
        print(f"✓ Filtered to {len(filtered_chunks)} applicable chunks")

        # Step 2  — Chain 1: intent identification
        _notify("Analyzing case facts to identify primary criminal intent using LLM...")
        intent_input = {
            "complainant": fir_data.get("complainant_name", "Unknown"),
            "accused": ", ".join(fir_data.get("accused_names", [])),
            "incident": fir_data.get("incident_description", "Unknown"),
            "victim_impact": fir_data.get("victim_impact", "Unknown"),
            "evidence": fir_data.get("evidence", "Unknown"),
        }
        intent_result_str = self._invoke_with_fallback("intent", self.intent_prompt, intent_input)
        try:
            intent_result = json.loads(intent_result_str)
        except Exception:
            intent_result = {"primary_intent": "Unknown", "confidence": 0, "secondary_intents": []}

        print(f"✓ Primary Intent: {intent_result.get('primary_intent')}")

        # Step 3  — intent-driven second retrieval pass
        _notify(f"Performing targeted retrieval for intent: {intent_result.get('primary_intent')}...")
        for iq in intent_to_retrieval_queries(
            intent_result.get("primary_intent", ""),
            intent_result.get("secondary_intents", []),
        ):
            extra = self.retrieve_relevant_statutes(iq, top_k=8)
            filtered_chunks = self._merge_chunks(filtered_chunks, extra)
        filtered_chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
        print(f"✓ After intent-driven retrieval: {len(filtered_chunks)} unique chunks")

        # Step 4  — Chain 2: legal reasoning
        _notify("Synthesizing legal reasoning and mapping applicable IPC/BNS sections...")
        statute_ctx = "\n".join(
            f"[{c['law']} {c['section_id']}] {c['section_text']}" for c in filtered_chunks[:20]
        )
        reasoning_input = {
            "incident": fir_data.get("incident_description", "Unknown"),
            "victim_impact": fir_data.get("victim_impact", "None reported"),
            "evidence": fir_data.get("evidence", "None reported"),
            "accused": ", ".join(fir_data.get("accused_names", [])),
            "victim": fir_data.get("victim_name", "Unknown"),
            "primary_intent": intent_result.get("primary_intent", "Unknown"),
            "statute_context": statute_ctx,
        }
        reasoning_str = self._invoke_with_fallback("reasoning", self.reasoning_prompt, reasoning_input)
        try:
            reasoning_result = json.loads(reasoning_str)
        except Exception:
            reasoning_result = {
                "applicable_statutes": [], "legal_basis": "Error parsing",
                "severity_assessment": "unknown", "confidence": 0,
            }

        print(f"✓ Applicable Statutes: {len(reasoning_result.get('applicable_statutes', []))}")

        # Step 5  — enrich statutes with mappings & extracts
        enriched = self._enrich_statutes(reasoning_result.get("applicable_statutes", []))

        return {
            "status": "success",
            "chain_prompting_stages": 2,
            "fir_summary": {
                "complainant": fir_data.get("complainant_name"),
                "accused": fir_data.get("accused_names"),
                "incident": fir_data.get("incident_description", "")[:150],
            },
            "analysis": {
                "intent_identification": intent_result,
                "legal_reasoning": reasoning_result,
            },
            "retrieved_data": {
                "total_chunks_retrieved": len(retrieved_chunks),
                "chunks_after_filtering": len(filtered_chunks),
                "top_chunks_used": [
                    {"law": c["law"], "section": c["section_id"],
                     "score": round(c["similarity_score"], 4),
                     "preview": c["section_text"][:100]}
                    for c in filtered_chunks[:3]
                ],
            },
            "applicable_statutes": enriched,
            "severity": reasoning_result.get("severity_assessment", "unknown"),
            "confidence": reasoning_result.get("confidence", 0),
        }

    # ------------------------------------------------------------------
    #  Statute enrichment (private helper)
    # ------------------------------------------------------------------
    def _enrich_statutes(self, raw_statutes: list[dict]) -> list[dict]:
        enriched = []
        for statute in raw_statutes:
            law, section_id = self._parse_section_ref(statute.get("section", ""))
            corresponding = self.find_corresponding_sections(law, section_id)
            extract = self.get_section_extract(law, section_id)

            entry: dict = {
                "primary": {
                    "law": law, "section": section_id,
                    "title": statute.get("law", ""),
                    "reasoning": statute.get("reasoning", ""),
                    "extract": extract,
                },
                "corresponding_sections": [
                    {"law": c["law"], "section": c["section_id"],
                     "extract": self.get_section_extract(c["law"], c["section_id"])}
                    for c in corresponding
                ],
            }
            enriched.append(entry)
        return enriched


# ---------------------------------------------------------------------------
#  Convenience loader (used by server.py for sample FIR)
# ---------------------------------------------------------------------------
def create_sample_fir() -> dict:
    fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
    with open(fir_path, "r", encoding="utf-8") as f:
        return json.load(f)
