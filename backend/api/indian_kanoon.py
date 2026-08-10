#!/usr/bin/env python3

"""

Indian Kanoon API — Case Law Search & Pipeline Orchestration (Stage 2)

======================================================================

Searches Indian Kanoon for real case law matching IPC sections from Stage 1,

fetches judgment text, and delegates to groq_prompts for summarisation,

verdict prediction, and section-influence ranking.



API: https://api.indiankanoon.org  (500 free calls/day)

"""



import os

import re

import time

from collections import Counter
from pathlib import Path

import json

import numpy as np

import requests

from dotenv import load_dotenv



from kanoon_cache import cache_key, load_cache, save_cache

from groq_prompts import (

    predict_verdict,

    rank_section_influence,

    build_fact_query,

    build_broad_fact_query,

    summarize_case_with_llm,

)



REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()



# ---------------------------------------------------------------------------

#  Config

# ---------------------------------------------------------------------------

KANOON_BASE_URL = "https://api.indiankanoon.org"

KANOON_API_KEY = os.environ.get("KANOON_API_KEY", "")



MAX_PER_SECTION = 2       # cases per IPC section

MAX_TOTAL_CASES = 5       # total cases to fetch & summarise





# ---------------------------------------------------------------------------

#  Low-level helpers

# ---------------------------------------------------------------------------

def _headers() -> dict:

    return {"Authorization": f"Token {KANOON_API_KEY}", "Accept": "application/json"}





def _clean_html(text: str) -> str:

    return re.sub(r"<[^>]+>", "", text).strip()





def _extract_date_from_title(title: str) -> str:

    m = re.search(r"on\s+(\d{1,2}\s+\w+,?\s+\d{4})", title)

    if m:

        return m.group(1)

    m = re.search(r"\((\d{4})\)", title)

    return m.group(1) if m else ""





def _extract_ipc_sections(mapped_sections: list[str]) -> list[str]:

    sections = []

    for s in mapped_sections:

        m = re.match(r"(?:IPC|Indian Penal Code)\s+(?:Section\s+)?(\d+[A-Za-z]*)", s.strip(), re.IGNORECASE)

        if m:

            sections.append(m.group(1))

    return list(dict.fromkeys(sections))





def _tokenize_words(text: str) -> list[str]:

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def preprocess_judgment(text: str) -> str:
    """Refined preprocessor: Facts, Held, Judgment extraction."""
    markers = [
        r"(?i)facts\s+of\s+the\s+case",
        r"(?i)brief\s+facts",
        r"(?i)held",
        r"(?i)conclusion",
        r"(?i)judgment",
        r"(?i)order",
        r"(?i)findings",
    ]
    parts = re.split(r"\n\s*(?:\d+\.|\b[A-Z][A-Z\s]+\b)\s*", text)
    kept = []
    for part in parts:
        if any(re.search(marker, part[:100]) for marker in markers):
            kept.append(part.strip())

    if not kept:
        if len(text) > 2500:
            return text[:1500] + "\n... [truncated] ...\n" + text[-1000:]
        return text
    return "\n\n".join(kept)[:2500]


def summarize_text(text: str) -> str:

    """

    Extractive summary via per-sentence TF–IDF (matrix over sentences × terms).

    Top-K sentences (K = max(3, n//5), capped at n), returned in original order.

    """

    if not text or not str(text).strip():

        return ""



    raw = str(text).strip()

    parts = re.split(r"(?<=[.!?])\s+", raw)

    sentences = [p.strip() for p in parts if p.strip()]

    if not sentences:

        return ""



    n = len(sentences)

    sentence_tokens = [_tokenize_words(s) for s in sentences]

    if not any(sentence_tokens):

        return sentences[0]



    vocab = sorted({w for tokens in sentence_tokens for w in tokens})

    if not vocab:

        return sentences[0]



    w2i = {w: i for i, w in enumerate(vocab)}

    V = len(vocab)

    tf = np.zeros((n, V), dtype=np.float64)

    for i, tokens in enumerate(sentence_tokens):

        if not tokens:

            continue

        den = len(tokens)

        for w, cnt in Counter(tokens).items():

            tf[i, w2i[w]] = cnt / den



    df = np.zeros(V, dtype=np.float64)

    for j, term in enumerate(vocab):

        df[j] = sum(1 for tokens in sentence_tokens if term in tokens)

    idf = np.log(n / np.maximum(df, 1.0))



    scores = (tf * idf).sum(axis=1)

    k = min(max(3, n // 5), n)

    top_idx = np.argsort(-scores)[:k]

    selected_indices = sorted(int(i) for i in top_idx)

    return " ".join(sentences[i] for i in selected_indices)





def _is_actual_case(title: str, docsource: str) -> bool:

    """Return True only if the document looks like a court judgment."""

    title_lower = title.lower()

    source_lower = docsource.lower()



    if re.match(r"^Section\s+\d+", title, re.IGNORECASE):

        return False



    skip_keywords = [

        "penal code", "ranbir penal code", "criminal procedure code",

        "- act", "- rules", "- regulation", "- ordinance", "- bill",

        "constitution of india", "bare act",

    ]

    for kw in skip_keywords:

        if kw in title_lower or kw in source_lower:

            return False



    act_sources = ["union of india - section", "state of ", "central government"]

    for src in act_sources:

        if source_lower.startswith(src) and "vs" not in title_lower:

            return False



    if docsource.endswith("- Act") or docsource.endswith("- Code"):

        return False



    return True





# ---------------------------------------------------------------------------

#  Kanoon API calls

# ---------------------------------------------------------------------------

def search_kanoon(query: str, page: int = 0, maxpages: int = 1) -> dict:

    ck = cache_key(f"search:{query}:p{page}:m{maxpages}")

    cached = load_cache(ck)

    if cached:

        return cached

    try:
        data_payload = {
            "formInput": query,
            "pagenum": page,
            "doctypes": "judgments",
            "fromdate": "01-01-2000",
            "sortby": "mostrecent",
            "maxpages": maxpages,
        }
        
        import urllib.parse
        full_url = f"{KANOON_BASE_URL}/search/?{urllib.parse.urlencode(data_payload)}"
        print(f"\n[Kanoon API] RAW URL: {full_url}")
        print(f"  Filter:   doctypes=judgments, fromdate=01-01-2000, sortby=mostrecent, maxpages={maxpages}")
        
        resp = requests.post(f"{KANOON_BASE_URL}/search/", headers=_headers(), data=data_payload, timeout=15)

        resp.raise_for_status()

        data = resp.json()
        if data.get("docs"):
            print(f"\n[Kanoon API] RAW FIRST RESULT JSON:\n{json.dumps(data['docs'][0], indent=2)}")

        save_cache(ck, data)

        return data
    except requests.exceptions.HTTPError as e:

        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}", "docs": []}

    except requests.exceptions.RequestException as e:

        return {"error": str(e), "docs": []}





def get_doc_detail(tid: int) -> dict:

    """Fetch full judgment text from Indian Kanoon by TID."""

    ck = cache_key(f"doc:{tid}")

    cached = load_cache(ck)

    if cached:

        return cached

    try:

        resp = requests.post(f"{KANOON_BASE_URL}/doc/{tid}/", headers=_headers(), timeout=20)

        resp.raise_for_status()

        data = resp.json()

        save_cache(ck, data)

        return data

    except Exception as e:

        return {"error": str(e)}





# ---------------------------------------------------------------------------

#  Main pipeline: search → fetch docs → summarise → predict verdict

# ---------------------------------------------------------------------------

def search_and_analyze(

    mapped_sections: list[str],

    fir_summary: str = "",

    max_per_section: int = MAX_PER_SECTION,

    max_total: int = MAX_TOTAL_CASES,
    applicable_statutes: list[dict] | None = None,
    callback: callable = None,

) -> dict:

    """

    Full Stage 2 pipeline:

    1. Search Indian Kanoon using fact-enriched queries with IPC sections

    2. Fetch judgment text for top cases

    3. Summarise each case with Groq LLM (via groq_prompts)

    4. Predict verdict/punishment based on all precedents

    5. Rank section influence on verdict

    """

    print("\n" + "=" * 80)

    print("INDIAN KANOON API - EXACT INPUTS")

    print("=" * 80)

    print(f"\n[INPUT 1] mapped_sections (from Stage 1):")

    for i, sec in enumerate(mapped_sections, 1):

        print(f"  {i}. {sec}")

    print(f"\n[INPUT 2] fir_summary (first 600 chars):")

    print(f"  {fir_summary}")

    print("\n" + "=" * 80)



    if not KANOON_API_KEY:

        return {"status": "error", "sections_searched": [], "cases": [],

                "verdict_prediction": None, "api_calls_used": 0,

                "error": "KANOON_API_KEY is not set in .env"}



    ipc_sections = _extract_ipc_sections(mapped_sections)

    if callback: callback(f"Extracted relevant IPC sections for search: {', '.join(ipc_sections)}")
    print(f"\n[STEP 1] Extracted IPC sections: {ipc_sections}")

    if not ipc_sections and not fir_summary:

        return {"status": "no_results", "sections_searched": [], "cases": [],

                "verdict_prediction": None, "api_calls_used": 0, "error": None}



    all_cases: list[dict] = []

    seen_tids: set = set()

    api_calls = 0



    def _add_cases_from_result(result, section_label, limit):

        added = 0

        if result.get("error"):

            print(f"[Kanoon] Search error: {result['error']}")

            return added

        for doc in result.get("docs", []):

            if added >= limit or len(all_cases) >= max_total:

                break

            tid = doc.get("tid")

            if tid in seen_tids:

                continue

            title = _clean_html(doc.get("title", "Unknown Case"))

            docsource = doc.get("docsource", "Unknown Court")

            if not _is_actual_case(title, docsource):

                continue

            seen_tids.add(tid)

            all_cases.append({

                "title": title, "tid": tid, "court": docsource,

                "snippet": _clean_html(doc.get("headline", ""))[:500],

                "section": section_label,

                "date": _extract_date_from_title(title),

                "url": f"https://indiankanoon.org/doc/{tid}/" if tid else "",

                "summary": "",

            })

            added += 1

        return added



    # Strategy 1: LLM-crafted fact+section query

    fact_query = build_fact_query(fir_summary, ipc_sections) if fir_summary else ""
    broad_fact_query = build_broad_fact_query(fir_summary, ipc_sections) if fir_summary else ""

    if fact_query:

        if callback: callback(f"Searching Indian Kanoon with fact-based query: {fact_query}")
        print(f"\n[KANOON SEARCH QUERY 1 - Deterministic]\n  Query: \"{fact_query}\"")

        result = search_kanoon(fact_query)

        api_calls += 1

        _add_cases_from_result(result, "Fact-match", min(3, max_total))

        if len(all_cases) < 2:
            page_2_result = search_kanoon(fact_query, page=1, maxpages=1)
            api_calls += 1
            _add_cases_from_result(page_2_result, "Fact-match", min(2, max_total - len(all_cases)))

    if len(all_cases) < 2 and broad_fact_query:
        if callback: callback(f"Broadening Indian Kanoon search with crime-cluster query: {broad_fact_query}")
        print(f"\n[KANOON SEARCH QUERY 2 - Broader]\n  Query: \"{broad_fact_query}\"")

        broad_result = search_kanoon(broad_fact_query)
        api_calls += 1
        _add_cases_from_result(broad_result, "Broad-match", min(3, max_total - len(all_cases)))

        if len(all_cases) < 2:
            broad_page_2 = search_kanoon(broad_fact_query, page=1, maxpages=1)
            api_calls += 1
            _add_cases_from_result(broad_page_2, "Broad-match", min(2, max_total - len(all_cases)))



    # Strategy 2: Section-focused fallback using only the primary IPC section

    primary_sections = ipc_sections[:1]

    for i, sec in enumerate(primary_sections, 1):

        if len(all_cases) >= max_total:

            break

        query = f'"Section {sec} IPC" ANDD (conviction ORR sentence ORR judgment)'

        result = search_kanoon(query)

        api_calls += 1

        _add_cases_from_result(result, f"IPC {sec}", max_per_section)



    # Strategy 3: Broader fallback

    if len(all_cases) < 2:

        for sec in primary_sections:

            if len(all_cases) >= max_total:

                break

            query = f'"Section {sec} IPC" ANDD (guilty ORR punishment ORR conviction)'

            result = search_kanoon(query)

            api_calls += 1

            _add_cases_from_result(result, f"IPC {sec}", max_per_section)



    # --- FILTERING LOGIC ---
    print(f"\n[Kanoon] Filtering {len(all_cases)} results...")
    
    BAD_KEYWORDS = ["Constitution", "TADA", "Transfer", "Article 21", "Article 377", "Special Courts Act"]
    
    filtered: list[dict] = []
    for c in all_cases:
        # Check Year
        title = c.get("title", "")
        m = re.search(r"\b(1\d{3}|2000)\b", title) # Quick check for years < 2000
        if m and int(m.group(1)) < 2000:
            continue
            
        # Check Keywords
        if any(kw.lower() in title.lower() for kw in BAD_KEYWORDS):
            continue
            
        filtered.append(c)
        
    if len(filtered) >= 3:
        print(f"✓ Filtered to {len(filtered)} cases.")
        all_cases = filtered
    else:
        print(f"! Filtering left only {len(filtered)} cases. Reverting to raw results.")


    # Fetch judgment text + summarise

    if callback: callback(f"Found {len(all_cases)} potential cases. Fetching and summarizing judgment texts...")
    print(f"[Kanoon] Fetching & summarizing {len(all_cases)} cases...")

    summarized_cases: list[dict] = []

    for case in all_cases:

        tid = case["tid"]

        if not tid:

            continue

        doc_data = get_doc_detail(tid)

        api_calls += 1

        if doc_data.get("error"):

            continue

        doc_text = _clean_html(doc_data.get("doc", "")) or case["snippet"]

        prepped_text = preprocess_judgment(doc_text)

        summary_result = summarize_case_with_llm(
            prepped_text,
            case_title=case["title"],
            fir_summary=fir_summary,
            ipc_sections=[f"IPC {s}" for s in ipc_sections],
            return_metadata=True,
        )

        if summary_result.get("summary"):
            case["summary"] = summary_result["summary"]
            case["relevance_reason"] = summary_result.get("reason", "")
            case["relevance_score"] = int(summary_result.get("relevance_score", 0) or 0)
            case["relevant"] = bool(summary_result.get("relevant", False))
            summarized_cases.append(case)

    relevant_cases = [c for c in summarized_cases if c.get("relevant")]
    if relevant_cases:
        final_cases = sorted(relevant_cases, key=lambda c: c.get("relevance_score", 0), reverse=True)
    else:
        final_cases = sorted(summarized_cases, key=lambda c: c.get("relevance_score", 0), reverse=True)



    # Predict verdict

    if callback: callback("Synthesizing verdict prediction based on retrieved case law...")
    print("[Kanoon] Predicting verdict based on precedents...")

    cases_with_summaries = [c for c in final_cases if c.get("summary")]

    verdict = predict_verdict(

        fir_summary=fir_summary,

        ipc_sections=[f"IPC {s}" for s in ipc_sections],

        case_summaries=cases_with_summaries or final_cases,

    )



    # Rank section influence

    if callback: callback("Analyzing the influence of each IPC section on the final verdict...")
    print("[Kanoon] Ranking section influence on verdict...")

    section_influence = rank_section_influence(

        mapped_sections=mapped_sections,

        fir_summary=fir_summary,

        verdict=verdict,

        case_summaries=cases_with_summaries or final_cases,

        applicable_statutes=applicable_statutes,

    )



    return {

        "status": "success" if relevant_cases else "partial",

        "sections_searched": [f"IPC {s}" for s in ipc_sections],

        "cases": final_cases,

        "verdict_prediction": verdict,

        "section_influence": section_influence,

        "api_calls_used": api_calls,

        "error": None,

    }





# ---------------------------------------------------------------------------

#  Standalone test

# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Indian Kanoon + Groq Verdict Prediction — Test")

    print("=" * 60)



    test_sections = ["IPC 356", "IPC 390", "IPC 394"]

    test_fir = (

        "On 2025-06-20 at around 09:15 PM, the victim was assaulted near "

        "the market by an unknown person who snatched her handbag containing "

        "cash and documents."

    )



    result = search_and_analyze(test_sections, test_fir)



    print(f"\nStatus: {result['status']}")

    print(f"Sections: {result['sections_searched']}")

    print(f"API calls: {result['api_calls_used']}")

    print(f"Cases: {len(result['cases'])}")



    for i, c in enumerate(result["cases"], 1):

        print(f"\n  {i}. {c['title']}")

        print(f"     Court: {c['court']} | {c['section']}")

        if c["summary"]:

            print(f"     Summary: {c['summary'][:250]}...")



    vp = result.get("verdict_prediction", {})

    if vp:

        print(f"\n{'=' * 60}")

        print(f"  Verdict:    {vp.get('predicted_verdict', 'N/A')}")

        print(f"  Punishment: {vp.get('predicted_punishment', 'N/A')}")


