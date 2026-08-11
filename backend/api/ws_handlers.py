"""
WebSocket message handlers for the LexIR pipeline.

Each handler receives:
    msg          — the parsed client JSON message
    session_id   — unique connection identifier
    send         — coroutine to send a JSON dict to the client
    send_status  — coroutine to send a status message (stage, text)
    deps         — ServerDeps dataclass with shared singletons
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable, Any, Optional

from formatters import format_stage1, extract_mapped_sections, extract_primary_sections
from groq_prompts import rank_section_influence
from precedent_qa import answer_question

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
#  Shared dependencies (injected by the server at startup)
# ---------------------------------------------------------------------------
@dataclass
class ServerDeps:
    """Singletons shared across all WS handlers."""
    qa_engine: Any = None                        # PrecedentQA
    kanoon_searcher: Optional[Callable] = None   # indian_kanoon.search_and_analyze
    ensure_rag_system: Optional[Callable] = None # lazy-loader for RAG system
    sessions: dict = field(default_factory=dict)
    audit: Optional[Callable] = None             # _audit()
    analysis_context: dict = field(default_factory=dict)
    mongo_sessions_col: Any = None

# Type aliases for send helpers
SendFn = Callable[[dict], Awaitable[None]]
StatusFn = Callable[[int, str], Awaitable[None]]


def _build_sessions_list_payload(deps: ServerDeps, user_email: str = None) -> dict:
    """Build { type, sessions } for list_sessions and push refreshes."""
    if deps.mongo_sessions_col is None:
        return {"type": "sessions_list", "sessions": []}
    
    match_stage = {}
    if user_email:
        match_stage = {"user_email": user_email}

    pipeline = [
        {"$match": match_stage},
        {"$project": {
            "_id": 1,
            "title": 1,
            "fir_preview": 1,
            "created_at": 1,
            "status": 1,
            "message_count": {"$size": {"$ifNull": ["$messages", []]}},
        }},
        {"$sort": {"created_at": -1}},
    ]
    rows = list(deps.mongo_sessions_col.aggregate(pipeline))
    out = [
        {
            "id": r["_id"],
            "title": r.get("title") or "Untitled Analysis",
            "fir_preview": r.get("fir_preview", ""),
            "created_at": r.get("created_at", ""),
            "message_count": r.get("message_count", 0),
            "status": r.get("status", "complete"),
        }
        for r in rows
    ]
    return {"type": "sessions_list", "sessions": out}


async def push_sessions_list(send: SendFn, deps: ServerDeps, user_email: str = None) -> None:
    try:
        await send(_build_sessions_list_payload(deps, user_email))
    except Exception as e:
        print(f"[Mongo] push_sessions_list: {e}")


async def handle_rename_session(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    sid = msg.get("session_id")
    new_title = msg.get("title", "").strip()
    user_email = msg.get("user_email")
    if not sid or not new_title:
        return

    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.update_one(
                {"_id": sid},
                {"$set": {"title": new_title}}
            )
            await send({"type": "session_renamed", "session_id": sid, "title": new_title})
            await push_sessions_list(send, deps, user_email)
    except Exception as e:
        print(f"[Mongo] rename_session failed: {e}")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _load_fir(msg: dict, send_status, is_default_ok: bool = True) -> dict | None:
    """Return FIR data from the message, or load the sample FIR as fallback."""
    fir_data = msg.get("fir")
    if fir_data is None and is_default_ok:
        fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
        with open(fir_path, "r", encoding="utf-8") as f:
            fir_data = json.load(f)
    return fir_data


async def _run_stage2(mapped_sections, fir_summary, deps: ServerDeps, applicable_statutes=None, thought_callback: Callable = None):
    """Run Indian Kanoon search + verdict prediction (Stage 2) in a thread."""
    loop = asyncio.get_event_loop()

    def _sync_callback(t):
        if thought_callback:
            asyncio.run_coroutine_threadsafe(thought_callback(t), loop)

    try:
        return await loop.run_in_executor(
            None,
            lambda: deps.kanoon_searcher(
                mapped_sections=mapped_sections,
                fir_summary=fir_summary,
                applicable_statutes=applicable_statutes,
                callback=_sync_callback
            ),
        )
    except Exception as e:
        return {"status": "error", "cases": [], "verdict_prediction": None, "error": str(e)}


async def _run_rag_analysis(fir_data, deps: ServerDeps, thought_callback: Callable = None):
    """Run the full RAG chain analysis in a thread."""
    system = deps.ensure_rag_system()
    loop = asyncio.get_event_loop()

    def _sync_callback(t):
        if thought_callback:
            asyncio.run_coroutine_threadsafe(thought_callback(t), loop)

    return await loop.run_in_executor(
        None,
        lambda: system.analyze_fir_with_chains(fir_data, callback=_sync_callback)
    )


# ---------------------------------------------------------------------------
#  Handler: start_analysis (stages 1 & 2)
# ---------------------------------------------------------------------------
async def handle_start_analysis(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]
    user_email = msg.get("user_email")

    analysis_sid = str(uuid.uuid4())
    ctx["analysis_session_id"] = analysis_sid
    deps.analysis_context[analysis_sid] = {"fir": "", "stage1": None, "stage2": None}

    # 1. Load FIR
    fir_data = msg.get("fir")
    is_sample = fir_data is None
    if is_sample:
        fir_data = _load_fir(msg, send_status)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    inc = (fir_data.get("incident_description", "") or "") if fir_data else ""
    deps.analysis_context[analysis_sid]["fir"] = inc

    # Generate initial title using LLM
    from intent_queries import generate_brief_title
    title = generate_brief_title(inc)

    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.insert_one({
                "_id": analysis_sid,
                "user_email": user_email,
                "fir": fir_data,
                "title": title,
                "fir_preview": inc[:100] or "Analyzing…",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "messages": [],
                "stage1_data": None,
                "stage2_data": None,
            })
            await push_sessions_list(send, deps, user_email)
    except Exception as e:
        print(f"[Mongo] insert session: {e}")

    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data, "session_id": analysis_sid, "title": title})
    deps.audit("fir_loaded", session_id, fir_data.get("fir_id", ""))

    # 2. Stage 1
    s1_t0 = time.time()
    if is_sample:
        await send_status(1, "Loading pre-computed FIR analysis...")
        analysis_path = REPO_ROOT / "output" / "fir_analysis_result_chains.json"
        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis = json.load(f)
        else:
            alt_path = REPO_ROOT / "output" / "fir_analysis_result.json"
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
            else:
                await send({"type": "error", "message": "No pre-computed analysis found."})
                return
    else:
        await send_status(1, "Running live RAG analysis on your FIR (Pinecone + LLM)... ~30-60 s")

        async def _thought(t):
            await send({"type": "thought", "stage": 1, "message": t})

        try:
            analysis = await _run_rag_analysis(fir_data, deps, thought_callback=_thought)
        except Exception as e:
            await send({"type": "error", "message": f"RAG analysis failed: {e}"})
            return

    s1_latency = max(round(time.time() - s1_t0, 2), 0.1)
    ctx["analysis"] = analysis
    deps.analysis_context[analysis_sid]["stage1"] = analysis
    mapped_sections = extract_mapped_sections(analysis)
    primary_sections = extract_primary_sections(analysis)
    ctx["mapped_sections"] = primary_sections or mapped_sections
    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    stage1_data = format_stage1(fir_data, analysis, mapped_sections)
    stage1_data["_raw_analysis"] = analysis
    stage1_data["telemetry"] = {
        "latency_s": s1_latency,
        "model": "Groq Llama-3.3-70b",
        "vectors_scanned": 865,
        "similarity_metric": "Cosine",
        "retrieval_mode": "Hybrid RAG + Rule Filtering"
    }
    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.update_one(
                {"_id": analysis_sid},
                {"$set": {"stage1_data": stage1_data, "fir_preview": (inc[:100] or "Analysis in progress…")}},
            )
    except Exception as e:
        print(f"[Mongo] save stage1: {e}")
    await send({"type": "remove_thought", "stage": 1})
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    deps.audit("stage1_complete", session_id)

    # 3. Stage 2
    s2_t0 = time.time()
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")

    async def _thought2(t):
        await send({"type": "thought", "stage": 2, "message": t})

    stage2_result = await _run_stage2(
        primary_sections or mapped_sections,
        fir_summary,
        deps,
        applicable_statutes=analysis.get("applicable_statutes", []),
        thought_callback=_thought2,
    )
    s2_latency = max(round(time.time() - s2_t0, 2), 0.1)
    stage2_result["telemetry"] = {
        "latency_s": s2_latency,
        "cases_found": len(stage2_result.get("cases", [])),
        "engine": "Indian Kanoon API + Groq Fast Summarizer"
    }
    ctx["sim_result"] = stage2_result
    deps.analysis_context[analysis_sid]["stage2"] = stage2_result
    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.update_one(
                {"_id": analysis_sid},
                {"$set": {
                    "stage2_data": stage2_result,
                    "status": "complete",
                    "fir_preview": inc[:100] or deps.analysis_context[analysis_sid].get("fir", "")[:100],
                }},
            )
            await push_sessions_list(send, deps, user_email)
    except Exception as e:
        print(f"[Mongo] save stage2: {e}")
    await send({"type": "remove_thought", "stage": 2})
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    deps.audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: run_full_analysis (live RAG chain — slow path)
# ---------------------------------------------------------------------------
async def handle_full_analysis(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]

    analysis_sid = str(uuid.uuid4())
    ctx["analysis_session_id"] = analysis_sid
    deps.analysis_context[analysis_sid] = {"fir": "", "stage1": None, "stage2": None}

    fir_data = msg.get("fir")
    if not fir_data:
        fir_data = _load_fir(msg, send_status)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    inc = (fir_data.get("incident_description", "") or "") if fir_data else ""
    deps.analysis_context[analysis_sid]["fir"] = inc

    # Generate initial title using LLM
    from intent_queries import generate_brief_title
    title = generate_brief_title(inc)

    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.insert_one({
                "_id": analysis_sid,
                "fir": fir_data,
                "title": title,
                "fir_preview": inc[:100] or "Analyzing…",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "messages": [],
                "stage1_data": None,
                "stage2_data": None,
            })
            await push_sessions_list(send, deps)
    except Exception as e:
        print(f"[Mongo] insert session: {e}")

    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data, "session_id": analysis_sid, "title": title})

    s1_t0 = time.time()
    await send_status(1, "Running full RAG chain analysis (Pinecone + LLM)... This may take 30-60 seconds.")

    async def _thought(t):
        await send({"type": "thought", "stage": 1, "message": t})

    try:
        analysis = await _run_rag_analysis(fir_data, deps, thought_callback=_thought)
    except Exception as e:
        await send({"type": "error", "message": f"RAG analysis failed: {e}"})
        return

    s1_latency = max(round(time.time() - s1_t0, 2), 0.1)
    ctx["analysis"] = analysis
    deps.analysis_context[analysis_sid]["stage1"] = analysis
    mapped_sections = extract_mapped_sections(analysis)
    primary_sections = extract_primary_sections(analysis)
    ctx["mapped_sections"] = primary_sections or mapped_sections
    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    stage1_data = format_stage1(fir_data, analysis, mapped_sections)
    stage1_data["_raw_analysis"] = analysis
    stage1_data["telemetry"] = {
        "latency_s": s1_latency,
        "model": "Groq Llama-3.3-70b",
        "vectors_scanned": 865,
        "similarity_metric": "Cosine",
        "retrieval_mode": "Hybrid RAG + Rule Filtering"
    }
    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.update_one(
                {"_id": analysis_sid},
                {"$set": {"stage1_data": stage1_data, "fir_preview": (inc[:100] or "Analysis in progress…")}},
            )
    except Exception as e:
        print(f"[Mongo] save stage1: {e}")
    await send({"type": "remove_thought", "stage": 1})
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    deps.audit("stage1_live_complete", session_id)

    # Stage 2
    s2_t0 = time.time()
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")

    async def _thought2(t):
        await send({"type": "thought", "stage": 2, "message": t})

    stage2_result = await _run_stage2(
        primary_sections or mapped_sections,
        fir_summary,
        deps,
        applicable_statutes=analysis.get("applicable_statutes", []),
        thought_callback=_thought2,
    )
    s2_latency = max(round(time.time() - s2_t0, 2), 0.1)
    stage2_result["telemetry"] = {
        "latency_s": s2_latency,
        "cases_found": len(stage2_result.get("cases", [])),
        "engine": "Indian Kanoon API + Groq Fast Summarizer"
    }
    ctx["sim_result"] = stage2_result
    deps.analysis_context[analysis_sid]["stage2"] = stage2_result
    try:
        if deps.mongo_sessions_col is not None:
            deps.mongo_sessions_col.update_one(
                {"_id": analysis_sid},
                {"$set": {
                    "stage2_data": stage2_result,
                    "status": "complete",
                    "fir_preview": inc[:100] or deps.analysis_context[analysis_sid].get("fir", "")[:100],
                }},
            )
            await push_sessions_list(send, deps)
    except Exception as e:
        print(f"[Mongo] save stage2: {e}")
    await send({"type": "remove_thought", "stage": 2})
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    deps.audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: ask_question (Stage 3 Q&A)
# ---------------------------------------------------------------------------
async def handle_ask_question(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]
    question = msg.get("question", "").strip()

    if not question:
        await send({"type": "error", "message": "No question provided."})
        return

    lookup = msg.get("session_id") or ctx.get("analysis_session_id")
    if not lookup:
        answer_text = "Please submit a FIR first before asking questions."
        await send({
            "type": "qa_answer",
            "stage": 3,
            "data": {
                "question": question,
                "answer": answer_text,
                "is_no_match": True,
                "precedents_used": 0,
                "retrieval_status": "no_context",
            },
        })
        return

    ac_pre = deps.analysis_context.get(lookup, {})
    if not ctx.get("fir") and not ac_pre.get("stage1"):
        await send({"type": "error", "message": "No analysis started. Send 'start_analysis' first."})
        return

    if lookup not in deps.analysis_context:
        answer_text = "Please submit a FIR first before asking questions."
        await send({
            "type": "qa_answer",
            "stage": 3,
            "data": {
                "question": question,
                "answer": answer_text,
                "is_no_match": True,
                "precedents_used": 0,
                "retrieval_status": "no_context",
            },
        })
        return

    ac = deps.analysis_context[lookup]
    if not ac.get("stage1"):
        answer_text = "Please submit a FIR first before asking questions."
        await send({
            "type": "qa_answer",
            "stage": 3,
            "data": {
                "question": question,
                "answer": answer_text,
                "is_no_match": True,
                "precedents_used": 0,
                "retrieval_status": "no_context",
            },
        })
        return

    deps.audit("question", session_id, question[:100])
    await send_status(3, "Synthesizing answer from your analysis...")

    async def _thought(t):
        await send({"type": "thought", "stage": 3, "message": t})

    fir_text = ac.get("fir", "") or ""
    stage1_result = ac.get("stage1") or {}
    stage2_result = ac.get("stage2") or {}

    loop = asyncio.get_event_loop()

    answer_text = await loop.run_in_executor(
        None,
        lambda: answer_question(
            question=question,
            fir=fir_text,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            callback=lambda t: asyncio.run_coroutine_threadsafe(_thought(t), loop)
        ),
    )

    try:
        if deps.mongo_sessions_col is not None and lookup:
            deps.mongo_sessions_col.update_one(
                {"_id": lookup},
                {"$push": {"messages": {"$each": [
                    {"role": "user", "content": question, "timestamp": datetime.now(timezone.utc).isoformat()},
                    {"role": "assistant", "content": answer_text, "timestamp": datetime.now(timezone.utc).isoformat()},
                ]}}},
            )
    except Exception as e:
        print(f"[Mongo] update messages: {e}")

    await send({"type": "remove_thought", "stage": 3})
    await send({
        "type": "qa_answer",
        "stage": 3,
        "data": {
            "question": question,
            "answer": answer_text,
            "is_no_match": False,
            "precedents_used": len((stage2_result or {}).get("cases") or []),
            "retrieval_status": "context",
        },
    })
    deps.audit("qa_answer", session_id, f"q={question[:50]}")


# ---------------------------------------------------------------------------
#  Handlers: chat history (MongoDB)
# ---------------------------------------------------------------------------
async def handle_list_sessions(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    del session_id, send_status
    user_email = msg.get("user_email")
    try:
        await send(_build_sessions_list_payload(deps, user_email))
    except Exception as e:
        print(f"[Mongo] list_sessions: {e}")
        await send({"type": "sessions_list", "sessions": []})


async def handle_get_history(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    del send_status
    ws_conn_id = session_id
    sid = msg.get("session_id")
    try:
        if deps.mongo_sessions_col is None or not sid:
            await send({
                "type": "history",
                "session_id": sid,
                "fir": None,
                "stage1_data": None,
                "stage2_data": None,
                "messages": [],
                "status": None,
            })
            return
        doc = deps.mongo_sessions_col.find_one({"_id": sid})
        if not doc:
            await send({
                "type": "history",
                "session_id": sid,
                "fir": None,
                "stage1_data": None,
                "stage2_data": None,
                "messages": [],
                "status": None,
            })
            return
        fir_obj = doc.get("fir")
        fir_text = ""
        if isinstance(fir_obj, dict):
            fir_text = fir_obj.get("incident_description", "") or ""
        deps.analysis_context[sid] = {
            "fir": fir_text,
            "stage1": doc.get("stage1_data") or {},
            "stage2": doc.get("stage2_data") or {},
        }
        wctx = deps.sessions.get(ws_conn_id)
        if wctx is not None:
            wctx["analysis_session_id"] = sid
            wctx["fir"] = fir_obj if isinstance(fir_obj, dict) else None
        await send({
            "type": "history",
            "session_id": sid,
            "fir": doc.get("fir"),
            "title": doc.get("title", ""),
            "stage1_data": doc.get("stage1_data"),
            "stage2_data": doc.get("stage2_data"),
            "messages": doc.get("messages") or [],
            "status": doc.get("status"),
        })
    except Exception as e:
        print(f"[Mongo] get_history: {e}")
        await send({
            "type": "history",
            "session_id": sid,
            "fir": None,
            "stage1_data": None,
            "stage2_data": None,
            "messages": [],
            "status": None,
        })


async def handle_clear_session(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    del send_status
    sid = msg.get("session_id")
    user_email = msg.get("user_email")
    try:
        if deps.mongo_sessions_col is not None and sid:
            deps.mongo_sessions_col.delete_one({"_id": sid})
        deps.analysis_context.pop(sid, None)
        await send({"type": "session_cleared", "session_id": sid, "success": True})
        await push_sessions_list(send, deps, user_email)
    except Exception as e:
        print(f"[Mongo] clear_session: {e}")
        await send({"type": "error", "message": f"Could not clear session: {e}"})
