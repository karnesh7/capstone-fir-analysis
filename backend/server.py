#!/usr/bin/env python3
"""
LexIR Backend — FastAPI + WebSocket Server
===========================================
Thin server: defines REST endpoints, WebSocket routing, and startup.
Business logic lives in api/ modules (ws_handlers, formatters, schemas).

Usage:
    cd backend
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "backend" / "api"
sys.path.insert(0, str(API_DIR))

from schemas import FIRInput, FIRPdfRequest                      # noqa: E402
from ws_handlers import (                                          # noqa: E402
    ServerDeps,
    handle_start_analysis,
    handle_full_analysis,
    handle_ask_question,
    handle_list_sessions,
    handle_get_history,
    handle_clear_session,
)

from pydantic import BaseModel

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore

mongo_client = None
sessions_col = None
users_col = None
try:
    if MongoClient is not None:
        mongo_client = MongoClient("mongodb://localhost:27017")
        db = mongo_client["lexir"]
        sessions_col = db["chat_sessions"]
        users_col = db["users"]
except Exception as e:
    print(f"[SERVER] MongoDB unavailable: {e}")

class UserLogin(BaseModel):
    token: str
    name: str
    email: str
    picture: str

# ---------------------------------------------------------------------------
#  FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LexIR — Legal Intelligence & Retrieval API",
    version="1.0.0",
    description="WebSocket-powered legal FIR analysis pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Shared dependencies (populated at startup)
# ---------------------------------------------------------------------------
_deps = ServerDeps()
_rag_system = None  # lazy-loaded


def _ensure_rag_system():
    global _rag_system
    if _rag_system is None:
        print("[SERVER] Loading StatuteRAGChainSystem (Pinecone + LLM)...")
        from rag_llm_chain_prompting import StatuteRAGChainSystem
        # The constructor handles caching of the embedding model via sentence_transformers
        _rag_system = StatuteRAGChainSystem()
        print("[SERVER] ✓ RAG system loaded")
    return _rag_system


# ---------------------------------------------------------------------------
#  Audit logging
# ---------------------------------------------------------------------------
AUDIT_LOG = REPO_ROOT / "logs" / "audit_log.jsonl"


def _audit(action: str, session_id: str, detail: str = ""):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "session_id": session_id,
        "detail": detail,
    }
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # non-critical


# ---------------------------------------------------------------------------
#  Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    print("[SERVER] Initializing PrecedentQA (Groq LLM)...")
    from precedent_qa import PrecedentQA
    _deps.qa_engine = PrecedentQA()

    from indian_kanoon import search_and_analyze
    _deps.kanoon_searcher = search_and_analyze
    _deps.ensure_rag_system = _ensure_rag_system
    _deps.audit = _audit
    _deps.mongo_sessions_col = sessions_col
    print("[SERVER] ✓ All models loaded — server ready")


# ---------------------------------------------------------------------------
#  REST Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
async def api_login(user: UserLogin):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        # Check if user exists, if not create/update
        users_col.update_one(
            {"email": user.email},
            {"$set": {
                "name": user.name,
                "picture": user.picture,
                "last_login": datetime.now(timezone.utc).isoformat()
            }},
            upsert=True
        )
        return {"status": "ok"}
    except Exception as e:
        print(f"[SERVER] Auth failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _deps.qa_engine is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/sessions")
async def api_list_sessions(user_email: str = None):
    try:
        if sessions_col is None:
            return {"sessions": []}
        
        match_stage = {}
        if user_email:
            match_stage = {"user_email": user_email}
            
        pipeline = [
            {"$match": match_stage},
            {"$project": {
                "_id": 1,
                "fir_preview": 1,
                "created_at": 1,
                "status": 1,
                "message_count": {"$size": {"$ifNull": ["$messages", []]}},
            }},
            {"$sort": {"created_at": -1}},
        ]
        rows = list(sessions_col.aggregate(pipeline))
        return {
            "sessions": [
                {
                    "id": r["_id"],
                    "fir_preview": r.get("fir_preview", ""),
                    "message_count": r.get("message_count", 0),
                    "created_at": r.get("created_at", ""),
                    "status": r.get("status", "complete"),
                }
                for r in rows
            ],
        }
    except Exception as e:
        print(f"[SERVER] GET /api/sessions failed: {e}")
        return {"sessions": []}


@app.get("/api/fir/sample")
async def get_sample_fir():
    fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
    if not fir_path.exists():
        raise HTTPException(status_code=404, detail="Sample FIR not found")
    with open(fir_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/fir/json")
async def submit_fir_json(fir: FIRInput):
    return {"status": "ok", "fir": fir.dict()}


@app.post("/api/fir/pdf-payload")
async def get_fir_pdf_payload(fir: FIRInput):
    from fir_pdf_mapper import build_fir_pdf_payload
    payload = build_fir_pdf_payload(fir.dict())
    return {"status": "ok", "pdf_payload": payload}


@app.post("/api/fir/pdf")
async def generate_fir_pdf(req: FIRPdfRequest):
    from fir_pdf_mapper import build_fir_pdf_payload
    from fir_pdf_generator import generate_fir_pdf as _gen_pdf

    payload = build_fir_pdf_payload(req.fir, req.analysis)
    pdf_bytes = _gen_pdf(payload["fields"])
    fir_id = req.fir.get("fir_id", "FIR")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fir_id}.pdf"'},
    )





# ---------------------------------------------------------------------------
#  WebSocket Router
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    user_email = ws.query_params.get("user_email")
    session_id = uuid.uuid4().hex[:12]
    _deps.sessions[session_id] = {
        "fir": None, "analysis": None, "sim_result": None,
        "mapped_sections": [], "fir_summary": "",
    }
    _audit("ws_connect", session_id)

    async def send(msg: dict):
        await ws.send_json(msg)

    async def send_status(stage: int, message: str):
        await send({"type": "status", "stage": stage, "message": message})

    try:
        await send_status(0, "Connected to LexIR server. Send 'start_analysis' to begin.")

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start_analysis":
                msg["user_email"] = user_email
                await handle_start_analysis(msg, session_id, send, send_status, _deps)
            elif msg_type == "run_full_analysis":
                await handle_full_analysis(msg, session_id, send, send_status, _deps)
            elif msg_type == "ask_question":
                await handle_ask_question(msg, session_id, send, send_status, _deps)
            elif msg_type == "list_sessions":
                await handle_list_sessions(msg, session_id, send, send_status, _deps)
            elif msg_type == "get_history":
                await handle_get_history(msg, session_id, send, send_status, _deps)
            elif msg_type == "clear_session":
                await handle_clear_session(msg, session_id, send, send_status, _deps)
            elif msg_type == "rename_session":
                from ws_handlers import handle_rename_session
                await handle_rename_session(msg, session_id, send, send_status, _deps)
            elif msg_type in ("search_kanoon", "show_cases"):
                ctx = _deps.sessions[session_id]
                if ctx.get("sim_result"):
                    await send({"type": "stage2_result", "stage": 2, "data": ctx["sim_result"]})
                else:
                    await send({"type": "error", "message": "No analysis run yet."})
            else:
                await send({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        _audit("ws_disconnect", session_id)
    except Exception as e:
        _audit("ws_error", session_id, str(e))
        try:
            await send({"type": "error", "message": f"Server error: {e}"})
        except Exception:
            pass
    finally:
        _deps.sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
#  Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
