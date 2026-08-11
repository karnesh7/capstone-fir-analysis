# LexIR — Legal Intelligence & Retrieval System for FIR Analysis

> **Capstone Project** — An AI-powered pipeline that maps FIR allegations to IPC/BNS statutes, retrieves real Indian court precedents, predicts likely verdicts, and enables follow-up legal Q&A — all in a live, chat-style interface.

---

## Table of Contents

- [Overview](#overview)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Tech Stack](#tech-stack)
- [Repository Layout](#repository-layout)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Frontend UI](#frontend-ui)
- [Model Selection & Fallback System](#model-selection--fallback-system)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Data Storage](#data-storage)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

---

## Overview

**LexIR** is a full-stack legal analysis system built for analyzing First Information Reports (FIRs) under Indian law. Given a structured FIR with incident facts and allegations, LexIR:

1. Identifies applicable **IPC / BNS sections** using a RAG-based statute retrieval pipeline over **865+ vectorized provisions**.
2. Features an **interactive IPC ↔ BNS (2023) cross-comparison modal** highlighting statutory definitions, penalties, and modernization notes.
3. Distinguishes between **criminal matters** and **civil/consumer disputes** — skipping criminal section mapping when the facts indicate a contract or quality disagreement.
4. Searches **Indian Kanoon** for real precedent judgments and summarizes them.
5. **Predicts the likely verdict and punishment** based on retrieved court precedents.
6. Answers **follow-up legal questions** grounded in the FIR context and prior analysis.
7. Displays **live pipeline telemetry** (RAG latency, Groq LLM model info, vectors scanned).
8. Exports complete **Legal Case Briefs (.md)** and official **Form IF-1 FIR PDFs**.
9. Includes **1-click demo scenarios** (Cyber Fraud, Armed Robbery, Road Accident, Contract Dispute) for instant evaluation.

The system is split into a Python/FastAPI backend and a React 19 frontend, communicating over WebSockets.

---

## How the Pipeline Works

### Stage 1 — FIR Analysis & Statute Mapping

The backend receives FIR text over the WebSocket connection and runs the statute retrieval pipeline:

- Classifies the **primary legal nature** of the complaint (criminal vs. civil/consumer)
- For **criminal matters**: maps the FIR to applicable IPC/BNS sections using vector similarity search (Pinecone) and LLM-based legal reasoning
- For **civil/consumer disputes**: returns zero criminal sections and provides the appropriate civil/consumer legal basis instead
- The Stage 1 card in the UI shows statute numbers and corresponding BNS sections — not raw offense title text

Key files:
- `backend/api/rag_llm_chain_prompting.py` — RAG chain and statute retrieval
- `backend/api/intent_queries.py` — Intent classification queries
- `backend/api/formatters.py` — Output formatting

---

### Stage 2 — Indian Kanoon Precedent Search

Runs **only for criminal matters**. The pipeline:

1. Builds a search query from FIR facts and mapped sections
2. Calls the **Indian Kanoon API** to fetch real judgments
3. Retrieves and summarizes full judgment text for each case
4. **Predicts verdict and punishment** from the retrieved cases
5. Ranks the **influence of each applicable section** on the verdict

Key files:
- `backend/api/indian_kanoon.py` — Kanoon API client and caching
- `backend/api/groq_prompts.py` — LLM prompts for summarization and prediction

---

### Stage 3 — Legal Q&A

After the analysis completes, users can ask follow-up legal questions. Stage 3 reuses the already-computed session data (Stage 1 + Stage 2 results), so answers are grounded in the same case context without re-running the pipeline.

Key files:
- `backend/api/precedent_qa.py` — Q&A engine
- `backend/api/ws_handlers.py` — WebSocket message routing

---

## Tech Stack

### Backend

| Component | Library / Service |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Real-time Comms | WebSockets |
| LLM Provider | Groq (`openai/gpt-oss-120b`, `groq/compound-mini`, `qwen/qwen3.6-27b`) |
| Embeddings | `sentence-transformers` |
| Vector Search | Pinecone |
| Legal Database | Indian Kanoon API |
| Session Storage | MongoDB (`pymongo`) |
| PDF Generation | `fpdf2`, `PyPDF2` |
| Config | `python-dotenv` |
| Schema Validation | Pydantic v2 |

### Frontend

| Component | Library |
|---|---|
| Framework | React 19 |
| Routing | React Router v7 |
| Auth | `@react-oauth/google`, `jwt-decode` |
| Icons | `lucide-react` |
| Data Fetching | WebSocket (native) |
| Bundler | Create React App (`react-scripts`) |

---

## Repository Layout

```text
capstone-fir-analysis/
├── backend/
│   ├── server.py                        # FastAPI app, HTTP endpoints, WebSocket entrypoint
│   ├── api/
│   │   ├── ws_handlers.py               # WebSocket message dispatch & session lifecycle
│   │   ├── rag_llm_chain_prompting.py   # Stage 1 RAG chain & statute retrieval
│   │   ├── indian_kanoon.py             # Stage 2 Kanoon API client & caching
│   │   ├── groq_prompts.py              # LLM prompt builders (summarisation, verdict, Q&A)
│   │   ├── precedent_qa.py              # Stage 3 Q&A engine
│   │   ├── model_config.py              # Model fallback chains & auto-selection
│   │   ├── intent_queries.py            # Intent classification helper
│   │   ├── formatters.py                # Stage output formatters
│   │   ├── kanoon_cache.py              # In-process Kanoon response cache
│   │   ├── schemas.py                   # Shared Pydantic schemas
│   │   ├── fir_pdf_generator.py         # PDF rendering from FIR data
│   │   └── fir_pdf_mapper.py            # Maps raw FIR fields to PDF layout
│   └── evaluation/                      # Benchmark & evaluation scripts (see below)
│
├── frontend/
│   └── src/
│       ├── App.js                       # Top-level layout, routing, brief export & auth gate
│       ├── hooks/
│       │   └── useLexIR.js              # WebSocket state management & analysis orchestration
│       ├── data/
│       │   └── presetScenarios.js       # Pre-loaded 1-click test scenarios
│       └── components/
│           ├── ChatArea.js / .css       # Main chat + stage card rendering
│           ├── Stage1Card.js / .css     # Stage 1 output (statutes, telemetry & comparison modal)
│           ├── Stage2Card.js / .css     # Stage 2 output (precedents, verdict & Kanoon telemetry)
│           ├── VerdictCard.js           # Verdict & punishment display
│           ├── SectionInfluence.js      # Section influence ranking
│           ├── KanoonCaseList.js        # Precedent case list
│           ├── FIRForm.js / .css        # Multi-field FIR input form with 1-click scenario pills
│           ├── ChatInput.js / .css      # Q&A input box
│           ├── Sidebar.js / .css        # Session history & controls
│           ├── SessionList.js           # Session list inside sidebar
│           ├── ChatHistory.js           # Per-session message history
│           └── LoginPage.js / Login.css # Google OAuth & direct login screen
│
├── preprocessing/
│   ├── build_statute_dataset.py         # Parse IPC/BNS statutes, generate embeddings & deploy
│   ├── deploy_to_pinecone.py            # Upload statute vectors to Pinecone
│   └── test_vector_db.py               # Smoke-test the Pinecone index
│
├── src_dataset_files/                   # Raw statute and dataset files
├── output/                              # Benchmark outputs & generated artifacts
├── logs/                                # Audit and run logs
├── requirements.txt
└── .env                                 # API keys (not committed)
```

---

## Environment Variables

Create a `.env` file in the **repository root** with the following keys:

```env
# Required
GROQ_API_KEY=your_groq_api_key
KANOON_API_KEY=your_indian_kanoon_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional — Google Sign-In
REACT_APP_GOOGLE_CLIENT_ID=your_google_oauth_client_id

# Optional — Override the default Groq model
GROQ_MODEL=openai/gpt-oss-20b
```

> The frontend reads the `.env` at build time via `dotenv-cli` (configured in `package.json` scripts). Both backend and frontend pick up keys from the same root `.env`.

---

## Installation

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- MongoDB running locally at `mongodb://localhost:27017`
- API keys for Groq, Indian Kanoon, and Pinecone

---

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd capstone-fir-analysis
```

### 2. Set up environment variables

Create a `.env` file in the repository root and fill in your API keys (see [Environment Variables](#environment-variables)).

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize Pinecone vectors (first-time only)

Parse IPC/BNS statute files, generate embeddings, and automatically populate your Pinecone index:

```bash
python preprocessing/build_statute_dataset.py
```

Verify the index is populated:

```bash
python preprocessing/test_vector_db.py
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the App

### Step 1 — Start MongoDB

Ensure MongoDB is running locally at `mongodb://localhost:27017`.

If MongoDB is unavailable, the app will still start, but session/chat history persistence will be disabled.

### Step 2 — Start the backend

From the **`backend/`** directory:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running:

```
GET http://localhost:8000/health
```

### Step 3 — Start the frontend

From the **`frontend/`** directory:

```bash
npm start
```

The app opens at `http://localhost:3000`.

---

## API Reference

### HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check and loaded model status |
| `GET` | `/api/fir/sample` | Returns a sample FIR payload |
| `POST` | `/api/fir/json` | Submit FIR as JSON |
| `POST` | `/api/fir/pdf-payload` | Generate PDF payload from FIR data |
| `POST` | `/api/fir/pdf` | Generate and download Form IF-1 FIR PDF |
| `GET` | `/api/sessions` | List saved chat sessions |
| `POST` | `/api/auth/login` | Record user profile session in MongoDB |

### WebSocket — `/ws`

The primary communication channel. The frontend sends typed JSON messages; the backend streams status updates, thought steps, and stage results back in real time.

| Message Type | Direction | Description |
|---|---|---|
| `start_analysis` | Client → Server | Begin FIR analysis (Stage 1) |
| `run_full_analysis` | Client → Server | Run the full pipeline (Stages 1–2) |
| `ask_question` | Client → Server | Ask a follow-up legal question (Stage 3) |
| `list_sessions` | Client → Server | Fetch all saved sessions |
| `get_history` | Client → Server | Retrieve a specific session's messages |
| `clear_session` | Client → Server | Delete a saved session |
| `rename_session` | Client → Server | Rename a saved session |

The server emits `status`, `thought`, and stage-specific result events during analysis.

---

## Frontend UI

The frontend is a single-page application with a sidebar and a main chat area:

| Component | Role |
|---|---|
| `LoginPage` | Google OAuth & direct login fallback screen |
| `App.js` | Root layout, routing, legal brief export, and auth state |
| `useLexIR` | Central hook — WebSocket connection, analysis state, session management |
| `FIRForm` | Multi-field FIR input form with 1-click preset scenarios |
| `Stage1Card` | Displays statute mapping results, latency telemetry, and interactive IPC ↔ BNS comparison modal |
| `Stage2Card` | Displays precedent cases, verdict prediction, Kanoon telemetry, and section influence |
| `VerdictCard` | Dedicated verdict probability and punishment range summary |
| `SectionInfluence` | Ranks sections by their influence on the predicted verdict |
| `KanoonCaseList` | Lists and summarizes retrieved Indian Kanoon judgments |
| `Sidebar` | Session list, rename, delete, and navigation controls |
| `ChatInput` | Q&A input for Stage 3 follow-up questions |

---

## Model Selection & Fallback System

LexIR uses a **role-based model fallback chain** defined in `backend/api/model_config.py`. Each pipeline role has an ordered list of models; if the primary model is rate-limited or unavailable, the system automatically retries with the next model in the chain.

> **Last benchmarked: August 11, 2026** on 8 live Groq models (4 SLM, 4 LLM).
> Judge: `llama-3.3-70b-versatile`. Composite = (ROUGE-1 + BLEU + METEOR + Faithfulness + (1−Hallucination)) / 5.

| Role | Primary Model | Fallback 1 | Fallback 2 |
|---|---|---|---|
| `slm_intent` (intent classification) | `groq/compound-mini` (0.671) | `openai/gpt-oss-20b` (0.628) | `allam-2-7b` (0.614) |
| `llm_reasoning` (legal reasoning) | `openai/gpt-oss-120b` (0.548) | `qwen/qwen3.6-27b` (0.522) | `groq/compound` |
| `summarisation` | `groq/compound-mini` | `openai/gpt-oss-20b` | `qwen/qwen3.6-27b` |
| `qa` (Stage 3 Q&A) | `groq/compound-mini` | `openai/gpt-oss-20b` | `qwen/qwen3.6-27b` |

> **Note:** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were deprecated by Groq on August 16, 2026 and have been removed from all chains. `groq/compound` internally uses `openai/gpt-oss-120b` and shares its TPM budget — kept as last-resort only.

You can override the model for any session by setting `GROQ_MODEL` in `.env`.

---

## Evaluation & Benchmarking

The `backend/evaluation/` directory contains standalone benchmark scripts used during development to evaluate and tune the pipeline:

| Script | Purpose |
|---|---|
| `benchmark_algorithmic_vs_llm.py` | Compares rule-based vs. LLM statute mapping |
| `benchmark_groq_metrics.py` | Evaluates Groq models across pipeline roles |
| `benchmark_groq_pipeline_models.py` | End-to-end pipeline model comparison |
| `benchmark_feature2_kanoon_live.py` | Live Kanoon retrieval quality |
| `benchmark_feature_2_summarization.py` | Judgment summarization quality |
| `benchmark_feature_3_ranking.py` | Section influence ranking accuracy |
| `benchmark_groq_summarization.py` | Summarization model comparison |
| `benchmark_negative_rule_semantic.py` | Civil vs. criminal detection accuracy |
| `benchmark_response_time.py` | End-to-end latency profiling |
| `compare_summary_approaches.py` | Compares summarization strategies |
| `demo_negative_rules_filter.py` | Demonstrates civil/consumer dispute filtering |

Benchmark results are written to `output/` and can influence automatic model selection via `output/model_benchmark_latest.json`.

---

## Data Storage

### MongoDB

Chat sessions are stored in a `chat_sessions` collection. Each document contains:

- `session_id` — unique identifier
- `fir_preview` — short preview of the FIR text
- `created_at` — timestamp
- `messages` — full message history (user + assistant)

### Output & Logs

| Directory | Contents |
|---|---|
| `output/` | Benchmark JSON results, generated artifacts, `model_benchmark_latest.json` |
| `logs/` | Audit logs and run traces |

---

## Troubleshooting

### Backend fails to start

- Confirm all Python dependencies are installed: `pip install -r requirements.txt`
- Verify `.env` contains `GROQ_API_KEY`, `KANOON_API_KEY`, and `PINECONE_API_KEY`
- Ensure MongoDB is running (or set it to start automatically)
- Confirm port `8000` is not already in use

### Frontend shows no results

- Verify the backend is running and reachable at `http://localhost:8000/health`
- Check that the WebSocket connection at `ws://localhost:8000/ws` is not blocked by a firewall or proxy
- Confirm the `.env` file is in the **repository root**, not inside the `frontend/` directory

### Stage 2 returns no cases

This is expected if:
- The FIR describes a **civil or consumer dispute** — Stage 2 is intentionally skipped
- The Kanoon search query is too narrow for the available case database
- The `KANOON_API_KEY` is missing or invalid

### Google Sign-In not working

- Ensure `REACT_APP_GOOGLE_CLIENT_ID` is set in `.env`
- The frontend reads this at **build/start time** via `dotenv-cli`, so restart `npm start` after updating `.env`
- Verify the OAuth client ID in Google Cloud Console allows `http://localhost:3000` as an authorized origin

---

## Development Notes

- Keep backend changes focused — logic is split by stage and feature; avoid consolidating into `server.py`
- Use existing stage files (`rag_llm_chain_prompting.py`, `indian_kanoon.py`, `precedent_qa.py`) as the primary extension points
- When updating the retrieval pipeline, verify that the frontend stage card components still match the backend payload shape
- The frontend's `.env` is loaded via `dotenv-cli` in `npm start`/`npm run build` — it reads from the **project root**, not from the `frontend/` directory
- Benchmark scripts in `evaluation/` are standalone — run them from the `backend/` directory with the root `.env` active

---

## Key Files at a Glance

| File | Purpose |
|---|---|
| `backend/server.py` | FastAPI app entry point & REST endpoints |
| `backend/api/ws_handlers.py` | WebSocket session lifecycle, timing & dispatch |
| `backend/api/rag_llm_chain_prompting.py` | Stage 1 statute retrieval RAG pipeline |
| `backend/api/indian_kanoon.py` | Stage 2 Kanoon client & judgment caching |
| `backend/api/groq_prompts.py` | LLM prompt definitions (summarization, verdict) |
| `backend/api/precedent_qa.py` | Stage 3 Q&A engine |
| `backend/api/model_config.py` | Model fallback chains & dynamic routing |
| `frontend/src/hooks/useLexIR.js` | Frontend state & WS orchestration |
| `frontend/src/App.js` | Root app component, export brief handler & auth routing |
| `frontend/src/data/presetScenarios.js` | 1-Click realistic scenario cases for testing |
| `preprocessing/build_statute_dataset.py` | One-time statute dataset extraction & Pinecone vector build |

---

## Data Attribution & Acknowledgments

- **Indian Kanoon**: Court precedent judgments, case laws, and legal ruling extracts are retrieved via the [Indian Kanoon API](https://api.indiankanoon.org/).
- **Statute Data**: Bare acts for the Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS, 2023) sourced from official Gazette publications.
