# LexIR

LexIR is a legal intelligence and retrieval system for FIR analysis. It takes FIR details and allegations, maps them to applicable IPC/BNS sections, searches Indian Kanoon for precedent cases, summarizes the retrieved judgments, predicts the likely verdict, and supports follow-up legal questions in a chat-style interface.

The project is split into a Python backend and a React frontend:

- Backend: FastAPI, WebSockets, Groq, Pinecone, Indian Kanoon, MongoDB
- Frontend: React with a live chat / stage-based analysis UI

## What The App Does

LexIR runs in three stages:

1. Stage 1, FIR analysis
   - Reads the structured FIR incident facts
   - Identifies likely criminal intent and applicable IPC/BNS sections
   - Uses the RAG pipeline and statute retrieval to narrow down relevant legal provisions

2. Stage 2, precedent search
   - Builds a search query from the FIR facts and mapped sections
   - Calls the Indian Kanoon API to fetch real judgments
   - Summarizes each retrieved case
   - Predicts verdict, punishment, and section influence

3. Stage 3, legal Q&A
   - Answers follow-up legal questions using the FIR context and prior analysis
   - Reuses stage 1 and stage 2 results so answers stay grounded in the same case

There is also support for:

- FIR PDF generation
- Persistent chat/session history in MongoDB

## Repository Layout

```text
backend/
  server.py
  api/
    rag_llm_chain_prompting.py
    indian_kanoon.py
    precedent_qa.py
    ws_handlers.py
    groq_prompts.py
    fir_pdf_generator.py
    fir_pdf_mapper.py
frontend/
  src/
    components/
    hooks/
src_dataset_files/
output/
logs/
```

## Tech Stack

### Backend
- FastAPI
- WebSockets
- Groq LLM
- Pinecone vector search
- Indian Kanoon API
- MongoDB for chat/session storage
- python-dotenv for configuration
- sentence-transformers and numpy for retrieval / ranking helpers

### Frontend
- React 19
- react-scripts
- lucide-react icons
- WebSocket-based state updates

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- MongoDB running locally at `mongodb://localhost:27017`
- API keys in a `.env` file at the project root

## Environment Variables

Create a `.env` file in the repository root with the required keys:

```env
GROQ_API_KEY=your_groq_api_key
KANOON_API_KEY=your_indiankanoon_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional (for Google Sign-In)
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id
```

## Installation

### 1. Clone the project

```bash
git clone <your-repo-url>
cd capstone_project
```

### 2. Backend setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Initialize Statute Vectors & Pinecone (First-time setup only)

Run the one-time dataset extraction and vector deployment script to parse IPC/BNS statutes, generate embeddings, and populate your Pinecone index:

```bash
python preprocessing/build_statute_dataset.py
```

### 4. Frontend setup

Install Node dependencies:

```bash
cd frontend
npm install
```

## How To Run

### Start MongoDB

Make sure MongoDB is running locally. The backend expects:

```text
mongodb://localhost:27017
```

If MongoDB is unavailable, the app can still start, but chat history persistence will not work.

### Start the backend

From the `backend` directory:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Backend health endpoint:

```text
http://localhost:8000/health
```

### Start the frontend

From the `frontend` directory:

```bash
npm start
```

The frontend runs on:

```text
http://localhost:3000
```

## How The Pipeline Works

### Stage 1, FIR Analysis

The backend receives FIR text through the WebSocket connection and runs the statute retrieval pipeline. It identifies the primary legal nature of the matter, and then either maps the FIR to IPC/BNS sections or classifies it as a civil / consumer dispute when the facts show only a contract or quality disagreement without deception, force, or coercion.

Relevant code paths:

- `backend/api/rag_llm_chain_prompting.py`
- `backend/api/intent_queries.py`
- `backend/api/formatters.py`

### Stage 1 Output Notes

- Criminal matters show clean IPC/BNS section labels in the UI
- Civil / contract disputes return `Applicable Sections (0)` and a civil/consumer legal basis instead of cheating/fraud sections
- The stage 1 card displays the statute numbers and any corresponding BNS sections, not the offense title text

### Stage 2, Indian Kanoon Precedents

Stage 2 runs only when the matter is treated as criminal. It does the following:

1. Builds a Kanoon search query from the FIR facts and mapped sections
2. Searches Indian Kanoon for real precedent cases
3. Fetches full judgment text for each case
4. Summarizes the judgment text
5. Predicts verdict and punishment from the retrieved cases
6. Ranks the influence of applicable sections on the verdict

Relevant code paths:

- `backend/api/indian_kanoon.py`
- `backend/api/groq_prompts.py`

### Stage 3, Question Answering

Stage 3 uses the FIR analysis and precedent results to answer user questions in context. The stage runs over the already computed session data instead of starting from scratch.

Relevant code paths:

- `backend/api/precedent_qa.py`
- `backend/api/ws_handlers.py`

## WebSocket Message Flow

The frontend and backend communicate over `/ws`.

Common message types:

- `start_analysis` — begin FIR analysis
- `run_full_analysis` — run the full pipeline
- `ask_question` — ask a follow-up legal question
- `list_sessions` — fetch saved chat sessions
- `get_history` — fetch a saved session’s messages
- `clear_session` — delete a saved session
- `rename_session` — rename a saved session

During the analysis flow the backend sends status and thought updates, then stage results.

## HTTP Endpoints

### Health and FIR helpers

- `GET /health`
- `GET /api/fir/sample`
- `POST /api/fir/json`
- `POST /api/fir/pdf-payload`
- `POST /api/fir/pdf`
- `POST /api/fir/upload`

### Session storage

- `GET /api/sessions`

## Frontend UI Summary

The frontend is organized around the live chat and stage cards:

- `ChatArea` renders stage output, system messages, user messages, assistant messages, and loading/thought updates
- `Stage1Card` shows FIR mapping details
- `Stage2Card` shows precedent cases and verdict prediction
- `Sidebar` shows session history and controls
- `useLexIR` manages WebSocket state and app-level analysis state

## Data Storage

### MongoDB

Chat sessions are stored in a `chat_sessions` collection. Each session stores:

- Session ID
- FIR preview
- Created timestamp
- Message history

### Output and logs

The project also writes to:

- `output/` for generated artifacts and benchmark outputs
- `logs/` for audit logs

## Troubleshooting

### Backend fails to start

Check that:

- Python dependencies are installed
- `.env` contains the required API keys
- MongoDB is running locally
- Port 8000 is free

### Frontend shows no results

Check that:

- Backend is running on port 8000
- The WebSocket URL is reachable
- Your `.env` values are loaded correctly

### Stage 2 returns no cases

This usually means one of the following:

- The FIR query is too narrow
- Indian Kanoon returned no results for the current query terms
- The API key is missing or invalid

If the input is a civil / contract dispute, stage 2 is intentionally skipped and stage 1 will return zero criminal sections.

## Development Notes

- Keep backend changes focused; the project is already split by stage and feature
- Use the existing stage files rather than moving logic into `server.py`
- When updating the retrieval pipeline, verify the frontend stage cards still match the backend payload shape

## Useful Files

- [backend/server.py](backend/server.py)
- [backend/api/ws_handlers.py](backend/api/ws_handlers.py)
- [backend/api/indian_kanoon.py](backend/api/indian_kanoon.py)
- [backend/api/rag_llm_chain_prompting.py](backend/api/rag_llm_chain_prompting.py)
- [backend/api/precedent_qa.py](backend/api/precedent_qa.py)
- [frontend/src/hooks/useLexIR.js](frontend/src/hooks/useLexIR.js)
- [frontend/src/components/ChatArea.js](frontend/src/components/ChatArea.js)
- [frontend/src/components/Stage1Card.js](frontend/src/components/Stage1Card.js)
- [frontend/src/components/Stage2Card.js](frontend/src/components/Stage2Card.js)

## Current Status

The project currently supports:

- Structured FIR analysis and legal reasoning
- Statute mapping and stage-based reasoning
- Civil / contract dispute detection with zero criminal sections when appropriate
- Indian Kanoon precedent retrieval
- Case summaries and verdict prediction
- Chat history and session persistence
- Follow-up legal Q&A

## Next Steps

- Improve precedent retrieval quality for more specific FIR fact patterns
- Continue tuning stage 2 ranking and summarization
- Expand civil-vs-criminal detection coverage for more non-criminal complaint patterns
- Extend the frontend history and session management UX if needed
