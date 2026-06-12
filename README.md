# Fraud Analytics Agent

A production-grade multi-agent system built with **LangGraph** that analyzes transaction data and generates detailed fraud reports through a conversational interface.

---

## Overview

The agent accepts natural language requests like _"Give me the weekly fraud report for last week, focusing on discount abuse"_ and autonomously runs a 7-agent pipeline to retrieve knowledge, query data, reason about findings, validate them, and produce a structured report — all while maintaining a multi-turn conversation.

**Live demo:** `https://endpoint-6e655273-09b0-4d2f-9a2c-cb90d6e4d8bb.agentbase-runtime.aiplatform.vngcloud.vn`

---

## Architecture

### Agent Pipeline

```
User Request
     │
     ▼
┌─────────────┐     clarify/follow_up     ┌─────────────┐
│Conversation │ ◄────────────────────────► │ Human Input │
│   Manager   │                            │  (interrupt)│
└─────────────┘                            └─────────────┘
     │ proceed
     ▼
┌─────────────┐
│Orchestrator │  Parses intent → report_type, fraud_pillar, date_range
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │  Searches FAISS vector store for relevant fraud policies/SOPs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Query    │  Executes 9 data tools (transactions, merchants, users)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Summarizer  │  Converts raw query results into business-readable summaries
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reasoning  │  Identifies fraud findings with severity and evidence
└──────┬──────┘
       │
       ▼
┌─────────────┐     retry (query/retrieval)
│ Validation  │ ──────────────────────────►  loop back if findings are weak
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐
│   Report    │  Generates executive summary + 7-section analyst report
└──────┬──────┘
       │
       ▼
Conversation Manager  →  offers follow-up or ends session
```

### Conversation Flow

The **Conversation Manager** decides after every user message:

| Decision | Behaviour |
|---|---|
| `proceed` | Run the full fraud analysis pipeline |
| `clarify` | Ask the user a focused question before running |
| `follow_up` | After a report, offer to do more analysis |
| `end` | Gracefully close the session |

---

## Fraud Pillars

The agent can analyse any of these fraud dimensions:

| Pillar | Description |
|---|---|
| **Merchant Abuse** | Volume spikes, concentration risk, collusion signals |
| **Discount Abuse** | Promo code stacking, high discount ratios, coordinated abuse |
| **User Abuse** | High-frequency users, account farming, ATO signals |
| **Payment Risk** | Failure rate anomalies, card testing, BIN-level patterns |
| **Volume Risk** | Z-score spikes, synthetic transaction rings, bot activity |

---

## Data Tools

9 LangChain tools available to the Query agent:

| Tool | Data Returned |
|---|---|
| `query_transaction_summary` | Volume, amount, success/failure rates, avg/median value |
| `query_discount_analysis` | Discount ratio per merchant, top abusers, leakage |
| `query_payment_solution_breakdown` | Success rate by payment solution (pmcID) |
| `query_trend_comparison` | Current vs previous period delta |
| `query_daily_volume_anomalies` | Z-score per day, anomaly flags |
| `query_merchant_metrics` | Top merchants by volume, count, discount |
| `query_merchant_new_vs_existing` | New vs existing merchant share |
| `query_user_metrics` | Top users by transaction count/amount |
| `query_user_discount_behavior` | Users with highest discount ratios |

---

## Knowledge Base

The Retrieval agent uses a **FAISS vector store** backed by documents in `data/knowledge/`. Each `.txt` file is embedded with `all-MiniLM-L6-v2` at startup.

```
data/knowledge/
├── merchant_abuse_policy.txt
├── discount_abuse_sop.txt
├── volume_anomaly_playbook.txt
├── user_abuse_guidelines.txt
├── payment_risk_policy.txt
├── historical_report_q4_2024.txt
├── fraud_rules_v3.2.txt
└── emerging_risks_2025_q1.txt
```

**To add new knowledge:** drop a `.txt` file in `data/knowledge/` using this format, then restart the server:

```
source: my_source
type: my_type
version: 1.0

Your document content here...
```

---

## Project Structure

```
├── fraud_analytics/
│   ├── agents/
│   │   ├── conversation.py   # Conversation manager
│   │   ├── orchestrator.py   # Request parser
│   │   ├── retrieval.py      # FAISS knowledge search
│   │   ├── query.py          # Agentic tool-calling loop
│   │   ├── summarizer.py     # Data summarization
│   │   ├── reasoning.py      # Fraud finding extraction
│   │   ├── validation.py     # Finding validation
│   │   └── report.py         # Report generation
│   ├── graph/
│   │   └── fraud_graph.py    # LangGraph state machine
│   ├── tools/
│   │   ├── transaction_tools.py
│   │   ├── merchant_tools.py
│   │   └── user_tools.py
│   ├── knowledge/
│   │   └── vector_store.py   # FAISS knowledge base
│   ├── utils/
│   │   └── data_simulator.py # Mock transaction generator
│   ├── schemas/models.py     # Pydantic output schemas
│   ├── state.py              # LangGraph state definition
│   └── config.py             # LLM factory + structured_invoke()
├── data/
│   ├── knowledge/            # Source documents (.txt)
│   └── vector_store/         # FAISS index (auto-generated)
├── models/                   # Embedding model (downloaded locally)
├── frontend/
│   └── index.html            # Chat UI
├── server.py                 # FastAPI HTTP server
├── main.py                   # CLI entry point
├── Dockerfile
└── requirements.txt
```

---

## API

### `GET /health`
Liveness probe. Returns `{"status": "ok"}`.

### `POST /chat`

```json
{
  "message": "Weekly fraud report for last week, focusing on discount abuse",
  "session_id": "optional-on-first-call"
}
```

Response:
```json
{
  "session_id": "abc-123",
  "response": "Agent question or follow-up message",
  "report": "Full markdown report (non-empty only when a new report is ready)",
  "done": false
}
```

Pass the returned `session_id` on every subsequent call. When `done: true` the session is closed.

### `DELETE /chat/{session_id}`
Explicitly close a session.

### `GET /docs`
Interactive Swagger UI.

---

## Running Locally

### 1. Setup

```bash
git clone https://github.com/luongnn2311/claw-a-thon-demo-agent.git
cd claw-a-thon-demo-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API key:
# AI_PLATFORM_API_KEY=your_key_here
```

### 3. Download embedding model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')
"
```

### 4. Start the server

```bash
python server.py
# → http://localhost:8080
# → http://localhost:8080/docs  (Swagger UI)
```

### 5. CLI mode (optional)

```bash
# Multi-turn chat
python main.py

# Single-shot report
python main.py --single "Weekly fraud report for last week"
```

---

## Deployment (GreenNode AgentBase)

```bash
# 1. Build for linux/amd64
docker build --platform linux/amd64 -t vcr.vngcloud.vn/<repo>/fraud-analytics:latest .

# 2. Login to AgentBase Container Registry
bash ~/.claude/skills/agentbase/scripts/cr.sh credentials docker-login

# 3. Push
docker push vcr.vngcloud.vn/<repo>/fraud-analytics:latest

# 4. Create runtime
bash ~/.claude/skills/agentbase/scripts/runtime.sh create \
  --name fraud-analytics \
  --image vcr.vngcloud.vn/<repo>/fraud-analytics:latest \
  --flavor runtime-s2-general-2x4 \
  --env-file .env \
  --from-cr
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | GreenNode AI Platform (Gemma 4 31B) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| HTTP server | FastAPI + Uvicorn |
| Deployment | GreenNode AgentBase Runtime |
