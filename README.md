# ZaloPay Fraud Analytics Assistant

A production-grade multi-agent fraud assistant built with **LangGraph** for ZaloPay. The agent acts as a domain expert — it can answer general fraud questions, retrieve domain knowledge, and generate detailed fraud analysis reports through a conversational interface.

**Live demo:** `https://endpoint-6e655273-09b0-4d2f-9a2c-cb90d6e4d8bb.agentbase-runtime.aiplatform.vngcloud.vn`

---

## Overview

The assistant has two operating modes:

| Mode | Trigger | Behaviour |
|---|---|---|
| **Q&A / Chat** | Any question, domain query, data lookup | Answered directly using knowledge base + data tools + optional web search. No pipeline run. |
| **Report Generation** | Explicit report request ("generate a report for…") | Runs the full 7-agent analysis pipeline and produces a structured fraud report. |

On first entry the assistant shows a welcome screen listing all available report domains and quick-action chips.

---

## Report Domains

| Domain | Description |
|---|---|
| `fraud_loss` | Monthly + weekly loss by segment (domestic, international, VNPAY, domestic-card…) |
| `promo_abuse` | Promo abuse rate, BAD_V2 / FAD detection effectiveness |
| `coin2dd` | Coin-to-Direct-Debit abuse analysis |
| `appid_breakdown` | Fraud breakdown by merchant / appID |
| `general` | Full overview across all domains above |

---

## Architecture

### Conversation Flow

```
User Message
     │
     ▼
┌──────────────────┐
│  Conversation    │  First entry → welcome + domain list
│    Manager       │  General question  → answer (followup node)
└────────┬─────────┘  Explicit report   → proceed (full pipeline)
         │             Ambiguous         → clarify (recommend domains)
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
 proceed    answer         clarify
    │         │              │
    ▼         ▼              ▼
Full      Followup       Human Input
Pipeline    Node          (interrupt)
    │         │              │
    │    (answers from       └──► Conversation Manager
    │   knowledge + data          (next turn)
    │   tools + web search)
    ▼
Human Input (interrupt — follow-up or new request)
```

### Full Report Pipeline

```
Orchestrator  →  Retrieval  →  Query  →  Summarizer  →  Reasoning  →  Validation  →  Report
     │               │            │            │               │             │            │
  Parses         FAISS         Pandas       Converts       ZaloPay       Validates    Produces
  intent &      vector        pipeline     analysis       decision      findings    exec summary
  maps to       store         + suggest_*  outputs to     trees +       (retries    + analyst
  tables        lookup        tools        narrative      severity      if weak)    report
```

---

## Data Pipeline

Reads from `projectF/data input/` CSVs and produces 5 output tables:

| Table | Description |
|---|---|
| `fraud_monthly_loss` | MoM fraud loss by segment |
| `fraud_weekly_loss` | WoW fraud loss by segment |
| `promo_weekly_abuse` | Weekly promo abuse + detection metrics |
| `coin2dd_monthly` | Monthly Coin2DD abuse by category |
| `appid_fraud_breakdown` | Fraud by appID / merchant with MoM comparison |

Segment mapping: `454` → domestic_direct · `9999` → international · `1002` → domestic_card · `1023` → VNPAY · `7022` → domestic_wallet

---

## Analysis Tools

Deterministic `suggest_*` functions produce structured findings (no LLM) before the LLM writes the narrative:

| Tool | Output |
|---|---|
| `analyze_fraud_monthly` | MoM change flags — CRITICAL / ALERT / WATCH / STABLE |
| `analyze_fraud_weekly` | WoW change flags per segment |
| `analyze_promo_weekly` | Abuse rate vs threshold, detection health |
| `analyze_coin2dd` | Coin2DD % vs CRITICAL threshold (>20%) |
| `analyze_appid_breakdown` | Top appIDs with MoM delta flags |

Priority labels: **CRITICAL** (same-day action) · **ALERT** (24 h) · **WATCH** (this week) · **STABLE**

---

## Knowledge Base

8 ZaloPay-specific `.txt` files embedded into a FAISS vector store at startup:

```
data/knowledge/
├── zalopay_segment_definitions.txt       # appID → segment, pmcID mapping
├── zalopay_fraud_thresholds.txt          # Monthly/weekly loss thresholds, promo %, Coin2DD %
├── zalopay_fraud_patterns.txt            # 6 known attack patterns (Top-up Flow, Gaming/Telco, …)
├── zalopay_fraud_narrative_templates.txt # Fill-in-the-blank report templates + decision tree
├── zalopay_promo_abuse_patterns.txt      # BAD_V2, FAD, 5 abuse patterns, detection health
├── zalopay_promo_narrative_templates.txt # Promo narrative templates + decision tree
├── zalopay_cross_domain_principles.txt   # Team ownership, 5 table purposes, reporting principles
└── zalopay_glossary.txt                  # POM, BAD_V2, FAD, DVI, VAMP, TC40, ATO, CNP, Coin2DD…
```

**To add knowledge:** drop a `.txt` file in `data/knowledge/` and restart the server.

---

## Web Search

The follow-up agent can call `search_fintech_web` (DuckDuckGo, anchored to `fintech payment fraud risk`) when:
- The user explicitly asks to search the web, **or**
- The question covers an industry concept not in the local knowledge base

Web results are used as supplementary background context — never quoted directly.

---

## Project Structure

```
├── fraud_analytics/
│   ├── agents/
│   │   ├── conversation.py    # Conversation manager + welcome + routing
│   │   ├── orchestrator.py    # Intent parser → pillar + table mapping
│   │   ├── retrieval.py       # FAISS knowledge search
│   │   ├── query.py           # Pipeline + suggest_* tool-calling loop
│   │   ├── summarizer.py      # Converts analysis outputs → narrative
│   │   ├── reasoning.py       # ZaloPay decision trees + severity
│   │   ├── validation.py      # Finding validation + retry logic
│   │   ├── report.py          # Exec summary + analyst report (parallel)
│   │   └── followup.py        # Q&A — knowledge + data tools + web search
│   ├── graph/
│   │   └── fraud_graph.py     # LangGraph state machine
│   ├── tools/
│   │   ├── pipeline.py        # Pandas pipeline → 5 output tables
│   │   └── analysis.py        # Deterministic suggest_* functions
│   ├── knowledge/
│   │   ├── vector_store.py    # FAISS knowledge base
│   │   └── web_enrichment.py  # DuckDuckGo web search utility
│   ├── state.py               # LangGraph state definition
│   └── config.py              # LLM factory + structured_invoke()
├── data/
│   ├── knowledge/             # ZaloPay domain knowledge (.txt)
│   └── vector_store/          # FAISS index (auto-generated)
├── projectF/                  # Raw data input (git-ignored)
├── models/                    # Embedding model (downloaded locally)
├── frontend/
│   └── index.html             # Chat UI with domain list + quick chips
├── server.py                  # FastAPI HTTP server
├── main.py                    # CLI entry point
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
  "message": "Generate a fraud loss report for last month",
  "session_id": "optional-on-first-call"
}
```

Response:
```json
{
  "session_id": "abc-123",
  "response": "Agent question or answer",
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
# Edit .env — set AI_PLATFORM_API_KEY at minimum
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
python main.py
```

---

## Deployment (GreenNode AgentBase)

```bash
# 1. Login to AgentBase Container Registry
bash ~/.claude/skills/agentbase/scripts/cr.sh credentials docker-login

# 2. Build for linux/amd64
TAG="v$(date +%Y%m%d%H%M%S)"
docker build --platform linux/amd64 \
  -t vcr.vngcloud.vn/111480-abp111980/claw-a-thon-demo-agent:$TAG .

# 3. Push
docker push vcr.vngcloud.vn/111480-abp111980/claw-a-thon-demo-agent:$TAG

# 4. Update runtime
bash ~/.claude/skills/agentbase/scripts/runtime.sh update \
  runtime-357a5879-1fa1-4245-be42-4030d6569b60 \
  --image vcr.vngcloud.vn/111480-abp111980/claw-a-thon-demo-agent:$TAG \
  --flavor runtime-s2-general-4x8 \
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
| Data pipeline | Pandas |
| Web search | DuckDuckGo (`ddgs`) |
| HTTP server | FastAPI + Uvicorn |
| Deployment | GreenNode AgentBase Runtime |
