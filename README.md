# Investor Relations — Decision Cockpit & Agentic Scanner

An intelligent, multi-tenant capable investor relations platform and predictive agentic framework for earnings call question forecasting, transcript search, and analyst dossier analytics. Built to support banking and enterprise financial institutions.

---

## 🌟 Key Features

* **Analyst Question Intelligence**: Predicts upcoming earnings call questions per analyst using historical Q&A trajectories, narration novelty, and persona metrics.
* **Agentic Graph Core (LangGraph)**: Multi-agent execution graph consisting of a **Planning Agent**, **Overall Topic Layer**, **Analyst-Specific Reweighter**, and **Verifier Grounding Gate**.
* **Multi-Bank & Multi-Quarter Archival Intelligence**: Scalable document parser and graph indexer designed for cross-institutional earnings call indexing and peer comparison.
* **Interactive Dashboard & Search**: Hybrid search engine (BM25 + graph traversal) over historical earnings call transcripts with citation attribution.
* **FastAPI Live Application**: REST API backend for live predictions, document ingestion, and transcript archives.
* **Quarter View & News Feeds**: Multi-quarter metric comparison charts (NIM, GNPA, PAT, CET1, and more, grouped by topic) alongside bank-specific external news and analyst-mention coverage, as top-level workspace tabs.
* **Usage & Audit**: A durable, append-only audit trail for every LLM call the platform makes (provider, model, purpose, tokens, latency, estimated cost, success/failure), with a dashboard for daily volume, per-provider breakdown, and a filterable recent-calls log.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mrjaggu/IR-Agentic-Scanner.git
cd IR-Agentic-Scanner

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
MODEL_PROVIDER=GROQ # GROQ | GEMINI | OPENAI
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_groq_api_key_here

# Optional -- powers the News feeds tab's external headlines (thenewsapi.com,
# free tier: 100 requests/day). Without it, News feeds still loads but shows
# "External news isn't configured on this server" instead of erroring.
THE_NEWS_API_TOKEN=your_thenewsapi_token_here
```

### 3. Running the Application

Launch the live FastAPI server:

```bash
python run.py app
```

Open your browser at **http://localhost:8000**.

---

## 🛠️ CLI Usage (`run.py`)

The project includes a CLI utility `run.py`:

```bash
# Start FastAPI application
python run.py app --port 8000

# Compile UI HTML template
python run.py ui

# Run LangGraph Agentic Pipeline
python run.py agentic

# Run held-out semantic evaluations
python run.py evaluate --semantic
```

---

## 🌐 Deployment to Vercel

This repository includes a `vercel.json` configuration and a GitHub Action workflow for automatic deployment on push to `main`.

### Setup Secrets in GitHub Repository Settings:
1. Navigate to **Settings > Secrets and variables > Actions**.
2. Add the following secrets:
   * `VERCEL_TOKEN`: Your Vercel Personal Access Token.
   * `VERCEL_ORG_ID`: Your Vercel Organization / Team ID.
   * `VERCEL_PROJECT_ID`: Your Vercel Project ID.
   * `GROQ_API_KEY`: Groq API key for model predictions.

---

## 📂 Repository Structure

```
├── frontend/             # Compiled UI & HTML templates
├── src/
│   ├── agentic/          # LangGraph multi-agent orchestration
│   ├── api/              # FastAPI & HTTP server handlers
│   ├── config/           # Platform settings & env configurations
│   ├── data/             # Transcript parsing & dossier synthesis
│   ├── audit/            # Durable per-call LLM audit trail (data/logs/llm_calls_*.jsonl)
│   ├── model_provider/   # Unified LLM provider client (Groq/Gemini/OpenAI)
│   ├── news/             # External headlines + analyst-mention coverage (News feeds tab)
│   ├── search/           # Hybrid search & reciprocal rank fusion
│   └── signals/          # Topic salience & metric extraction
├── .env.example          # Sample environment template
├── requirements.txt      # Standard requirements file
├── run.py                # Command-line entrypoint
└── vercel.json           # Vercel deployment configuration
```

---

## 🗺️ Roadmap & Contributing

See [ROADMAP.md](ROADMAP.md) for what's shipped, what's next, and what's
explicitly out of scope. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
get a change into the app -- branching model, commit conventions, and what
to verify before opening a PR.

---

## 📜 License

[MIT License](LICENSE)
