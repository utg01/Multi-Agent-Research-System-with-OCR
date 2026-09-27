# Multi-Agent Research System with OCR

A LangGraph-based multi-agent research assistant with OCR-style PDF text and
figure extraction. Enter a topic, and it plans research queries, searches the
web (Tavily), Wikipedia, and academic papers (arXiv), then synthesizes
everything into a structured report.

## Architecture

<img src="https://raw.githubusercontent.com/utg01/Multi-Agent-Research-System-with-OCR/main/docs/architecture.png" alt="Multi-Agent Research System architecture" width="100%" />

This system uses LangGraph to coordinate a planner, three specialized research
agents, and a final synthesizer. The planner creates focused research tasks,
the web, Wikipedia, and paper agents investigate those tasks in parallel, and
the synthesizer combines their findings into a cited report.

## How it works

- A planner node generates web, Wikipedia, and paper research queries.
- Three subagents run in parallel: web search, Wikipedia lookup, and
  academic paper research (with relevance filtering, PDF text extraction, and
  Gemini vision for figures).
- A synthesizer combines everything into a final report: Overview, Key
  Findings, Recent Developments, Points of Disagreement, Conclusion.

## Tech Stack

- Backend: Python, FastAPI, LangGraph, LangChain, Gemini
  (langchain-google-genai), Tavily, arXiv, PyMuPDF
- Frontend: HTML, CSS, JavaScript (no framework, no build step)

## Environment Variables

Create a `.env` file in the backend with:

```env
GOOGLE_API_KEY=
TAVILY_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
```

## Setup

```powershell
git clone <repo-url>
cd research-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run the backend:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

Backend runs at http://127.0.0.1:8000

Serve the frontend:

```powershell
python -m http.server 5500 --directory frontend
```

Open http://localhost:5500

## Deployment

- Backend: Render
- Frontend: Vercel

Render's free tier spins down when idle, so the first request after a
period of inactivity can take up to a minute.
