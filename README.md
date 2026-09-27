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

The workflow starts with the planner node, which rewrites the user's topic,
uses an initial Tavily search for grounding, and creates focused queries for
each research source. LangGraph then runs the three research agents in
parallel before passing their findings to the synthesizer.

### Web Research Agent

The web agent sends each planned query to Tavily, excluding Wikipedia domains.
It keeps the source title, URL, and content while preparing the search results
for Gemini to summarize. The resulting findings are passed to the synthesizer
with the original research question and available source context.

### Wikipedia Agent

The Wikipedia agent searches for candidate article titles instead of relying
only on automatic title suggestions. It uses Gemini relevance validation to
choose an article related to the requested topic, handles disambiguation, and
returns a sourced summary with the selected page URL.

### Academic Paper Agent

The paper agent searches arXiv for multiple candidates for each planned paper
query. Before downloading anything, it uses Gemini to filter candidates by
relevance and avoids selecting the same paper more than once. Relevant papers
are downloaded temporarily, their PDF text is extracted with PyMuPDF, and
pages containing figures are rendered as images. Gemini Vision describes
charts and diagrams, after which Gemini summarizes the paper's contribution
and findings. Temporary PDF files are removed after processing.

### Synthesizer Agent

The synthesizer receives the web, Wikipedia, and paper findings after the
parallel research stage. It combines the evidence into one structured report
with the sections Overview, Key Findings, Recent Developments, Points of
Disagreement, and Conclusion, while preserving source citations and avoiding
unsupported claims.

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
