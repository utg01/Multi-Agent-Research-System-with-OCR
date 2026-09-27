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

## Features

- Topic-based research planning with LangGraph
- Parallel research across web, Wikipedia, and academic papers
- Tavily web search with source titles, URLs, and content
- Wikipedia article search with relevance validation
- arXiv paper search with duplicate prevention and relevance filtering
- PDF text extraction with PyMuPDF
- Gemini Vision analysis for paper figures and diagrams
- Temporary PDF cleanup after paper processing
- Structured report generation with source citations
- Simple static frontend with PDF report download

## Agent Roles

### Planner Agent

- Rewrites the user's topic into a focused research direction.
- Uses an initial Tavily search to ground the research plan.
- Creates separate web, Wikipedia, and paper queries.

### Web Research Agent

- Searches Tavily for relevant and recent web information.
- Excludes Wikipedia domains to keep the web source independent.
- Sends search content to Gemini for concise summaries.
- Preserves available source titles and URLs in the research findings.

### Wikipedia Agent

- Searches for candidate Wikipedia article titles.
- Uses Gemini to validate article relevance to the requested topic.
- Handles ambiguous or disambiguation pages.
- Returns a sourced article summary.

### Academic Paper Agent

- Searches arXiv for multiple candidate papers.
- Filters candidates for relevance before downloading PDFs.
- Prevents the same paper from being selected more than once.
- Extracts PDF text and detects pages containing figures.
- Uses Gemini Vision to describe charts and diagrams.
- Summarizes each selected paper and removes temporary PDF files.

### Synthesizer Agent

- Combines findings from all three research agents.
- Preserves source citations and avoids unsupported claims.
- Produces a structured report with Overview, Key Findings, Recent
  Developments, Points of Disagreement, and Conclusion.

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

## Sample Output

Sample input: `cricket in india`

See the generated [sample research report PDF](cricket_in_india.pdf).

## Deployment

- Backend: Render
- Frontend: Vercel

Render's free tier spins down when idle, so the first request after a
period of inactivity can take up to a minute.
