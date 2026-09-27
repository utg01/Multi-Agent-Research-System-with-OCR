"""
Changes from the original:
1. Concurrency: process_query() runs per paper_query, and up to MAX_PAPERS
   queries are worked on at once via a small worker pool, instead of one
   query fully finishing (download -> extract -> vision -> summarize)
   before the next starts. Workers stop pulling new queries once 3
   successful papers are banked, preserving the original cost-capping
   behavior while getting concurrency's speedup.
2. Relevance filter: each query now fetches CANDIDATES_PER_QUERY arXiv
   results (was 1) and a single LLM call scores all of them against the
   topic before any PDF is downloaded. Only a candidate judged relevant
   gets downloaded/extracted/vision-processed. If none qualify, the query
   is skipped (logged, not treated as a failure).
3. Rate limiting: the unconditional time.sleep(20) between every image
   batch is gone. Vision calls are capped to VISION_CONCURRENCY at a time
   via a semaphore, and only back off (exponentially) when a call actually
   fails with a rate-limit/quota error.
4. Storage: PDFs now download to the OS temp directory instead of a
   project-relative "papers" folder, and are deleted immediately after
   processing (success or failure). A persistent project folder doesn't
   survive on most deployment platforms' ephemeral filesystems, and even
   where it does, it never got cleaned up -- every paper ever processed
   would sit on disk forever.

IMPORTANT: papers_agent_node is now async. LangGraph runs async nodes fine
alongside sync ones, but you need to invoke the graph with
`await graph.ainvoke(state)` (or `.astream`) instead of `graph.invoke(state)`
for this to actually run without blocking the event loop. A sync wrapper is
provided at the bottom for call sites that can't be made async, but it will
raise if called from inside an already-running event loop (e.g. FastAPI,
Jupyter) -- in that case, call papers_agent_node_async directly.
"""

import os
import base64
import asyncio
import tempfile
from urllib.request import urlretrieve

import arxiv
import fitz  # PyMuPDF
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.system_prompts import SYNTHESIZER_PROMPT
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

vision_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

paper_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

# Reuses the same model for the pre-download relevance check. Swap for a
# lighter/cheaper model here if you have one available -- this call only
# ever sees titles + abstracts, never full paper content.
relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

MAX_PAPERS = 4
CANDIDATES_PER_QUERY = 5
VISION_CONCURRENCY = 2          # concurrent Gemini vision calls, across all papers
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 10       # seconds; doubles each retry

_vision_semaphore = asyncio.Semaphore(VISION_CONCURRENCY)


class FigureDescription(BaseModel):
    page: int = Field(description="Page number this figure appears on")
    figure_content: str = Field(description="Description of the chart/diagram/figure relevant to the topic")


class FigureDescriptions(BaseModel):
    figures: List[FigureDescription]


class CandidateScore(BaseModel):
    index: int = Field(description="Index of the candidate in the provided list")
    relevant: bool = Field(description="Whether this paper is genuinely relevant to the topic (not just keyword-adjacent)")
    reason: str = Field(description="One short sentence justifying the verdict")


class RelevanceAssessment(BaseModel):
    scores: List[CandidateScore]


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "rate limit", "quota", "resource_exhausted"))


async def _with_rate_limit_retry(coro_fn, *args, **kwargs):
    """Await coro_fn(*args, **kwargs), backing off only on genuine rate-limit errors."""
    delay = RATE_LIMIT_BASE_DELAY
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < RATE_LIMIT_MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise


def _find_candidates_sync(query: str, max_results: int):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    return list(client.results(search))


async def find_paper_candidates(query: str, max_results: int = CANDIDATES_PER_QUERY):
    return await asyncio.to_thread(_find_candidates_sync, query, max_results)


async def pick_relevant_paper(topic: str, candidates: list, used_ids: set):
    """Score all candidates in a single call and return the first one judged
    relevant AND not already picked for another query this run, in arXiv's
    own relevance order. Returns None if nothing clears the bar -- that's a
    legitimate "no good match" outcome, not an error.

    The used_ids check-and-reserve happens in the same synchronous block
    (no `await` in between), so concurrent workers can't both claim the
    same paper even though they may be scoring candidates at the same time."""
    if not candidates:
        return None

    listing = "\n\n".join(
        f"[{i}] Title: {c.title}\nAbstract: {c.summary[:800]}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""Topic: {topic}

Below are candidate papers found for a research query. For EACH one, judge whether
it is genuinely relevant to the topic (not just keyword-adjacent).

{listing}

Return a verdict for every index listed above."""

    structured = relevance_llm.with_structured_output(RelevanceAssessment)
    try:
        result = await _with_rate_limit_retry(structured.ainvoke, [HumanMessage(content=prompt)])
    except Exception as e:
        print(f"relevance check failed for topic '{topic}': {e}")
        return None

    verdicts = {s.index: s.relevant for s in result.scores}
    for i, c in enumerate(candidates):
        if verdicts.get(i) and c.entry_id not in used_ids:
            used_ids.add(c.entry_id)
            return c
    return None


def _download_paper_sync(paper) -> str:
    """Downloads to the OS temp directory (not a project-relative folder) --
    on most deployment platforms (Render included) the app filesystem is
    ephemeral and/or has a small disk quota, and a permanent 'papers' folder
    would accumulate forever with no cleanup. tempfile.mkstemp gives a
    unique path per paper (safe under concurrent workers) that's guaranteed
    writable in virtually any containerized environment."""
    if not paper.pdf_url:
        raise ValueError("paper has no PDF URL")
    fd, path = tempfile.mkstemp(
        suffix=".pdf",
        prefix=f"paper_{paper.get_short_id().replace('/', '_')}_",
    )
    os.close(fd)  # urlretrieve reopens/writes the path itself
    urlretrieve(paper.pdf_url, path)
    return path


async def download_paper(paper):
    return await asyncio.to_thread(_download_paper_sync, paper)


def _extract_text_and_figure_pages_sync(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    figure_pages = []

    for i, page in enumerate(doc):
        content = page.get_text()
        pages.append({"page_no": i, "content": content})
        if page.get_images():
            figure_pages.append(i)

    return doc, pages, figure_pages


async def extract_text_and_figure_pages(pdf_path):
    return await asyncio.to_thread(_extract_text_and_figure_pages_sync, pdf_path)


def _render_pages_sync(doc, page_nums):
    """CPU-bound pixmap rendering + base64 encoding, kept off the event loop."""
    content = []
    for page_num in page_nums:
        pix = doc[page_num].get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        content.append({"type": "text", "text": f"(above image is page {page_num + 1})"})
    return content


async def describe_rendered_batch(rendered_content: list, topic: str):
    """Takes already-rendered page content (see _render_pages_sync) and runs
    the vision LLM call. No `doc` access here, so this is safe to run
    concurrently across batches -- only the actual network call needs to
    wait on the semaphore."""
    content = [
        {"type": "text", "text": f"These are pages from a research paper. For each page, describe "
                                  f"any charts, diagrams or figures relevant to '{topic}' in a few sentences. "
                                  f"Match each description to its correct page number."}
    ] + rendered_content

    structured_vision_llm = vision_llm.with_structured_output(FigureDescriptions)

    async with _vision_semaphore:
        result = await _with_rate_limit_retry(structured_vision_llm.ainvoke, [HumanMessage(content=content)])

    return [f.model_dump() for f in result.figures]


async def summarize_paper(title, content, topic):
    prompt = f"""Topic: {topic}
Paper: {title}

Content (body text + figure descriptions):
{content}

Summarize this paper's key contribution and findings in 3-4 sentences, focused on what's relevant to the topic."""

    resp = await _with_rate_limit_retry(
        paper_llm.ainvoke,
        [SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=prompt)],
    )
    content = resp.content
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content.strip()


async def process_query(query: str, topic: str, used_ids: set) -> Optional[str]:
    """Full pipeline for one paper_query. Returns a finding string, or None
    if nothing relevant was found (a legitimate skip, not a failure)."""
    candidates = await find_paper_candidates(query)
    paper = await pick_relevant_paper(topic, candidates, used_ids)
    if not paper:
        return None

    path = await download_paper(paper)
    try:
        doc, pages, figure_pages = await extract_text_and_figure_pages(path)
        try:
            figure_batches = [figure_pages[i:i + 3] for i in range(0, len(figure_pages), 3)]

            # Rendering touches the fitz `doc` object directly, so it stays
            # sequential (it's fast, CPU-bound work anyway). The actual vision
            # model calls -- the slow, network-bound part -- run concurrently,
            # bounded by _vision_semaphore. This was the real bottleneck for
            # image-heavy papers: previously each batch's network call waited
            # for the previous one to finish even though nothing required that.
            rendered_batches = [
                await asyncio.to_thread(_render_pages_sync, doc, batch)
                for batch in figure_batches
            ]
            batch_results = await asyncio.gather(
                *(describe_rendered_batch(rendered, topic) for rendered in rendered_batches)
            )
            figure_notes = [item for batch in batch_results for item in batch]

            full_text = "\n".join(p["content"] for p in pages)
            combined = full_text[:6000]

            if figure_notes:
                figure_text = "\n".join(f"Page {f['page']}: {f['figure_content']}" for f in figure_notes)
                combined += "\n\nFigures:\n" + figure_text

            summary = await summarize_paper(paper.title, combined, topic)
            return f"[Paper] {paper.title} ({paper.published.date()})\n{summary}\n(Source: {paper.entry_id})"
        finally:
            doc.close()
    finally:
        # Always clean up the temp PDF, whether processing succeeded, raised,
        # or was interrupted -- this is what actually prevents disk growth,
        # not just moving the download location.
        try:
            os.remove(path)
        except OSError:
            pass


async def papers_agent_node_async(state: dict) -> dict:
    topic = state["topic"]
    paper_queries = list(state.get("plan", {}).get("paper_queries", []))

    findings: List[str] = []
    papers_done = 0
    used_ids: set = set()  # arXiv entry_ids already claimed, shared across all workers

    async def worker():
        nonlocal papers_done
        # No `await` between the loop condition and the pop below, so under
        # asyncio's single-threaded cooperative model this can't double-pop
        # or race past MAX_PAPERS across workers.
        while paper_queries and papers_done < MAX_PAPERS:
            query = paper_queries.pop(0)
            try:
                result = await process_query(query, topic, used_ids)
            except Exception as e:
                print(f"couldn't process paper for query '{query}': {e}")
                findings.append(f"[Paper] {query}\n(failed to process, skipped)")
                continue
            if result is None:
                print(f"no relevant paper found for query '{query}'")
                continue
            if papers_done < MAX_PAPERS:
                findings.append(result)
                papers_done += 1

    num_workers = min(MAX_PAPERS, len(paper_queries)) or 1
    await asyncio.gather(*(worker() for _ in range(num_workers)))

    return {"paper_findings": findings}


def papers_agent_node(state: dict) -> dict:
    """Sync-compatible entry point for call sites that haven't moved to
    ainvoke yet. Prefer calling papers_agent_node_async directly via
    `await graph.ainvoke(...)`. This wrapper raises if called from inside an
    already-running event loop (FastAPI, Jupyter, etc.) -- use the async
    version directly there."""
    return asyncio.run(papers_agent_node_async(state))