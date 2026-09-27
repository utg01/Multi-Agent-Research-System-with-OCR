"""
Root cause of the espionage-article bug: wikipedia.page(topic, auto_suggest=True)
can silently resolve to an unrelated page with NO exception raised -- auto_suggest
does its own fuzzy text match internally and just returns whatever it thinks is
closest, with zero relevance validation. That's how "intelligent agent" ended up
fetching content about a spy/intelligence agent.

The second bug: the DisambiguationError handler took e.options[0] blindly, which
is Wikipedia's arbitrary internal ordering, not a relevance ranking.

Neither of those explains the outright fetch failures on "autonomous agents" /
"multi-agent systems" -- those are almost certainly PageErrors from auto_suggest
failing to match oddly-phrased multi-word queries to Wikipedia's actual titles.

Fix: replace auto_suggest entirely with wikipedia.search() (returns real
candidate titles, never raises DisambiguationError) + the same relevance-gate
pattern used in papers_agent.py -- score all candidates against the topic in one
LLM call, pick the first one judged relevant, then fetch that exact title with
auto_suggest=False. If the picked title itself turns out to be a disambiguation
page, the same relevance scoring is applied to its options instead of taking
the first one blindly.
"""

import os
from typing import List, Optional

import wikipedia
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

wiki_relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

CANDIDATES_PER_TOPIC = 5


class TitleScore(BaseModel):
    index: int = Field(description="Index of the candidate title in the provided list")
    relevant: bool = Field(description="Whether this article is actually about the intended topic")
    reason: str = Field(description="One short sentence justifying the verdict")


class TitleAssessment(BaseModel):
    scores: List[TitleScore]


def _search_candidates(topic: str, max_results: int = CANDIDATES_PER_TOPIC) -> List[str]:
    try:
        results = wikipedia.search(topic, results=max_results)
    except Exception as e:
        print(f"wiki search failed for '{topic}': {e}")
        return []
    if not results:
        print(f"wiki search returned ZERO candidates for '{topic}'")
    return results


def _pick_relevant_title(topic: str, candidates: List[str]) -> Optional[str]:
    """Score candidate titles against the topic and return the first one judged
    relevant, in Wikipedia's own search-relevance order. This is what replaces
    both auto_suggest's silent guessing and the old e.options[0] fallback."""
    if not candidates:
        return None

    listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = f"""Topic: {topic}

Below are candidate Wikipedia article titles found for this topic. For EACH one,
judge whether it is actually about the intended topic -- watch especially for
same-word-different-meaning pages (e.g. "agent" meaning a software/AI agent vs.
a spy, insurance, or real-estate agent; or any other overloaded term).

{listing}

Return a verdict for every index listed above."""

    structured = wiki_relevance_llm.with_structured_output(TitleAssessment)
    try:
        result = structured.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        print(f"wiki relevance check failed for '{topic}': {e}")
        # fall back to the top search result rather than failing the topic outright --
        # search's own relevance ranking is a reasonable fallback, just unverified
        return candidates[0]

    verdicts = {s.index: s.relevant for s in result.scores}
    for i, c in enumerate(candidates):
        if verdicts.get(i):
            return c

    print(f"wiki relevance check REJECTED all {len(candidates)} candidates for '{topic}':")
    for s in result.scores:
        title = candidates[s.index] if s.index < len(candidates) else "?"
        print(f"  [{s.index}] '{title}' -> relevant={s.relevant} ({s.reason})")
    return None


def get_wiki_summary(topic):
    candidates = _search_candidates(topic)
    if not candidates:
        return f"[Wikipedia] {topic}\n(no page found)"

    title = _pick_relevant_title(topic, candidates)
    if not title:
        return f"[Wikipedia] {topic}\n(no matching page found among candidates)"

    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        # the picked title turned out to itself be a disambiguation page --
        # score its options the same way instead of taking the first one blindly
        sub_title = _pick_relevant_title(topic, e.options[:CANDIDATES_PER_TOPIC])
        if not sub_title:
            return f"[Wikipedia] {topic}\n(couldn't resolve disambiguation, skipping)"
        try:
            page = wikipedia.page(sub_title, auto_suggest=False)
        except Exception as err:
            print(f"disambiguation fallback failed for '{topic}': {err}")
            return f"[Wikipedia] {topic}\n(couldn't resolve disambiguation, skipping)"
    except wikipedia.exceptions.PageError:
        print(f"no wiki page found for title '{title}' (topic '{topic}')")
        return f"[Wikipedia] {topic}\n(no page found)"
    except Exception as e:
        print(f"wiki fetch failed for '{topic}': {e}")
        return f"[Wikipedia] {topic}\n(fetch failed)"

    try:
        summary = wikipedia.summary(page.title, sentences=8, auto_suggest=False)
    except Exception as e:
        print(f"wiki summary fetch failed for '{page.title}': {e}")
        return f"[Wikipedia] {topic}\n(fetch failed)"

    return f"[Wikipedia] {topic}\n{summary}\n(Source: {page.url})"


def wiki_agent_node(state: dict) -> dict:
    topics = state.get("plan", {}).get("wikipedia_topics", [])

    if not topics:
        return {"wiki_findings": []}

    findings = []
    for t in topics:
        findings.append(get_wiki_summary(t))

    return {"wiki_findings": findings}