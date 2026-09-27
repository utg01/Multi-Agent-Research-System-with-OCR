import os
from typing import List, Optional

import requests
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

wiki_relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

CANDIDATES_PER_TOPIC = 5
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = (
    "ResearchAgent/1.0 "
    "(https://github.com/utg01/Multi-Agent-Research-System-with-OCR)"
)
WIKIPEDIA_TIMEOUT = 10


class WikiPageError(Exception):
    pass


class WikiDisambiguationError(Exception):
    def __init__(self, title: str, options: List[str]):
        super().__init__(title)
        self.title = title
        self.options = options


def _wiki_request(params: dict) -> dict:
    response = requests.get(
        WIKIPEDIA_API_URL,
        params={**params, "format": "json"},
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=WIKIPEDIA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_page(title: str) -> dict:
    data = _wiki_request({
        "action": "query",
        "prop": "extracts|info|pageprops",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "8",
        "inprop": "url",
        "redirects": "1",
        "titles": title,
    })
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page:
        raise WikiPageError(title)
    if "pageprops" in page and "disambiguation" in page["pageprops"]:
        options_data = _wiki_request({
            "action": "query",
            "prop": "links",
            "plnamespace": "0",
            "pllimit": str(CANDIDATES_PER_TOPIC),
            "titles": page["title"],
        })
        options_page = next(iter(options_data["query"]["pages"].values()))
        options = [link["title"] for link in options_page.get("links", [])]
        raise WikiDisambiguationError(page["title"], options)
    return {
        "title": page["title"],
        "extract": page.get("extract", ""),
        "url": page["fullurl"],
    }


class TitleScore(BaseModel):
    index: int = Field(description="Index of the candidate title in the provided list")
    relevant: bool = Field(description="Whether this article is actually about the intended topic")
    reason: str = Field(description="One short sentence justifying the verdict")


class TitleAssessment(BaseModel):
    scores: List[TitleScore]


def _search_candidates(topic: str, max_results: int = CANDIDATES_PER_TOPIC) -> List[str]:
    try:
        data = _wiki_request({
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": max_results,
        })
        results = [item["title"] for item in data["query"]["search"]]
    except Exception as e:
        print(f"wiki search failed for '{topic}': {e}")
        return []
    if not results:
        print(f"wiki search returned ZERO candidates for '{topic}'")
    return results


def _pick_relevant_title(topic: str, candidates: List[str]) -> Optional[str]:
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
        page = _fetch_page(title)
    except WikiDisambiguationError as e:
        # the picked title turned out to itself be a disambiguation page --
        # score its options the same way instead of taking the first one blindly
        sub_title = _pick_relevant_title(topic, e.options[:CANDIDATES_PER_TOPIC])
        if not sub_title:
            return f"[Wikipedia] {topic}\n(couldn't resolve disambiguation, skipping)"
        try:
            page = _fetch_page(sub_title)
        except Exception as err:
            print(f"disambiguation fallback failed for '{topic}': {err}")
            return f"[Wikipedia] {topic}\n(couldn't resolve disambiguation, skipping)"
    except WikiPageError:
        print(f"no wiki page found for title '{title}' (topic '{topic}')")
        return f"[Wikipedia] {topic}\n(no page found)"
    except Exception as e:
        print(f"wiki fetch failed for '{topic}': {e}")
        return f"[Wikipedia] {topic}\n(fetch failed)"

    try:
        summary = page["extract"]
    except Exception as e:
        print(f"wiki summary fetch failed for '{page.title}': {e}")
        return f"[Wikipedia] {topic}\n(fetch failed)"

    return f"[Wikipedia] {topic}\n{summary}\n(Source: {page['url']})"


def wiki_agent_node(state: dict) -> dict:
    topics = state.get("plan", {}).get("wikipedia_topics", [])

    if not topics:
        return {"wiki_findings": []}

    findings = []
    for t in topics:
        findings.append(get_wiki_summary(t))

    return {"wiki_findings": findings}