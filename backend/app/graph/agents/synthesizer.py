import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

from app.system_prompts import SYNTHESIZER_PROMPT


PLACEHOLDER_SOURCE = re.compile(
    r"\s*\(Source:\s*(?:\[)?https?://doi\.org/10\.20935/xxx(?:\])?(?:\([^)]*\))?\s*\)",
    re.IGNORECASE,
)


def _remove_placeholder_sources(text: str) -> str:
    return PLACEHOLDER_SOURCE.sub("", text)

def build_report(topic, web_findings, wiki_findings, paper_findings):
    web_text = _remove_placeholder_sources("\n\n".join(web_findings)) if web_findings else "(nothing came back from web search)"
    wiki_text = _remove_placeholder_sources("\n\n".join(wiki_findings)) if wiki_findings else "(nothing came back from wikipedia)"
    paper_text = _remove_placeholder_sources("\n\n".join(paper_findings)) if paper_findings else "(nothing came back from papers)"

    user_msg = f"""Topic: {topic}

    WEB FINDINGS:
    {web_text}

    WIKIPEDIA FINDINGS:
    {wiki_text}

    PAPER FINDINGS:
    {paper_text}

    Write the final report now."""

    resp = llm.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=user_msg),
    ])
    content = resp.content
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return _remove_placeholder_sources(content.strip())


def synthesizer_node(state: dict) -> dict:
    topic = state["topic"]
    web_findings = state.get("web_findings", [])
    wiki_findings = state.get("wiki_findings", [])
    paper_findings = state.get("paper_findings", [])

    report = build_report(topic, web_findings, wiki_findings, paper_findings)
    return {"final_report": report}