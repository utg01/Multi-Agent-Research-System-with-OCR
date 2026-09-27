import os
import json
from app.system_prompts import WEB_SUMMARY_SYSTEM_PROMPT
from tavily import TavilyClient
from langchain_core.messages import SystemMessage, HumanMessage

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
from app.graph.build_graph import llm

def _search_and_summarize(question: str) -> str:
    try:
        response = tavily_client.search(
            query=question,
            max_results=4,
            search_depth="advanced",
            exclude_domains=["wikipedia.org"],
        )
        raw_results = response.get("results", [])
    except Exception as e:
        print(f"Tavily search failed for question '{question}': {e}")
        return f"[Web] {question}\n(search failed, skipped)"

    if not raw_results:
        return f"[Web] {question}\n(no results found)"

    # keeping only what the LLM actually needs — title, url, content — drop score/raw_content/etc noise
    clean_results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in raw_results
        if r.get("content")
    ]
    results_json = json.dumps(clean_results, indent=2)

    try:
        response = llm.invoke([
            SystemMessage(content=WEB_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nSearch results (JSON):\n{results_json}"),
        ])
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        summary = content.strip()
    except Exception as e:
        print(f"Summarization failed for question '{question}': {e}")
        return f"[Web] {question}\n(summarization failed)"

    return f"[Web] {question}\n{summary}"


def web_agent_node(state: dict) -> dict:
    web_queries = state.get("plan", {}).get("web_queries", [])

    if not web_queries:
        return {"web_findings": []}

    findings = [_search_and_summarize(question) for question in web_queries]
    return {"web_findings": findings}