from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
import os
load_dotenv()
from tavily import TavilyClient
from app.graph.build_graph import ResearchPlan, ResearchState
from app.system_prompts import PLANNER_SYSTEM_PROMPT
from app.graph.build_graph import llm
import arxiv

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class RewrittenQuery(BaseModel):
    search_query: str = Field(description="A concise, effective web search query rewritten from the user's raw input")


def planner_node(state: ResearchState) -> dict:
    # LLM rewrites the raw human query into a better search query
    rewrite_llm = llm.with_structured_output(RewrittenQuery)
    rewritten = rewrite_llm.invoke([
        SystemMessage(content="Rewrite the user's query into a concise, effective web search query/" \
                              "The results of that query will be used for further research"),
        HumanMessage(content=state["human_input"]),
    ])

    # exactly 1 Tavily call, using the LLM's own rewritten query
    results = tavily_client.search(
        query=rewritten.search_query,
        max_results=3,
        search_depth="fast",
        exclude_domains=["wikipedia.org"]
    )
    grounding_text = "\n".join(r["content"] for r in results.get("results", []))

    # Step 3: LLM builds the actual plan from the grounded context
    planner_llm = llm.with_structured_output(ResearchPlan)
    plan = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {state['human_input']}\n\nGrounding search results:\n{grounding_text}"),
    ])

    return {
        "plan": dict(plan),
        "topic": plan.get("topic", ""),
    }

def find_paper(query):
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    results = list(client.results(search))
    return results[0] if results else None



