from typing import TypedDict, List, Annotated
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END

class ResearchPlan(TypedDict):
    web_queries: List[str]        # deep-search queries, web agent only
    wikipedia_topics: List[str]   # article titles, wiki agent only
    paper_queries: List[str]      # arXiv search queries, NOT exact paper titles — see below
    topic:str

class ResearchState(TypedDict):
    human_input:str
    topic: str
    plan: ResearchPlan
    web_findings: Annotated[List[str], operator.add]
    wiki_findings: Annotated[List[str], operator.add]
    paper_findings: Annotated[List[str], operator.add]
    final_report: str

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2,
)

from app.graph.agents.planner import planner_node
from app.graph.agents.web_agent import web_agent_node
from app.graph.agents.wiki_agent import wiki_agent_node
from app.graph.agents.papers_agent import papers_agent_node
from app.graph.agents.synthesizer import synthesizer_node

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("wiki_agent", wiki_agent_node)
    graph.add_node("papers_agent", papers_agent_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")

    # fan-out: planner feeds all three source agents in parallel
    graph.add_edge("planner", "web_agent")
    graph.add_edge("planner", "wiki_agent")
    graph.add_edge("planner", "papers_agent")

    # fan-in: LangGraph waits for all three branches before running synthesizer
    graph.add_edge("web_agent", "synthesizer")
    graph.add_edge("wiki_agent", "synthesizer")
    graph.add_edge("papers_agent", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()

compiled_graph = build_graph()

async def run_research(human_input: str) -> ResearchState:
    initial_state: ResearchState = {
        "human_input": human_input,
        "topic": "",
        "plan": {},
        "web_findings": [],
        "wiki_findings": [],
        "paper_findings": [],
        "final_report": "",
    }
    result = await compiled_graph.ainvoke(initial_state)
    return result