# backend/app/models/schemas.py
from typing import Any, Dict
from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str  # raw user input, planner will normalize this into a clean topic


class ResearchResponse(BaseModel):
    topic: str                    # cleaned topic, from the planner
    plan: Dict[str, Any]          # web_queries, wikipedia_topics, paper_queries, topic
    report: str