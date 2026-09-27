# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException

from app.models.schemas import ResearchRequest, ResearchResponse
from app.graph.build_graph import run_research

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def research(payload: ResearchRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        result = await run_research(payload.query.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"research failed: {e}")

    return ResearchResponse(
        topic=result.get("topic", payload.query.strip()),
        plan=result.get("plan", {}),
        report=result.get("final_report", ""),
    )