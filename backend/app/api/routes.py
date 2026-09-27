# backend/app/api/routes.py
import re

from fastapi import APIRouter, HTTPException, Response

from app.models.schemas import PdfExportRequest, ResearchRequest, ResearchResponse
from app.graph.build_graph import run_research
from app.pdf_export import create_report_pdf

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


@router.post("/export-pdf")
async def export_pdf(payload: PdfExportRequest):
    if not payload.topic.strip() or not payload.report.strip():
        raise HTTPException(status_code=400, detail="topic and report are required")

    try:
        pdf = create_report_pdf(payload.topic.strip(), payload.report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", payload.topic.strip()).strip("_")
    filename = f"{filename or 'research-report'}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )