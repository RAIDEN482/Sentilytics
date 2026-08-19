# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request):
    """
    Live progress via SSE.
    """
    pass
