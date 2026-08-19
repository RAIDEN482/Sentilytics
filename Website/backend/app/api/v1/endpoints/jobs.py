from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import JobStatusResponse
from app.schemas.results import PaginatedResults

router = APIRouter()

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Job status & progress snapshot.
    """
    pass

@router.get("/{job_id}/results", response_model=PaginatedResults)
async def get_job_results(
    job_id: str,
    request: Request,
    page: int = 1,
    size: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Paginated row-level results.
    """
    pass

@router.get("/{job_id}/export")
async def export_job_results(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Streamed CSV/JSON download.
    """
    pass
