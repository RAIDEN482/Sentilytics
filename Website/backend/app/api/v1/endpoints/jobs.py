import io
import csv
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.core.database import get_db
from app.schemas.analysis import JobStatusResponse
from app.schemas.results import PaginatedResults, AnalysisResultResponse
from app.models.analysis import AnalysisJob, AnalysisResult

router = APIRouter()
logger = logging.getLogger(__name__)

def parse_uuid(job_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Job status & progress snapshot.
    """
    uuid_val = parse_uuid(job_id)
    result = await db.execute(select(AnalysisJob).filter_by(id=uuid_val))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
        
    # Estimate completion time
    estimated_completion = None
    if job.status == "processing" and job.processed_rows > 0:
        elapsed = (datetime.now(timezone.utc) - job.created_at).total_seconds()
        rate = job.processed_rows / elapsed
        if rate > 0:
            remaining = job.total_rows - job.processed_rows
            est_seconds = remaining / rate
            estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=est_seconds)
    elif job.status == "queued":
        estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=job.total_rows * 0.1)
    elif job.status in ["completed", "failed"]:
        estimated_completion = job.completed_at
        
    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        failed_rows=job.failed_rows,
        estimated_completion=estimated_completion,
        summary_distribution=job.summary_distribution
    )

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
    uuid_val = parse_uuid(job_id)
    
    # Check job exists
    job_check = await db.execute(select(AnalysisJob).filter_by(id=uuid_val))
    if not job_check.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
        
    # Count total results
    total_result = await db.execute(
        select(func.count(AnalysisResult.id)).filter_by(job_id=uuid_val)
    )
    total = total_result.scalar_one()
    
    # Fetch paginated results
    offset = (page - 1) * size
    query = select(AnalysisResult).filter_by(job_id=uuid_val).order_by(AnalysisResult.row_index).offset(offset).limit(size)
    results = await db.execute(query)
    items = results.scalars().all()
    
    # Form response items
    response_items = []
    for item in items:
        response_items.append(
            AnalysisResultResponse(
                id=item.id,
                job_id=item.job_id,
                row_index=item.row_index,
                raw_text=item.raw_text,
                compound_score=float(item.compound_score) if item.compound_score is not None else 0.0,
                positive_score=float(item.positive_score) if item.positive_score is not None else 0.0,
                neutral_score=float(item.neutral_score) if item.neutral_score is not None else 0.0,
                negative_score=float(item.negative_score) if item.negative_score is not None else 0.0,
                overall_sentiment=item.overall_sentiment or "neutral",
                aspects=item.aspects or [],
                key_phrases=item.key_phrases or []
            )
        )
        
    return PaginatedResults(
        items=response_items,
        total=total,
        page=page,
        size=size
    )

@router.get("/{job_id}/export")
async def export_job_results(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Streamed CSV download.
    """
    uuid_val = parse_uuid(job_id)
    
    # Check job exists
    result = await db.execute(select(AnalysisJob).filter_by(id=uuid_val))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
        
    # Generate CSV rows
    async def csv_generator():
        header = [
            "row_index", "raw_text", "overall_sentiment", "compound_score",
            "positive_score", "neutral_score", "negative_score", "key_phrases", "aspects"
        ]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        yield output.getvalue()
        
        # Query results
        query = select(AnalysisResult).filter_by(job_id=uuid_val).order_by(AnalysisResult.row_index)
        results = await db.execute(query)
        
        for item in results.scalars().all():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                item.row_index,
                item.raw_text,
                item.overall_sentiment,
                float(item.compound_score) if item.compound_score is not None else 0.0,
                float(item.positive_score) if item.positive_score is not None else 0.0,
                float(item.neutral_score) if item.neutral_score is not None else 0.0,
                float(item.negative_score) if item.negative_score is not None else 0.0,
                json.dumps(item.key_phrases),
                json.dumps(item.aspects)
            ])
            yield output.getvalue()
            
    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_{job_id}.csv"}
    )

