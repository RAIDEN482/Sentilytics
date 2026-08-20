# pyrefly: ignore [missing-import]
import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.future import select
from app.core.database import async_session_maker
from app.models.analysis import AnalysisJob

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request):
    """
    Live progress via SSE.
    """
    try:
        uuid_val = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )
        
    async def event_generator():
        last_status = None
        last_processed = -1
        
        while True:
            # Terminate if the client disconnects
            if await request.is_disconnected():
                logger.info(f"Client disconnected from SSE stream of job {job_id}")
                break
                
            async with async_session_maker() as session:
                result = await session.execute(select(AnalysisJob).filter_by(id=uuid_val))
                job = result.scalar_one_or_none()
                
                if not job:
                    yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
                    break
                    
                # Calculate estimated completion
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
                    
                est_comp_str = estimated_completion.isoformat() if estimated_completion else None
                
                status_data = {
                    "job_id": str(job.id),
                    "status": job.status,
                    "total_rows": job.total_rows,
                    "processed_rows": job.processed_rows,
                    "failed_rows": job.failed_rows,
                    "estimated_completion": est_comp_str,
                    "summary_distribution": job.summary_distribution
                }
                
                # Only yield if progress or status changed
                if job.status != last_status or job.processed_rows != last_processed:
                    yield f"data: {json.dumps(status_data)}\n\n"
                    last_status = job.status
                    last_processed = job.processed_rows
                    
                # Exit stream if job finished
                if job.status in ["completed", "failed"]:
                    break
                    
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

