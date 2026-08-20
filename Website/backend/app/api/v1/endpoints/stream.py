# pyrefly: ignore [missing-import]
import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.future import select
import redis.asyncio as redis_async

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.analysis import AnalysisJob
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request, current_user: User = Depends(get_current_user)):
    """
    Live progress via SSE using Redis Pub/Sub.
    """
    try:
        uuid_val = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )

    # Validate that job belongs to current user
    async with async_session_maker() as session:
        result = await session.execute(select(AnalysisJob).filter_by(id=uuid_val))
        job = result.scalar_one_or_none()
        
        if not job or job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        if job.status in ["completed", "failed"]:
            # If already done, we can just return one event and close
            async def single_event():
                yield f"data: {json.dumps({'job_id': str(job.id), 'status': job.status, 'total_rows': job.total_rows, 'processed_rows': job.processed_rows, 'failed_rows': job.failed_rows, 'summary_distribution': job.summary_distribution})}\n\n"
            return StreamingResponse(single_event(), media_type="text/event-stream")

    async def event_generator():
        r = redis_async.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        channel_name = f"channel:job_progress:{job_id}"
        await pubsub.subscribe(channel_name)

        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from SSE stream of job {job_id}")
                    break

                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    data = message["data"].decode("utf-8")
                    yield f"data: {data}\n\n"
                    
                    data_dict = json.loads(data)
                    if data_dict.get("status") in ["completed", "failed"]:
                        break
                else:
                    # Keep-alive or just yield nothing
                    pass
                    
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            await r.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
