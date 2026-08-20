import asyncio
import csv
import json
import logging
import uuid
import redis.asyncio as redis_async
from datetime import datetime, timezone
from celery import shared_task
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy.future import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.analysis import AnalysisJob, AnalysisResult
from app.services.analysis import analysis_service
from app.schemas.analysis import AnalyzeOptions

logger = logging.getLogger(__name__)

# Redis pubsub helper
async def publish_progress(job_id: str, payload: dict):
    try:
        r = redis_async.from_url(settings.REDIS_URL)
        channel = f"channel:job_progress:{job_id}"
        await r.publish(channel, json.dumps(payload))
        await r.aclose()
    except Exception as e:
        logger.error(f"Failed to publish progress to Redis: {e}")

async def async_process_chunk(chunk, text_column, include_aspects, include_key_phrases, semaphore):
    """
    Process a single chunk of rows. Uses a semaphore to cap concurrent executions.
    """
    async with semaphore:
        results = []
        for idx, row in chunk:
            text = row.get(text_column, "")
            if not text:
                lowercase_row = {k.lower(): v for k, v in row.items()}
                text = lowercase_row.get(text_column.lower(), "")
            if not text and len(row) > 0:
                text = list(row.values())[0]

            if not text:
                results.append((idx, None, None))
                continue

            try:
                # Wrap synchronous analyze_raw in a thread to unblock async event loop if it becomes heavy
                # or just run it synchronously as it currently is fast dictionary lookups.
                raw_res = analysis_service.analyze_raw(
                    text=text,
                    options=AnalyzeOptions(
                        include_aspects=include_aspects,
                        include_key_phrases=include_key_phrases
                    )
                )
                results.append((idx, text, raw_res))
            except Exception as e:
                logger.error(f"Error analyzing row {idx}: {e}")
                results.append((idx, text, None))
        return results

async def process_batch_job_async(job_id_str, file_content, text_column, include_aspects, include_key_phrases):
    job_id = uuid.UUID(job_id_str)
    reader = list(csv.DictReader(file_content.splitlines()))
    
    # Micro-batching: split into chunks of 15 rows
    chunk_size = 15
    chunks = []
    current_chunk = []
    
    for idx, row in enumerate(reader):
        current_chunk.append((idx, row))
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)
        
    semaphore = asyncio.Semaphore(5) # Concurrency semaphore capped at 5
    
    async with async_session_maker() as session:
        try:
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one_or_none()
            if not job:
                return

            job.status = "processing"
            await session.commit()

            processed = 0
            failed = 0
            dist = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}

            for chunk in chunks:
                chunk_results = await async_process_chunk(
                    chunk, text_column, include_aspects, include_key_phrases, semaphore
                )
                
                for idx, text, raw_res in chunk_results:
                    if raw_res is None:
                        failed += 1
                        continue
                        
                    analysis_result = AnalysisResult(
                        job_id=job_id,
                        row_index=idx,
                        raw_text=text,
                        compound_score=raw_res["overall_sentiment"]["compound_score"],
                        positive_score=raw_res["overall_sentiment"]["positive_score"],
                        neutral_score=raw_res["overall_sentiment"]["neutral_score"],
                        negative_score=raw_res["overall_sentiment"]["negative_score"],
                        overall_sentiment=raw_res["overall_sentiment"]["label"],
                        aspects=raw_res["aspects"],
                        key_phrases=raw_res["key_phrases"]
                    )
                    session.add(analysis_result)

                    label = raw_res["overall_sentiment"]["label"]
                    if label in dist:
                        dist[label] += 1
                    else:
                        dist["mixed"] += 1

                    processed += 1
                    
                # Commit after each chunk
                result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
                job = result.scalar_one()
                job.processed_rows = processed
                job.failed_rows = failed
                job.summary_distribution = dist.copy()
                await session.commit()
                
                # Publish progress to Redis via Pub/Sub
                await publish_progress(str(job_id), {
                    "job_id": str(job.id),
                    "status": job.status,
                    "total_rows": job.total_rows,
                    "processed_rows": job.processed_rows,
                    "failed_rows": job.failed_rows,
                    "summary_distribution": job.summary_distribution
                })
                
            # Mark completed
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one()
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
            
            # Publish final completed state
            await publish_progress(str(job_id), {
                "job_id": str(job.id),
                "status": job.status,
                "total_rows": job.total_rows,
                "processed_rows": job.processed_rows,
                "failed_rows": job.failed_rows,
                "summary_distribution": job.summary_distribution
            })
            
            logger.info(f"Batch analysis job {job_id} successfully completed.")
        except Exception as e:
            logger.error(f"Fatal error in batch job {job_id}: {e}")
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
            
            # Publish failed state
            await publish_progress(str(job_id), {
                "job_id": str(job_id),
                "status": "failed",
                "detail": str(e)
            })
            raise e

@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def process_batch_job_task(self, job_id: str, file_content: str, text_column: str, include_aspects: bool, include_key_phrases: bool):
    """
    Celery task with exponential backoff logic (3 attempts on transient failures)
    """
    try:
        asyncio.run(process_batch_job_async(job_id, file_content, text_column, include_aspects, include_key_phrases))
    except Exception as exc:
        # Exponential backoff calculation
        delay = self.default_retry_delay * (2 ** self.request.retries)
        try:
            self.retry(exc=exc, countdown=delay)
        except MaxRetriesExceededError:
            logger.error(f"Job {job_id} failed after {self.max_retries} retries.")
