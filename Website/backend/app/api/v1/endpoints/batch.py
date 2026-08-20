import csv
import json
import os
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import get_db, async_session_maker
from app.schemas.analysis import JobStatusResponse, AnalyzeOptions
from app.models.analysis import AnalysisJob, AnalysisResult
from app.services.sentiment import analyze_sentiment_local

router = APIRouter()

DEFAULT_USER_ID = uuid.UUID("4a3b2c1d-5e6f-7a8b-9c0d-1e2f3a4b5c6d")
logger = logging.getLogger(__name__)

async def process_batch_job(
    job_id: uuid.UUID,
    file_content: str,
    text_column: str,
    include_aspects: bool,
    include_key_phrases: bool
):
    # Parse rows from file content
    reader = csv.DictReader(file_content.splitlines())
    rows = list(reader)
    
    async with async_session_maker() as session:
        try:
            # 1. Update status to "processing"
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one_or_none()
            if not job:
                return
            
            job.status = "processing"
            await session.commit()
            
            processed = 0
            failed = 0
            dist = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
            
            for idx, row in enumerate(rows):
                text = row.get(text_column, "")
                if not text:
                    # Case insensitive lookup fallback
                    lowercase_row = {k.lower(): v for k, v in row.items()}
                    text = lowercase_row.get(text_column.lower(), "")
                    
                if not text and len(row) > 0:
                    # Fallback to the first column
                    text = list(row.values())[0]
                    
                if not text:
                    failed += 1
                    continue
                
                try:
                    res = analyze_sentiment_local(
                        text=text,
                        include_aspects=include_aspects,
                        include_key_phrases=include_key_phrases
                    )
                    
                    # Create AnalysisResult record
                    analysis_result = AnalysisResult(
                        job_id=job_id,
                        row_index=idx,
                        raw_text=text,
                        compound_score=res["overall_sentiment"]["compound_score"],
                        positive_score=res["overall_sentiment"]["positive_score"],
                        neutral_score=res["overall_sentiment"]["neutral_score"],
                        negative_score=res["overall_sentiment"]["negative_score"],
                        overall_sentiment=res["overall_sentiment"]["label"],
                        aspects=res["aspects"],
                        key_phrases=res["key_phrases"]
                    )
                    session.add(analysis_result)
                    
                    label = res["overall_sentiment"]["label"]
                    if label in dist:
                        dist[label] += 1
                    else:
                        dist["mixed"] += 1
                        
                    processed += 1
                except Exception as e:
                    logger.error(f"Row {idx} processing failed: {e}")
                    failed += 1
                
                # Frequently update database to allow SSE updates to be real-time
                result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
                job = result.scalar_one()
                job.processed_rows = processed
                job.failed_rows = failed
                job.summary_distribution = dist.copy()
                await session.commit()
                
            # Complete Job
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one()
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info(f"Batch job {job_id} completed successfully.")
            
        except Exception as e:
            logger.error(f"Error executing batch job {job_id}: {e}")
            result = await session.execute(select(AnalysisJob).filter_by(id=job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()

@router.post("", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column_mapping: str = Form(...),
    options: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Bulk file upload (async).
    """
    job_id = uuid.uuid4()
    
    # Parse column mapping
    text_column = "text"
    try:
        mapping = json.loads(column_mapping)
        if isinstance(mapping, dict):
            text_column = mapping.get("text_column") or mapping.get("text") or list(mapping.values())[0]
        else:
            text_column = str(mapping)
    except json.JSONDecodeError:
        text_column = column_mapping

    # Parse options
    opt = AnalyzeOptions()
    try:
        opt_dict = json.loads(options)
        opt = AnalyzeOptions(**opt_dict)
    except Exception:
        pass
        
    # Read file contents
    contents = await file.read()
    file_content_str = contents.decode("utf-8")
    
    # Save file on local workspace
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{job_id}.csv"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content_str)
        
    # Calculate row count
    reader = csv.DictReader(file_content_str.splitlines())
    rows = list(reader)
    total_rows = len(rows)
    
    # Create AnalysisJob
    job = AnalysisJob(
        id=job_id,
        user_id=DEFAULT_USER_ID,
        file_name=file.filename or "uploaded_file.csv",
        file_path=file_path,
        status="queued",
        total_rows=total_rows,
        processed_rows=0,
        failed_rows=0,
        summary_distribution={"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Enqueue background task
    background_tasks.add_task(
        process_batch_job,
        job_id=job_id,
        file_content=file_content_str,
        text_column=text_column,
        include_aspects=opt.include_aspects,
        include_key_phrases=opt.include_key_phrases
    )
    
    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        failed_rows=job.failed_rows,
        estimated_completion=datetime.now(timezone.utc),
        summary_distribution=job.summary_distribution
    )

