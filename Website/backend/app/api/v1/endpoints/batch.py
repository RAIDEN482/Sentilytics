import csv
import json
import os
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import JobStatusResponse, AnalyzeOptions
from app.models.analysis import AnalysisJob
from app.services.analysis import analysis_service
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()


@router.post("", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column_mapping: str = Form(...),
    options: str = Form(...),
    current_user: User = Depends(get_current_user),
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
    
    # Save file to local uploads directory
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
        user_id=current_user.id,

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
    
    # Enqueue background task handled by AnalysisService
    background_tasks.add_task(
        analysis_service.process_batch_job,
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


