from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import JobStatusResponse

router = APIRouter()

@router.post("", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_upload(
    request: Request,
    file: UploadFile = File(...),
    column_mapping: str = Form(...),
    options: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Bulk file upload (async).
    """
    pass
