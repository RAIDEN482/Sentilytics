from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.analysis import analysis_service
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("", response_model=AnalyzeResponse)
async def analyze_text(
    request: Request,
    payload: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Single-text synchronous analysis.
    """
    return await analysis_service.analyze_text(
        text=payload.text,
        options=payload.options,
        user_id=current_user.id,
        db=db
    )



