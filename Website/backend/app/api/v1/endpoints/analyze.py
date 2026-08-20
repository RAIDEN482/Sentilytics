import uuid
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.sentiment import analyze_sentiment_local
from app.models.log import SingleAnalysis

router = APIRouter()

DEFAULT_USER_ID = uuid.UUID("4a3b2c1d-5e6f-7a8b-9c0d-1e2f3a4b5c6d")

@router.post("", response_model=AnalyzeResponse)
async def analyze_text(
    request: Request,
    payload: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Single-text synchronous analysis.
    """
    include_aspects = payload.options.include_aspects if payload.options else True
    include_key_phrases = payload.options.include_key_phrases if payload.options else True
    
    # Perform local sentiment analysis
    analysis_res = analyze_sentiment_local(
        text=payload.text,
        include_aspects=include_aspects,
        include_key_phrases=include_key_phrases
    )
    
    # Log to single_analyses table
    single_analysis = SingleAnalysis(
        user_id=DEFAULT_USER_ID,
        raw_text=payload.text,
        overall_sentiment=analysis_res["overall_sentiment"]["label"],
        compound_score=analysis_res["overall_sentiment"]["compound_score"],
        aspects=analysis_res["aspects"]
    )
    
    db.add(single_analysis)
    await db.commit()
    await db.refresh(single_analysis)
    
    return AnalyzeResponse(
        id=str(single_analysis.id),
        overall_sentiment=analysis_res["overall_sentiment"],
        aspects=analysis_res["aspects"],
        key_phrases=analysis_res["key_phrases"],
        metadata=analysis_res["metadata"]
    )

