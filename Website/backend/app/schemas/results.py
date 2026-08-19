from pydantic import BaseModel, ConfigDict
from typing import List
from uuid import UUID
from app.schemas.analysis import AspectDetail

class AnalysisResultResponse(BaseModel):
    id: UUID
    job_id: UUID
    row_index: int
    raw_text: str
    compound_score: float
    positive_score: float
    neutral_score: float
    negative_score: float
    overall_sentiment: str
    aspects: List[AspectDetail]
    key_phrases: List[str]

    model_config = ConfigDict(from_attributes=True)

class PaginatedResults(BaseModel):
    items: List[AnalysisResultResponse]
    total: int
    page: int
    size: int
