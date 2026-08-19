from pydantic import BaseModel, ConfigDict
from typing import List
from uuid import UUID

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
    aspects: list
    key_phrases: list

    model_config = ConfigDict(from_attributes=True)

class PaginatedResults(BaseModel):
    items: List[AnalysisResultResponse]
    total: int
    page: int
    size: int
