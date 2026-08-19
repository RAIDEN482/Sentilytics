from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from uuid import UUID
from datetime import datetime

class AnalyzeOptions(BaseModel):
    include_aspects: bool = True
    include_key_phrases: bool = True
    language: str = "en"

class AnalyzeRequest(BaseModel):
    text: str
    options: Optional[AnalyzeOptions] = AnalyzeOptions()

class SentimentScore(BaseModel):
    label: str
    positive_score: float
    neutral_score: float
    negative_score: float
    compound_score: float

class AspectDetail(BaseModel):
    aspect: str
    category: str
    sentiment: str
    confidence: float
    evidence: str

class AnalyzeMetadata(BaseModel):
    model_version: str
    processing_time_ms: int
    llm_used: bool

class AnalyzeResponse(BaseModel):
    id: str
    overall_sentiment: SentimentScore
    aspects: List[AspectDetail]
    key_phrases: List[str]
    metadata: AnalyzeMetadata

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total_rows: int
    processed_rows: int
    failed_rows: int
    estimated_completion: Optional[datetime] = None
    summary_distribution: dict

    model_config = ConfigDict(from_attributes=True)
