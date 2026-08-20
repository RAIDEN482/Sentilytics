from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Numeric, Text, Uuid, JSON
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime, timezone
from app.models.base import Base

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(String(50), default="queued")
    total_rows = Column(Integer, nullable=False, default=0)
    processed_rows = Column(Integer, nullable=False, default=0)
    failed_rows = Column(Integer, nullable=False, default=0)
    summary_distribution = Column(JSON, default={"positive": 0, "neutral": 0, "negative": 0, "mixed": 0})
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="jobs")
    results = relationship("AnalysisResult", back_populates="job", cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False)
    row_index = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)
    compound_score = Column(Numeric(5, 4))
    positive_score = Column(Numeric(5, 4))
    neutral_score = Column(Numeric(5, 4))
    negative_score = Column(Numeric(5, 4))
    overall_sentiment = Column(String(20))
    aspects = Column(JSON, default=[])
    key_phrases = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    job = relationship("AnalysisJob", back_populates="results")

