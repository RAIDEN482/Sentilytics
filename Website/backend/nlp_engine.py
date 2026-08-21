"""
Re-export module for Sentilytics NLPEngine.
Allows direct root-level imports:
    from nlp_engine import NLPEngine, nlp_engine, LLMAspectAnalysis, LLMAspectItem
"""

from app.services.nlp_engine import (
    NLPEngine,
    nlp_engine,
    LLMAspectAnalysis,
    LLMAspectItem,
    get_spacy_nlp,
    CATEGORY_KEYWORDS,
)

__all__ = [
    "NLPEngine",
    "nlp_engine",
    "LLMAspectAnalysis",
    "LLMAspectItem",
    "get_spacy_nlp",
    "CATEGORY_KEYWORDS",
]
