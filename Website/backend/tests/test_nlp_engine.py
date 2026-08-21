"""
Standalone unit tests for Sentilytics Three-Tier Hybrid ABSA Orchestrator (nlp_engine.py).
Tests execute independently of the API and database layers.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from app.services import nlp_engine as nlp_mod
from app.services.nlp_engine import (
    NLPEngine,
    LLMAspectAnalysis,
    LLMAspectItem,
    get_spacy_nlp,
    CATEGORY_KEYWORDS,
)
from app.schemas.analysis import AnalyzeOptions, AnalyzeResponse, SentimentScore, AspectDetail


@pytest.fixture
def engine():
    """Returns a clean instance of NLPEngine."""
    return NLPEngine()


# ===========================================================================
# 1. Tier 1: Lexicon-based VADER Tests
# ===========================================================================

def test_tier1_positive_sentiment(engine):
    text = "The product is absolutely amazing, fast, and extremely easy to use!"
    score = engine.analyze_tier1_vader(text)

    assert isinstance(score, SentimentScore)
    assert score.label == "positive"
    assert score.compound_score > 0.5
    assert score.positive_score > 0.0
    assert score.negative_score == 0.0


def test_tier1_negative_sentiment(engine):
    text = "Terrible experience. The system is horribly slow, buggy, and completely broken."
    score = engine.analyze_tier1_vader(text)

    assert isinstance(score, SentimentScore)
    assert score.label == "negative"
    assert score.compound_score < -0.5
    assert score.negative_score > 0.0


def test_tier1_neutral_sentiment(engine):
    text = "The server is located in data center room 4B."
    score = engine.analyze_tier1_vader(text)

    assert isinstance(score, SentimentScore)
    assert score.label == "neutral"
    assert -0.05 <= score.compound_score <= 0.05


def test_tier1_empty_text(engine):
    score = engine.analyze_tier1_vader("")
    assert score.label == "neutral"
    assert score.compound_score == 0.0
    assert score.neutral_score == 1.0


# ===========================================================================
# 2. Tier 2: Strict Pydantic JSON-Schema LLM Tests
# ===========================================================================

def test_llm_schema_valid_item():
    item = LLMAspectItem(
        aspect="billing system",
        category="Billing & Price",
        sentiment="negative",
        confidence=0.92,
        evidence="The billing system charged me twice."
    )
    assert item.aspect == "billing system"
    assert item.category == "Billing & Price"
    assert item.sentiment == "negative"
    assert item.confidence == 0.92


def test_llm_schema_invalid_confidence():
    with pytest.raises(ValidationError):
        LLMAspectItem(
            aspect="pricing",
            category="Billing & Price",
            sentiment="positive",
            confidence=1.5, # Out of range (must be <= 1.0)
            evidence="Great price."
        )


def test_llm_schema_invalid_sentiment():
    with pytest.raises(ValidationError):
        LLMAspectItem(
            aspect="speed",
            category="Speed & Performance",
            sentiment="super-happy", # Invalid literal
            confidence=0.8,
            evidence="Very fast."
        )


def test_tier2_llm_successful_parsing(engine):
    mock_item = LLMAspectItem(
        aspect="customer support",
        category="Customer Support",
        sentiment="positive",
        confidence=0.95,
        evidence="Customer support was very helpful and responsive."
    )
    mock_analysis = LLMAspectAnalysis(
        aspects=[mock_item],
        key_phrases=["customer support", "helpful"]
    )

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_analysis
    mock_client.beta.chat.completions.parse.return_value.choices = [mock_choice]

    with patch.object(engine, "_get_openai_client", return_value=mock_client):
        result = engine.analyze_tier2_llm("Customer support was very helpful and responsive.")
        assert result is not None
        assert len(result.aspects) == 1
        assert result.aspects[0].aspect == "customer support"
        assert result.aspects[0].sentiment == "positive"
        assert result.key_phrases == ["customer support", "helpful"]


# ===========================================================================
# 3. Tier 3: spaCy Noun-Chunk & VADER Fallback Tests
# ===========================================================================

def test_tier3_spacy_fallback_extraction(engine):
    text = "The user interface is very beautiful and easy, but the subscription price is too high."
    overall = engine.analyze_tier1_vader(text)

    res = engine.analyze_tier3_spacy_fallback(
        text=text,
        overall_sentiment=overall,
        include_aspects=True,
        include_key_phrases=True,
    )

    assert "aspects" in res
    assert "key_phrases" in res
    aspects = res["aspects"]
    assert len(aspects) > 0

    aspect_names = [a.aspect.lower() for a in aspects]
    categories = [a.category for a in aspects]

    # Should detect UI or usability and pricing/subscription
    assert any("interface" in a or "user" in a for a in aspect_names) or "UI & Usability" in categories
    assert any("price" in a or "subscription" in a for a in aspect_names) or "Billing & Price" in categories


def test_tier3_category_mapping(engine):
    assert engine._categorize_aspect("pricing plan", "the pricing plan is affordable") == "Billing & Price"
    assert engine._categorize_aspect("page load speed", "the speed is blazing fast") == "Speed & Performance"
    assert engine._categorize_aspect("help desk", "the support rep helped me") == "Customer Support"
    assert engine._categorize_aspect("button design", "the ui layout is clean") == "UI & Usability"
    assert engine._categorize_aspect("system crash", "the bug caused a server error") == "Stability & Reliability"
    assert engine._categorize_aspect("unknown entity", "this is something else") == "General"


# ===========================================================================
# 4. Lazy-Loading spaCy Verification Constraint
# ===========================================================================

def test_spacy_lazy_loading():
    """
    Verify that spaCy model is NOT loaded upon initial import or during Tier 1 only execution.
    It must be strictly loaded on-demand.
    """
    # Reset singleton to test lazy loading
    nlp_mod._spacy_nlp_instance = None
    assert nlp_mod._spacy_nlp_instance is None

    test_engine = NLPEngine()
    # Tier 1 only should NOT load spaCy
    test_engine.analyze_tier1_vader("The product is great!")
    assert nlp_mod._spacy_nlp_instance is None, "spaCy model should NOT load during Tier 1 VADER execution!"

    # Now invoke Tier 3 or get_spacy_nlp()
    loaded_nlp = get_spacy_nlp()
    assert loaded_nlp is not None
    assert nlp_mod._spacy_nlp_instance is not None


# ===========================================================================
# 5. Hybrid Orchestration & Cascading Fallback Tests
# ===========================================================================

def test_orchestrator_cascade_on_llm_failure(engine):
    """
    When LLM raises an exception (e.g. rate limit / network error / timeout),
    the orchestrator must catch it and cascade seamlessly to Tier 3.
    """
    text = "The customer support was lightning fast and very polite."
    
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = Exception("RateLimitError: 429 Too Many Requests")

    with patch.object(engine, "_get_openai_client", return_value=mock_client):
        response = engine.analyze(text)

        assert isinstance(response, AnalyzeResponse)
        assert response.overall_sentiment.label == "positive"
        assert len(response.aspects) > 0
        assert response.metadata.llm_used is False
        assert "tier3" in response.metadata.model_version


def test_orchestrator_end_to_end_output_contract(engine):
    text = "The dashboard looks clean and responsive, but we encountered a major bug in export reports."
    response = engine.analyze(text, options=AnalyzeOptions(include_aspects=True, include_key_phrases=True))

    assert isinstance(response, AnalyzeResponse)
    assert response.id is not None
    assert response.overall_sentiment is not None
    assert isinstance(response.overall_sentiment.compound_score, float)
    assert isinstance(response.aspects, list)
    assert len(response.aspects) > 0

    for aspect in response.aspects:
        assert isinstance(aspect, AspectDetail)
        assert aspect.aspect != ""
        assert aspect.category != ""
        assert aspect.sentiment in ("positive", "negative", "neutral")
        assert 0.0 <= aspect.confidence <= 1.0
        assert len(aspect.evidence) > 0

    assert isinstance(response.key_phrases, list)
    assert response.metadata.processing_time_ms >= 0


def test_root_module_reexport():
    import nlp_engine
    assert hasattr(nlp_engine, "NLPEngine")
    assert hasattr(nlp_engine, "nlp_engine")
    assert hasattr(nlp_engine, "LLMAspectAnalysis")
    assert hasattr(nlp_engine, "LLMAspectItem")
    assert hasattr(nlp_engine, "get_spacy_nlp")


def test_analysis_service_integration():
    from app.services.analysis import analysis_service
    res = analysis_service.analyze_raw("The application performance is incredibly fast, but customer support is unhelpful.")
    assert "overall_sentiment" in res
    assert "aspects" in res
    assert "key_phrases" in res
    assert "metadata" in res
    assert len(res["aspects"]) > 0


