"""
Sentilytics Three-Tier Hybrid ABSA (Aspect-Based Sentiment Analysis) Orchestrator.

Hierarchy:
  - Tier 1: Lexicon-based VADER analysis (instant, zero-cost, baseline sentiment polarity).
  - Tier 2: Strict Pydantic JSON-schema constrained LLM extraction (GPT-4o-mini / Gemini Flash / Ollama).
  - Tier 3: Lazy-loaded spaCy noun-chunk extraction + VADER sentence valence fallback.
"""

from __future__ import annotations

import re
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Literal
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core.config import settings
from app.schemas.analysis import (
    AnalyzeOptions,
    AnalyzeResponse,
    SentimentScore,
    AspectDetail,
    AnalyzeMetadata,
)

logger = logging.getLogger(__name__)

# Predefined domain category keywords for Tier 3 categorization
CATEGORY_KEYWORDS = {
    "Billing & Price": [
        "price", "pricing", "cost", "expensive", "cheap", "billing", "money",
        "subscription", "fee", "invoice", "charge", "affordable", "overpriced",
        "rate", "tier", "payment", "discount", "refund", "plan"
    ],
    "Speed & Performance": [
        "speed", "slow", "fast", "performance", "lag", "laggy", "loading",
        "load", "responsive", "hang", "delay", "quick", "latency", "throughput",
        "sluggish", "snappy", "instant", "fps", "render", "runtime"
    ],
    "UI & Usability": [
        "ui", "ux", "design", "interface", "layout", "navigation", "easy",
        "intuitive", "confusing", "cluttered", "button", "dashboard", "look",
        "theme", "visual", "simple", "complex", "clunky", "ergonomic", "workflow"
    ],
    "Customer Support": [
        "support", "help", "customer service", "agent", "ticket", "rep",
        "representative", "assistance", "response time", "team", "guidance",
        "service", "care", "helpline", "resolution"
    ],
    "Feature Set": [
        "feature", "features", "capability", "capabilities", "integration",
        "export", "report", "tool", "filter", "search", "customization",
        "option", "sync", "webhook", "api", "functionality", "plugin", "extension"
    ],
    "Stability & Reliability": [
        "bug", "bugs", "error", "errors", "fail", "failure", "failed", "crash",
        "crashes", "uptime", "downtime", "stable", "stability", "reliable",
        "reliability", "glitch", "glitches", "broken", "defect", "issue", "freeze"
    ],
}

# ---------------------------------------------------------------------------
# Strict Pydantic JSON Schemas for Tier 2 LLM Extraction
# ---------------------------------------------------------------------------

class LLMAspectItem(BaseModel):
    aspect: str = Field(
        ...,
        description="The specific aspect, component, or entity mentioned (e.g. 'battery life', 'customer service', 'checkout flow')."
    )
    category: str = Field(
        ...,
        description="High-level category. Choose from: 'Billing & Price', 'Speed & Performance', 'UI & Usability', 'Customer Support', 'Feature Set', 'Stability & Reliability', or 'General'."
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Strict sentiment polarity expressed towards this specific aspect."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting certainty."
    )
    evidence: str = Field(
        ...,
        description="Exact verbatim sentence or clause from the input text providing direct evidence."
    )


class LLMAspectAnalysis(BaseModel):
    aspects: List[LLMAspectItem] = Field(
        default_factory=list,
        description="List of aspect-level sentiment items identified in the text."
    )
    key_phrases: List[str] = Field(
        default_factory=list,
        description="List of salient key phrases or topics from the text."
    )


# ---------------------------------------------------------------------------
# Lazy-loaded spaCy singleton for Tier 3
# ---------------------------------------------------------------------------

_spacy_nlp_instance = None


def get_spacy_nlp():
    """
    Lazy-load spaCy model only when Tier 3 fallback actually fires.
    Keeps baseline server memory minimal.
    """
    global _spacy_nlp_instance
    if _spacy_nlp_instance is None:
        # pyrefly: ignore [missing-import]
        import spacy
        try:
            _spacy_nlp_instance = spacy.load("en_core_web_sm")
            logger.info("spaCy 'en_core_web_sm' model loaded successfully on demand.")
        except Exception as e:
            logger.warning(f"Could not load 'en_core_web_sm' ({e}), falling back to blank 'en' with sentencizer.")
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _spacy_nlp_instance = nlp
    return _spacy_nlp_instance


# ---------------------------------------------------------------------------
# NLP Engine Implementation
# ---------------------------------------------------------------------------

class NLPEngine:
    """
    Three-Tier Aspect-Based Sentiment Analysis Orchestrator.
    """

    def __init__(self):
        # Tier 1 VADER instance is fast, pure Python, and lightweight
        self.vader = SentimentIntensityAnalyzer()
        self._openai_client = None

    # -----------------------------------------------------------------------
    # Tier 1: Lexicon-based VADER Analysis
    # -----------------------------------------------------------------------

    def analyze_tier1_vader(self, text: str) -> SentimentScore:
        """
        Extract baseline compound, positive, neutral, and negative polarity scores.
        """
        if not text or not text.strip():
            return SentimentScore(
                label="neutral",
                positive_score=0.0,
                neutral_score=1.0,
                negative_score=0.0,
                compound_score=0.0,
            )

        scores = self.vader.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        pos = float(scores.get("pos", 0.0))
        neu = float(scores.get("neu", 0.0))
        neg = float(scores.get("neg", 0.0))

        # Standard VADER threshold classification
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScore(
            label=label,
            positive_score=round(pos, 4),
            neutral_score=round(neu, 4),
            negative_score=round(neg, 4),
            compound_score=round(compound, 4),
        )

    # -----------------------------------------------------------------------
    # Tier 2: Strict JSON-Schema LLM Aspect Extraction
    # -----------------------------------------------------------------------

    def _get_openai_client(self):
        if self._openai_client is None:
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            api_key = settings.OPENAI_API_KEY or settings.GEMINI_API_KEY
            base_url = None
            if settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            elif settings.OLLAMA_BASE_URL:
                base_url = settings.OLLAMA_BASE_URL
                api_key = api_key or "ollama"

            if not api_key and not settings.OLLAMA_BASE_URL:
                return None

            self._openai_client = OpenAI(
                api_key=api_key or "not-set",
                base_url=base_url,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
        return self._openai_client

    def analyze_tier2_llm(self, text: str) -> Optional[LLMAspectAnalysis]:
        """
        Extract aspect-level sentiments constrained strictly by Pydantic JSON schema.
        Returns None if LLM is disabled, unconfigured, or fails (triggers cascade to Tier 3).
        """
        if not settings.ENABLE_LLM_TIER:
            return None

        client = self._get_openai_client()
        if client is None:
            return None

        # Determine target model
        if settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
            model = settings.GEMINI_MODEL
        elif settings.OLLAMA_BASE_URL:
            model = settings.OLLAMA_MODEL
        else:
            model = settings.OPENAI_MODEL

        system_prompt = (
            "You are Sentilytics' expert Aspect-Based Sentiment Analysis (ABSA) system. "
            "Analyze the given text and extract all distinct aspects, their categories, sentiment polarities, "
            "confidence scores, and verbatim evidence snippets. Output must strictly conform to the JSON schema."
        )

        user_prompt = f"Text to analyze:\n\"\"\"{text}\"\"\""

        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=LLMAspectAnalysis,
                temperature=0.0,
            )
            parsed_result: LLMAspectAnalysis = completion.choices[0].message.parsed
            if parsed_result:
                return parsed_result
            return None
        except Exception as e:
            logger.warning(f"Tier 2 LLM execution failed ({type(e).__name__}: {e}). Cascading to Tier 3 fallback.")
            return None

    # -----------------------------------------------------------------------
    # Tier 3: spaCy Noun-Chunk & VADER Valence Fallback Chain
    # -----------------------------------------------------------------------

    def analyze_tier3_spacy_fallback(
        self,
        text: str,
        overall_sentiment: SentimentScore,
        include_aspects: bool = True,
        include_key_phrases: bool = True,
    ) -> Dict[str, Any]:
        """
        Fallback chain using lazy-loaded spaCy noun-chunk extraction and sentence VADER valence.
        """
        if not text or not text.strip():
            return {"aspects": [], "key_phrases": []}

        nlp = get_spacy_nlp()
        doc = nlp(text)

        aspects: List[AspectDetail] = []
        key_phrases: List[str] = []
        seen_aspects = set()

        if include_aspects:
            sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
            for sent in sentences:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                sent_scores = self.vader.polarity_scores(sent_text)
                sent_compound = sent_scores.get("compound", 0.0)

                # Extract noun chunks if syntax parser is available
                noun_chunks = []
                if hasattr(sent, "noun_chunks"):
                    try:
                        noun_chunks = list(sent.noun_chunks)
                    except Exception:
                        noun_chunks = []

                if not noun_chunks:
                    # Fallback noun / keyword detection if noun_chunks not available
                    noun_chunks = [token for token in sent if token.pos_ in ("NOUN", "PROPN")]

                for chunk in noun_chunks:
                    chunk_text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
                    clean_aspect = re.sub(r'^(the|a|an|this|that|my|our|their|its)\s+', '', chunk_text, flags=re.IGNORECASE).strip()

                    # Filter short / generic noise
                    if len(clean_aspect) < 3 or clean_aspect.lower() in seen_aspects:
                        continue
                    if clean_aspect.lower() in {"everything", "something", "anything", "nothing", "everyone", "someone", "it", "they", "them", "we", "i", "you"}:
                        continue

                    seen_aspects.add(clean_aspect.lower())

                    # Map to category
                    category = self._categorize_aspect(clean_aspect, sent_text)

                    # Determine aspect sentiment from local context & sentence valence
                    aspect_sentiment, confidence = self._compute_aspect_sentiment(
                        aspect_token=clean_aspect,
                        sentence_text=sent_text,
                        sentence_compound=sent_compound,
                        overall_label=overall_sentiment.label,
                    )

                    aspects.append(
                        AspectDetail(
                            aspect=clean_aspect,
                            category=category,
                            sentiment=aspect_sentiment,
                            confidence=round(confidence, 2),
                            evidence=sent_text[:200],
                        )
                    )

        # Extract Key Phrases
        if include_key_phrases:
            # Pick prominent noun chunks and strong sentiment adjectives
            phrases = []
            for token in doc:
                if token.pos_ in ("ADJ", "NOUN", "PROPN") and not token.is_stop and len(token.text) > 3:
                    if token.text.lower() not in [p.lower() for p in phrases]:
                        phrases.append(token.text)
                if len(phrases) >= 6:
                    break
            key_phrases = phrases[:5]

        return {
            "aspects": aspects,
            "key_phrases": key_phrases,
        }

    def _categorize_aspect(self, aspect: str, context: str) -> str:
        """
        Classify aspect into standard domain category based on keyword matches.
        """
        combined = f"{aspect} {context}".lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', combined):
                    return category
        return "General"

    def _compute_aspect_sentiment(
        self,
        aspect_token: str,
        sentence_text: str,
        sentence_compound: float,
        overall_label: str,
    ) -> tuple[str, float]:
        """
        Compute sentiment for an aspect within its sentence context.
        """
        # Score the sentence valence
        if sentence_compound >= 0.05:
            sentiment = "positive"
            confidence = min(0.95, 0.70 + abs(sentence_compound) * 0.25)
        elif sentence_compound <= -0.05:
            sentiment = "negative"
            confidence = min(0.95, 0.70 + abs(sentence_compound) * 0.25)
        else:
            # Fallback to overall label
            sentiment = overall_label
            confidence = 0.75

        return sentiment, confidence

    # -----------------------------------------------------------------------
    # Master Orchestrator: Hybrid Execution
    # -----------------------------------------------------------------------

    def analyze_raw(
        self,
        text: str,
        options: Optional[AnalyzeOptions] = None,
        force_tier: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate Tier 1 + Tier 2 (or Tier 3 fallback).
        Returns dictionary formatted for AnalyzeResponse.
        """
        start_time = time.perf_counter()

        include_aspects = options.include_aspects if options else True
        include_key_phrases = options.include_key_phrases if options else True

        # Tier 1: Baseline VADER Polarity (Always executed)
        tier1_sentiment = self.analyze_tier1_vader(text)

        aspects_list: List[Dict[str, Any]] = []
        key_phrases_list: List[str] = []
        tier_used = 1
        llm_used = False

        if include_aspects or include_key_phrases:
            # Check if forced to Tier 3 or Tier 1
            if force_tier == 3 or (force_tier is None and not settings.ENABLE_LLM_TIER):
                tier3_res = self.analyze_tier3_spacy_fallback(
                    text=text,
                    overall_sentiment=tier1_sentiment,
                    include_aspects=include_aspects,
                    include_key_phrases=include_key_phrases,
                )
                aspects_list = [a.model_dump() for a in tier3_res["aspects"]]
                key_phrases_list = tier3_res["key_phrases"]
                tier_used = 3
            elif force_tier == 1:
                # Lexicon baseline only
                tier_used = 1
            else:
                # Attempt Tier 2 Structured LLM
                llm_result = self.analyze_tier2_llm(text)
                if llm_result is not None:
                    aspects_list = [
                        AspectDetail(
                            aspect=item.aspect,
                            category=item.category,
                            sentiment=item.sentiment,
                            confidence=item.confidence,
                            evidence=item.evidence,
                        ).model_dump()
                        for item in llm_result.aspects
                    ] if include_aspects else []
                    key_phrases_list = llm_result.key_phrases if include_key_phrases else []
                    tier_used = 2
                    llm_used = True
                else:
                    # Cascade to Tier 3 spaCy fallback
                    tier3_res = self.analyze_tier3_spacy_fallback(
                        text=text,
                        overall_sentiment=tier1_sentiment,
                        include_aspects=include_aspects,
                        include_key_phrases=include_key_phrases,
                    )
                    aspects_list = [a.model_dump() for a in tier3_res["aspects"]]
                    key_phrases_list = tier3_res["key_phrases"]
                    tier_used = 3

        processing_time_ms = max(1, int((time.perf_counter() - start_time) * 1000))

        return {
            "overall_sentiment": tier1_sentiment.model_dump(),
            "aspects": aspects_list,
            "key_phrases": key_phrases_list,
            "metadata": {
                "model_version": f"sentilytics-absa-tier{tier_used}",
                "processing_time_ms": processing_time_ms,
                "llm_used": llm_used,
            },
        }

    def analyze(
        self,
        text: str,
        options: Optional[AnalyzeOptions] = None,
        analysis_id: Optional[str] = None,
    ) -> AnalyzeResponse:
        """
        Synchronously analyze text and return structured AnalyzeResponse schema.
        """
        raw_res = self.analyze_raw(text, options)
        aid = analysis_id or str(uuid.uuid4())

        return AnalyzeResponse(
            id=aid,
            overall_sentiment=SentimentScore(**raw_res["overall_sentiment"]),
            aspects=[AspectDetail(**a) for a in raw_res["aspects"]],
            key_phrases=raw_res["key_phrases"],
            metadata=AnalyzeMetadata(**raw_res["metadata"]),
        )


# Global Singleton
nlp_engine = NLPEngine()
