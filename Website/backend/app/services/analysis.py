import re
import csv
import io
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import async_session_maker
from app.schemas.analysis import (
    AnalyzeOptions,
    AnalyzeResponse,
    SentimentScore,
    AspectDetail,
    AnalyzeMetadata
)
from app.models.analysis import AnalysisJob, AnalysisResult
from app.models.log import SingleAnalysis

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = uuid.UUID("4a3b2c1d-5e6f-7a8b-9c0d-1e2f3a4b5c6d")

# Sentiment Lexicons
POSITIVE_WORDS = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "love": 2.0, "awesome": 1.8,
    "fantastic": 1.9, "amazing": 1.8, "happy": 1.2, "helpful": 1.3, "friendly": 1.2,
    "satisfied": 1.4, "satisfaction": 1.4, "recommend": 1.5, "best": 2.0, "perfect": 2.0,
    "easy": 1.2, "smooth": 1.3, "intuitive": 1.4, "fast": 1.3, "speedy": 1.3,
    "efficient": 1.4, "superb": 1.8, "wonderful": 1.8, "delight": 1.6, "delighted": 1.6,
    "glad": 1.2, "pleased": 1.3, "outstanding": 1.9, "exceptional": 2.0, "reliable": 1.5,
    "affordable": 1.3, "cheap": 1.0, "solid": 1.2, "clean": 1.1, "valuable": 1.4,
    "flawless": 2.0, "seamless": 1.5, "brilliant": 1.8, "impressive": 1.6
}

NEGATIVE_WORDS = {
    "bad": 1.0, "terrible": 2.0, "hate": 2.0, "worst": 2.2, "poor": 1.4,
    "unhappy": 1.4, "frustrated": 1.6, "frustrating": 1.6, "slow": 1.3, "bug": 1.3,
    "bugs": 1.3, "broken": 1.8, "fail": 1.6, "failure": 1.7, "failed": 1.6,
    "useless": 1.9, "difficult": 1.3, "hard": 1.1, "confusing": 1.5, "error": 1.3,
    "errors": 1.4, "issue": 1.1, "issues": 1.2, "crash": 1.9, "crashes": 1.9,
    "expensive": 1.4, "costly": 1.3, "waste": 1.7, "disappointed": 1.6, "annoyed": 1.4,
    "pain": 1.5, "suck": 1.8, "sucks": 1.8, "awful": 2.0, "horrible": 2.0,
    "defect": 1.5, "clunky": 1.4, "lag": 1.4, "laggy": 1.5, "overpriced": 1.6,
    "glitch": 1.3, "glitches": 1.4, "unresponsive": 1.7, "delayed": 1.2
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "hardly", "scarcely", "barely",
    "without", "don't", "dont", "didn't", "didnt", "doesn't", "doesnt",
    "won't", "wont", "wasn't", "wasnt", "isn't", "isnt", "cannot", "can't", "cant"
}

INTENSIFIERS = {
    "very": 1.4, "extremely": 1.8, "super": 1.5, "highly": 1.5, "really": 1.3,
    "absolutely": 1.7, "totally": 1.4, "completely": 1.5, "incredibly": 1.7,
    "exceptionally": 1.8, "deeply": 1.4
}

DAMPENERS = {
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "a bit": 0.7,
    "marginally": 0.6, "partially": 0.7
}

# Aspect Domain Definitions
ASPECT_CATEGORIES = {
    "pricing": {
        "category": "Billing & Price",
        "keywords": ["price", "cost", "expensive", "cheap", "billing", "money", "subscription", "pricing", "affordable", "fee", "invoice", "charge", "overpriced"],
        "pos_indicators": ["cheap", "affordable", "fair", "reasonable", "worth", "inexpensive"],
        "neg_indicators": ["expensive", "costly", "overpriced", "high", "hidden", "rip off", "waste"]
    },
    "performance": {
        "category": "Speed & Performance",
        "keywords": ["speed", "slow", "fast", "performance", "lag", "load", "responsive", "hang", "delay", "quick", "latency", "throughput"],
        "pos_indicators": ["fast", "quick", "speedy", "responsive", "instant", "snappy"],
        "neg_indicators": ["slow", "lag", "laggy", "delay", "hang", "latency", "sluggish"]
    },
    "usability": {
        "category": "UI & Usability",
        "keywords": ["ui", "ux", "design", "interface", "layout", "easy", "confusing", "hard", "intuitive", "clunky", "navigation", "simple"],
        "pos_indicators": ["easy", "intuitive", "simple", "clean", "modern", "beautiful", "smooth"],
        "neg_indicators": ["confusing", "hard", "ugly", "cluttered", "clunky", "complex", "difficult"]
    },
    "support": {
        "category": "Customer Support",
        "keywords": ["support", "help", "customer service", "agent", "ticket", "assistance", "representative", "rep"],
        "pos_indicators": ["helpful", "fast", "friendly", "polite", "quick", "responsive", "resolved"],
        "neg_indicators": ["poor", "slow", "bad", "unhelpful", "rude", "ignored", "unresponsive"]
    },
    "features": {
        "category": "Feature Set & Capabilities",
        "keywords": ["feature", "features", "capability", "capabilities", "integration", "export", "report", "dashboard", "tool", "customization"],
        "pos_indicators": ["powerful", "versatile", "comprehensive", "rich", "flexible", "useful"],
        "neg_indicators": ["missing", "limited", "lacking", "basic", "restricted", "insufficient"]
    },
    "reliability": {
        "category": "Stability & Reliability",
        "keywords": ["reliable", "reliability", "bug", "bugs", "error", "errors", "fail", "failure", "crash", "crashes", "stable", "uptime", "downtime", "broken"],
        "pos_indicators": ["reliable", "stable", "solid", "flawless", "dependable"],
        "neg_indicators": ["bug", "crash", "broken", "unstable", "frequent", "downtime", "defect"]
    }
}


class AnalysisService:
    """
    Core Analysis Service providing sentiment analysis, aspect categorization,
    key phrase extraction, and async batch job processing.
    """

    def analyze_raw(self, text: str, options: Optional[AnalyzeOptions] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        include_aspects = options.include_aspects if options else True
        include_key_phrases = options.include_key_phrases if options else True
        
        # Tokenize words preserving order for context parsing
        words = re.findall(r"\b[\w'-]+\b", text.lower())
        
        pos_weight = 0.0
        neg_weight = 0.0
        
        n_words = len(words)
        for i, word in enumerate(words):
            # Check context window (preceding 2 words) for negations or intensifiers
            multiplier = 1.0
            is_negated = False
            
            for back_idx in [i - 1, i - 2]:
                if back_idx >= 0:
                    prev_word = words[back_idx]
                    if prev_word in NEGATION_WORDS:
                        is_negated = True
                    if prev_word in INTENSIFIERS:
                        multiplier *= INTENSIFIERS[prev_word]
                    if prev_word in DAMPENERS:
                        multiplier *= DAMPENERS[prev_word]
            
            if word in POSITIVE_WORDS:
                score = POSITIVE_WORDS[word] * multiplier
                if is_negated:
                    neg_weight += score * 0.8
                else:
                    pos_weight += score
            elif word in NEGATIVE_WORDS:
                score = NEGATIVE_WORDS[word] * multiplier
                if is_negated:
                    pos_weight += score * 0.8
                else:
                    neg_weight += score

        # Compute normalized distribution scores
        total_weight = pos_weight + neg_weight
        if n_words == 0 or total_weight == 0:
            pos_score = 0.0
            neg_score = 0.0
            neu_score = 1.0
            compound = 0.0
        else:
            neutral_base = max(0.2, (n_words - (pos_weight + neg_weight)) / max(n_words, 1))
            total_sum = pos_weight + neg_weight + neutral_base
            pos_score = round(pos_weight / total_sum, 4)
            neg_score = round(neg_weight / total_sum, 4)
            neu_score = round(max(0.0, 1.0 - pos_score - neg_score), 4)
            
            # Compound normalization from -1.0 to 1.0
            diff = pos_weight - neg_weight
            compound = round(diff / (pos_weight + neg_weight + 1.0), 4)
            compound = max(-1.0, min(1.0, compound))
            
        if compound >= 0.05:
            overall_label = "positive"
        elif compound <= -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        # Aspect extraction
        aspects: List[Dict[str, Any]] = []
        if include_aspects:
            aspects = self._extract_aspects(text, overall_label)

        # Key phrases extraction
        key_phrases: List[str] = []
        if include_key_phrases:
            key_phrases = self._extract_key_phrases(text, words)

        processing_time_ms = max(1, int((time.perf_counter() - start_time) * 1000))

        return {
            "overall_sentiment": {
                "label": overall_label,
                "positive_score": float(pos_score),
                "neutral_score": float(neu_score),
                "negative_score": float(neg_score),
                "compound_score": float(compound)
            },
            "aspects": aspects,
            "key_phrases": key_phrases,
            "metadata": {
                "model_version": "sentilytics-engine-2.0",
                "processing_time_ms": processing_time_ms,
                "llm_used": False
            }
        }

    def _extract_aspects(self, text: str, fallback_sentiment: str) -> List[Dict[str, Any]]:
        aspects = []
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        lower_text = text.lower()

        for aspect_key, aspect_def in ASPECT_CATEGORIES.items():
            matched_keywords = [kw for kw in aspect_def["keywords"] if re.search(r'\b' + re.escape(kw) + r'\b', lower_text)]
            if not matched_keywords:
                continue

            # Find best evidence sentence
            evidence = next(
                (s for s in sentences if any(kw in s.lower() for kw in matched_keywords)),
                text
            )[:150]
            evidence_lower = evidence.lower()

            # Determine polarity for this specific aspect
            has_pos = any(w in evidence_lower for w in aspect_def["pos_indicators"])
            has_neg = any(w in evidence_lower for w in aspect_def["neg_indicators"])

            if has_neg and not has_pos:
                sentiment = "negative"
                confidence = 0.90
            elif has_pos and not has_neg:
                sentiment = "positive"
                confidence = 0.90
            else:
                sentiment = fallback_sentiment
                confidence = 0.80

            aspects.append({
                "aspect": aspect_key,
                "category": aspect_def["category"],
                "sentiment": sentiment,
                "confidence": confidence,
                "evidence": evidence
            })

        return aspects

    def _extract_key_phrases(self, text: str, words: List[str]) -> List[str]:
        phrases: List[str] = []
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]

        for sentence in sentences:
            sentence_words = re.findall(r"\b[\w'-]+\b", sentence)
            for idx, word in enumerate(sentence_words):
                w_lower = word.lower()
                if (w_lower in POSITIVE_WORDS or w_lower in NEGATIVE_WORDS or w_lower in INTENSIFIERS) and idx < len(sentence_words) - 1:
                    next_word = sentence_words[idx + 1]
                    if len(next_word) > 2 and next_word.lower() not in NEGATION_WORDS:
                        phrase = f"{word} {next_word}"
                        if phrase.lower() not in [p.lower() for p in phrases]:
                            phrases.append(phrase)

        if not phrases:
            # Informative word fallback
            for w in words:
                if len(w) > 4 and w not in POSITIVE_WORDS and w not in NEGATIVE_WORDS and w not in NEGATION_WORDS:
                    if w not in phrases:
                        phrases.append(w)
                    if len(phrases) >= 3:
                        break

        return phrases[:5]

    async def analyze_text(
        self,
        text: str,
        options: Optional[AnalyzeOptions] = None,
        user_id: Optional[uuid.UUID] = None,
        db: Optional[AsyncSession] = None
    ) -> AnalyzeResponse:
        """
        Synchronous single text analysis with optional persistence.
        """
        uid = user_id or DEFAULT_USER_ID
        raw_res = self.analyze_raw(text, options)

        analysis_id = str(uuid.uuid4())
        if db:
            single_analysis = SingleAnalysis(
                id=uuid.UUID(analysis_id),
                user_id=uid,
                raw_text=text,
                overall_sentiment=raw_res["overall_sentiment"]["label"],
                compound_score=raw_res["overall_sentiment"]["compound_score"],
                aspects=raw_res["aspects"]
            )
            db.add(single_analysis)
            await db.commit()
            await db.refresh(single_analysis)
            analysis_id = str(single_analysis.id)

        return AnalyzeResponse(
            id=analysis_id,
            overall_sentiment=SentimentScore(**raw_res["overall_sentiment"]),
            aspects=[AspectDetail(**a) for a in raw_res["aspects"]],
            key_phrases=raw_res["key_phrases"],
            metadata=AnalyzeMetadata(**raw_res["metadata"])
        )


    async def export_results_csv(self, job_id: uuid.UUID, db: AsyncSession) -> AsyncGenerator[str, None]:
        """
        Streamed CSV generator for job results.
        """
        header = [
            "row_index", "raw_text", "overall_sentiment", "compound_score",
            "positive_score", "neutral_score", "negative_score", "key_phrases", "aspects"
        ]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        yield output.getvalue()

        query = select(AnalysisResult).filter_by(job_id=job_id).order_by(AnalysisResult.row_index)
        results = await db.execute(query)

        for item in results.scalars().all():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                item.row_index,
                item.raw_text,
                item.overall_sentiment,
                float(item.compound_score) if item.compound_score is not None else 0.0,
                float(item.positive_score) if item.positive_score is not None else 0.0,
                float(item.neutral_score) if item.neutral_score is not None else 0.0,
                float(item.negative_score) if item.negative_score is not None else 0.0,
                json.dumps(item.key_phrases),
                json.dumps(item.aspects)
            ])
            yield output.getvalue()


# Global Singleton
analysis_service = AnalysisService()
