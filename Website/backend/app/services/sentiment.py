import re
import time
from typing import Dict, List, Any

# Simple rule-based/dictionary-based sentiment analyzer
POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "fantastic", "amazing", "happy", 
    "helpful", "friendly", "satisfied", "satisfaction", "recommend", "best", "perfect",
    "easy", "smooth", "intuitive", "fast", "speedy", "efficient", "superb", "wonderful",
    "delight", "delighted", "glad", "pleased", "outstanding", "exceptional"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "hate", "worst", "poor", "unhappy", "frustrated", "slow", 
    "bug", "broken", "fail", "failure", "useless", "difficult", "hard", "confusing",
    "error", "issue", "crash", "expensive", "costly", "waste", "disappointed", "annoyed",
    "pain", "useless", "suck", "sucks", "awful", "horrible", "defect"
}

def analyze_sentiment_local(text: str, include_aspects: bool = True, include_key_phrases: bool = True) -> Dict[str, Any]:
    start_time = time.perf_counter()
    
    # Lowercase and clean words
    words = re.findall(r'\b\w+\b', text.lower())
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total = len(words)
    if total == 0:
        pos_score = 0.0
        neg_score = 0.0
        neu_score = 1.0
        compound = 0.0
    else:
        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = (total - pos_count - neg_count) / total
        
        # Simple compound formula scaled between -1 and 1
        diff = pos_count - neg_count
        if diff == 0:
            compound = 0.0
        else:
            compound = diff / (pos_count + neg_count + 2.0)
            
    if compound > 0.05:
        overall_sentiment = "positive"
    elif compound < -0.05:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
        
    # Extracted aspects
    aspects = []
    if include_aspects:
        # Detect pricing aspect
        if any(w in text.lower() for w in ["price", "cost", "expensive", "cheap", "billing", "money", "subscription", "pricing"]):
            sentiment = "negative" if any(w in text.lower() for w in ["expensive", "costly", "high", "overpriced"]) else ("positive" if any(w in text.lower() for w in ["cheap", "fair", "affordable"]) else overall_sentiment)
            aspects.append({
                "aspect": "pricing",
                "category": "Billing & Price",
                "sentiment": sentiment,
                "confidence": 0.85,
                "evidence": next((line.strip() for line in text.split('.') if any(w in line.lower() for w in ["price", "cost", "expensive", "cheap", "billing", "pricing"])), text)[:150]
            })
        # Detect performance aspect
        if any(w in text.lower() for w in ["speed", "slow", "fast", "performance", "lag", "load", "responsive", "hang"]):
            sentiment = "negative" if any(w in text.lower() for w in ["slow", "lag", "hang", "delay"]) else "positive"
            aspects.append({
                "aspect": "performance",
                "category": "Speed & Performance",
                "sentiment": sentiment,
                "confidence": 0.90,
                "evidence": next((line.strip() for line in text.split('.') if any(w in line.lower() for w in ["speed", "slow", "fast", "performance", "lag", "responsive"])), text)[:150]
            })
        # Detect usability/UI aspect
        if any(w in text.lower() for w in ["ui", "ux", "design", "interface", "layout", "easy", "confusing", "hard", "intuitive"]):
            sentiment = "negative" if any(w in text.lower() for w in ["confusing", "hard", "ugly", "cluttered"]) else "positive"
            aspects.append({
                "aspect": "usability",
                "category": "UI & Usability",
                "sentiment": sentiment,
                "confidence": 0.80,
                "evidence": next((line.strip() for line in text.split('.') if any(w in line.lower() for w in ["ui", "ux", "design", "interface", "easy", "confusing", "hard", "intuitive"])), text)[:150]
            })
        # Detect customer support aspect
        if any(w in text.lower() for w in ["support", "help", "customer service", "agent", "ticket", "assistance"]):
            sentiment = "negative" if any(w in text.lower() for w in ["poor", "slow", "bad", "unhelpful"]) else "positive"
            aspects.append({
                "aspect": "support",
                "category": "Customer Support",
                "sentiment": sentiment,
                "confidence": 0.88,
                "evidence": next((line.strip() for line in text.split('.') if any(w in line.lower() for w in ["support", "help", "customer service", "assistance"])), text)[:150]
            })
            
    # Key phrase extraction
    key_phrases = []
    if include_key_phrases:
        sentences = text.split('.')
        for sentence in sentences:
            sentence_words = re.findall(r'\b\w+\b', sentence.strip())
            for idx, word in enumerate(sentence_words):
                if idx < len(sentence_words) - 1 and word.lower() in POSITIVE_WORDS.union(NEGATIVE_WORDS):
                    phrase = f"{word} {sentence_words[idx+1]}"
                    if len(phrase) > 5 and phrase not in key_phrases:
                        key_phrases.append(phrase)
        if not key_phrases:
            key_phrases = [w for w in words if len(w) > 5 and w not in POSITIVE_WORDS and w not in NEGATIVE_WORDS][:3]

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    
    return {
        "overall_sentiment": {
            "label": overall_sentiment,
            "positive_score": float(pos_score),
            "neutral_score": float(neu_score),
            "negative_score": float(neg_score),
            "compound_score": float(compound)
        },
        "aspects": aspects,
        "key_phrases": key_phrases,
        "metadata": {
            "model_version": "sentilytics-rule-1.0",
            "processing_time_ms": max(1, processing_time_ms),
            "llm_used": False
        }
    }
