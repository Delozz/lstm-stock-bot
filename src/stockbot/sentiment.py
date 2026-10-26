"""VADER headline sentiment (FR-13)."""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score(headline: str) -> float:
    return _analyzer.polarity_scores(headline)["compound"]
