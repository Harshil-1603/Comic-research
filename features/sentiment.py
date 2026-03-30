"""
Emotion labelling using j-hartmann/emotion-english-distilroberta-base.
Outputs 7 emotions directly — no heuristic mapping needed.

Emotions: anger, disgust, fear, joy, neutral, sadness, surprise
(exactly matches config.LABEL_MAP)
"""
from transformers import pipeline

_emotion_pipeline = None


def _get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,          # return only the top label
        )
    return _emotion_pipeline


def get_emotion(text: str) -> str:
    """
    Predict emotion from text using a dedicated emotion model.

    Args:
        text: dialogue / caption extracted from a comic panel

    Returns:
        One of: anger, disgust, fear, joy, neutral, sadness, surprise
    """
    if not text or not text.strip():
        return "neutral"

    try:
        result = _get_pipeline()(text[:512])
        # pipeline with top_k=1 returns [[{'label': ..., 'score': ...}]]
        label = result[0][0]["label"].lower()
        return label
    except Exception:
        return "neutral"
