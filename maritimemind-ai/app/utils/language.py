"""
Language Detection Utility — Phase 15.1 Multilingual Support

Lightweight wrapper around `langdetect` for detecting the language
of text at ingestion time (chunk tagging) and query time (response language).

Returns ISO 639-1 codes (e.g., "en", "es", "ar", "fr", "de").
"""

from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.language")

# Human-readable names for common maritime languages
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "no": "Norwegian",
    "da": "Danish",
    "sv": "Swedish",
    "fi": "Finnish",
    "el": "Greek",
    "tr": "Turkish",
    "pl": "Polish",
    "hi": "Hindi",
    "tl": "Filipino",
}

_DEFAULT_LANGUAGE = "en"


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.

    Args:
        text: Input text (at least ~20 characters for reliable detection).

    Returns:
        ISO 639-1 language code (e.g., "en", "es", "ar").
        Falls back to "en" on detection failure or very short text.
    """
    if not text or len(text.strip()) < 15:
        return _DEFAULT_LANGUAGE

    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception as e:
        logger.debug(f"Language detection failed, defaulting to '{_DEFAULT_LANGUAGE}': {e}")
        return _DEFAULT_LANGUAGE


def get_language_name(code: str) -> str:
    """Return a human-readable language name for an ISO 639-1 code."""
    return LANGUAGE_NAMES.get(code, code.upper())
