"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share one instance instead of
each loading their own copy from disk.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_nlp_full = None
_nlp_lemma = None
_load_failed_full = False
_load_failed_lemma = False
_lock = threading.Lock()


def _ensure_model_available():
    """Verify the optional spaCy model without performing a network download.

    Model installation is an operator-managed deployment step.  A retrieval
    or indexing call must never turn a missing optional model into an
    unsolicited network request.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        )

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "spaCy model en_core_web_sm is unavailable; automatic model download "
            "is disabled. Install the approved model before enabling NLP features."
        )


def get_nlp_full():
    """Return spaCy model with all pipelines (NER, tagger, etc.) for entity extraction."""
    global _nlp_full, _load_failed_full
    if _load_failed_full:
        return None
    if _nlp_full is not None:
        return _nlp_full
    with _lock:
        if _nlp_full is not None:
            return _nlp_full
        if _load_failed_full:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_full = spacy.load("en_core_web_sm")
            logger.info("spaCy full model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy full model: {e}")
            _load_failed_full = True
            return None
    return _nlp_full


def get_nlp_lemma():
    """Return spaCy model with only lemmatizer for BM25 text processing."""
    global _nlp_lemma, _load_failed_lemma
    if _load_failed_lemma:
        return None
    if _nlp_lemma is not None:
        return _nlp_lemma
    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        if _load_failed_lemma:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_lemma = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.info("spaCy lemma model loaded")
        except Exception as e:
            logger.warning(f"Failed to load spaCy lemma model: {e}")
            _load_failed_lemma = True
            return None
    return _nlp_lemma
