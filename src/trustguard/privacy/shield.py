"""PII detection and redaction middleware built on Microsoft Presidio.

Every user query passes through `PrivacyShield.mask` before it reaches the
vector store or the LLM. Detected values are replaced with typed placeholders
such as ``[ACCOUNT_NUMBER]`` and are never returned to callers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Presidio entity -> placeholder written into the masked text.
PLACEHOLDERS: dict[str, str] = {
    "PERSON": "[PERSON]",
    "US_SSN": "[SSN]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
    "US_BANK_NUMBER": "[ACCOUNT_NUMBER]",
    "IBAN_CODE": "[ACCOUNT_NUMBER]",
    "CREDIT_CARD": "[CARD_NUMBER]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE_NUMBER]",
    "IP_ADDRESS": "[IP_ADDRESS]",
}

# Entities we deliberately do not mask: locations, dates and nationalities are
# needed to answer policy questions ("international wire to Canada") and are
# not personal identifiers on their own.
DETECTED_ENTITIES: list[str] = sorted(PLACEHOLDERS)

_ACCOUNT_CONTEXT = ["account", "acct", "a/c", "routing", "member", "card", "iban", "number", "no"]


def _custom_recognizers() -> list[PatternRecognizer]:
    """Pattern recognizers that close gaps in Presidio's defaults.

    Presidio's built-in SSN and credit-card recognizers reject values that fail
    checksum or validity rules, which most mock data does. A privacy shield
    should fail closed, so these patterns match on shape alone.
    """
    ssn = PatternRecognizer(
        supported_entity="US_SSN",
        name="trustguard_ssn",
        patterns=[
            Pattern("ssn_dashed", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
            Pattern("ssn_spaced", r"\b\d{3} \d{2} \d{4}\b", 0.8),
        ],
        context=["ssn", "social", "security"],
    )
    card = PatternRecognizer(
        supported_entity="CREDIT_CARD",
        name="trustguard_card",
        patterns=[
            Pattern("card_grouped", r"\b(?:\d{4}[- ]){3}\d{4}\b", 0.85),
            Pattern("amex_grouped", r"\b\d{4}[- ]\d{6}[- ]\d{5}\b", 0.85),
        ],
        context=["card", "credit", "debit"],
    )
    account = PatternRecognizer(
        supported_entity="ACCOUNT_NUMBER",
        name="trustguard_account",
        patterns=[
            # Any standalone run of 8-17 digits. Amounts like "$10,000" do not
            # match because of the separator.
            Pattern("long_digit_run", r"\b\d{8,17}\b", 0.6),
            # Prefixed forms such as "acct #4421-88"; the identifier must contain a
            # digit so phrases like "account records" are not masked.
            Pattern(
                "prefixed_account",
                r"(?i)\b(?:acct|account)\s*(?:no\.?|number|#)?\s*[:#]?\s*(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{3,}\b",
                0.7,
            ),
        ],
        context=_ACCOUNT_CONTEXT,
    )
    return [ssn, card, account]


@dataclass(frozen=True)
class DetectedEntity:
    """Metadata about a redacted span. The original value is intentionally omitted."""

    entity_type: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Serialize for logs and API responses."""
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class MaskResult:
    """Output of `PrivacyShield.mask`."""

    masked_text: str
    entities: list[DetectedEntity] = field(default_factory=list)

    @property
    def pii_detected(self) -> bool:
        """True if at least one span was redacted."""
        return bool(self.entities)


class PrivacyShield:
    """Detects and anonymizes PII in free text."""

    def __init__(self, spacy_model: str = "en_core_web_md", score_threshold: float = 0.35) -> None:
        """Build Presidio engines.

        Args:
            spacy_model: spaCy pipeline used for named-entity recognition.
            score_threshold: Minimum Presidio confidence for a span to be masked.
        """
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        )
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["en"]
        )
        for recognizer in _custom_recognizers():
            self.analyzer.registry.add_recognizer(recognizer)
        self.anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]
        self.score_threshold = score_threshold
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder})
            for entity, placeholder in PLACEHOLDERS.items()
        }

    def analyze(self, text: str) -> list[RecognizerResult]:
        """Run detection only."""
        return self.analyzer.analyze(
            text=text,
            language="en",
            entities=DETECTED_ENTITIES,
            score_threshold=self.score_threshold,
        )

    def mask(self, text: str) -> MaskResult:
        """Replace detected PII with typed placeholders.

        Args:
            text: Raw user input.

        Returns:
            The masked text and metadata about each redacted span.
        """
        if not text.strip():
            return MaskResult(masked_text=text)
        results = self.analyze(text)
        if not results:
            return MaskResult(masked_text=text)
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,  # type: ignore[arg-type]
            operators=self._operators,
        )
        entities = [
            DetectedEntity(entity_type=r.entity_type, start=r.start, end=r.end, score=r.score)
            for r in sorted(results, key=lambda r: r.start)
        ]
        return MaskResult(masked_text=anonymized.text, entities=entities)


_shield: PrivacyShield | None = None
_lock = threading.Lock()


def get_shield(spacy_model: str = "en_core_web_md") -> PrivacyShield:
    """Return a process-wide shield; loading spaCy is slow, so do it once."""
    global _shield
    with _lock:
        if _shield is None:
            _shield = PrivacyShield(spacy_model=spacy_model)
        return _shield
