# =============================================================================
# Financial RAG Agent — Text Parser
# src/financial_rag/ingestion/parsers/text_parser.py
#
# Cleans and normalises plain text extracted from SEC filings.
# Called after HTMLParser produces raw text — before chunking.
# =============================================================================

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Patterns that are noise in financial text
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(https?|ftp)://\S+", re.I),  # URLs
    re.compile(r"\S+@\S+\.\S+"),  # email addresses
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),  # control characters
    re.compile(r"\n{4,}"),  # 4+ consecutive newlines
]

# Financial number normalisation — preserve structure, clean formatting
_NUMBER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Remove dollar signs before numbers: $1,234 → 1234
    (re.compile(r"\$\s*([\d,]+(?:\.\d+)?)"), r"\1"),
    # Remove commas in numbers: 1,234,567 → 1234567
    (re.compile(r"(\d),(\d{3})"), r"\1\2"),
    # Normalise percentages: 12.5 % → 12.5%
    (re.compile(r"(\d)\s+%"), r"\1%"),
]


class TextParser:
    """
    Cleans and normalises plain text for downstream chunking and embedding.

    Does NOT chunk — that is the responsibility of TextProcessor.
    Does NOT embed — that is the responsibility of the embeddings module.

    Usage:
        parser = TextParser()
        clean = parser.clean(raw_text)
        normalised = parser.normalise_numbers(clean)
    """

    def clean(self, text: str) -> str:
        """
        Full cleaning pipeline.

        Steps:
          1. Unicode normalisation (NFKC)
          2. Remove noise patterns (URLs, emails, control chars)
          3. Normalise whitespace
          4. Strip leading/trailing whitespace per line
        """
        if not text or not text.strip():
            return ""

        # Step 1: Unicode normalisation
        # NFKC converts ligatures, normalises accented characters
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Remove noise
        for pattern in _NOISE_PATTERNS:
            text = pattern.sub(" ", text)

        # Step 3: Normalise whitespace within lines
        lines = []
        for line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)

        # Step 4: Collapse excessive blank lines (max 2 consecutive)
        cleaned_lines: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def normalise_numbers(self, text: str) -> str:
        """
        Normalise numeric formatting in financial text.

        Preserves the semantic value of numbers while removing
        formatting that fragments tokenisation (commas, currency symbols).

        Example:
            "$1,234,567 million" → "1234567 million"
            "12.5 %" → "12.5%"
        """
        for pattern, replacement in _NUMBER_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def extract_metrics(self, text: str) -> dict[str, float]:
        """
        Extract numerical financial metrics from text using regex patterns.

        Returns a dict of metric_name → value for downstream storage
        in the `metrics` JSONB column of financial_chunks.

        Patterns cover:
          - Revenue / net income / EPS figures
          - Percentage changes
          - Dollar amounts with scale qualifiers (million, billion)
        """
        metrics: dict[str, float] = {}

        # Revenue patterns: "revenue of $X billion/million"
        revenue_match = re.search(
            r"(?:revenue|net\s+revenue|total\s+revenue)[^\d]*"
            r"([\d,]+(?:\.\d+)?)\s*"
            r"(billion|million|thousand)?",
            text,
            re.I,
        )
        if revenue_match:
            value = float(revenue_match.group(1).replace(",", ""))
            scale = revenue_match.group(2) or ""
            metrics["revenue"] = _apply_scale(value, scale)

        # Net income
        income_match = re.search(
            r"net\s+(?:income|earnings|loss)[^\d]*"
            r"([\d,]+(?:\.\d+)?)\s*"
            r"(billion|million|thousand)?",
            text,
            re.I,
        )
        if income_match:
            value = float(income_match.group(1).replace(",", ""))
            scale = income_match.group(2) or ""
            metrics["net_income"] = _apply_scale(value, scale)

        # EPS: "earnings per share of $X.XX" or "EPS of $X.XX"
        eps_match = re.search(
            r"(?:earnings\s+per\s+(?:diluted\s+)?share|eps)[^\d]*"
            r"\$?([\d]+(?:\.\d+)?)",
            text,
            re.I,
        )
        if eps_match:
            metrics["eps"] = float(eps_match.group(1))

        # Operating margin: "operating margin of X%"
        margin_match = re.search(
            r"(?:operating|gross|net)\s+margin[^\d]*" r"([\d]+(?:\.\d+)?)\s*%",
            text,
            re.I,
        )
        if margin_match:
            metrics["margin_pct"] = float(margin_match.group(1))

        return metrics


# =============================================================================
# Helpers
# =============================================================================


def _apply_scale(value: float, scale: str) -> float:
    """Convert a scaled value to its full numeric form."""
    scale = scale.lower()
    if scale == "billion":
        return value * 1_000_000_000
    if scale == "million":
        return value * 1_000_000
    if scale == "thousand":
        return value * 1_000
    return value
