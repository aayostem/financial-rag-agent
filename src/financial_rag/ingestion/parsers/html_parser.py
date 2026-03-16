# =============================================================================
# Financial RAG Agent — HTML Parser
# src/financial_rag/ingestion/parsers/html_parser.py
#
# Parses raw SEC filing HTML into clean section-tagged text.
# SEC filings are messy — SGML headers, XBRL inline tags, exhibit noise.
# This parser strips all of that and returns structured, clean text.
# =============================================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# =============================================================================
# Section detection — maps heading patterns to canonical section names
# Order matters: first match wins
# =============================================================================

_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"management.{0,20}discussion.{0,20}analysis", re.I), "MD&A"),
    (re.compile(r"risk\s+factors", re.I), "Risk Factors"),
    (
        re.compile(r"quantitative.{0,20}qualitative.{0,20}market\s+risk", re.I),
        "Market Risk",
    ),
    (re.compile(r"business\s+overview|item\s+1[.\s]+business", re.I), "Business"),
    (re.compile(r"financial\s+statements", re.I), "Financial Statements"),
    (re.compile(r"balance\s+sheet|financial\s+position", re.I), "Balance Sheet"),
    (
        re.compile(r"income\s+statement|results\s+of\s+operations", re.I),
        "Income Statement",
    ),
    (re.compile(r"cash\s+flow", re.I), "Cash Flow"),
    (re.compile(r"legal\s+proceedings", re.I), "Legal Proceedings"),
    (re.compile(r"properties", re.I), "Properties"),
    (re.compile(r"selected\s+financial\s+data", re.I), "Selected Financial Data"),
    (re.compile(r"notes\s+to\s+(the\s+)?financial", re.I), "Notes to Financials"),
]

# Tags that never contain useful text
_DISCARD_TAGS = frozenset(
    {
        "script",
        "style",
        "meta",
        "link",
        "head",
        "noscript",
        "svg",
        "img",
        "figure",
        "ix:nonfraction",
        "ix:nonnumeric",  # XBRL inline tags
        "xbrl",
        "xbrli",
    }
)

# Maximum consecutive blank lines to preserve
_MAX_BLANK_LINES = 2


@dataclass
class ParsedSection:
    """A single identified section from a filing."""

    name: str
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def __repr__(self) -> str:
        return f"<ParsedSection '{self.name}' chars={self.char_count}>"


@dataclass
class ParsedFiling:
    """
    Result of parsing a raw SEC filing HTML document.
    Contains both the full cleaned text and per-section breakdown.
    """

    ticker: str
    filing_type: str
    fiscal_year: int | None
    full_text: str
    sections: list[ParsedSection]
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.full_text)

    def get_section(self, name: str) -> str | None:
        """Return text for a named section, or None if not found."""
        for s in self.sections:
            if s.name == name:
                return s.text
        return None

    def __repr__(self) -> str:
        section_names = [s.name for s in self.sections]
        return (
            f"<ParsedFiling {self.ticker} {self.filing_type} "
            f"FY{self.fiscal_year} chars={self.char_count} "
            f"sections={section_names}>"
        )


# =============================================================================
# HTMLParser
# =============================================================================


class HTMLParser:
    """
    Parses raw SEC filing HTML into clean, section-tagged text.

    Usage:
        parser = HTMLParser()
        parsed = parser.parse(raw_html, ticker="AAPL",
                              filing_type="10-K", fiscal_year=2023)
    """

    def parse(
        self,
        raw_html: str,
        *,
        ticker: str,
        filing_type: str,
        fiscal_year: int | None = None,
    ) -> ParsedFiling:
        """
        Parse raw HTML content into a structured ParsedFiling.

        Args:
            raw_html:    Raw HTML/text content from EDGAR
            ticker:      Company ticker (for metadata)
            filing_type: SEC form type (for metadata)
            fiscal_year: Fiscal year (for metadata)

        Returns:
            ParsedFiling with full_text and sections populated.
        """
        logger.debug(
            "Parsing %s %s FY%s (%d chars)",
            ticker,
            filing_type,
            fiscal_year,
            len(raw_html),
        )

        # Step 1: Strip SGML header (SEC filings start with SGML metadata)
        html_content = self._strip_sgml_header(raw_html)

        # Step 2: Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        # Step 3: Remove noise tags entirely
        self._remove_noise_tags(soup)

        # Step 4: Extract clean text
        full_text = self._extract_text(soup)

        # Step 5: Detect sections
        sections = self._detect_sections(full_text)

        logger.debug(
            "Parsed %s %s — %d chars, %d sections detected",
            ticker,
            filing_type,
            len(full_text),
            len(sections),
        )

        return ParsedFiling(
            ticker=ticker,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            full_text=full_text,
            sections=sections,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _strip_sgml_header(self, content: str) -> str:
        """
        SEC filings begin with an SGML header block before the HTML.
        Strip everything before the first <html> or <HTML> tag.
        """
        match = re.search(r"<html", content, re.I)
        if match:
            return content[match.start() :]
        # No HTML tag found — treat as plain text
        return content

    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        """Remove tags that never contain useful content."""
        for tag_name in _DISCARD_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove empty tags
        for tag in soup.find_all(True):
            if isinstance(tag, Tag) and not tag.get_text(strip=True):
                tag.decompose()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from the parsed soup.
        Preserves paragraph breaks, normalises whitespace.
        """
        # Use get_text with a separator to preserve line structure
        raw_text = soup.get_text(separator="\n")

        # Normalise: collapse runs of spaces (not newlines)
        lines = []
        for line in raw_text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)

        # Collapse excessive blank lines
        cleaned_lines: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= _MAX_BLANK_LINES:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        # Remove SEC boilerplate patterns
        text = self._remove_boilerplate(text)

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common SEC filing boilerplate that adds noise."""
        patterns = [
            # Page numbers: "- 42 -" or "42"
            r"\n\s*-\s*\d+\s*-\s*\n",
            # Table of contents markers
            r"\.{5,}\s*\d+",
            # EDGAR filing header fields
            r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?FORM\s+\S+",
            # Exhibit separators
            r"={10,}",
            r"-{10,}",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "\n", text, flags=re.S)

        return text.strip()

    def _detect_sections(self, text: str) -> list[ParsedSection]:
        """
        Identify major sections within the filing text.

        Strategy:
          - Split text into lines
          - For each line, check if it matches a known section heading
          - Group all text between headings under that section name
          - Sections shorter than 200 chars are discarded as false positives
        """
        lines = text.splitlines()

        # Find heading positions
        heading_positions: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 200:
                # Section headings are short
                continue
            for pattern, section_name in _SECTION_PATTERNS:
                if pattern.search(stripped):
                    heading_positions.append((i, section_name))
                    break

        if not heading_positions:
            # No sections detected — return full text as "General"
            return [ParsedSection(name="General", text=text)]

        # Build sections from heading positions
        sections: list[ParsedSection] = []
        seen_sections: set[str] = set()

        for idx, (line_idx, section_name) in enumerate(heading_positions):
            # Determine end of this section
            if idx + 1 < len(heading_positions):
                end_idx = heading_positions[idx + 1][0]
            else:
                end_idx = len(lines)

            section_text = "\n".join(lines[line_idx:end_idx]).strip()

            # Skip duplicates and very short sections
            if section_name in seen_sections:
                continue
            if len(section_text) < 200:
                continue

            seen_sections.add(section_name)
            sections.append(ParsedSection(name=section_name, text=section_text))

        return sections if sections else [ParsedSection(name="General", text=text)]
