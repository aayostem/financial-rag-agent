# =============================================================================
# Financial RAG Agent — SEC EDGAR Ingestor
# src/financial_rag/ingestion/sec_ingestor.py
#
# Async SEC EDGAR client.
# Provides:
#   - Rate-limited filing downloads (EDGAR hard limit: 10 req/sec)
#   - SHA-256 deduplication via filings table
#   - Retry with exponential backoff on transient errors
#   - Structured logging on every operation
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date
from typing import TYPE_CHECKING

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from financial_rag.config import get_settings
from financial_rag.utils.exceptions import SECFetchError

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

# EDGAR base URLs
_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILINGS = f"{_EDGAR_BASE}/submissions"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

# Valid filing types for this system
SUPPORTED_FILING_TYPES = frozenset({"10-K", "10-Q", "8-K", "20-F", "DEF 14A", "S-1"})


# =============================================================================
# Rate limiter — EDGAR enforces 10 req/sec per IP
# =============================================================================


class _EdgarRateLimiter:
    """
    Token-bucket rate limiter for EDGAR API calls.
    Initialised once per SECIngestor instance.
    """

    def __init__(self, rps: int) -> None:
        self._semaphore = asyncio.Semaphore(rps)
        self._interval = 1.0 / rps

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *_: object) -> None:
        await asyncio.sleep(self._interval)
        self._semaphore.release()


# =============================================================================
# Filing metadata dataclass
# =============================================================================


class FilingMetadata:
    """
    Lightweight value object returned by list_filings().
    Passed to download_filing() to fetch actual content.
    """

    __slots__ = (
        "ticker",
        "filing_type",
        "fiscal_year",
        "fiscal_quarter",
        "filed_at",
        "accession_number",
        "primary_document",
        "source_url",
        "cik",
    )

    def __init__(
        self,
        *,
        ticker: str,
        filing_type: str,
        fiscal_year: int | None,
        fiscal_quarter: int | None,
        filed_at: date | None,
        accession_number: str,
        primary_document: str,
        source_url: str,
        cik: str,
    ) -> None:
        self.ticker = ticker
        self.filing_type = filing_type
        self.fiscal_year = fiscal_year
        self.fiscal_quarter = fiscal_quarter
        self.filed_at = filed_at
        self.accession_number = accession_number
        self.primary_document = primary_document
        self.source_url = source_url
        self.cik = cik

    def __repr__(self) -> str:
        return (
            f"<FilingMetadata {self.ticker} {self.filing_type} "
            f"FY{self.fiscal_year} filed={self.filed_at}>"
        )


# =============================================================================
# SECIngestor
# =============================================================================


class SECIngestor:
    """
    Async SEC EDGAR ingestor.

    Usage:
        async with SECIngestor() as ingestor:
            filings = await ingestor.list_filings("AAPL", "10-K", years=3)
            for meta in filings:
                raw_html = await ingestor.download_filing(meta)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._rate_limiter = _EdgarRateLimiter(self._settings.EDGAR_RATE_LIMIT_RPS)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SECIngestor:
        self._client = httpx.AsyncClient(
            headers={
                # EDGAR requires a descriptive User-Agent identifying the app
                # and a contact email. Requests without this may be blocked.
                "User-Agent": self._settings.EDGAR_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=self._settings.EDGAR_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_filings(
        self,
        ticker: str,
        filing_type: str,
        *,
        years: int = 3,
    ) -> list[FilingMetadata]:
        """
        Return metadata for the most recent N years of filings.

        Args:
            ticker:       Stock ticker symbol (e.g. 'AAPL')
            filing_type:  SEC form type (e.g. '10-K', '10-Q')
            years:        How many years of filings to fetch (max 5)

        Returns:
            List of FilingMetadata, newest first.

        Raises:
            SECFetchError on network or API errors.
        """
        if filing_type not in SUPPORTED_FILING_TYPES:
            raise SECFetchError(
                f"Unsupported filing type '{filing_type}'. "
                f"Supported: {sorted(SUPPORTED_FILING_TYPES)}"
            )

        years = min(years, self._settings.EDGAR_MAX_RETRIES * 2)  # cap at 5
        ticker = ticker.upper()

        logger.info("Listing %s filings for %s (years=%d)", filing_type, ticker, years)

        cik = await self._resolve_cik(ticker)
        submissions = await self._fetch_submissions(cik)
        filings = self._parse_submissions(
            submissions,
            ticker=ticker,
            filing_type=filing_type,
            cik=cik,
            years=years,
        )

        logger.info("Found %d %s filings for %s", len(filings), filing_type, ticker)
        return filings

    async def download_filing(
        self,
        meta: FilingMetadata,
        *,
        raw_dir: Path | None = None,
    ) -> tuple[str, str]:
        """
        Download the primary document for a filing.

        Args:
            meta:    FilingMetadata from list_filings()
            raw_dir: Optional directory to cache raw files.
                     If provided and file exists, returns cached content.

        Returns:
            (raw_html_content, file_hash) tuple.
            raw_html_content is the unprocessed HTML/text from EDGAR.
            file_hash is the SHA-256 of the content for deduplication.

        Raises:
            SECFetchError on download failure.
            DuplicateFilingError if this hash already exists (caller handles).
        """
        # Check local cache first
        if raw_dir:
            cached = self._check_cache(meta, raw_dir)
            if cached:
                content, file_hash = cached
                logger.info(
                    "Cache hit for %s %s FY%s",
                    meta.ticker,
                    meta.filing_type,
                    meta.fiscal_year,
                )
                return content, file_hash

        content = await self._fetch_document(meta.source_url)
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Persist to cache if requested
        if raw_dir:
            self._write_cache(meta, raw_dir, content)

        logger.info(
            "Downloaded %s %s FY%s — %d chars hash=%s",
            meta.ticker,
            meta.filing_type,
            meta.fiscal_year,
            len(content),
            file_hash[:12],
        )
        return content, file_hash

    async def check_duplicate(
        self,
        file_hash: str,
        ticker: str,
        filing_type: str,
    ) -> bool:
        """
        Return True if this file_hash already exists in the filings table.
        Pure utility — caller decides whether to skip or raise.
        """
        from financial_rag.storage.database import get_db_client
        from financial_rag.storage.repositories.filings import FilingsRepository

        client = await get_db_client()
        async with client.session() as session:
            repo = FilingsRepository(session)
            return await repo.exists_by_hash(file_hash)

    # ── EDGAR API calls ───────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _resolve_cik(self, ticker: str) -> str:
        """
        Resolve a ticker symbol to its SEC CIK number.
        EDGAR's company_tickers.json maps ticker → CIK.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            async with self._rate_limiter:
                response = await self._client.get(url)  # type: ignore[union-attr]
                response.raise_for_status()

            data = response.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.debug("Resolved %s → CIK %s", ticker, cik)
                    return cik

            raise SECFetchError(
                f"Ticker '{ticker}' not found in EDGAR company registry. "
                f"Verify the ticker is correct."
            )
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to resolve CIK for '{ticker}': {exc}") from exc

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_submissions(self, cik: str) -> dict:
        """Fetch the full submissions JSON for a CIK from EDGAR."""
        url = f"{_EDGAR_FILINGS}/CIK{cik}.json"
        try:
            async with self._rate_limiter:
                response = await self._client.get(url)  # type: ignore[union-attr]
                response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to fetch submissions for CIK {cik}: {exc}") from exc

    @retry(
        retry=retry_if_exception_type(SECFetchError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_document(self, url: str) -> str:
        """Fetch the raw content of a filing document."""
        try:
            async with self._rate_limiter:
                response = await self._client.get(  # type: ignore[union-attr]
                    url,
                    headers={"Host": "www.sec.gov"},
                )
                response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise SECFetchError(f"Failed to download filing from {url}: {exc}") from exc

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_submissions(
        self,
        data: dict,
        *,
        ticker: str,
        filing_type: str,
        cik: str,
        years: int,
    ) -> list[FilingMetadata]:
        """
        Parse EDGAR submissions JSON into FilingMetadata list.
        Filters to the requested filing_type and year limit.
        """
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found for %s", ticker)
            return []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        documents = recent.get("primaryDocument", [])

        results: list[FilingMetadata] = []
        seen_years: set[int] = set()

        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            if len(results) >= years:
                break

            try:
                filed_at = date.fromisoformat(filed_dates[i]) if filed_dates[i] else None
                fiscal_year = filed_at.year if filed_at else None
                accession = accessions[i].replace("-", "")
                primary_doc = documents[i] if i < len(documents) else ""

                source_url = f"{_EDGAR_ARCHIVES}/{int(cik)}" f"/{accession}/{primary_doc}"

                if fiscal_year and fiscal_year in seen_years:
                    continue
                if fiscal_year:
                    seen_years.add(fiscal_year)

                results.append(
                    FilingMetadata(
                        ticker=ticker,
                        filing_type=filing_type,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=None,  # set for 10-Q later
                        filed_at=filed_at,
                        accession_number=accession,
                        primary_document=primary_doc,
                        source_url=source_url,
                        cik=cik,
                    )
                )
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed filing entry at index %d: %s", i, exc)
                continue

        return results

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, meta: FilingMetadata, raw_dir: Path) -> Path:
        safe_ticker = meta.ticker.replace("/", "_")
        return raw_dir / safe_ticker / meta.filing_type / f"FY{meta.fiscal_year}.html"

    def _check_cache(self, meta: FilingMetadata, raw_dir: Path) -> tuple[str, str] | None:
        path = self._cache_path(meta, raw_dir)
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            return content, hashlib.sha256(content.encode()).hexdigest()
        return None

    def _write_cache(self, meta: FilingMetadata, raw_dir: Path, content: str) -> None:
        path = self._cache_path(meta, raw_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Cached filing to %s", path)
