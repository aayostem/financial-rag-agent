import os
from sec_edgar_downloader import Downloader
from loguru import logger
from financial_rag.config import config  # Fixed import


class SECIngestor:
    def __init__(self):
        self.dl = Downloader(config.RAW_DATA_PATH)
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)

    def download_filings(self, ticker, filing_type="10-K", years=5):
        """Download SEC filings for a given ticker"""
        try:
            logger.info(f"Downloading {filing_type} filings for {ticker}")

            # Download filings
            num_filings = self.dl.get(
                filing_type, ticker, amount=years, download_details=True
            )

            logger.success(
                f"Successfully downloaded {num_filings} {filing_type} filings for {ticker}"
            )
            return num_filings

        except Exception as e:
            logger.error(f"Error downloading filings for {ticker}: {str(e)}")
            raise

    def get_filing_paths(self, ticker, filing_type="10-K"):
        """Get paths to downloaded filings"""
        ticker_path = os.path.join(
            config.RAW_DATA_PATH, "sec-edgar-filings", ticker, filing_type
        )
        if not os.path.exists(ticker_path):
            return []

        filing_paths = []
        for root, dirs, files in os.walk(ticker_path):
            for file in files:
                if file.endswith(".txt") or file.endswith(".html"):
                    filing_paths.append(os.path.join(root, file))

        return filing_paths
