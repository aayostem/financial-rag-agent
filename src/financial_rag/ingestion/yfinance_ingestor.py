import yfinance as yf
import pandas as pd
import os
import json
from loguru import logger
from financial_rag.config import config  # Fixed import


class YFinanceIngestor:
    def __init__(self):
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)

    def download_stock_data(self, ticker, period="1y"):
        """Download stock price data and company info"""
        try:
            logger.info(f"Downloading data for {ticker}")

            # Get stock data
            stock = yf.Ticker(ticker)

            # Historical prices
            hist = stock.history(period=period)

            # Company info
            info = stock.info

            # Save data
            data = {
                "ticker": ticker,
                "historical_prices": hist.to_dict(),
                "company_info": info,
            }

            file_path = os.path.join(config.RAW_DATA_PATH, f"{ticker}_yfinance.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.success(f"Successfully downloaded data for {ticker}")
            return data

        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            raise

    def get_financial_news(self, ticker, num_articles=10):
        """Get recent news for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:num_articles]
            return news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
