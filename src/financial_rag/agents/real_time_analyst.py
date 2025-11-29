from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
from loguru import logger

from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.data.real_time_sources import (
    RealTimeMarketData,
    AlternativeDataSources,
)
from financial_rag.monitoring.tracing import AgentMonitor


class RealTimeAnalystAgent(FinancialAgent):
    """Enhanced agent with real-time market intelligence"""

    def __init__(self, vector_store, enable_monitoring: bool = True):
        super().__init__(vector_store, enable_monitoring)
        self.real_time_data = RealTimeMarketData()
        self.alternative_data = AlternativeDataSources()
        self.market_alerts = MarketAlertSystem()

    async def analyze_with_market_context(
        self, question: str, tickers: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze with real-time market context"""
        start_time = datetime.now()

        try:
            logger.info(f"Real-time analysis for: {question}")

            # Extract tickers from question if not provided
            if not tickers:
                tickers = self.extract_tickers_from_question(question)

            # Get real-time data
            real_time_context = await self.get_real_time_context(tickers)

            # Enhance question with real-time context
            enhanced_question = self.enhance_question_with_context(
                question, real_time_context
            )

            # Use parent agent for analysis
            analysis_result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.analyze, enhanced_question
            )

            # Add real-time insights
            analysis_result["real_time_insights"] = self.generate_real_time_insights(
                analysis_result, real_time_context
            )

            # Check for market alerts
            analysis_result["alerts"] = await self.market_alerts.check_alerts(
                tickers, real_time_context
            )

            # Log monitoring
            if self.monitor.enabled:
                self.monitor.log_query_analysis(
                    question=question,
                    answer=analysis_result["answer"],
                    total_latency=(datetime.now() - start_time).total_seconds(),
                    source_count=len(analysis_result.get("source_documents", [])),
                    agent_type="real_time_analyst",
                )

            return analysis_result

        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            # Fallback to standard analysis
            return await super().analyze(question)

    def extract_tickers_from_question(self, question: str) -> List[str]:
        """Extract stock tickers from natural language question"""
        # Simple extraction - in production, use more sophisticated NLP
        import re

        # Common ticker patterns
        ticker_pattern = r"\b[A-Z]{1,5}\b"
        potential_tickers = re.findall(ticker_pattern, question.upper())

        # Filter likely tickers (exclude common words)
        common_words = {
            "THE",
            "AND",
            "FOR",
            "WHAT",
            "HOW",
            "WHY",
            "WILL",
            "THIS",
            "THAT",
        }
        tickers = [
            ticker
            for ticker in potential_tickers
            if ticker not in common_words and len(ticker) >= 2
        ]

        # Add major indices if discussing markets generally
        market_terms = ["market", "markets", "stock market", "dow", "nasdaq", "s&p"]
        if any(term in question.lower() for term in market_terms):
            tickers.extend(["^GSPC", "^DJI", "^IXIC"])

        return list(set(tickers))  # Remove duplicates

    async def get_real_time_context(self, tickers: List[str]) -> Dict[str, Any]:
        """Get comprehensive real-time context"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "market_data": {},
            "sentiment_analysis": {},
            "market_summary": {},
        }

        if tickers:
            # Get real-time market data
            context["market_data"] = await self.real_time_data.get_live_market_data(
                tickers
            )

            # Get sentiment analysis
            for ticker in tickers:
                news = self.real_time_data.get_real_time_news(ticker)
                social_data = await self.alternative_data.get_social_sentiment(ticker)
                sentiment = self.alternative_data.sentiment_analyzer.analyze_comprehensive_sentiment(
                    ticker, news, social_data
                )
                context["sentiment_analysis"][ticker] = sentiment

        # Get overall market summary
        context["market_summary"] = await self.real_time_data.get_market_summary()

        return context

    def enhance_question_with_context(
        self, question: str, context: Dict[str, Any]
    ) -> str:
        """Enhance the question with real-time context"""

        enhanced_prompt = f"""
Original Question: {question}

Real-Time Market Context (as of {context['timestamp']}):

"""

        # Add market data if available
        if context["market_data"]:
            enhanced_prompt += "Current Market Prices:\n"
            for ticker, data in context["market_data"].items():
                enhanced_prompt += (
                    f"- {ticker}: ${data['price']} ({data['change_pct']:+.2f}%)\n"
                )
            enhanced_prompt += "\n"

        # Add market summary
        if context["market_summary"]:
            enhanced_prompt += f"Market Summary: {context['market_summary'].get('market_summary', 'N/A')}\n"
            enhanced_prompt += f"Overall Sentiment: {context['market_summary'].get('market_sentiment', 'N/A')}\n\n"

        # Add sentiment analysis
        if context["sentiment_analysis"]:
            enhanced_prompt += "Sentiment Analysis:\n"
            for ticker, sentiment in context["sentiment_analysis"].items():
                enhanced_prompt += f"- {ticker}: {sentiment['sentiment_label']} (confidence: {sentiment['confidence']:.2f})\n"
            enhanced_prompt += "\n"

        enhanced_prompt += f"Please provide analysis considering this real-time context for: {question}"

        return enhanced_prompt

    def generate_real_time_insights(
        self, analysis_result: Dict, context: Dict
    ) -> List[str]:
        """Generate real-time insights based on analysis and market data"""
        insights = []

        # Price movement insights
        for ticker, data in context.get("market_data", {}).items():
            if abs(data["change_pct"]) > 2.0:  # Significant movement
                direction = "up" if data["change_pct"] > 0 else "down"
                insights.append(
                    f"🚨 {ticker} is trading {direction} {abs(data['change_pct']):.1f}% "
                    f"at ${data['price']} with high volume"
                )

        # Sentiment insights
        for ticker, sentiment in context.get("sentiment_analysis", {}).items():
            if sentiment["sentiment_label"] in ["strongly_bullish", "strongly_bearish"]:
                insights.append(
                    f"📊 {ticker} shows {sentiment['sentiment_label']} sentiment "
                    f"with {sentiment['confidence']:.1%} confidence"
                )

        # Market condition insights
        market_sentiment = context.get("market_summary", {}).get("market_sentiment")
        if market_sentiment:
            insights.append(f"🌍 Overall market sentiment is {market_sentiment}")

        return insights

    async def stream_market_updates(self, tickers: List[str], callback):
        """Stream real-time market updates"""
        while True:
            try:
                context = await self.get_real_time_context(tickers)
                await callback(context)
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in market updates stream: {e}")
                await asyncio.sleep(60)  # Wait longer on error


class MarketAlertSystem:
    """System for generating market alerts"""

    def __init__(self):
        self.alert_rules = self.initialize_alert_rules()

    def initialize_alert_rules(self) -> List[Dict]:
        """Initialize alert rules"""
        return [
            {
                "name": "large_price_move",
                "condition": lambda data: abs(data["change_pct"]) > 3.0,
                "message": lambda ticker, data: f"Large price movement: {ticker} {data['change_pct']:+.1f}%",
            },
            {
                "name": "high_volume",
                "condition": lambda data: data.get("volume", 0) > 1000000,
                "message": lambda ticker, data: f"High volume: {ticker} with {data['volume']:,} shares",
            },
            {
                "name": "extreme_sentiment",
                "condition": lambda data: data.get("sentiment", {}).get(
                    "sentiment_label"
                )
                in ["strongly_bullish", "strongly_bearish"],
                "message": lambda ticker, data: f"Extreme sentiment: {ticker} is {data['sentiment']['sentiment_label']}",
            },
        ]

    async def check_alerts(self, tickers: List[str], context: Dict) -> List[Dict]:
        """Check for market alerts"""
        alerts = []

        for ticker in tickers:
            market_data = context.get("market_data", {}).get(ticker, {})
            sentiment_data = context.get("sentiment_analysis", {}).get(ticker, {})

            data = {**market_data, "sentiment": sentiment_data}

            for rule in self.alert_rules:
                try:
                    if rule["condition"](data):
                        alert = {
                            "ticker": ticker,
                            "type": rule["name"],
                            "message": rule["message"](ticker, data),
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        }
                        alerts.append(alert)
                        logger.info(f"Alert triggered: {alert['message']}")
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule['name']}: {e}")

        return alerts
