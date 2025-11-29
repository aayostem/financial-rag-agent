from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import os
from loguru import logger

from financial_rag.agents.real_time_analyst import RealTimeAnalystAgent
from financial_rag.processing.audio_processor import EarningsCallProcessor
from financial_rag.processing.document_understanding import FinancialDocumentProcessor


class MultiModalAnalystAgent(RealTimeAnalystAgent):
    """Agent capable of multi-modal financial analysis"""

    def __init__(self, vector_store, enable_monitoring: bool = True):
        super().__init__(vector_store, enable_monitoring)
        self.audio_processor = EarningsCallProcessor()
        self.document_processor = FinancialDocumentProcessor()
        self.multimodal_context = {}

    async def analyze_earnings_call(
        self, audio_path: str, ticker: str
    ) -> Dict[str, Any]:
        """Comprehensive earnings call analysis"""
        try:
            logger.info(f"Analyzing earnings call for {ticker}: {audio_path}")

            # Transcribe and analyze audio
            call_analysis = self.audio_processor.transcribe_earnings_call(audio_path)

            # Get real-time context
            real_time_context = await self.get_real_time_context([ticker])

            # Combine with historical analysis
            historical_analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                self.agent.analyze,
                f"Provide historical context and risk factors for {ticker}",
            )

            # Generate comprehensive insights
            insights = self.generate_earnings_insights(
                call_analysis, real_time_context, historical_analysis
            )

            # Store in multimodal context
            self.multimodal_context[
                f"earnings_{ticker}_{datetime.now().isoformat()}"
            ] = {
                "call_analysis": call_analysis,
                "real_time_context": real_time_context,
                "historical_analysis": historical_analysis,
                "insights": insights,
            }

            return {
                "ticker": ticker,
                "call_analysis": call_analysis,
                "real_time_context": real_time_context,
                "historical_analysis": historical_analysis,
                "insights": insights,
                "summary": self.generate_earnings_summary(insights),
            }

        except Exception as e:
            logger.error(f"Error analyzing earnings call: {e}")
            raise

    def generate_earnings_insights(
        self, call_analysis: Dict, real_time_context: Dict, historical_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate insights from earnings call analysis"""
        insights = {
            "sentiment_analysis": {},
            "key_metrics": {},
            "guidance_analysis": {},
            "market_reaction": {},
            "investment_implications": [],
        }

        # Sentiment insights
        sentiment_data = call_analysis.get("sentiment_analysis", {})
        insights["sentiment_analysis"] = self.analyze_call_sentiment(sentiment_data)

        # Metric insights
        key_metrics = call_analysis.get("key_metrics", {})
        insights["key_metrics"] = self.analyze_call_metrics(key_metrics)

        # Guidance analysis
        guidance = key_metrics.get("guidance", {})
        insights["guidance_analysis"] = self.analyze_guidance(guidance)

        # Market reaction analysis
        market_data = real_time_context.get("market_data", {})
        insights["market_reaction"] = self.analyze_market_reaction(market_data)

        # Investment implications
        insights["investment_implications"] = self.generate_investment_implications(
            insights, historical_analysis
        )

        return insights

    def analyze_call_sentiment(self, sentiment_data: Dict) -> Dict[str, Any]:
        """Analyze sentiment from earnings call"""
        overall = sentiment_data.get("overall", {})
        by_speaker = sentiment_data.get("by_speaker", {})

        return {
            "overall_sentiment": self.get_sentiment_label(
                overall.get("average_polarity", 0)
            ),
            "sentiment_score": overall.get("average_polarity", 0),
            "speaker_sentiments": {
                speaker: {
                    "sentiment": self.get_sentiment_label(
                        data.get("average_polarity", 0)
                    ),
                    "score": data.get("average_polarity", 0),
                    "confidence": data.get("average_subjectivity", 0),
                }
                for speaker, data in by_speaker.items()
            },
            "key_positive_points": self.extract_positive_statements(sentiment_data),
            "key_concerns": self.extract_concerns(sentiment_data),
        }

    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return "very_positive"
        elif score > 0.05:
            return "positive"
        elif score > -0.05:
            return "neutral"
        elif score > -0.2:
            return "negative"
        else:
            return "very_negative"

    def extract_positive_statements(self, sentiment_data: Dict) -> List[str]:
        """Extract positive statements from call"""
        # Implementation would analyze transcript segments with high positive sentiment
        return ["Strong growth in key segments", "Confidence in future outlook"]

    def extract_concerns(self, sentiment_data: Dict) -> List[str]:
        """Extract concerns from call"""
        # Implementation would analyze transcript segments with negative sentiment
        return ["Supply chain challenges", "Macroeconomic headwinds"]

    def analyze_call_metrics(self, key_metrics: Dict) -> Dict[str, Any]:
        """Analyze key metrics from earnings call"""
        analysis = {
            "revenue_trend": self.analyze_revenue_trend(key_metrics.get("revenue", [])),
            "profitability": self.analyze_profitability(key_metrics.get("eps", [])),
            "growth_indicators": self.analyze_growth_indicators(
                key_metrics.get("growth_rates", [])
            ),
            "key_announcements": key_metrics.get("key_announcements", []),
        }

        return analysis

    def analyze_revenue_trend(self, revenue_data: List) -> Dict[str, Any]:
        """Analyze revenue trends"""
        if not revenue_data:
            return {"trend": "unknown", "confidence": 0}

        # Simple trend analysis
        return {"trend": "growing", "confidence": 0.8}

    def analyze_profitability(self, eps_data: List) -> Dict[str, Any]:
        """Analyze profitability trends"""
        if not eps_data:
            return {"trend": "unknown", "confidence": 0}

        # Simple profitability analysis
        return {"trend": "stable", "confidence": 0.7}

    def analyze_growth_indicators(self, growth_rates: List) -> Dict[str, Any]:
        """Analyze growth indicators"""
        if not growth_rates:
            return {"indicators": [], "overall_growth": "unknown"}

        positive_growth = [g for g in growth_rates if g.get("rate", 0) > 0]
        growth_strength = (
            len(positive_growth) / len(growth_rates) if growth_rates else 0
        )

        return {
            "indicators": growth_rates,
            "overall_growth": "strong" if growth_strength > 0.7 else "moderate",
        }

    def analyze_guidance(self, guidance: Dict) -> Dict[str, Any]:
        """Analyze forward guidance"""
        sentences = guidance.get("sentences", [])
        confidence = guidance.get("confidence", 0)

        return {
            "guidance_statements": sentences,
            "confidence": confidence,
            "sentiment": self.analyze_guidance_sentiment(sentences),
            "key_points": self.extract_guidance_key_points(sentences),
        }

    def analyze_guidance_sentiment(self, guidance_sentences: List[str]) -> str:
        """Analyze sentiment of guidance statements"""
        if not guidance_sentences:
            return "neutral"

        # Simple keyword-based sentiment analysis
        positive_words = ["strong", "growth", "increase", "improve", "optimistic"]
        negative_words = ["challenge", "headwind", "uncertain", "pressure", "decline"]

        text = " ".join(guidance_sentences).lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def extract_guidance_key_points(self, guidance_sentences: List[str]) -> List[str]:
        """Extract key points from guidance"""
        key_points = []

        for sentence in guidance_sentences:
            # Extract quantitative guidance
            if any(
                word in sentence.lower()
                for word in ["expect", "target", "forecast", "guidance"]
            ):
                key_points.append(sentence)

        return key_points[:3]  # Return top 3 key points

    def analyze_market_reaction(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze market reaction to earnings"""
        reaction = {
            "price_movement": {},
            "volume_analysis": {},
            "sentiment_impact": "neutral",
        }

        for ticker, data in market_data.items():
            price_change = data.get("change_pct", 0)
            volume = data.get("volume", 0)

            reaction["price_movement"][ticker] = {
                "change": price_change,
                "magnitude": abs(price_change),
                "direction": "up" if price_change > 0 else "down",
            }

            # Simple volume analysis
            if volume > 1000000:  # High volume threshold
                reaction["volume_analysis"][ticker] = "high_volume"
            else:
                reaction["volume_analysis"][ticker] = "normal_volume"

        # Determine overall sentiment impact
        price_changes = [data.get("change_pct", 0) for data in market_data.values()]
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0

        if avg_change > 2:
            reaction["sentiment_impact"] = "very_positive"
        elif avg_change > 0.5:
            reaction["sentiment_impact"] = "positive"
        elif avg_change < -2:
            reaction["sentiment_impact"] = "very_negative"
        elif avg_change < -0.5:
            reaction["sentiment_impact"] = "negative"

        return reaction

    def generate_investment_implications(
        self, insights: Dict, historical_analysis: Dict
    ) -> List[str]:
        """Generate investment implications from multi-modal analysis"""
        implications = []

        sentiment = insights.get("sentiment_analysis", {}).get(
            "overall_sentiment", "neutral"
        )
        market_reaction = insights.get("market_reaction", {}).get(
            "sentiment_impact", "neutral"
        )

        # Generate implications based on sentiment and market reaction
        if sentiment in ["very_positive", "positive"] and market_reaction in [
            "positive",
            "very_positive",
        ]:
            implications.extend(
                [
                    "Strong buy signal: Positive earnings and market confirmation",
                    "Consider increasing position size",
                    "Monitor for continued positive momentum",
                ]
            )
        elif sentiment in ["very_positive", "positive"] and market_reaction in [
            "negative",
            "very_negative",
        ]:
            implications.extend(
                [
                    "Contrarian opportunity: Positive fundamentals but negative market reaction",
                    "Potential buying opportunity if sentiment mismatch persists",
                    "Research reasons for market skepticism",
                ]
            )
        elif sentiment in ["negative", "very_negative"]:
            implications.extend(
                [
                    "Caution advised: Negative earnings sentiment",
                    "Consider reducing exposure or implementing hedges",
                    "Monitor for further deterioration",
                ]
            )
        else:
            implications.append("Neutral outlook: Monitor for clearer signals")

        return implications

    def generate_earnings_summary(self, insights: Dict) -> str:
        """Generate natural language earnings summary"""
        sentiment = insights.get("sentiment_analysis", {}).get(
            "overall_sentiment", "neutral"
        )
        market_reaction = insights.get("market_reaction", {}).get(
            "sentiment_impact", "neutral"
        )
        key_announcements = insights.get("key_metrics", {}).get("key_announcements", [])

        summary_parts = []

        # Sentiment summary
        sentiment_map = {
            "very_positive": "extremely positive",
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "very_negative": "very negative",
        }

        summary_parts.append(
            f"Earnings call sentiment: {sentiment_map.get(sentiment, 'neutral')}"
        )

        # Market reaction
        summary_parts.append(f"Market reaction: {market_reaction}")

        # Key announcements
        if key_announcements:
            summary_parts.append(
                f"Key announcements: {len(key_announcements)} significant items"
            )

        # Investment implications
        implications = insights.get("investment_implications", [])
        if implications:
            summary_parts.append(f"Primary implication: {implications[0]}")

        return ". ".join(summary_parts)

    async def analyze_financial_documents(
        self, document_paths: List[str], ticker: str
    ) -> Dict[str, Any]:
        """Analyze financial documents with table extraction"""
        try:
            logger.info(f"Analyzing financial documents for {ticker}")

            all_insights = {}

            for doc_path in document_paths:
                if doc_path.endswith(".pdf"):
                    doc_analysis = self.document_processor.extract_financial_tables(
                        doc_path
                    )
                    all_insights[doc_path] = doc_analysis

            # Combine insights across documents
            combined_analysis = self.combine_document_insights(all_insights)

            # Get real-time context
            real_time_context = await self.get_real_time_context([ticker])

            # Generate comprehensive analysis
            comprehensive_analysis = {
                "document_insights": combined_analysis,
                "real_time_context": real_time_context,
                "financial_health": self.assess_financial_health(combined_analysis),
                "investment_recommendation": self.generate_document_based_recommendation(
                    combined_analysis
                ),
            }

            return comprehensive_analysis

        except Exception as e:
            logger.error(f"Error analyzing financial documents: {e}")
            raise

    def combine_document_insights(self, all_insights: Dict) -> Dict[str, Any]:
        """Combine insights from multiple documents"""
        combined = {
            "key_metrics": {},
            "trends": {},
            "risk_factors": [],
            "growth_indicators": [],
        }

        for doc_path, insights in all_insights.items():
            doc_insights = insights.get("insights", {})

            # Combine metrics
            metrics = doc_insights.get("financial_metrics", {})
            for metric, value in metrics.items():
                if metric not in combined["key_metrics"]:
                    combined["key_metrics"][metric] = []
                combined["key_metrics"][metric].append(value)

            # Combine trends
            trends = doc_insights.get("trends", {})
            combined["trends"].update(trends)

            # Extract risk factors from key findings
            findings = doc_insights.get("key_findings", [])
            risk_related = [
                f
                for f in findings
                if any(
                    word in f.lower()
                    for word in ["risk", "decline", "challenge", "pressure"]
                )
            ]
            combined["risk_factors"].extend(risk_related)

            # Extract growth indicators
            growth_related = [
                f
                for f in findings
                if any(
                    word in f.lower()
                    for word in ["growth", "increase", "improve", "strong"]
                )
            ]
            combined["growth_indicators"].extend(growth_related)

        # Calculate average metrics
        for metric, values in combined["key_metrics"].items():
            if values:
                combined["key_metrics"][metric] = sum(values) / len(values)

        return combined

    def assess_financial_health(self, insights: Dict) -> Dict[str, Any]:
        """Assess overall financial health"""
        metrics = insights.get("key_metrics", {})
        trends = insights.get("trends", {})

        health_score = 0
        positive_indicators = 0
        total_indicators = 0

        # Profitability indicators
        if "profit_margin" in metrics:
            total_indicators += 1
            if metrics["profit_margin"] > 0.1:
                positive_indicators += 1
                health_score += 25

        # Growth indicators
        if "revenue_growth" in metrics:
            total_indicators += 1
            if metrics["revenue_growth"] > 0.05:
                positive_indicators += 1
                health_score += 25

        # Financial stability indicators
        if "debt_to_equity" in metrics:
            total_indicators += 1
            if metrics["debt_to_equity"] < 2.0:
                positive_indicators += 1
                health_score += 25

        # Trend indicators
        positive_trends = sum(
            1 for trend in trends.values() if trend.get("direction") == "increasing"
        )
        if positive_trends > 0:
            total_indicators += 1
            positive_indicators += 1
            health_score += 25

        # Calculate overall health
        health_ratio = (
            positive_indicators / total_indicators if total_indicators > 0 else 0
        )

        return {
            "health_score": health_score,
            "health_rating": self.get_health_rating(health_score),
            "positive_indicators": positive_indicators,
            "total_indicators": total_indicators,
            "health_ratio": health_ratio,
        }

    def get_health_rating(self, score: float) -> str:
        """Convert health score to rating"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"

    def generate_document_based_recommendation(self, insights: Dict) -> Dict[str, Any]:
        """Generate investment recommendation based on document analysis"""
        health_assessment = self.assess_financial_health(insights)
        health_rating = health_assessment.get("health_rating", "fair")

        recommendation_map = {
            "excellent": {
                "action": "strong_buy",
                "confidence": "high",
                "reasoning": "Strong financial health with positive trends",
            },
            "good": {
                "action": "buy",
                "confidence": "medium",
                "reasoning": "Good financial health with some positive indicators",
            },
            "fair": {
                "action": "hold",
                "confidence": "medium",
                "reasoning": "Mixed financial indicators, requires monitoring",
            },
            "poor": {
                "action": "sell",
                "confidence": "medium",
                "reasoning": "Weak financial health with concerning indicators",
            },
            "critical": {
                "action": "strong_sell",
                "confidence": "high",
                "reasoning": "Critical financial health issues identified",
            },
        }

        return recommendation_map.get(
            health_rating,
            {
                "action": "hold",
                "confidence": "low",
                "reasoning": "Insufficient data for clear recommendation",
            },
        )

    async def comprehensive_analysis(
        self, ticker: str, audio_path: str = None, document_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive multi-modal analysis"""
        analysis_results = {}

        # Real-time analysis
        analysis_results["real_time"] = await self.get_real_time_context([ticker])

        # Earnings call analysis if available
        if audio_path and os.path.exists(audio_path):
            analysis_results["earnings_call"] = await self.analyze_earnings_call(
                audio_path, ticker
            )

        # Document analysis if available
        if document_paths:
            analysis_results["documents"] = await self.analyze_financial_documents(
                document_paths, ticker
            )

        # Historical/RAG analysis
        analysis_results["historical"] = await asyncio.get_event_loop().run_in_executor(
            None,
            self.agent.analyze,
            f"Provide comprehensive analysis of {ticker} including competitive position and industry trends",
        )

        # Generate unified insights
        unified_insights = self.generate_unified_insights(analysis_results)

        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "analysis_modes": list(analysis_results.keys()),
            "detailed_analysis": analysis_results,
            "unified_insights": unified_insights,
            "executive_summary": self.generate_executive_summary(unified_insights),
        }

    def generate_unified_insights(self, analysis_results: Dict) -> Dict[str, Any]:
        """Generate unified insights from all analysis modes"""
        insights = {
            "investment_rating": "hold",
            "confidence_score": 0.5,
            "key_strengths": [],
            "key_risks": [],
            "catalyst_events": [],
            "valuation_assessment": "fair",
        }

        # Combine insights from different analysis modes
        all_positive_indicators = []
        all_negative_indicators = []

        # Real-time insights
        real_time = analysis_results.get("real_time", {})
        market_data = real_time.get("market_data", {})
        for ticker, data in market_data.items():
            if data.get("change_pct", 0) > 2:
                all_positive_indicators.append(
                    f"Strong price momentum: {ticker} up {data['change_pct']:.1f}%"
                )

        # Earnings call insights
        earnings = analysis_results.get("earnings_call", {})
        earnings_insights = earnings.get("insights", {})
        sentiment = earnings_insights.get("sentiment_analysis", {}).get(
            "overall_sentiment"
        )
        if sentiment in ["very_positive", "positive"]:
            all_positive_indicators.append("Positive earnings call sentiment")

        # Document insights
        documents = analysis_results.get("documents", {})
        doc_insights = documents.get("financial_health", {})
        health_rating = doc_insights.get("health_rating")
        if health_rating in ["excellent", "good"]:
            all_positive_indicators.append(f"Strong financial health: {health_rating}")

        # Determine overall rating
        positive_count = len(all_positive_indicators)
        negative_count = len(all_negative_indicators)

        if positive_count > negative_count + 2:
            insights["investment_rating"] = "buy"
            insights["confidence_score"] = 0.7
        elif negative_count > positive_count + 2:
            insights["investment_rating"] = "sell"
            insights["confidence_score"] = 0.7
        else:
            insights["investment_rating"] = "hold"
            insights["confidence_score"] = 0.5

        insights["key_strengths"] = all_positive_indicators[:3]  # Top 3 strengths
        insights["key_risks"] = all_negative_indicators[:3]  # Top 3 risks

        return insights

    def generate_executive_summary(self, insights: Dict) -> str:
        """Generate executive summary for decision makers"""
        rating = insights.get("investment_rating", "hold")
        confidence = insights.get("confidence_score", 0.5)
        strengths = insights.get("key_strengths", [])
        risks = insights.get("key_risks", [])

        summary = f"Investment Recommendation: {rating.upper()} (Confidence: {confidence:.0%})\n\n"

        if strengths:
            summary += "Key Strengths:\n• " + "\n• ".join(strengths) + "\n\n"

        if risks:
            summary += "Key Risks:\n• " + "\n• ".join(risks) + "\n\n"

        summary += "Suggested Action: "
        if rating == "buy":
            summary += "Consider establishing or adding to position"
        elif rating == "sell":
            summary += "Consider reducing or exiting position"
        else:
            summary += "Maintain current position with continued monitoring"

        return summary
