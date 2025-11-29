from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger

from financial_rag.agents.multi_modal_analyst import MultiModalAnalystAgent
from financial_rag.analytics.forecasting import TimeSeriesAnalyzer, FinancialForecaster


class PredictiveAnalystAgent(MultiModalAnalystAgent):
    """Advanced agent with predictive analytics and forecasting capabilities"""

    def __init__(self, vector_store, enable_monitoring: bool = True):
        super().__init__(vector_store, enable_monitoring)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.forecaster = FinancialForecaster()
        self.prediction_cache = {}

    async def predictive_analysis(
        self, ticker: str, analysis_horizon: str = "30d"
    ) -> Dict[str, Any]:
        """Comprehensive predictive analysis combining multiple forecasts"""
        try:
            logger.info(
                f"Conducting predictive analysis for {ticker} over {analysis_horizon}"
            )

            # Parse horizon
            horizon_days = self.parse_analysis_horizon(analysis_horizon)

            # Execute multiple predictive analyses in parallel
            tasks = [
                self.forecast_stock_price(ticker, horizon_days),
                self.analyze_momentum(ticker),
                self.assess_technical_outlook(ticker),
                self.get_fundamental_forecast(ticker),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine predictions
            combined_forecast = self.combine_predictions(ticker, results, horizon_days)

            # Generate investment thesis
            investment_thesis = self.generate_investment_thesis(combined_forecast)

            return {
                "ticker": ticker,
                "analysis_horizon": analysis_horizon,
                "horizon_days": horizon_days,
                "price_forecast": combined_forecast["price_forecast"],
                "momentum_analysis": combined_forecast["momentum_analysis"],
                "technical_outlook": combined_forecast["technical_outlook"],
                "fundamental_forecast": combined_forecast["fundamental_forecast"],
                "composite_score": combined_forecast["composite_score"],
                "investment_thesis": investment_thesis,
                "key_risks": self.identify_prediction_risks(combined_forecast),
                "monitoring_recommendations": self.generate_monitoring_recommendations(
                    combined_forecast
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in predictive analysis for {ticker}: {e}")
            raise

    def parse_analysis_horizon(self, horizon: str) -> int:
        """Parse analysis horizon string to days"""
        horizon = horizon.lower().strip()

        if horizon.endswith("d"):
            return int(horizon[:-1])
        elif horizon.endswith("w"):
            return int(horizon[:-1]) * 7
        elif horizon.endswith("m"):
            return int(horizon[:-1]) * 30
        elif horizon.endswith("y"):
            return int(horizon[:-1]) * 365
        else:
            return 30  # Default to 30 days

    async def forecast_stock_price(
        self, ticker: str, horizon_days: int
    ) -> Dict[str, Any]:
        """Forecast stock price with advanced analytics"""
        try:
            # Use cached forecast if recent and available
            cache_key = f"{ticker}_price_{horizon_days}"
            if (
                cache_key in self.prediction_cache
                and datetime.now() - self.prediction_cache[cache_key]["timestamp"]
                < timedelta(hours=1)
            ):
                return self.prediction_cache[cache_key]["forecast"]

            forecast = await self.forecaster.forecast_stock_price(ticker, horizon_days)

            # Cache the forecast
            self.prediction_cache[cache_key] = {
                "forecast": forecast,
                "timestamp": datetime.now(),
            }

            return forecast

        except Exception as e:
            logger.error(f"Error in price forecasting for {ticker}: {e}")
            return {"error": str(e)}

    async def analyze_momentum(self, ticker: str) -> Dict[str, Any]:
        """Analyze price momentum and trend strength"""
        try:
            trend_analysis = await self.time_series_analyzer.analyze_stock_trends(
                ticker
            )

            momentum_score = self.calculate_momentum_score(trend_analysis)

            return {
                "trend_analysis": trend_analysis,
                "momentum_score": momentum_score,
                "momentum_strength": self.assess_momentum_strength(momentum_score),
                "trend_continuation_probability": self.calculate_trend_continuation_probability(
                    trend_analysis
                ),
                "key_momentum_indicators": self.extract_momentum_indicators(
                    trend_analysis
                ),
            }

        except Exception as e:
            logger.error(f"Error in momentum analysis for {ticker}: {e}")
            return {"error": str(e)}

    def calculate_momentum_score(self, trend_analysis: Dict) -> float:
        """Calculate composite momentum score"""
        score = 0.0
        factors = 0

        # Trend direction factor
        consensus = trend_analysis.get("consensus", {})
        if consensus.get("direction") == "upward":
            score += consensus.get("confidence", 0)
        elif consensus.get("direction") == "downward":
            score -= consensus.get("confidence", 0)
        factors += 1

        # Volatility factor (lower volatility is better for momentum)
        volatility = trend_analysis.get("volatility_analysis", {}).get(
            "current_volatility", 0
        )
        if volatility > 0:
            volatility_score = max(
                0, 1 - (volatility / 0.05)
            )  # Normalize to 5% volatility
            score += volatility_score
            factors += 1

        # Technical factor
        technicals = trend_analysis.get("technical_analysis", {})
        moving_averages = technicals.get("moving_averages", {})
        if moving_averages.get("golden_cross"):
            score += 0.3
            factors += 1

        return score / factors if factors > 0 else 0.0

    def assess_momentum_strength(self, momentum_score: float) -> str:
        """Assess momentum strength based on score"""
        if momentum_score > 0.3:
            return "strong_positive"
        elif momentum_score > 0.1:
            return "positive"
        elif momentum_score > -0.1:
            return "neutral"
        elif momentum_score > -0.3:
            return "negative"
        else:
            return "strong_negative"

    def calculate_trend_continuation_probability(self, trend_analysis: Dict) -> float:
        """Calculate probability of trend continuation"""
        # Simplified implementation
        consensus = trend_analysis.get("consensus", {})
        confidence = consensus.get("confidence", 0.5)
        volatility = trend_analysis.get("volatility_analysis", {}).get(
            "current_volatility", 0.02
        )

        # Higher confidence and lower volatility increase continuation probability
        continuation_prob = confidence * (1 - min(1, volatility / 0.05))

        return round(continuation_prob, 2)

    def extract_momentum_indicators(self, trend_analysis: Dict) -> List[str]:
        """Extract key momentum indicators"""
        indicators = []

        # Trend indicators
        consensus = trend_analysis.get("consensus", {})
        if consensus.get("agreement_level") == "high":
            indicators.append(f"Strong {consensus.get('direction')} trend consensus")

        # Volatility indicators
        volatility_regime = trend_analysis.get("volatility_analysis", {}).get(
            "volatility_regime"
        )
        if volatility_regime == "low_volatility":
            indicators.append("Low volatility environment supportive of trends")

        # Technical indicators
        technicals = trend_analysis.get("technical_analysis", {})
        if technicals.get("moving_averages", {}).get("golden_cross"):
            indicators.append("Golden cross technical pattern")

        return indicators

    async def assess_technical_outlook(self, ticker: str) -> Dict[str, Any]:
        """Assess technical analysis outlook"""
        try:
            trend_analysis = await self.time_series_analyzer.analyze_stock_trends(
                ticker
            )
            technicals = trend_analysis.get("technical_analysis", {})

            outlook_score = self.calculate_technical_score(technicals)

            return {
                "technical_analysis": technicals,
                "outlook_score": outlook_score,
                "technical_bias": self.determine_technical_bias(outlook_score),
                "key_levels": self.identify_key_technical_levels(technicals),
                "pattern_analysis": self.analyze_chart_patterns(technicals),
            }

        except Exception as e:
            logger.error(f"Error in technical outlook for {ticker}: {e}")
            return {"error": str(e)}

    def calculate_technical_score(self, technicals: Dict) -> float:
        """Calculate technical analysis score"""
        score = 0.0
        factors = 0

        # Moving average factors
        moving_averages = technicals.get("moving_averages", {})
        if moving_averages.get("price_vs_sma_20") == "above":
            score += 0.2
            factors += 1
        if moving_averages.get("price_vs_sma_50") == "above":
            score += 0.3
            factors += 1
        if moving_averages.get("golden_cross"):
            score += 0.5
            factors += 1

        # Support/resistance factors
        support_resistance = technicals.get("support_resistance", {})
        distance_to_resistance = support_resistance.get("distance_to_resistance", 0)
        distance_to_support = support_resistance.get("distance_to_support", 0)

        if distance_to_resistance > distance_to_support:
            score += 0.3  # More room to rise than fall
            factors += 1

        # Volume factors
        volume_analysis = technicals.get("volume_analysis", {})
        if volume_analysis.get("volume_confirmation"):
            score += 0.2
            factors += 1

        return score / factors if factors > 0 else 0.5

    def determine_technical_bias(self, technical_score: float) -> str:
        """Determine technical bias from score"""
        if technical_score > 0.7:
            return "strongly_bullish"
        elif technical_score > 0.6:
            return "bullish"
        elif technical_score > 0.4:
            return "neutral"
        elif technical_score > 0.3:
            return "bearish"
        else:
            return "strongly_bearish"

    def identify_key_technical_levels(self, technicals: Dict) -> Dict[str, float]:
        """Identify key technical levels"""
        support_resistance = technicals.get("support_resistance", {})

        return {
            "resistance": support_resistance.get("resistance_level"),
            "support": support_resistance.get("support_level"),
            "current_price": support_resistance.get("resistance_level", 0)
            - support_resistance.get("distance_to_resistance", 0)
            * (support_resistance.get("resistance_level", 1)),
        }

    def analyze_chart_patterns(self, technicals: Dict) -> List[str]:
        """Analyze chart patterns (simplified)"""
        patterns = []

        moving_averages = technicals.get("moving_averages", {})
        if moving_averages.get("golden_cross"):
            patterns.append("Golden Cross (Bullish)")

        # Add more pattern detection logic here
        patterns.append("Trend following established moving averages")

        return patterns

    async def get_fundamental_forecast(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental analysis forecast"""
        try:
            # Use multi-modal analysis for fundamental forecast
            comprehensive = await self.comprehensive_analysis(ticker)
            unified_insights = comprehensive.get("unified_insights", {})

            fundamental_score = self.calculate_fundamental_score(unified_insights)

            return {
                "fundamental_analysis": unified_insights,
                "fundamental_score": fundamental_score,
                "growth_outlook": self.assess_growth_outlook(unified_insights),
                "valuation_assessment": self.assess_valuation(unified_insights),
                "quality_metrics": self.extract_quality_metrics(unified_insights),
            }

        except Exception as e:
            logger.error(f"Error in fundamental forecast for {ticker}: {e}")
            return {"error": str(e)}

    def calculate_fundamental_score(self, unified_insights: Dict) -> float:
        """Calculate fundamental analysis score"""
        score = 0.0
        factors = 0

        # Investment rating factor
        rating = unified_insights.get("investment_rating", "hold")
        if rating == "buy":
            score += 0.8
        elif rating == "strong_buy":
            score += 1.0
        elif rating == "sell":
            score += 0.2
        elif rating == "strong_sell":
            score += 0.0
        else:
            score += 0.5
        factors += 1

        # Confidence factor
        confidence = unified_insights.get("confidence_score", 0.5)
        score *= confidence
        factors += 1

        # Strengths vs risks factor
        strengths = len(unified_insights.get("key_strengths", []))
        risks = len(unified_insights.get("key_risks", []))

        if strengths + risks > 0:
            strength_ratio = strengths / (strengths + risks)
            score += strength_ratio
            factors += 1

        return score / factors if factors > 0 else 0.5

    def assess_growth_outlook(self, unified_insights: Dict) -> str:
        """Assess growth outlook from fundamental analysis"""
        strengths = unified_insights.get("key_strengths", [])
        growth_indicators = [
            s
            for s in strengths
            if any(word in s.lower() for word in ["growth", "expanding", "increasing"])
        ]

        if len(growth_indicators) >= 2:
            return "strong_growth"
        elif len(growth_indicators) >= 1:
            return "moderate_growth"
        else:
            return "limited_growth"

    def assess_valuation(self, unified_insights: Dict) -> str:
        """Assess valuation from fundamental analysis"""
        # Simplified valuation assessment
        return "fair"  # In production, implement proper valuation analysis

    def extract_quality_metrics(self, unified_insights: Dict) -> List[str]:
        """Extract quality metrics from fundamental analysis"""
        metrics = []
        strengths = unified_insights.get("key_strengths", [])

        for strength in strengths[:3]:  # Top 3 strengths as quality indicators
            metrics.append(strength)

        return metrics

    def combine_predictions(
        self, ticker: str, results: List, horizon_days: int
    ) -> Dict[str, Any]:
        """Combine predictions from multiple analysis types"""
        # Extract successful results
        price_forecast = next(
            (
                r
                for r in results
                if isinstance(r, dict) and "point_forecast" in r.get("ensemble", {})
            ),
            {},
        )
        momentum_analysis = next(
            (r for r in results if isinstance(r, dict) and "momentum_score" in r), {}
        )
        technical_outlook = next(
            (r for r in results if isinstance(r, dict) and "technical_bias" in r), {}
        )
        fundamental_forecast = next(
            (r for r in results if isinstance(r, dict) and "fundamental_score" in r), {}
        )

        # Calculate composite score
        composite_score = self.calculate_composite_score(
            price_forecast, momentum_analysis, technical_outlook, fundamental_forecast
        )

        return {
            "price_forecast": price_forecast,
            "momentum_analysis": momentum_analysis,
            "technical_outlook": technical_outlook,
            "fundamental_forecast": fundamental_forecast,
            "composite_score": composite_score,
            "overall_bias": self.determine_overall_bias(composite_score),
            "confidence_level": self.calculate_confidence_level(
                price_forecast,
                momentum_analysis,
                technical_outlook,
                fundamental_forecast,
            ),
        }

    def calculate_composite_score(
        self,
        price_forecast: Dict,
        momentum_analysis: Dict,
        technical_outlook: Dict,
        fundamental_forecast: Dict,
    ) -> float:
        """Calculate composite predictive score"""
        scores = []
        weights = []

        # Price forecast score
        if price_forecast.get("ensemble"):
            forecast = price_forecast["ensemble"]["point_forecast"]
            # Normalize to -1 to 1 scale based on expected return
            # This is simplified - in production, use proper normalization
            scores.append(0.1)  # Placeholder
            weights.append(0.3)

        # Momentum score
        momentum_score = momentum_analysis.get("momentum_score", 0.5)
        scores.append(momentum_score)
        weights.append(0.25)

        # Technical score
        technical_score = technical_outlook.get("outlook_score", 0.5)
        scores.append(technical_score)
        weights.append(0.25)

        # Fundamental score
        fundamental_score = fundamental_forecast.get("fundamental_score", 0.5)
        scores.append(fundamental_score)
        weights.append(0.2)

        # Calculate weighted average
        if scores and weights:
            composite = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return composite
        else:
            return 0.5

    def determine_overall_bias(self, composite_score: float) -> str:
        """Determine overall predictive bias"""
        if composite_score > 0.7:
            return "strongly_bullish"
        elif composite_score > 0.6:
            return "bullish"
        elif composite_score > 0.4:
            return "neutral"
        elif composite_score > 0.3:
            return "bearish"
        else:
            return "strongly_bearish"

    def calculate_confidence_level(
        self,
        price_forecast: Dict,
        momentum_analysis: Dict,
        technical_outlook: Dict,
        fundamental_forecast: Dict,
    ) -> str:
        """Calculate overall confidence level"""
        confidence_factors = []

        # Price forecast confidence
        if price_forecast.get("ensemble", {}).get("confidence_interval", {}):
            ci_width = (
                price_forecast["ensemble"]["confidence_interval"]["upper"]
                - price_forecast["ensemble"]["confidence_interval"]["lower"]
            )
            if ci_width < 10:  # Narrow confidence interval
                confidence_factors.append("high")
            else:
                confidence_factors.append("medium")

        # Model agreement
        if (
            len(
                [
                    f
                    for f in [
                        price_forecast,
                        momentum_analysis,
                        technical_outlook,
                        fundamental_forecast,
                    ]
                    if f and not f.get("error")
                ]
            )
            >= 3
        ):
            confidence_factors.append("high")
        else:
            confidence_factors.append("medium")

        # Determine overall confidence
        if confidence_factors.count("high") >= 2:
            return "high"
        elif confidence_factors.count("medium") >= 2:
            return "medium"
        else:
            return "low"

    def generate_investment_thesis(self, combined_forecast: Dict) -> str:
        """Generate investment thesis based on combined forecasts"""
        overall_bias = combined_forecast.get("overall_bias", "neutral")
        confidence = combined_forecast.get("confidence_level", "medium")

        thesis_templates = {
            "strongly_bullish": {
                "high": "Strong bullish conviction with high confidence across multiple analytical frameworks.",
                "medium": "Bullish outlook supported by converging signals from technical, momentum, and fundamental analysis.",
                "low": "Cautiously optimistic view with some analytical support but limited confidence.",
            },
            "bullish": {
                "high": "Positive investment case with solid analytical foundation and reasonable confidence.",
                "medium": "Moderately bullish view with balanced risk-reward characteristics.",
                "low": "Tentative positive bias requiring confirmation from additional data points.",
            },
            "neutral": {
                "high": "Neutral stance with high conviction due to offsetting positive and negative factors.",
                "medium": "Balanced outlook with mixed signals from different analytical approaches.",
                "low": "Uncertain environment with insufficient clear directional signals.",
            },
            "bearish": {
                "high": "Defensive positioning recommended based on concerning signals across multiple frameworks.",
                "medium": "Cautious outlook with several risk factors outweighing potential opportunities.",
                "low": "Mildly negative view that warrants monitoring for deterioration.",
            },
            "strongly_bearish": {
                "high": "Strong risk-off recommendation with high-confidence bearish signals.",
                "medium": "Significant concerns across analytical dimensions supporting defensive stance.",
                "low": "Substantial risks identified but with some uncertainty about timing and magnitude.",
            },
        }

        template = thesis_templates.get(overall_bias, thesis_templates["neutral"])
        return template.get(
            confidence, "Insufficient data for clear investment thesis."
        )

    def identify_prediction_risks(
        self, combined_forecast: Dict
    ) -> List[Dict[str, str]]:
        """Identify risks to predictive accuracy"""
        risks = []

        confidence = combined_forecast.get("confidence_level", "medium")
        if confidence == "low":
            risks.append(
                {
                    "type": "low_confidence",
                    "description": "Low confidence in predictive signals across analytical frameworks",
                    "mitigation": "Wait for higher conviction signals or reduce position size",
                }
            )

        # Model disagreement risk
        components = [
            "price_forecast",
            "momentum_analysis",
            "technical_outlook",
            "fundamental_forecast",
        ]
        valid_components = sum(
            1
            for comp in components
            if combined_forecast.get(comp) and not combined_forecast[comp].get("error")
        )

        if valid_components < 3:
            risks.append(
                {
                    "type": "limited_analysis",
                    "description": f"Only {valid_components} analytical frameworks provided reliable signals",
                    "mitigation": "Seek additional data sources or analytical perspectives",
                }
            )

        return risks

    def generate_monitoring_recommendations(self, combined_forecast: Dict) -> List[str]:
        """Generate recommendations for ongoing monitoring"""
        recommendations = []

        overall_bias = combined_forecast.get("overall_bias", "neutral")
        confidence = combined_forecast.get("confidence_level", "medium")

        if confidence == "low":
            recommendations.append(
                "Monitor for strengthening signals before taking significant action"
            )

        if overall_bias in ["bullish", "strongly_bullish"]:
            recommendations.append(
                "Watch for technical breakdown below key support levels"
            )
            recommendations.append(
                "Monitor earnings reports for fundamental confirmation"
            )
        elif overall_bias in ["bearish", "strongly_bearish"]:
            recommendations.append(
                "Watch for technical breakout above key resistance levels"
            )
            recommendations.append("Monitor for positive fundamental catalyst changes")

        recommendations.append("Review analysis weekly for material changes in signals")

        return recommendations
