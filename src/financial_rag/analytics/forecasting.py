import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf


class TimeSeriesAnalyzer:
    """Advanced time series analysis for financial data"""

    def __init__(self):
        self.models = {}
        self.scalers = {}

    async def analyze_stock_trends(
        self, ticker: str, period: str = "2y"
    ) -> Dict[str, Any]:
        """Comprehensive stock trend analysis"""
        try:
            logger.info(f"Analyzing trends for {ticker}")

            # Get historical data
            stock_data = await self.get_historical_data(ticker, period)

            if stock_data.empty:
                return {"error": f"No data available for {ticker}"}

            # Perform multiple analyses
            trend_analysis = self.analyze_price_trends(stock_data)
            volatility_analysis = self.analyze_volatility(stock_data)
            seasonality_analysis = self.analyze_seasonality(stock_data)
            technical_analysis = self.perform_technical_analysis(stock_data)

            # Generate insights
            insights = self.generate_trend_insights(
                trend_analysis,
                volatility_analysis,
                seasonality_analysis,
                technical_analysis,
            )

            return {
                "ticker": ticker,
                "analysis_period": period,
                "trend_analysis": trend_analysis,
                "volatility_analysis": volatility_analysis,
                "seasonality_analysis": seasonality_analysis,
                "technical_analysis": technical_analysis,
                "insights": insights,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing trends for {ticker}: {e}")
            raise

    async def get_historical_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Get historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return pd.DataFrame()

            # Calculate additional features
            hist["Daily_Return"] = hist["Close"].pct_change()
            hist["Volume_SMA"] = hist["Volume"].rolling(window=20).mean()
            hist["Price_SMA_20"] = hist["Close"].rolling(window=20).mean()
            hist["Price_SMA_50"] = hist["Close"].rolling(window=50).mean()
            hist["Volatility"] = hist["Daily_Return"].rolling(window=20).std()

            return hist.dropna()

        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return pd.DataFrame()

    def analyze_price_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends and patterns"""
        trends = {}

        # Short-term trend (20 days)
        if len(data) >= 20:
            short_trend = self.calculate_trend_direction(data["Close"].tail(20))
            trends["short_term"] = {
                "direction": short_trend,
                "strength": self.calculate_trend_strength(data["Close"].tail(20)),
                "duration_days": 20,
            }

        # Medium-term trend (50 days)
        if len(data) >= 50:
            medium_trend = self.calculate_trend_direction(data["Close"].tail(50))
            trends["medium_term"] = {
                "direction": medium_trend,
                "strength": self.calculate_trend_strength(data["Close"].tail(50)),
                "duration_days": 50,
            }

        # Long-term trend (200 days)
        if len(data) >= 200:
            long_trend = self.calculate_trend_direction(data["Close"].tail(200))
            trends["long_term"] = {
                "direction": long_trend,
                "strength": self.calculate_trend_strength(data["Close"].tail(200)),
                "duration_days": 200,
            }

        # Overall trend consensus
        trends["consensus"] = self.determine_trend_consensus(trends)

        return trends

    def calculate_trend_direction(self, prices: pd.Series) -> str:
        """Calculate trend direction using linear regression"""
        if len(prices) < 2:
            return "neutral"

        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values

        model = LinearRegression()
        model.fit(x, y)

        slope = model.coef_[0]

        if slope > 0.001:
            return "upward"
        elif slope < -0.001:
            return "downward"
        else:
            return "neutral"

    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using R-squared"""
        if len(prices) < 2:
            return 0.0

        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values

        model = LinearRegression()
        model.fit(x, y)

        return model.score(x, y)  # R-squared

    def determine_trend_consensus(self, trends: Dict) -> Dict[str, Any]:
        """Determine overall trend consensus"""
        directions = [
            trend["direction"] for trend in trends.values() if isinstance(trend, dict)
        ]

        if not directions:
            return {"direction": "neutral", "confidence": 0.0}

        upward_count = directions.count("upward")
        downward_count = directions.count("downward")
        neutral_count = directions.count("neutral")

        total = len(directions)

        if upward_count > downward_count and upward_count > neutral_count:
            direction = "upward"
            confidence = upward_count / total
        elif downward_count > upward_count and downward_count > neutral_count:
            direction = "downward"
            confidence = downward_count / total
        else:
            direction = "neutral"
            confidence = neutral_count / total

        return {
            "direction": direction,
            "confidence": confidence,
            "agreement_level": (
                "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
            ),
        }

    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price volatility"""
        returns = data["Daily_Return"].dropna()

        if len(returns) == 0:
            return {}

        volatility_metrics = {
            "current_volatility": returns.tail(20).std(),
            "historical_volatility": returns.std(),
            "volatility_trend": self.analyze_volatility_trend(returns),
            "volatility_regime": self.determine_volatility_regime(returns),
            "max_drawdown": self.calculate_max_drawdown(data["Close"]),
        }

        return volatility_metrics

    def analyze_volatility_trend(self, returns: pd.Series) -> str:
        """Analyze whether volatility is increasing or decreasing"""
        if len(returns) < 40:
            return "stable"

        recent_vol = returns.tail(20).std()
        historical_vol = returns.head(len(returns) - 20).std()

        if recent_vol > historical_vol * 1.2:
            return "increasing"
        elif recent_vol < historical_vol * 0.8:
            return "decreasing"
        else:
            return "stable"

    def determine_volatility_regime(self, returns: pd.Series) -> str:
        """Determine current volatility regime"""
        current_vol = returns.tail(20).std()
        historical_vol = returns.std()

        if current_vol > historical_vol * 1.5:
            return "high_volatility"
        elif current_vol < historical_vol * 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        try:
            # Analyze monthly seasonality
            data["Month"] = data.index.month
            monthly_returns = data.groupby("Month")["Daily_Return"].mean()

            # Analyze day-of-week seasonality
            data["DayOfWeek"] = data.index.dayofweek
            dow_returns = data.groupby("DayOfWeek")["Daily_Return"].mean()

            return {
                "monthly_patterns": monthly_returns.to_dict(),
                "weekly_patterns": dow_returns.to_dict(),
                "strongest_seasonal_month": (
                    monthly_returns.idxmax() if not monthly_returns.empty else None
                ),
                "seasonal_strength": (
                    monthly_returns.std() if not monthly_returns.empty else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {}

    def perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis"""
        technicals = {}

        # Moving averages
        if len(data) >= 50:
            sma_20 = data["Price_SMA_20"].iloc[-1]
            sma_50 = data["Price_SMA_50"].iloc[-1]
            current_price = data["Close"].iloc[-1]

            technicals["moving_averages"] = {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "price_vs_sma_20": "above" if current_price > sma_20 else "below",
                "price_vs_sma_50": "above" if current_price > sma_50 else "below",
                "golden_cross": sma_20 > sma_50
                and data["Price_SMA_20"].iloc[-2] <= data["Price_SMA_50"].iloc[-2],
            }

        # Support and resistance levels
        technicals["support_resistance"] = self.identify_support_resistance(data)

        # Volume analysis
        technicals["volume_analysis"] = self.analyze_volume_patterns(data)

        return technicals

    def identify_support_resistance(
        self, data: pd.DataFrame, window: int = 20
    ) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        if len(data) < window:
            return {}

        recent_data = data.tail(window)

        resistance = recent_data["High"].max()
        support = recent_data["Low"].min()
        current_price = data["Close"].iloc[-1]

        return {
            "resistance_level": resistance,
            "support_level": support,
            "distance_to_resistance": (resistance - current_price) / current_price,
            "distance_to_support": (current_price - support) / current_price,
            "trading_range": resistance - support,
        }

    def analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading volume patterns"""
        if len(data) < 20:
            return {}

        current_volume = data["Volume"].iloc[-1]
        avg_volume = data["Volume_SMA"].iloc[-1]

        return {
            "volume_trend": (
                "high"
                if current_volume > avg_volume * 1.2
                else "low" if current_volume < avg_volume * 0.8 else "normal"
            ),
            "volume_ratio": current_volume / avg_volume,
            "volume_confirmation": self.check_volume_confirmation(data),
        }

    def check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Check if volume confirms price movement"""
        if len(data) < 2:
            return False

        price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
        volume_ratio = data["Volume"].iloc[-1] / data["Volume_SMA"].iloc[-1]

        # Volume should confirm price movement
        if price_change > 0 and volume_ratio > 1.0:
            return True
        elif price_change < 0 and volume_ratio > 1.0:
            return True
        else:
            return False

    def generate_trend_insights(
        self,
        trend_analysis: Dict,
        volatility_analysis: Dict,
        seasonality_analysis: Dict,
        technical_analysis: Dict,
    ) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []

        # Trend insights
        trend_consensus = trend_analysis.get("consensus", {})
        if (
            trend_consensus.get("direction") == "upward"
            and trend_consensus.get("confidence", 0) > 0.7
        ):
            insights.append("Strong upward trend with high confidence")
        elif (
            trend_consensus.get("direction") == "downward"
            and trend_consensus.get("confidence", 0) > 0.7
        ):
            insights.append("Strong downward trend with high confidence")

        # Volatility insights
        volatility_regime = volatility_analysis.get("volatility_regime")
        if volatility_regime == "high_volatility":
            insights.append(
                "High volatility regime - consider risk management strategies"
            )
        elif volatility_regime == "low_volatility":
            insights.append("Low volatility regime - potential for breakout")

        # Technical insights
        moving_averages = technical_analysis.get("moving_averages", {})
        if moving_averages.get("golden_cross"):
            insights.append("Golden cross detected - bullish technical signal")

        # Seasonal insights
        seasonal_strength = seasonality_analysis.get("seasonal_strength", 0)
        if seasonal_strength > 0.02:  # 2% average monthly return difference
            insights.append("Significant seasonal patterns detected")

        return insights


class FinancialForecaster:
    """AI-powered financial forecasting"""

    def __init__(self):
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.model_registry = {}

    async def forecast_stock_price(
        self, ticker: str, horizon_days: int = 30, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Forecast stock price with confidence intervals"""
        try:
            logger.info(f"Forecasting {ticker} for {horizon_days} days")

            # Get historical data
            historical_data = await self.time_series_analyzer.get_historical_data(
                ticker, "2y"
            )

            if historical_data.empty:
                return {"error": f"Insufficient data for {ticker}"}

            # Prepare features for forecasting
            features, target = self.prepare_forecasting_features(historical_data)

            if len(features) < 30:
                return {"error": "Insufficient data for reliable forecasting"}

            # Train multiple models for ensemble forecasting
            forecasts = await self.ensemble_forecast(
                features, target, horizon_days, confidence_level
            )

            # Generate forecast insights
            insights = self.generate_forecast_insights(forecasts, historical_data)

            return {
                "ticker": ticker,
                "forecast_horizon_days": horizon_days,
                "confidence_level": confidence_level,
                "point_forecast": forecasts["ensemble"]["point_forecast"],
                "confidence_interval": forecasts["ensemble"]["confidence_interval"],
                "model_performance": forecasts["model_performance"],
                "forecast_insights": insights,
                "key_assumptions": self.get_forecast_assumptions(),
                "risk_factors": self.identify_forecast_risks(
                    forecasts, historical_data
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error forecasting {ticker}: {e}")
            raise

    def prepare_forecasting_features(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for forecasting model"""
        features = data[
            [
                "Close",
                "Volume",
                "Daily_Return",
                "Volatility",
                "Price_SMA_20",
                "Price_SMA_50",
            ]
        ].copy()

        # Create lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f"Close_Lag_{lag}"] = features["Close"].shift(lag)
            features[f"Volume_Lag_{lag}"] = features["Volume"].shift(lag)
            features[f"Return_Lag_{lag}"] = features["Daily_Return"].shift(lag)

        # Create rolling statistics
        features["Rolling_Mean_5"] = features["Close"].rolling(5).mean()
        features["Rolling_Std_5"] = features["Close"].rolling(5).std()
        features["Rolling_Mean_10"] = features["Close"].rolling(10).mean()
        features["Rolling_Std_10"] = features["Close"].rolling(10).std()

        # Target variable (future price)
        target = features["Close"].shift(-1)  # Next day's price

        # Drop NaN values
        valid_data = features.dropna()
        target = target.loc[valid_data.index]

        return valid_data, target

    async def ensemble_forecast(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        horizon: int,
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Perform ensemble forecasting using multiple models"""
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "linear_regression": LinearRegression(),
        }

        forecasts = {}
        model_performance = {}

        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate performance metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                model_performance[model_name] = {
                    "mae": mae,
                    "rmse": rmse,
                    "r_squared": model.score(X_test, y_test),
                }

                # Store model for future use
                self.model_registry[model_name] = model

                # Generate forecast (using most recent data)
                latest_features = features.iloc[-1:].copy()
                point_forecast = model.predict(latest_features)[0]

                # Simple confidence interval (in production, use proper methods)
                confidence_width = rmse * 2  # Approximate 95% CI

                forecasts[model_name] = {
                    "point_forecast": point_forecast,
                    "confidence_interval": {
                        "lower": point_forecast - confidence_width,
                        "upper": point_forecast + confidence_width,
                    },
                }

            except Exception as e:
                logger.error(f"Error with {model_name} forecasting: {e}")
                continue

        # Ensemble forecast (weighted average)
        if forecasts:
            ensemble_forecast = self.create_ensemble_forecast(
                forecasts, model_performance
            )
            forecasts["ensemble"] = ensemble_forecast

        forecasts["model_performance"] = model_performance

        return forecasts

    def create_ensemble_forecast(
        self, individual_forecasts: Dict, model_performance: Dict
    ) -> Dict[str, Any]:
        """Create ensemble forecast from individual model forecasts"""
        # Weight forecasts by model performance (inverse of RMSE)
        weights = {}
        total_weight = 0

        for model_name, performance in model_performance.items():
            if model_name in individual_forecasts and performance["rmse"] > 0:
                weight = 1 / performance["rmse"]
                weights[model_name] = weight
                total_weight += weight

        if total_weight == 0:
            # Equal weighting if no performance data
            for model_name in individual_forecasts:
                weights[model_name] = 1
            total_weight = len(individual_forecasts)

        # Calculate weighted average forecast
        weighted_forecast = 0
        for model_name, weight in weights.items():
            forecast = individual_forecasts[model_name]["point_forecast"]
            weighted_forecast += (weight / total_weight) * forecast

        # Calculate ensemble confidence interval
        all_lower = [
            f["confidence_interval"]["lower"] for f in individual_forecasts.values()
        ]
        all_upper = [
            f["confidence_interval"]["upper"] for f in individual_forecasts.values()
        ]

        return {
            "point_forecast": weighted_forecast,
            "confidence_interval": {"lower": min(all_lower), "upper": max(all_upper)},
            "model_weights": weights,
        }

    def generate_forecast_insights(
        self, forecasts: Dict, historical_data: pd.DataFrame
    ) -> List[str]:
        """Generate insights from forecast results"""
        insights = []

        if "ensemble" not in forecasts:
            return ["Insufficient data for reliable forecasting"]

        current_price = historical_data["Close"].iloc[-1]
        forecast_price = forecasts["ensemble"]["point_forecast"]
        expected_return = (forecast_price - current_price) / current_price

        # Return-based insights
        if expected_return > 0.05:
            insights.append("Strong positive return forecast")
        elif expected_return > 0.02:
            insights.append("Moderate positive return forecast")
        elif expected_return < -0.05:
            insights.append("Significant negative return forecast")
        elif expected_return < -0.02:
            insights.append("Moderate negative return forecast")
        else:
            insights.append("Neutral return forecast")

        # Confidence interval insights
        ci = forecasts["ensemble"]["confidence_interval"]
        ci_width = (ci["upper"] - ci["lower"]) / current_price

        if ci_width > 0.15:
            insights.append("High forecast uncertainty - wide confidence interval")
        elif ci_width < 0.05:
            insights.append("Low forecast uncertainty - narrow confidence interval")

        # Model agreement insights
        individual_forecasts = [
            f["point_forecast"]
            for f in forecasts.values()
            if isinstance(f, dict) and "point_forecast" in f
        ]
        forecast_std = np.std(individual_forecasts) if individual_forecasts else 0

        if forecast_std / current_price < 0.01:
            insights.append("High model agreement on forecast direction")
        elif forecast_std / current_price > 0.03:
            insights.append("Significant model disagreement on forecast")

        return insights

    def get_forecast_assumptions(self) -> List[str]:
        """Get key assumptions used in forecasting"""
        return [
            "Historical patterns will continue in the near future",
            "No major unexpected market events",
            "Company fundamentals remain stable",
            "Market conditions similar to historical period",
            "Technical patterns provide meaningful signals",
        ]

    def identify_forecast_risks(
        self, forecasts: Dict, historical_data: pd.DataFrame
    ) -> List[Dict[str, str]]:
        """Identify risks to forecast accuracy"""
        risks = []

        # Data quality risks
        if len(historical_data) < 100:
            risks.append(
                {
                    "type": "data_sufficiency",
                    "description": "Limited historical data for reliable forecasting",
                    "impact": "high",
                }
            )

        # Model performance risks
        model_performance = forecasts.get("model_performance", {})
        for model_name, performance in model_performance.items():
            if performance.get("r_squared", 0) < 0.5:
                risks.append(
                    {
                        "type": "model_accuracy",
                        "description": f"Low explanatory power in {model_name} model",
                        "impact": "medium",
                    }
                )

        # Market condition risks
        volatility = historical_data["Daily_Return"].std()
        if volatility > 0.03:  # 3% daily volatility
            risks.append(
                {
                    "type": "market_volatility",
                    "description": "High market volatility reduces forecast reliability",
                    "impact": "high",
                }
            )

        return risks

    async def predict_earnings(
        self, ticker: str, next_quarter: bool = True
    ) -> Dict[str, Any]:
        """Predict company earnings using multiple data sources"""
        try:
            logger.info(f"Predicting earnings for {ticker}")

            # Get company information
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get historical earnings data (simplified - in production, use actual earnings data)
            historical_earnings = await self.get_historical_earnings(ticker)

            # Analyze trends and make prediction
            earnings_prediction = self.analyze_earnings_trends(
                historical_earnings, info
            )

            # Get analyst consensus for comparison
            analyst_consensus = self.get_analyst_consensus(ticker)

            return {
                "ticker": ticker,
                "period": "next_quarter" if next_quarter else "next_year",
                "predicted_eps": earnings_prediction["predicted_eps"],
                "confidence": earnings_prediction["confidence"],
                "key_drivers": earnings_prediction["key_drivers"],
                "analyst_consensus": analyst_consensus,
                "deviation_from_consensus": earnings_prediction["predicted_eps"]
                - analyst_consensus.get("consensus_eps", 0),
                "surprise_probability": self.calculate_surprise_probability(
                    earnings_prediction, analyst_consensus
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error predicting earnings for {ticker}: {e}")
            raise

    async def get_historical_earnings(self, ticker: str) -> Dict[str, Any]:
        """Get historical earnings data (simplified implementation)"""
        # In production, this would fetch actual historical earnings data
        # For now, return mock data structure
        return {
            "quarters": [
                {"date": "2023-Q4", "eps": 2.18, "revenue": 119.58},
                {"date": "2023-Q3", "eps": 1.46, "revenue": 89.50},
                {"date": "2023-Q2", "eps": 1.26, "revenue": 81.80},
                {"date": "2023-Q1", "eps": 1.52, "revenue": 94.84},
            ],
            "growth_rates": {
                "eps_qoq": 0.15,
                "revenue_qoq": 0.08,
                "eps_yoy": 0.12,
                "revenue_yoy": 0.05,
            },
        }

    def analyze_earnings_trends(
        self, historical_earnings: Dict, company_info: Dict
    ) -> Dict[str, Any]:
        """Analyze earnings trends and make prediction"""
        # Simplified prediction logic
        # In production, use more sophisticated time series analysis

        last_eps = historical_earnings["quarters"][0]["eps"]
        growth_rate = historical_earnings["growth_rates"]["eps_qoq"]

        # Adjust growth rate based on company size and industry
        market_cap = company_info.get("marketCap", 0)
        if market_cap > 200e9:  # Large cap
            growth_rate *= 0.8  # Slower growth for large companies
        elif market_cap < 10e9:  # Small cap
            growth_rate *= 1.2  # Faster growth for small companies

        predicted_eps = last_eps * (1 + growth_rate)

        return {
            "predicted_eps": round(predicted_eps, 2),
            "confidence": 0.7,  # Based on historical accuracy
            "key_drivers": [
                f"Historical growth rate: {growth_rate:.1%}",
                "Seasonal patterns in earnings",
                "Industry growth trends",
            ],
        }

    def get_analyst_consensus(self, ticker: str) -> Dict[str, Any]:
        """Get analyst consensus estimates (simplified)"""
        # In production, fetch from financial data provider
        return {
            "consensus_eps": 2.35,
            "high_estimate": 2.60,
            "low_estimate": 2.10,
            "number_of_analysts": 25,
            "revision_trend": "upward",
        }

    def calculate_surprise_probability(
        self, prediction: Dict, consensus: Dict
    ) -> float:
        """Calculate probability of earnings surprise"""
        deviation = abs(prediction["predicted_eps"] - consensus["consensus_eps"])
        consensus_range = consensus["high_estimate"] - consensus["low_estimate"]

        if consensus_range == 0:
            return 0.5

        # Higher deviation from consensus increases surprise probability
        surprise_prob = min(0.9, deviation / (consensus_range / 2))

        return round(surprise_prob, 2)
