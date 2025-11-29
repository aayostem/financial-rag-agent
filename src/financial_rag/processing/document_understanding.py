import pdfplumber
import pandas as pd
import camelot
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from loguru import logger


class FinancialDocumentProcessor:
    """Process financial documents with table extraction and understanding"""

    def __init__(self):
        self.table_analyzer = FinancialTableAnalyzer()
        self.chart_processor = ChartProcessor()

    def extract_financial_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract and analyze financial tables from PDF"""
        try:
            tables_data = {
                "income_statements": [],
                "balance_sheets": [],
                "cash_flow_statements": [],
                "other_tables": [],
            }

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables using camelot for better accuracy
                    camelot_tables = camelot.read_pdf(
                        pdf_path, pages=str(page_num + 1), flavor="lattice"
                    )

                    for table in camelot_tables:
                        table_data = self.process_financial_table(
                            table.df, page_num, table.parsing_report
                        )

                        if table_data:
                            table_type = self.classify_table_type(table_data)
                            tables_data[table_type].append(table_data)

                    # Fallback: pdfplumber tables
                    pdfplumber_tables = page.extract_tables()
                    for table in pdfplumber_tables:
                        if table and len(table) > 1:  # Valid table with header
                            df = pd.DataFrame(table[1:], columns=table[0])
                            table_data = self.process_financial_table(
                                df, page_num, {"accuracy": 0.7}
                            )

                            if table_data:
                                table_type = self.classify_table_type(table_data)
                                tables_data[table_type].append(table_data)

            # Analyze extracted tables
            analyzed_tables = self.analyze_all_tables(tables_data)

            return analyzed_tables

        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            return {}

    def process_financial_table(
        self, df: pd.DataFrame, page_num: int, parsing_report: Dict
    ) -> Optional[Dict]:
        """Process and clean financial table"""
        try:
            # Clean the dataframe
            df_clean = self.clean_dataframe(df)

            if df_clean.empty or len(df_clean.columns) < 2:
                return None

            # Extract metadata
            table_metadata = {
                "page_number": page_num + 1,
                "parsing_accuracy": parsing_report.get("accuracy", 0),
                "shape": df_clean.shape,
                "columns": df_clean.columns.tolist(),
                "data_types": self.infer_data_types(df_clean),
            }

            # Convert to structured data
            structured_data = self.convert_to_structured_data(df_clean)

            return {
                "metadata": table_metadata,
                "raw_data": df_clean.to_dict("records"),
                "structured_data": structured_data,
                "cleaned_df": df_clean,
            }

        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return None

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize dataframe"""
        # Remove empty rows and columns
        df_clean = df.dropna(how="all").dropna(axis=1, how="all")

        # Clean column names
        df_clean.columns = [self.clean_column_name(col) for col in df_clean.columns]

        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()

        # Convert numeric columns
        for col in df_clean.columns:
            if col != "line_item":  # Assuming first column is description
                df_clean[col] = df_clean[col].apply(self.convert_financial_value)

        return df_clean

    def clean_column_name(self, name: Any) -> str:
        """Clean column names"""
        if pd.isna(name):
            return "unknown"

        name_str = str(name).strip().lower()
        # Remove special characters but keep periods for dates
        name_str = re.sub(r"[^\w\s.]", "", name_str)
        return name_str

    def convert_financial_value(self, value: Any) -> Optional[float]:
        """Convert financial values to numbers"""
        if pd.isna(value):
            return None

        value_str = str(value).strip()

        # Remove common financial notation
        value_str = (
            value_str.replace("$", "")
            .replace(",", "")
            .replace("(", "-")
            .replace(")", "")
        )

        # Handle percentage values
        if "%" in value_str:
            value_str = value_str.replace("%", "")
            try:
                return float(value_str) / 100
            except:
                return None

        # Handle text representations
        if "million" in value_str.lower():
            value_str = value_str.lower().replace("million", "").strip()
            try:
                return float(value_str) * 1_000_000
            except:
                return None

        if "billion" in value_str.lower():
            value_str = value_str.lower().replace("billion", "").strip()
            try:
                return float(value_str) * 1_000_000_000
            except:
                return None

        try:
            return float(value_str)
        except:
            return None

    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer data types for columns"""
        data_types = {}

        for col in df.columns:
            # Check if column contains mostly numeric values
            numeric_count = df[col].apply(lambda x: isinstance(x, (int, float))).sum()

            if numeric_count / len(df) > 0.7:
                data_types[col] = "numeric"
            else:
                data_types[col] = "text"

        return data_types

    def convert_to_structured_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert table to structured financial data"""
        structured = {"line_items": [], "periods": [], "values": {}}

        # Assume first column is line items
        if len(df.columns) > 0:
            line_items = df.iloc[:, 0].tolist()
            structured["line_items"] = [
                str(item) for item in line_items if pd.notna(item)
            ]

        # Assume other columns are time periods
        if len(df.columns) > 1:
            periods = df.columns[1:].tolist()
            structured["periods"] = periods

            # Extract values for each line item and period
            for i, line_item in enumerate(structured["line_items"]):
                if i < len(df):
                    values = {}
                    for j, period in enumerate(periods):
                        if j + 1 < len(df.columns):
                            value = df.iloc[i, j + 1]
                            if pd.notna(value):
                                values[period] = value

                    if values:
                        structured["values"][line_item] = values

        return structured

    def classify_table_type(self, table_data: Dict) -> str:
        """Classify the type of financial table"""
        line_items = table_data["structured_data"].get("line_items", [])
        line_items_lower = [str(item).lower() for item in line_items]

        # Income statement indicators
        income_indicators = [
            "revenue",
            "sales",
            "gross profit",
            "operating income",
            "net income",
            "eps",
        ]
        if any(
            indicator in " ".join(line_items_lower) for indicator in income_indicators
        ):
            return "income_statements"

        # Balance sheet indicators
        balance_indicators = [
            "assets",
            "liabilities",
            "equity",
            "cash",
            "inventory",
            "debt",
        ]
        if any(
            indicator in " ".join(line_items_lower) for indicator in balance_indicators
        ):
            return "balance_sheets"

        # Cash flow indicators
        cash_flow_indicators = [
            "operating activities",
            "investing activities",
            "financing activities",
            "cash flow",
        ]
        if any(
            indicator in " ".join(line_items_lower)
            for indicator in cash_flow_indicators
        ):
            return "cash_flow_statements"

        return "other_tables"

    def analyze_all_tables(self, tables_data: Dict) -> Dict[str, Any]:
        """Analyze all extracted tables for insights"""
        insights = {"financial_metrics": {}, "trends": {}, "key_findings": []}

        # Analyze income statements
        for table in tables_data["income_statements"]:
            metrics = self.table_analyzer.analyze_income_statement(table)
            insights["financial_metrics"].update(metrics)

        # Analyze balance sheets
        for table in tables_data["balance_sheets"]:
            metrics = self.table_analyzer.analyze_balance_sheet(table)
            insights["financial_metrics"].update(metrics)

        # Analyze trends
        insights["trends"] = self.table_analyzer.analyze_trends(tables_data)

        # Generate key findings
        insights["key_findings"] = self.generate_key_findings(insights)

        return {"tables": tables_data, "insights": insights}

    def generate_key_findings(self, insights: Dict) -> List[str]:
        """Generate natural language key findings"""
        findings = []
        metrics = insights.get("financial_metrics", {})
        trends = insights.get("trends", {})

        # Revenue findings
        if "revenue_growth" in metrics:
            growth = metrics["revenue_growth"]
            if growth > 0.1:
                findings.append(f"Strong revenue growth of {growth:.1%}")
            elif growth < 0:
                findings.append(f"Revenue decline of {abs(growth):.1%}")

        # Profitability findings
        if "profit_margin" in metrics:
            margin = metrics["profit_margin"]
            if margin > 0.2:
                findings.append(f"High profit margin of {margin:.1%}")
            elif margin < 0.05:
                findings.append(f"Low profit margin of {margin:.1%}")

        # Trend findings
        for metric, trend in trends.items():
            if trend.get("direction") == "increasing":
                findings.append(f"Increasing trend in {metric}")
            elif trend.get("direction") == "decreasing":
                findings.append(f"Decreasing trend in {metric}")

        return findings


class FinancialTableAnalyzer:
    """Analyze financial tables for insights"""

    def analyze_income_statement(self, table_data: Dict) -> Dict[str, float]:
        """Analyze income statement table"""
        metrics = {}
        structured_data = table_data["structured_data"]

        # Extract key metrics
        revenue = self.extract_metric(structured_data, "revenue")
        net_income = self.extract_metric(structured_data, "net income")
        gross_profit = self.extract_metric(structured_data, "gross profit")
        operating_income = self.extract_metric(structured_data, "operating income")

        if revenue and net_income:
            metrics["profit_margin"] = net_income / revenue

        if revenue and gross_profit:
            metrics["gross_margin"] = gross_profit / revenue

        if revenue and operating_income:
            metrics["operating_margin"] = operating_income / revenue

        # Calculate growth if multiple periods
        periods = structured_data.get("periods", [])
        if len(periods) >= 2 and revenue:
            current_rev = self.extract_metric_for_period(
                structured_data, "revenue", periods[-1]
            )
            prev_rev = self.extract_metric_for_period(
                structured_data, "revenue", periods[-2]
            )

            if current_rev and prev_rev and prev_rev != 0:
                metrics["revenue_growth"] = (current_rev - prev_rev) / prev_rev

        return metrics

    def analyze_balance_sheet(self, table_data: Dict) -> Dict[str, float]:
        """Analyze balance sheet table"""
        metrics = {}
        structured_data = table_data["structured_data"]

        # Extract key metrics
        total_assets = self.extract_metric(structured_data, "total assets")
        total_liabilities = self.extract_metric(structured_data, "total liabilities")
        equity = self.extract_metric(structured_data, "total equity")
        cash = self.extract_metric(structured_data, "cash")

        if total_assets and total_liabilities:
            metrics["debt_to_assets"] = total_liabilities / total_assets

        if equity and total_liabilities:
            metrics["debt_to_equity"] = total_liabilities / equity

        if cash and total_assets:
            metrics["cash_ratio"] = cash / total_assets

        return metrics

    def extract_metric(
        self, structured_data: Dict, metric_name: str
    ) -> Optional[float]:
        """Extract a metric from structured data"""
        values = structured_data.get("values", {})

        for line_item, period_values in values.items():
            if metric_name.lower() in line_item.lower():
                # Get the most recent value
                periods = structured_data.get("periods", [])
                if periods:
                    latest_period = periods[-1]
                    return period_values.get(latest_period)

        return None

    def extract_metric_for_period(
        self, structured_data: Dict, metric_name: str, period: str
    ) -> Optional[float]:
        """Extract a metric for a specific period"""
        values = structured_data.get("values", {})

        for line_item, period_values in values.items():
            if metric_name.lower() in line_item.lower():
                return period_values.get(period)

        return None

    def analyze_trends(self, tables_data: Dict) -> Dict[str, Any]:
        """Analyze trends across multiple periods"""
        trends = {}

        # Analyze income statement trends
        for table in tables_data["income_statements"]:
            structured_data = table["structured_data"]
            periods = structured_data.get("periods", [])

            if len(periods) >= 2:
                for line_item, values in structured_data.get("values", {}).items():
                    if len(values) >= 2:
                        period_values = list(values.items())
                        current_val = period_values[-1][1]
                        prev_val = period_values[-2][1]

                        if prev_val and prev_val != 0:
                            growth = (current_val - prev_val) / prev_val
                            direction = "increasing" if growth > 0 else "decreasing"

                            trends[line_item] = {
                                "growth_rate": growth,
                                "direction": direction,
                                "current_value": current_val,
                                "previous_value": prev_val,
                            }

        return trends


class ChartProcessor:
    """Process and extract data from financial charts"""

    def extract_chart_data(self, image_path: str) -> Dict[str, Any]:
        """Extract data from financial charts (simplified implementation)"""
        # In production, use computer vision models
        # For now, return mock data structure

        return {
            "chart_type": self.detect_chart_type(image_path),
            "data_points": [],
            "trend": "unknown",
            "confidence": 0.0,
        }

    def detect_chart_type(self, image_path: str) -> str:
        """Detect the type of financial chart"""
        # Simplified implementation
        return "line_chart"  # In production, use CV to detect chart type
