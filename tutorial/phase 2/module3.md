Let's proceed with the next major advancement: **Multi-Modal Financial Analysis**. This will add earnings call analysis, document understanding, and advanced data extraction capabilities.

## 🚀 **Advancement 2: Multi-Modal Financial Analysis**

### **Step 1: Audio Processing for Earnings Calls**

#### Create `src/financial_rag/processing/audio_processor.py`

```python
import whisper
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
import os
from loguru import logger
from financial_rag.config import config

class EarningsCallProcessor:
    """Process earnings call audio and extract insights"""
    
    def __init__(self):
        self.model = None
        self.speaker_diarization = SpeakerDiarization()
        self.sentiment_analyzer = AudioSentimentAnalyzer()
        
    def load_models(self):
        """Load Whisper model for speech recognition"""
        try:
            if not self.model:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = whisper.load_model("base", device=device)
                logger.success("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe_earnings_call(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe earnings call audio with speaker identification"""
        try:
            self.load_models()
            
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language='en',
                fp16=False  # More stable on CPU
            )
            
            # Perform speaker diarization
            segments_with_speakers = self.speaker_diarization.identify_speakers(
                audio_path, result['segments']
            )
            
            # Analyze sentiment per speaker
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(
                segments_with_speakers
            )
            
            # Extract key metrics
            key_metrics = self.extract_financial_metrics(result['text'])
            
            return {
                'transcript': result['text'],
                'segments': segments_with_speakers,
                'sentiment_analysis': sentiment_analysis,
                'key_metrics': key_metrics,
                'duration': result.get('duration', 0),
                'language': result.get('language', 'en')
            }
            
        except Exception as e:
            logger.error(f"Error transcribing earnings call: {e}")
            raise
    
    def extract_financial_metrics(self, transcript: str) -> Dict[str, Any]:
        """Extract financial metrics from earnings call transcript"""
        import re
        
        metrics = {
            'revenue': self.extract_revenue(transcript),
            'eps': self.extract_eps(transcript),
            'guidance': self.extract_guidance(transcript),
            'growth_rates': self.extract_growth_rates(transcript),
            'key_announcements': self.extract_announcements(transcript)
        }
        
        return metrics
    
    def extract_revenue(self, text: str) -> List[Dict]:
        """Extract revenue figures from transcript"""
        revenue_patterns = [
            r'revenue\s*(?:of|was|\$)\s*(\d+\.?\d*)\s*(billion|million|B|M)',
            r'(\d+\.?\d*)\s*(billion|million|B|M)\s*in\s*revenue',
            r'revenue\s*(?:growth|increased|decreased)\s*(?:by|to)\s*(\d+\.?\d*)%'
        ]
        
        revenues = []
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                revenues.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else 'unknown',
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return revenues
    
    def extract_eps(self, text: str) -> List[Dict]:
        """Extract EPS figures from transcript"""
        eps_patterns = [
            r'eps\s*(?:of|was|\$)\s*(\d+\.?\d*)',
            r'earnings\s*per\s*share\s*(?:of|was|\$)\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*eps'
        ]
        
        eps_figures = []
        for pattern in eps_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                eps_figures.append({
                    'value': float(match.group(1)),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return eps_figures
    
    def extract_guidance(self, text: str) -> Dict[str, Any]:
        """Extract forward guidance from transcript"""
        guidance_keywords = [
            'guidance', 'outlook', 'expect', 'forecast', 'project',
            'anticipate', 'target', 'estimate'
        ]
        
        guidance_sentences = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in guidance_keywords):
                guidance_sentences.append(sentence.strip())
        
        return {
            'sentences': guidance_sentences,
            'confidence': len(guidance_sentences) / max(len(sentences), 1)
        }
    
    def extract_growth_rates(self, text: str) -> List[Dict]:
        """Extract growth rate mentions"""
        growth_pattern = r'(\d+\.?\d*)%\s*(?:growth|increase|decrease|change)'
        
        growth_rates = []
        matches = re.finditer(growth_pattern, text, re.IGNORECASE)
        
        for match in matches:
            growth_rates.append({
                'rate': float(match.group(1)),
                'context': text[max(0, match.start()-30):match.end()+30]
            })
        
        return growth_rates
    
    def extract_announcements(self, text: str) -> List[str]:
        """Extract key announcements"""
        announcement_indicators = [
            'announce', 'launch', 'introduce', 'release', 'new',
            'partnership', 'acquisition', 'investment', 'expansion'
        ]
        
        announcements = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in announcement_indicators):
                # Filter out very short sentences
                if len(sentence.split()) > 5:
                    announcements.append(sentence.strip())
        
        return announcements


class SpeakerDiarization:
    """Identify different speakers in earnings calls"""
    
    def identify_speakers(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """Identify speakers in audio segments (simplified implementation)"""
        # In production, use pyannote.audio or similar
        # For now, use a rule-based approach
        
        speaker_segments = []
        current_speaker = "Speaker_1"
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').lower()
            
            # Simple speaker change detection based on content
            if i > 0:
                prev_text = segments[i-1].get('text', '').lower()
                
                # Speaker change indicators
                change_indicators = [
                    'thank you', 'questions', 'operator', 'next question',
                    'good morning', 'good afternoon', 'hello everyone'
                ]
                
                if any(indicator in text for indicator in change_indicators):
                    current_speaker = f"Speaker_{len(set([s['speaker'] for s in speaker_segments])) + 1}"
            
            speaker_segments.append({
                **segment,
                'speaker': current_speaker,
                'speaker_role': self.infer_speaker_role(text, current_speaker)
            })
        
        return speaker_segments
    
    def infer_speaker_role(self, text: str, speaker_id: str) -> str:
        """Infer speaker role based on content"""
        text_lower = text.lower()
        
        # CEO indicators
        if any(phrase in text_lower for phrase in [
            'our strategy', 'company vision', 'long-term', 'shareholders',
            'transformative', 'market leadership'
        ]):
            return 'CEO'
        
        # CFO indicators
        elif any(phrase in text_lower for phrase in [
            'financial results', 'revenue', 'eps', 'margin', 'guidance',
            'cash flow', 'balance sheet', 'capital allocation'
        ]):
            return 'CFO'
        
        # Analyst indicators
        elif any(phrase in text_lower for phrase in [
            'question', 'could you', 'can you', 'what about', 'how about'
        ]):
            return 'Analyst'
        
        # Operator indicators
        elif any(phrase in text_lower for phrase in [
            'thank you', 'next question', 'question and answer'
        ]):
            return 'Operator'
        
        return 'Unknown'


class AudioSentimentAnalyzer:
    """Analyze sentiment in audio segments"""
    
    def analyze_sentiment(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment per speaker and overall"""
        from textblob import TextBlob
        
        speaker_sentiments = {}
        overall_sentiment = {
            'positive_segments': 0,
            'negative_segments': 0,
            'neutral_segments': 0,
            'average_polarity': 0,
            'average_subjectivity': 0
        }
        
        polarities = []
        subjectivities = []
        
        for segment in segments:
            text = segment.get('text', '')
            speaker = segment.get('speaker', 'Unknown')
            
            if text.strip():
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                polarities.append(polarity)
                subjectivities.append(subjectivity)
                
                # Categorize sentiment
                if polarity > 0.1:
                    sentiment = 'positive'
                    overall_sentiment['positive_segments'] += 1
                elif polarity < -0.1:
                    sentiment = 'negative'
                    overall_sentiment['negative_segments'] += 1
                else:
                    sentiment = 'neutral'
                    overall_sentiment['neutral_segments'] += 1
                
                # Track by speaker
                if speaker not in speaker_sentiments:
                    speaker_sentiments[speaker] = {
                        'polarity_sum': 0,
                        'subjectivity_sum': 0,
                        'segment_count': 0,
                        'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                    }
                
                speaker_sentiments[speaker]['polarity_sum'] += polarity
                speaker_sentiments[speaker]['subjectivity_sum'] += subjectivity
                speaker_sentiments[speaker]['segment_count'] += 1
                speaker_sentiments[speaker]['sentiment_distribution'][sentiment] += 1
        
        # Calculate averages
        if polarities:
            overall_sentiment['average_polarity'] = sum(polarities) / len(polarities)
            overall_sentiment['average_subjectivity'] = sum(subjectivities) / len(subjectivities)
        
        # Finalize speaker sentiments
        for speaker, data in speaker_sentiments.items():
            data['average_polarity'] = data['polarity_sum'] / data['segment_count']
            data['average_subjectivity'] = data['subjectivity_sum'] / data['segment_count']
            del data['polarity_sum']
            del data['subjectivity_sum']
        
        return {
            'overall': overall_sentiment,
            'by_speaker': speaker_sentiments
        }
```

### **Step 2: Document Understanding & Table Extraction**

#### Create `src/financial_rag/processing/document_understanding.py`

```python
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
                'income_statements': [],
                'balance_sheets': [],
                'cash_flow_statements': [],
                'other_tables': []
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables using camelot for better accuracy
                    camelot_tables = camelot.read_pdf(
                        pdf_path, 
                        pages=str(page_num+1),
                        flavor='lattice'
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
                                df, page_num, {'accuracy': 0.7}
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
    
    def process_financial_table(self, df: pd.DataFrame, page_num: int, 
                              parsing_report: Dict) -> Optional[Dict]:
        """Process and clean financial table"""
        try:
            # Clean the dataframe
            df_clean = self.clean_dataframe(df)
            
            if df_clean.empty or len(df_clean.columns) < 2:
                return None
            
            # Extract metadata
            table_metadata = {
                'page_number': page_num + 1,
                'parsing_accuracy': parsing_report.get('accuracy', 0),
                'shape': df_clean.shape,
                'columns': df_clean.columns.tolist(),
                'data_types': self.infer_data_types(df_clean)
            }
            
            # Convert to structured data
            structured_data = self.convert_to_structured_data(df_clean)
            
            return {
                'metadata': table_metadata,
                'raw_data': df_clean.to_dict('records'),
                'structured_data': structured_data,
                'cleaned_df': df_clean
            }
            
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return None
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize dataframe"""
        # Remove empty rows and columns
        df_clean = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df_clean.columns = [self.clean_column_name(col) for col in df_clean.columns]
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Convert numeric columns
        for col in df_clean.columns:
            if col != 'line_item':  # Assuming first column is description
                df_clean[col] = df_clean[col].apply(self.convert_financial_value)
        
        return df_clean
    
    def clean_column_name(self, name: Any) -> str:
        """Clean column names"""
        if pd.isna(name):
            return 'unknown'
        
        name_str = str(name).strip().lower()
        # Remove special characters but keep periods for dates
        name_str = re.sub(r'[^\w\s.]', '', name_str)
        return name_str
    
    def convert_financial_value(self, value: Any) -> Optional[float]:
        """Convert financial values to numbers"""
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        # Remove common financial notation
        value_str = value_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        
        # Handle percentage values
        if '%' in value_str:
            value_str = value_str.replace('%', '')
            try:
                return float(value_str) / 100
            except:
                return None
        
        # Handle text representations
        if 'million' in value_str.lower():
            value_str = value_str.lower().replace('million', '').strip()
            try:
                return float(value_str) * 1_000_000
            except:
                return None
        
        if 'billion' in value_str.lower():
            value_str = value_str.lower().replace('billion', '').strip()
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
                data_types[col] = 'numeric'
            else:
                data_types[col] = 'text'
        
        return data_types
    
    def convert_to_structured_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert table to structured financial data"""
        structured = {
            'line_items': [],
            'periods': [],
            'values': {}
        }
        
        # Assume first column is line items
        if len(df.columns) > 0:
            line_items = df.iloc[:, 0].tolist()
            structured['line_items'] = [str(item) for item in line_items if pd.notna(item)]
        
        # Assume other columns are time periods
        if len(df.columns) > 1:
            periods = df.columns[1:].tolist()
            structured['periods'] = periods
            
            # Extract values for each line item and period
            for i, line_item in enumerate(structured['line_items']):
                if i < len(df):
                    values = {}
                    for j, period in enumerate(periods):
                        if j + 1 < len(df.columns):
                            value = df.iloc[i, j + 1]
                            if pd.notna(value):
                                values[period] = value
                    
                    if values:
                        structured['values'][line_item] = values
        
        return structured
    
    def classify_table_type(self, table_data: Dict) -> str:
        """Classify the type of financial table"""
        line_items = table_data['structured_data'].get('line_items', [])
        line_items_lower = [str(item).lower() for item in line_items]
        
        # Income statement indicators
        income_indicators = ['revenue', 'sales', 'gross profit', 'operating income', 'net income', 'eps']
        if any(indicator in ' '.join(line_items_lower) for indicator in income_indicators):
            return 'income_statements'
        
        # Balance sheet indicators
        balance_indicators = ['assets', 'liabilities', 'equity', 'cash', 'inventory', 'debt']
        if any(indicator in ' '.join(line_items_lower) for indicator in balance_indicators):
            return 'balance_sheets'
        
        # Cash flow indicators
        cash_flow_indicators = ['operating activities', 'investing activities', 'financing activities', 'cash flow']
        if any(indicator in ' '.join(line_items_lower) for indicator in cash_flow_indicators):
            return 'cash_flow_statements'
        
        return 'other_tables'
    
    def analyze_all_tables(self, tables_data: Dict) -> Dict[str, Any]:
        """Analyze all extracted tables for insights"""
        insights = {
            'financial_metrics': {},
            'trends': {},
            'key_findings': []
        }
        
        # Analyze income statements
        for table in tables_data['income_statements']:
            metrics = self.table_analyzer.analyze_income_statement(table)
            insights['financial_metrics'].update(metrics)
        
        # Analyze balance sheets
        for table in tables_data['balance_sheets']:
            metrics = self.table_analyzer.analyze_balance_sheet(table)
            insights['financial_metrics'].update(metrics)
        
        # Analyze trends
        insights['trends'] = self.table_analyzer.analyze_trends(tables_data)
        
        # Generate key findings
        insights['key_findings'] = self.generate_key_findings(insights)
        
        return {
            'tables': tables_data,
            'insights': insights
        }
    
    def generate_key_findings(self, insights: Dict) -> List[str]:
        """Generate natural language key findings"""
        findings = []
        metrics = insights.get('financial_metrics', {})
        trends = insights.get('trends', {})
        
        # Revenue findings
        if 'revenue_growth' in metrics:
            growth = metrics['revenue_growth']
            if growth > 0.1:
                findings.append(f"Strong revenue growth of {growth:.1%}")
            elif growth < 0:
                findings.append(f"Revenue decline of {abs(growth):.1%}")
        
        # Profitability findings
        if 'profit_margin' in metrics:
            margin = metrics['profit_margin']
            if margin > 0.2:
                findings.append(f"High profit margin of {margin:.1%}")
            elif margin < 0.05:
                findings.append(f"Low profit margin of {margin:.1%}")
        
        # Trend findings
        for metric, trend in trends.items():
            if trend.get('direction') == 'increasing':
                findings.append(f"Increasing trend in {metric}")
            elif trend.get('direction') == 'decreasing':
                findings.append(f"Decreasing trend in {metric}")
        
        return findings


class FinancialTableAnalyzer:
    """Analyze financial tables for insights"""
    
    def analyze_income_statement(self, table_data: Dict) -> Dict[str, float]:
        """Analyze income statement table"""
        metrics = {}
        structured_data = table_data['structured_data']
        
        # Extract key metrics
        revenue = self.extract_metric(structured_data, 'revenue')
        net_income = self.extract_metric(structured_data, 'net income')
        gross_profit = self.extract_metric(structured_data, 'gross profit')
        operating_income = self.extract_metric(structured_data, 'operating income')
        
        if revenue and net_income:
            metrics['profit_margin'] = net_income / revenue
        
        if revenue and gross_profit:
            metrics['gross_margin'] = gross_profit / revenue
        
        if revenue and operating_income:
            metrics['operating_margin'] = operating_income / revenue
        
        # Calculate growth if multiple periods
        periods = structured_data.get('periods', [])
        if len(periods) >= 2 and revenue:
            current_rev = self.extract_metric_for_period(structured_data, 'revenue', periods[-1])
            prev_rev = self.extract_metric_for_period(structured_data, 'revenue', periods[-2])
            
            if current_rev and prev_rev and prev_rev != 0:
                metrics['revenue_growth'] = (current_rev - prev_rev) / prev_rev
        
        return metrics
    
    def analyze_balance_sheet(self, table_data: Dict) -> Dict[str, float]:
        """Analyze balance sheet table"""
        metrics = {}
        structured_data = table_data['structured_data']
        
        # Extract key metrics
        total_assets = self.extract_metric(structured_data, 'total assets')
        total_liabilities = self.extract_metric(structured_data, 'total liabilities')
        equity = self.extract_metric(structured_data, 'total equity')
        cash = self.extract_metric(structured_data, 'cash')
        
        if total_assets and total_liabilities:
            metrics['debt_to_assets'] = total_liabilities / total_assets
        
        if equity and total_liabilities:
            metrics['debt_to_equity'] = total_liabilities / equity
        
        if cash and total_assets:
            metrics['cash_ratio'] = cash / total_assets
        
        return metrics
    
    def extract_metric(self, structured_data: Dict, metric_name: str) -> Optional[float]:
        """Extract a metric from structured data"""
        values = structured_data.get('values', {})
        
        for line_item, period_values in values.items():
            if metric_name.lower() in line_item.lower():
                # Get the most recent value
                periods = structured_data.get('periods', [])
                if periods:
                    latest_period = periods[-1]
                    return period_values.get(latest_period)
        
        return None
    
    def extract_metric_for_period(self, structured_data: Dict, metric_name: str, 
                                period: str) -> Optional[float]:
        """Extract a metric for a specific period"""
        values = structured_data.get('values', {})
        
        for line_item, period_values in values.items():
            if metric_name.lower() in line_item.lower():
                return period_values.get(period)
        
        return None
    
    def analyze_trends(self, tables_data: Dict) -> Dict[str, Any]:
        """Analyze trends across multiple periods"""
        trends = {}
        
        # Analyze income statement trends
        for table in tables_data['income_statements']:
            structured_data = table['structured_data']
            periods = structured_data.get('periods', [])
            
            if len(periods) >= 2:
                for line_item, values in structured_data.get('values', {}).items():
                    if len(values) >= 2:
                        period_values = list(values.items())
                        current_val = period_values[-1][1]
                        prev_val = period_values[-2][1]
                        
                        if prev_val and prev_val != 0:
                            growth = (current_val - prev_val) / prev_val
                            direction = 'increasing' if growth > 0 else 'decreasing'
                            
                            trends[line_item] = {
                                'growth_rate': growth,
                                'direction': direction,
                                'current_value': current_val,
                                'previous_value': prev_val
                            }
        
        return trends


class ChartProcessor:
    """Process and extract data from financial charts"""
    
    def extract_chart_data(self, image_path: str) -> Dict[str, Any]:
        """Extract data from financial charts (simplified implementation)"""
        # In production, use computer vision models
        # For now, return mock data structure
        
        return {
            'chart_type': self.detect_chart_type(image_path),
            'data_points': [],
            'trend': 'unknown',
            'confidence': 0.0
        }
    
    def detect_chart_type(self, image_path: str) -> str:
        """Detect the type of financial chart"""
        # Simplified implementation
        return 'line_chart'  # In production, use CV to detect chart type
```

### **Step 3: Multi-Modal Analysis Agent**

#### Create `src/financial_rag/agents/multi_modal_analyst.py`

```python
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
        
    async def analyze_earnings_call(self, audio_path: str, ticker: str) -> Dict[str, Any]:
        """Comprehensive earnings call analysis"""
        try:
            logger.info(f"Analyzing earnings call for {ticker}: {audio_path}")
            
            # Transcribe and analyze audio
            call_analysis = self.audio_processor.transcribe_earnings_call(audio_path)
            
            # Get real-time context
            real_time_context = await self.get_real_time_context([ticker])
            
            # Combine with historical analysis
            historical_analysis = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.analyze,
                f"Provide historical context and risk factors for {ticker}"
            )
            
            # Generate comprehensive insights
            insights = self.generate_earnings_insights(
                call_analysis, real_time_context, historical_analysis
            )
            
            # Store in multimodal context
            self.multimodal_context[f'earnings_{ticker}_{datetime.now().isoformat()}'] = {
                'call_analysis': call_analysis,
                'real_time_context': real_time_context,
                'historical_analysis': historical_analysis,
                'insights': insights
            }
            
            return {
                'ticker': ticker,
                'call_analysis': call_analysis,
                'real_time_context': real_time_context,
                'historical_analysis': historical_analysis,
                'insights': insights,
                'summary': self.generate_earnings_summary(insights)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing earnings call: {e}")
            raise
    
    def generate_earnings_insights(self, call_analysis: Dict, real_time_context: Dict, 
                                 historical_analysis: Dict) -> Dict[str, Any]:
        """Generate insights from earnings call analysis"""
        insights = {
            'sentiment_analysis': {},
            'key_metrics': {},
            'guidance_analysis': {},
            'market_reaction': {},
            'investment_implications': []
        }
        
        # Sentiment insights
        sentiment_data = call_analysis.get('sentiment_analysis', {})
        insights['sentiment_analysis'] = self.analyze_call_sentiment(sentiment_data)
        
        # Metric insights
        key_metrics = call_analysis.get('key_metrics', {})
        insights['key_metrics'] = self.analyze_call_metrics(key_metrics)
        
        # Guidance analysis
        guidance = key_metrics.get('guidance', {})
        insights['guidance_analysis'] = self.analyze_guidance(guidance)
        
        # Market reaction analysis
        market_data = real_time_context.get('market_data', {})
        insights['market_reaction'] = self.analyze_market_reaction(market_data)
        
        # Investment implications
        insights['investment_implications'] = self.generate_investment_implications(
            insights, historical_analysis
        )
        
        return insights
    
    def analyze_call_sentiment(self, sentiment_data: Dict) -> Dict[str, Any]:
        """Analyze sentiment from earnings call"""
        overall = sentiment_data.get('overall', {})
        by_speaker = sentiment_data.get('by_speaker', {})
        
        return {
            'overall_sentiment': self.get_sentiment_label(overall.get('average_polarity', 0)),
            'sentiment_score': overall.get('average_polarity', 0),
            'speaker_sentiments': {
                speaker: {
                    'sentiment': self.get_sentiment_label(data.get('average_polarity', 0)),
                    'score': data.get('average_polarity', 0),
                    'confidence': data.get('average_subjectivity', 0)
                }
                for speaker, data in by_speaker.items()
            },
            'key_positive_points': self.extract_positive_statements(sentiment_data),
            'key_concerns': self.extract_concerns(sentiment_data)
        }
    
    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return 'very_positive'
        elif score > 0.05:
            return 'positive'
        elif score > -0.05:
            return 'neutral'
        elif score > -0.2:
            return 'negative'
        else:
            return 'very_negative'
    
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
            'revenue_trend': self.analyze_revenue_trend(key_metrics.get('revenue', [])),
            'profitability': self.analyze_profitability(key_metrics.get('eps', [])),
            'growth_indicators': self.analyze_growth_indicators(key_metrics.get('growth_rates', [])),
            'key_announcements': key_metrics.get('key_announcements', [])
        }
        
        return analysis
    
    def analyze_revenue_trend(self, revenue_data: List) -> Dict[str, Any]:
        """Analyze revenue trends"""
        if not revenue_data:
            return {'trend': 'unknown', 'confidence': 0}
        
        # Simple trend analysis
        return {'trend': 'growing', 'confidence': 0.8}
    
    def analyze_profitability(self, eps_data: List) -> Dict[str, Any]:
        """Analyze profitability trends"""
        if not eps_data:
            return {'trend': 'unknown', 'confidence': 0}
        
        # Simple profitability analysis
        return {'trend': 'stable', 'confidence': 0.7}
    
    def analyze_growth_indicators(self, growth_rates: List) -> Dict[str, Any]:
        """Analyze growth indicators"""
        if not growth_rates:
            return {'indicators': [], 'overall_growth': 'unknown'}
        
        positive_growth = [g for g in growth_rates if g.get('rate', 0) > 0]
        growth_strength = len(positive_growth) / len(growth_rates) if growth_rates else 0
        
        return {
            'indicators': growth_rates,
            'overall_growth': 'strong' if growth_strength > 0.7 else 'moderate'
        }
    
    def analyze_guidance(self, guidance: Dict) -> Dict[str, Any]:
        """Analyze forward guidance"""
        sentences = guidance.get('sentences', [])
        confidence = guidance.get('confidence', 0)
        
        return {
            'guidance_statements': sentences,
            'confidence': confidence,
            'sentiment': self.analyze_guidance_sentiment(sentences),
            'key_points': self.extract_guidance_key_points(sentences)
        }
    
    def analyze_guidance_sentiment(self, guidance_sentences: List[str]) -> str:
        """Analyze sentiment of guidance statements"""
        if not guidance_sentences:
            return 'neutral'
        
        # Simple keyword-based sentiment analysis
        positive_words = ['strong', 'growth', 'increase', 'improve', 'optimistic']
        negative_words = ['challenge', 'headwind', 'uncertain', 'pressure', 'decline']
        
        text = ' '.join(guidance_sentences).lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_guidance_key_points(self, guidance_sentences: List[str]) -> List[str]:
        """Extract key points from guidance"""
        key_points = []
        
        for sentence in guidance_sentences:
            # Extract quantitative guidance
            if any(word in sentence.lower() for word in ['expect', 'target', 'forecast', 'guidance']):
                key_points.append(sentence)
        
        return key_points[:3]  # Return top 3 key points
    
    def analyze_market_reaction(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze market reaction to earnings"""
        reaction = {
            'price_movement': {},
            'volume_analysis': {},
            'sentiment_impact': 'neutral'
        }
        
        for ticker, data in market_data.items():
            price_change = data.get('change_pct', 0)
            volume = data.get('volume', 0)
            
            reaction['price_movement'][ticker] = {
                'change': price_change,
                'magnitude': abs(price_change),
                'direction': 'up' if price_change > 0 else 'down'
            }
            
            # Simple volume analysis
            if volume > 1000000: