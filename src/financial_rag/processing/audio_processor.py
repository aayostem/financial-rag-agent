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
                audio_path, language="en", fp16=False  # More stable on CPU
            )

            # Perform speaker diarization
            segments_with_speakers = self.speaker_diarization.identify_speakers(
                audio_path, result["segments"]
            )

            # Analyze sentiment per speaker
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(
                segments_with_speakers
            )

            # Extract key metrics
            key_metrics = self.extract_financial_metrics(result["text"])

            return {
                "transcript": result["text"],
                "segments": segments_with_speakers,
                "sentiment_analysis": sentiment_analysis,
                "key_metrics": key_metrics,
                "duration": result.get("duration", 0),
                "language": result.get("language", "en"),
            }

        except Exception as e:
            logger.error(f"Error transcribing earnings call: {e}")
            raise

    def extract_financial_metrics(self, transcript: str) -> Dict[str, Any]:
        """Extract financial metrics from earnings call transcript"""
        import re

        metrics = {
            "revenue": self.extract_revenue(transcript),
            "eps": self.extract_eps(transcript),
            "guidance": self.extract_guidance(transcript),
            "growth_rates": self.extract_growth_rates(transcript),
            "key_announcements": self.extract_announcements(transcript),
        }

        return metrics

    def extract_revenue(self, text: str) -> List[Dict]:
        """Extract revenue figures from transcript"""
        revenue_patterns = [
            r"revenue\s*(?:of|was|\$)\s*(\d+\.?\d*)\s*(billion|million|B|M)",
            r"(\d+\.?\d*)\s*(billion|million|B|M)\s*in\s*revenue",
            r"revenue\s*(?:growth|increased|decreased)\s*(?:by|to)\s*(\d+\.?\d*)%",
        ]

        revenues = []
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                revenues.append(
                    {
                        "value": match.group(1),
                        "unit": (
                            match.group(2) if len(match.groups()) > 1 else "unknown"
                        ),
                        "context": text[max(0, match.start() - 50) : match.end() + 50],
                    }
                )

        return revenues

    def extract_eps(self, text: str) -> List[Dict]:
        """Extract EPS figures from transcript"""
        eps_patterns = [
            r"eps\s*(?:of|was|\$)\s*(\d+\.?\d*)",
            r"earnings\s*per\s*share\s*(?:of|was|\$)\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*eps",
        ]

        eps_figures = []
        for pattern in eps_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                eps_figures.append(
                    {
                        "value": float(match.group(1)),
                        "context": text[max(0, match.start() - 50) : match.end() + 50],
                    }
                )

        return eps_figures

    def extract_guidance(self, text: str) -> Dict[str, Any]:
        """Extract forward guidance from transcript"""
        guidance_keywords = [
            "guidance",
            "outlook",
            "expect",
            "forecast",
            "project",
            "anticipate",
            "target",
            "estimate",
        ]

        guidance_sentences = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in guidance_keywords):
                guidance_sentences.append(sentence.strip())

        return {
            "sentences": guidance_sentences,
            "confidence": len(guidance_sentences) / max(len(sentences), 1),
        }

    def extract_growth_rates(self, text: str) -> List[Dict]:
        """Extract growth rate mentions"""
        growth_pattern = r"(\d+\.?\d*)%\s*(?:growth|increase|decrease|change)"

        growth_rates = []
        matches = re.finditer(growth_pattern, text, re.IGNORECASE)

        for match in matches:
            growth_rates.append(
                {
                    "rate": float(match.group(1)),
                    "context": text[max(0, match.start() - 30) : match.end() + 30],
                }
            )

        return growth_rates

    def extract_announcements(self, text: str) -> List[str]:
        """Extract key announcements"""
        announcement_indicators = [
            "announce",
            "launch",
            "introduce",
            "release",
            "new",
            "partnership",
            "acquisition",
            "investment",
            "expansion",
        ]

        announcements = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if any(
                indicator in sentence.lower() for indicator in announcement_indicators
            ):
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
            text = segment.get("text", "").lower()

            # Simple speaker change detection based on content
            if i > 0:
                prev_text = segments[i - 1].get("text", "").lower()

                # Speaker change indicators
                change_indicators = [
                    "thank you",
                    "questions",
                    "operator",
                    "next question",
                    "good morning",
                    "good afternoon",
                    "hello everyone",
                ]

                if any(indicator in text for indicator in change_indicators):
                    current_speaker = f"Speaker_{len(set([s['speaker'] for s in speaker_segments])) + 1}"

            speaker_segments.append(
                {
                    **segment,
                    "speaker": current_speaker,
                    "speaker_role": self.infer_speaker_role(text, current_speaker),
                }
            )

        return speaker_segments

    def infer_speaker_role(self, text: str, speaker_id: str) -> str:
        """Infer speaker role based on content"""
        text_lower = text.lower()

        # CEO indicators
        if any(
            phrase in text_lower
            for phrase in [
                "our strategy",
                "company vision",
                "long-term",
                "shareholders",
                "transformative",
                "market leadership",
            ]
        ):
            return "CEO"

        # CFO indicators
        elif any(
            phrase in text_lower
            for phrase in [
                "financial results",
                "revenue",
                "eps",
                "margin",
                "guidance",
                "cash flow",
                "balance sheet",
                "capital allocation",
            ]
        ):
            return "CFO"

        # Analyst indicators
        elif any(
            phrase in text_lower
            for phrase in [
                "question",
                "could you",
                "can you",
                "what about",
                "how about",
            ]
        ):
            return "Analyst"

        # Operator indicators
        elif any(
            phrase in text_lower
            for phrase in ["thank you", "next question", "question and answer"]
        ):
            return "Operator"

        return "Unknown"


class AudioSentimentAnalyzer:
    """Analyze sentiment in audio segments"""

    def analyze_sentiment(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment per speaker and overall"""
        from textblob import TextBlob

        speaker_sentiments = {}
        overall_sentiment = {
            "positive_segments": 0,
            "negative_segments": 0,
            "neutral_segments": 0,
            "average_polarity": 0,
            "average_subjectivity": 0,
        }

        polarities = []
        subjectivities = []

        for segment in segments:
            text = segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")

            if text.strip():
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                polarities.append(polarity)
                subjectivities.append(subjectivity)

                # Categorize sentiment
                if polarity > 0.1:
                    sentiment = "positive"
                    overall_sentiment["positive_segments"] += 1
                elif polarity < -0.1:
                    sentiment = "negative"
                    overall_sentiment["negative_segments"] += 1
                else:
                    sentiment = "neutral"
                    overall_sentiment["neutral_segments"] += 1

                # Track by speaker
                if speaker not in speaker_sentiments:
                    speaker_sentiments[speaker] = {
                        "polarity_sum": 0,
                        "subjectivity_sum": 0,
                        "segment_count": 0,
                        "sentiment_distribution": {
                            "positive": 0,
                            "negative": 0,
                            "neutral": 0,
                        },
                    }

                speaker_sentiments[speaker]["polarity_sum"] += polarity
                speaker_sentiments[speaker]["subjectivity_sum"] += subjectivity
                speaker_sentiments[speaker]["segment_count"] += 1
                speaker_sentiments[speaker]["sentiment_distribution"][sentiment] += 1

        # Calculate averages
        if polarities:
            overall_sentiment["average_polarity"] = sum(polarities) / len(polarities)
            overall_sentiment["average_subjectivity"] = sum(subjectivities) / len(
                subjectivities
            )

        # Finalize speaker sentiments
        for speaker, data in speaker_sentiments.items():
            data["average_polarity"] = data["polarity_sum"] / data["segment_count"]
            data["average_subjectivity"] = (
                data["subjectivity_sum"] / data["segment_count"]
            )
            del data["polarity_sum"]
            del data["subjectivity_sum"]

        return {"overall": overall_sentiment, "by_speaker": speaker_sentiments}
