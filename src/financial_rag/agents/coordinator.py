from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio
from loguru import logger
from enum import Enum

from financial_rag.agents.specialized_agents import (
    ResearchAnalystAgent,
    QuantitativeAnalystAgent,
    RiskOfficerAgent,
)


class AnalysisType(Enum):
    COMPREHENSIVE = "comprehensive"
    DEEP_RESEARCH = "deep_research"
    QUANTITATIVE = "quantitative"
    RISK_ASSESSMENT = "risk_assessment"
    VALUATION = "valuation"


class AgentCoordinator:
    """Coordinates multiple specialized agents for collaborative analysis"""

    def __init__(self, vector_store, enable_monitoring: bool = True):
        self.vector_store = vector_store
        self.enable_monitoring = enable_monitoring

        # Initialize specialized agents
        self.research_analyst = ResearchAnalystAgent(vector_store, enable_monitoring)
        self.quantitative_analyst = QuantitativeAnalystAgent(
            vector_store, enable_monitoring
        )
        self.risk_officer = RiskOfficerAgent(vector_store, enable_monitoring)

        self.agent_registry = {
            "research_analyst": self.research_analyst,
            "quantitative_analyst": self.quantitative_analyst,
            "risk_officer": self.risk_officer,
        }

        self.analysis_history = []
        logger.success("Multi-agent coordinator initialized")

    async def coordinate_analysis(
        self,
        ticker: str,
        analysis_type: AnalysisType,
        research_focus: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Coordinate analysis across multiple specialized agents"""
        try:
            logger.info(f"Coordinating {analysis_type.value} analysis for {ticker}")

            # Determine which agents to involve based on analysis type
            agents_to_use = self.select_agents_for_analysis(analysis_type)

            # Execute agent analyses in parallel
            agent_tasks = []
            for agent_name in agents_to_use:
                task = self.execute_agent_analysis(
                    agent_name, ticker, analysis_type, research_focus
                )
                agent_tasks.append(task)

            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            # Combine and synthesize results
            combined_analysis = await self.synthesize_agent_results(
                ticker, agents_to_use, agent_results, analysis_type
            )

            # Record analysis in history
            self.record_analysis_history(ticker, analysis_type, combined_analysis)

            return combined_analysis

        except Exception as e:
            logger.error(f"Error coordinating analysis for {ticker}: {e}")
            raise

    def select_agents_for_analysis(self, analysis_type: AnalysisType) -> List[str]:
        """Select which agents to involve based on analysis type"""
        agent_mapping = {
            AnalysisType.COMPREHENSIVE: [
                "research_analyst",
                "quantitative_analyst",
                "risk_officer",
            ],
            AnalysisType.DEEP_RESEARCH: ["research_analyst"],
            AnalysisType.QUANTITATIVE: ["quantitative_analyst"],
            AnalysisType.RISK_ASSESSMENT: ["risk_officer", "research_analyst"],
            AnalysisType.VALUATION: ["quantitative_analyst", "research_analyst"],
        }

        return agent_mapping.get(analysis_type, ["research_analyst"])

    async def execute_agent_analysis(
        self,
        agent_name: str,
        ticker: str,
        analysis_type: AnalysisType,
        research_focus: str,
    ) -> Dict[str, Any]:
        """Execute analysis using a specific agent"""
        try:
            agent = self.agent_registry[agent_name]

            if agent_name == "research_analyst":
                return await agent.conduct_deep_research(ticker, research_focus)

            elif agent_name == "quantitative_analyst":
                if analysis_type == AnalysisType.VALUATION:
                    return await agent.build_valuation_model(ticker, "dcf")
                else:
                    return await agent.analyze_financial_ratios(ticker)

            elif agent_name == "risk_officer":
                return await agent.conduct_risk_assessment(ticker)

            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        except Exception as e:
            logger.error(f"Error in {agent_name} analysis for {ticker}: {e}")
            return {
                "agent": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def synthesize_agent_results(
        self,
        ticker: str,
        agents_used: List[str],
        agent_results: List[Dict],
        analysis_type: AnalysisType,
    ) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        synthesized = {
            "ticker": ticker,
            "analysis_type": analysis_type.value,
            "agents_involved": agents_used,
            "agent_results": {},
            "consensus_analysis": {},
            "conflicting_viewpoints": [],
            "overall_recommendation": {},
            "synthesis_timestamp": datetime.now().isoformat(),
        }

        # Organize results by agent
        for i, agent_name in enumerate(agents_used):
            if i < len(agent_results) and not isinstance(agent_results[i], Exception):
                synthesized["agent_results"][agent_name] = agent_results[i]

        # Generate consensus analysis
        synthesized["consensus_analysis"] = self.generate_consensus(
            synthesized["agent_results"]
        )

        # Identify conflicts
        synthesized["conflicting_viewpoints"] = self.identify_conflicts(
            synthesized["agent_results"]
        )

        # Generate overall recommendation
        synthesized["overall_recommendation"] = self.generate_overall_recommendation(
            synthesized["agent_results"], analysis_type
        )

        return synthesized

    def generate_consensus(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consensus across agent analyses"""
        consensus = {
            "key_strengths": [],
            "key_risks": [],
            "valuation_outlook": "neutral",
            "risk_profile": "medium",
            "growth_prospects": "moderate",
        }

        # Aggregate strengths and risks from all agents
        all_strengths = []
        all_risks = []

        for agent_name, results in agent_results.items():
            if agent_name == "research_analyst":
                implications = results.get("investment_implications", {})
                all_strengths.extend(implications.get("positive_factors", []))
                all_risks.extend(implications.get("negative_factors", []))

            elif agent_name == "quantitative_analyst":
                # Extract quantitative insights
                ratio_analysis = results.get("ratio_interpretation", {})
                if "strong" in str(ratio_analysis).lower():
                    all_strengths.append("Strong financial ratios")

            elif agent_name == "risk_officer":
                risk_profile = results.get("overall_risk_profile", {})
                if risk_profile.get("overall_risk_score", 0) > 0.7:
                    all_risks.append("Elevated risk profile")

        # Deduplicate and select top items
        consensus["key_strengths"] = list(set(all_strengths))[:5]
        consensus["key_risks"] = list(set(all_risks))[:5]

        return consensus

    def identify_conflicts(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify conflicting viewpoints between agents"""
        conflicts = []

        # Compare research analyst and quantitative analyst views
        research_view = agent_results.get("research_analyst", {})
        quant_view = agent_results.get("quantitative_analyst", {})

        if research_view and quant_view:
            research_implication = research_view.get("investment_implications", {}).get(
                "overall_implication"
            )
            quant_ratios = quant_view.get("ratio_interpretation", {})

            # Check for fundamental vs quantitative disagreement
            if (
                research_implication == "bullish"
                and "weak" in str(quant_ratios).lower()
            ):
                conflicts.append(
                    {
                        "type": "fundamental_quantitative_mismatch",
                        "description": "Research analysis is bullish but quantitative metrics show weakness",
                        "agents_involved": ["research_analyst", "quantitative_analyst"],
                        "resolution_suggestion": "Reconcile qualitative strengths with quantitative weaknesses",
                    }
                )

        return conflicts

    def generate_overall_recommendation(
        self, agent_results: Dict[str, Any], analysis_type: AnalysisType
    ) -> Dict[str, Any]:
        """Generate overall investment recommendation"""
        recommendation_scores = {
            "strong_buy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strong_sell": 0,
        }

        # Score recommendations from each agent
        for agent_name, results in agent_results.items():
            agent_score = self.score_agent_recommendation(agent_name, results)
            recommendation_scores[agent_score] += 1

        # Determine overall recommendation
        max_score = max(recommendation_scores.values())
        overall_recommendation = [
            rec for rec, score in recommendation_scores.items() if score == max_score
        ][0]

        confidence = max_score / len(agent_results) if agent_results else 0.5

        return {
            "action": overall_recommendation,
            "confidence": confidence,
            "supporting_agents": [
                agent
                for agent, results in agent_results.items()
                if self.score_agent_recommendation(agent, results)
                == overall_recommendation
            ],
            "rationale": self.generate_recommendation_rationale(
                agent_results, overall_recommendation
            ),
            "time_horizon": self.determine_time_horizon(analysis_type),
        }

    def score_agent_recommendation(self, agent_name: str, results: Dict) -> str:
        """Convert agent analysis to recommendation score"""
        if agent_name == "research_analyst":
            implication = results.get("investment_implications", {}).get(
                "overall_implication"
            )
            return (
                "buy"
                if implication == "bullish"
                else "sell" if implication == "bearish" else "hold"
            )

        elif agent_name == "quantitative_analyst":
            ratios = results.get("ratio_interpretation", {})
            if "strong" in str(ratios).lower():
                return "buy"
            elif "weak" in str(ratios).lower():
                return "sell"
            else:
                return "hold"

        elif agent_name == "risk_officer":
            risk_score = results.get("overall_risk_profile", {}).get(
                "overall_risk_score", 0.5
            )
            if risk_score > 0.7:
                return "sell"
            elif risk_score < 0.3:
                return "buy"
            else:
                return "hold"

        return "hold"

    def generate_recommendation_rationale(
        self, agent_results: Dict, recommendation: str
    ) -> str:
        """Generate rationale for the overall recommendation"""
        rationales = []

        for agent_name, results in agent_results.items():
            agent_rationale = self.get_agent_rationale(
                agent_name, results, recommendation
            )
            if agent_rationale:
                rationales.append(agent_rationale)

        return (
            ". ".join(rationales)
            if rationales
            else "Insufficient consensus for strong recommendation"
        )

    def get_agent_rationale(
        self, agent_name: str, results: Dict, overall_rec: str
    ) -> Optional[str]:
        """Get rationale from specific agent's analysis"""
        agent_rec = self.score_agent_recommendation(agent_name, results)

        if agent_rec == overall_rec:
            if agent_name == "research_analyst":
                return (
                    "Research analysis supports this view based on fundamental factors"
                )
            elif agent_name == "quantitative_analyst":
                return "Quantitative metrics align with this recommendation"
            elif agent_name == "risk_officer":
                return "Risk assessment consistent with this position"

        return None

    def determine_time_horizon(self, analysis_type: AnalysisType) -> str:
        """Determine appropriate time horizon for recommendation"""
        horizon_map = {
            AnalysisType.COMPREHENSIVE: "long_term",
            AnalysisType.DEEP_RESEARCH: "long_term",
            AnalysisType.QUANTITATIVE: "medium_term",
            AnalysisType.RISK_ASSESSMENT: "short_term",
            AnalysisType.VALUATION: "medium_term",
        }

        return horizon_map.get(analysis_type, "medium_term")

    def record_analysis_history(
        self, ticker: str, analysis_type: AnalysisType, analysis_result: Dict[str, Any]
    ):
        """Record analysis in history for tracking and learning"""
        history_entry = {
            "ticker": ticker,
            "analysis_type": analysis_type.value,
            "timestamp": datetime.now().isoformat(),
            "agents_used": analysis_result.get("agents_involved", []),
            "recommendation": analysis_result.get("overall_recommendation", {}).get(
                "action"
            ),
            "confidence": analysis_result.get("overall_recommendation", {}).get(
                "confidence"
            ),
            "key_findings": analysis_result.get("consensus_analysis", {}).get(
                "key_strengths", []
            )[:3],
        }

        self.analysis_history.append(history_entry)

        # Keep only recent history (last 100 analyses)
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

    async def get_analysis_history(self, ticker: str = None) -> List[Dict[str, Any]]:
        """Get analysis history, optionally filtered by ticker"""
        if ticker:
            return [
                entry for entry in self.analysis_history if entry["ticker"] == ticker
            ]
        else:
            return self.analysis_history

    async def conduct_investment_committee(self, ticker: str) -> Dict[str, Any]:
        """Simulate investment committee meeting with all agents"""
        try:
            logger.info(f"Conducting investment committee for {ticker}")

            # Get analyses from all agents
            research_analysis = await self.research_analyst.conduct_deep_research(
                ticker
            )
            quant_analysis = await self.quantitative_analyst.analyze_financial_ratios(
                ticker
            )
            risk_analysis = await self.risk_officer.conduct_risk_assessment(ticker)

            # Generate committee discussion
            committee_discussion = self.simulate_committee_discussion(
                research_analysis, quant_analysis, risk_analysis
            )

            # Reach committee decision
            committee_decision = self.reach_committee_decision(
                research_analysis, quant_analysis, risk_analysis, committee_discussion
            )

            return {
                "ticker": ticker,
                "committee_members": list(self.agent_registry.keys()),
                "analyses_presented": {
                    "research": research_analysis,
                    "quantitative": quant_analysis,
                    "risk": risk_analysis,
                },
                "committee_discussion": committee_discussion,
                "final_decision": committee_decision,
                "meeting_minutes": self.generate_meeting_minutes(
                    ticker, committee_discussion, committee_decision
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in investment committee for {ticker}: {e}")
            raise

    def simulate_committee_discussion(
        self, research: Dict, quant: Dict, risk: Dict
    ) -> List[Dict[str, str]]:
        """Simulate discussion between committee members"""
        discussion = []

        # Research analyst presents findings
        research_implication = research.get("investment_implications", {}).get(
            "overall_implication"
        )
        discussion.append(
            {
                "speaker": "research_analyst",
                "message": f"Based on my comprehensive research, I'm {research_implication} on this investment due to...",
            }
        )

        # Quantitative analyst responds
        ratio_health = (
            "strong"
            if "strong" in str(quant.get("ratio_interpretation")).lower()
            else "mixed"
        )
        discussion.append(
            {
                "speaker": "quantitative_analyst",
                "message": f"The quantitative metrics show {ratio_health} financial health. Key ratios indicate...",
            }
        )

        # Risk officer provides risk perspective
        risk_score = risk.get("overall_risk_profile", {}).get("overall_risk_score", 0.5)
        risk_level = (
            "elevated"
            if risk_score > 0.6
            else "moderate" if risk_score > 0.4 else "low"
        )
        discussion.append(
            {
                "speaker": "risk_officer",
                "message": f"My risk assessment shows {risk_level} overall risk. Key concerns include...",
            }
        )

        return discussion

    def reach_committee_decision(
        self, research: Dict, quant: Dict, risk: Dict, discussion: List[Dict]
    ) -> Dict[str, Any]:
        """Reach final committee decision"""
        # Simple voting mechanism
        votes = {
            "research_analyst": self.score_agent_recommendation(
                "research_analyst", research
            ),
            "quantitative_analyst": self.score_agent_recommendation(
                "quantitative_analyst", quant
            ),
            "risk_officer": self.score_agent_recommendation("risk_officer", risk),
        }

        # Count votes
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Determine decision (simple majority)
        decision = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[decision] / len(votes)

        return {
            "decision": decision,
            "confidence": confidence,
            "vote_breakdown": votes,
            "unanimous": len(set(votes.values())) == 1,
            "implementation_guidance": self.generate_implementation_guidance(
                decision, confidence
            ),
        }

    def generate_implementation_guidance(self, decision: str, confidence: float) -> str:
        """Generate guidance for implementing the decision"""
        guidance_map = {
            "strong_buy": "Consider aggressive position building with staggered entry",
            "buy": "Establish core position with potential for tactical additions",
            "hold": "Maintain current exposure with close monitoring",
            "sell": "Reduce exposure systematically while monitoring for better exit points",
            "strong_sell": "Expedite position reduction with tight risk controls",
        }

        base_guidance = guidance_map.get(decision, "Maintain current strategy")

        if confidence > 0.8:
            return f"High confidence: {base_guidance}"
        elif confidence > 0.6:
            return f"Moderate confidence: {base_guidance} with regular review"
        else:
            return f"Low confidence: {base_guidance}. Reassess frequently."

    def generate_meeting_minutes(
        self, ticker: str, discussion: List[Dict], decision: Dict
    ) -> str:
        """Generate formal meeting minutes"""
        minutes = f"INVESTMENT COMMITTEE MEETING MINUTES\n"
        minutes += f"Ticker: {ticker}\n"
        minutes += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        minutes += "COMMITTEE DISCUSSION:\n"
        for i, turn in enumerate(discussion, 1):
            minutes += (
                f"{i}. {turn['speaker'].replace('_', ' ').title()}: {turn['message']}\n"
            )

        minutes += (
            f"\nFINAL DECISION: {decision['decision'].replace('_', ' ').title()}\n"
        )
        minutes += f"Confidence: {decision['confidence']:.0%}\n"
        minutes += f"Implementation: {decision['implementation_guidance']}\n"

        return minutes
