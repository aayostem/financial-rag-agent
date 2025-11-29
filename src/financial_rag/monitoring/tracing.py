import wandb
from loguru import logger
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from financial_rag.config import config


class AgentMonitor:
    """Monitoring and tracing for the financial agent"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.wandb_run = None

        if (
            enabled
            and config.WANDB_API_KEY
            and config.WANDB_API_KEY != "your_wandb_api_key_here"
        ):
            try:
                self.wandb_run = wandb.init(
                    project="financial-rag-agent",
                    config={
                        "embedding_model": config.EMBEDDING_MODEL,
                        "llm_model": config.LLM_MODEL,
                        "chunk_size": config.CHUNK_SIZE,
                        "top_k_results": config.TOP_K_RESULTS,
                    },
                )
                logger.success("Weights & Biases monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.enabled = False
        else:
            logger.info("Monitoring disabled - no WandB API key found")

    def log_retrieval(self, query: str, documents: List, scores: List[float]):
        """Log retrieval performance"""
        if not self.enabled:
            return

        try:
            retrieval_metrics = {
                "retrieval/query_length": len(query),
                "retrieval/documents_retrieved": len(documents),
                "retrieval/avg_score": sum(scores) / len(scores) if scores else 0,
                "retrieval/max_score": max(scores) if scores else 0,
                "retrieval/timestamp": datetime.now().isoformat(),
            }

            # Log source distribution
            sources = {}
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

            if self.wandb_run:
                self.wandb_run.log(retrieval_metrics)

            logger.debug(
                f"Retrieval logged: {len(documents)} docs for query: {query[:50]}..."
            )

        except Exception as e:
            logger.error(f"Error logging retrieval: {e}")

    def log_llm_call(
        self, prompt: str, response: str, latency: float, token_usage: Dict = None
    ):
        """Log LLM call details"""
        if not self.enabled:
            return

        try:
            llm_metrics = {
                "llm/prompt_length": len(prompt),
                "llm/response_length": len(response),
                "llm/latency_seconds": latency,
                "llm/timestamp": datetime.now().isoformat(),
            }

            if token_usage:
                llm_metrics.update(
                    {
                        "llm/prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "llm/completion_tokens": token_usage.get(
                            "completion_tokens", 0
                        ),
                        "llm/total_tokens": token_usage.get("total_tokens", 0),
                    }
                )

            if self.wandb_run:
                self.wandb_run.log(llm_metrics)

            logger.debug(
                f"LLM call logged: {len(prompt)} chars -> {len(response)} chars in {latency:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error logging LLM call: {e}")

    def log_agent_step(
        self,
        step_type: str,
        tool_name: str,
        input_data: str,
        output: str,
        success: bool,
    ):
        """Log agent tool usage steps"""
        if not self.enabled:
            return

        try:
            agent_metrics = {
                f"agent/{step_type}_tool": tool_name,
                f"agent/{step_type}_input_length": len(input_data),
                f"agent/{step_type}_output_length": len(output),
                f"agent/{step_type}_success": success,
                f"agent/{step_type}_timestamp": datetime.now().isoformat(),
            }

            if self.wandb_run:
                self.wandb_run.log(agent_metrics)

            logger.debug(f"Agent step logged: {tool_name} - Success: {success}")

        except Exception as e:
            logger.error(f"Error logging agent step: {e}")

    def log_query_analysis(
        self,
        question: str,
        answer: str,
        total_latency: float,
        source_count: int,
        agent_type: str,
    ):
        """Log complete query analysis"""
        if not self.enabled:
            return

        try:
            query_metrics = {
                "query/question_length": len(question),
                "query/answer_length": len(answer),
                "query/total_latency_seconds": total_latency,
                "query/source_count": source_count,
                "query/agent_type": agent_type,
                "query/success": len(answer) > 0,
                "query/timestamp": datetime.now().isoformat(),
            }

            if self.wandb_run:
                self.wandb_run.log(query_metrics)

                # Log the actual Q&A for analysis
                self.wandb_run.log(
                    {
                        "query_samples": wandb.Table(
                            columns=["Question", "Answer", "Latency", "Sources"],
                            data=[[question, answer, total_latency, source_count]],
                        )
                    }
                )

            logger.info(
                f"Query analysis logged: {question[:50]}... -> {len(answer)} chars in {total_latency:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error logging query analysis: {e}")

    def cleanup(self):
        """Clean up monitoring resources"""
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("Monitoring session ended")
