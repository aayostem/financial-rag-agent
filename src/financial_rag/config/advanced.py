import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, validator
from loguru import logger


class AdvancedConfig(BaseSettings):
    """Advanced configuration with validation"""

    # API Settings
    OPENAI_API_KEY: str
    WANDB_API_KEY: Optional[str] = None

    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    SEARCH_TYPE: str = "similarity"  # "similarity" or "mmr"

    # Agent Settings
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_ENABLE_MONITORING: bool = True

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_LOG_LEVEL: str = "info"

    # Storage Settings
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    RAW_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"

    # Kubernetes Settings
    K8S_NAMESPACE: str = "financial-rag"
    K8S_DEPLOYMENT_NAME: str = "financial-rag-api"

    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = True
    WANDB_ENABLED: bool = True

    @validator("CHUNK_SIZE")
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError("CHUNK_SIZE must be between 100 and 2000")
        return v

    @validator("LLM_TEMPERATURE")
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 1")
        return v

    @validator("TOP_K_RESULTS")
    def validate_top_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError("TOP_K_RESULTS must be between 1 and 10")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global advanced config
advanced_config = AdvancedConfig()
