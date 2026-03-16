import os
from ..base import BaseConfig


class ProductionConfig(BaseConfig):
    """Production environment settings"""

    # Production overrides
    DEBUG = False
    LOG_LEVEL = "WARNING"

    # Use production-ready models
    EMBEDDING_MODEL = os.getenv("PROD_EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("PROD_LLM_MODEL", "gpt-4")

    # Production vector store (cloud-based)
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pinecone")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")  # Cloud URL

    # Stricter production settings
    TOP_K_RESULTS = 3  # Optimize for latency/cost

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY")  # Must be set in environment
    REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "SECRET_KEY"]

    def __init__(self):
        # Validate required environment variables
        missing = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
