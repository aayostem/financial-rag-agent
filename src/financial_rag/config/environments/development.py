from ..base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development environment settings"""

    # Override base settings for development
    DEBUG = True
    LOG_LEVEL = "DEBUG"

    # Use local/cheaper models for development
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Keep local for speed
    LLM_MODEL = "gpt-3.5-turbo"  # Cheaper for testing

    # Local vector store
    VECTOR_STORE_TYPE = "chroma"
    VECTOR_STORE_PATH = str(BaseConfig.DATA_DIR / "chroma_db_dev")

    # More verbose retrieval for debugging
    TOP_K_RESULTS = 5

    # Development-specific settings
    RELOAD_ON_CHANGE = True
    SECRET_KEY = "dev-secret-key-change-in-production"
