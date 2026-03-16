import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseConfig:
    """Base configuration shared across all environments"""

    # Base paths - using pathlib for better path handling
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"

    # API Keys (will be overridden by environment-specific configs)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Vector Database - environment specific
    VECTOR_STORE_PATH = str(DATA_DIR / "chroma_db")
    VECTOR_STORE_TYPE = "chroma"  # chroma, pinecone, weaviate, etc.

    # Data Paths
    RAW_DATA_PATH = str(DATA_DIR / "raw")
    PROCESSED_DATA_PATH = str(DATA_DIR / "processed")

    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default local model
    LLM_MODEL = "gpt-3.5-turbo"

    # Chunking Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval Settings
    TOP_K_RESULTS = 3

    # Application Settings
    DEBUG = False
    TESTING = False
    LOG_LEVEL = "INFO"


# Don't instantiate here - let __init__.py handle this
