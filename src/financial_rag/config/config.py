import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Vector Database
    VECTOR_STORE_PATH = "./data/chroma_db"

    # Data Paths
    RAW_DATA_PATH = "./data/raw"
    PROCESSED_DATA_PATH = "./data/processed"

    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model
    # EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
    LLM_MODEL = "gpt-3.5-turbo"  # Start with 3.5, upgrade to gpt-4 later

    # Chunking Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval Settings
    TOP_K_RESULTS = 3


config = Config()
