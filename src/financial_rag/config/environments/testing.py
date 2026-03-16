from ..base import BaseConfig


class TestingConfig(BaseConfig):
    """Testing environment settings"""

    TESTING = True
    DEBUG = True

    # Use in-memory or temporary storage for tests
    VECTOR_STORE_PATH = ":memory:"  # For chroma in-memory
    DATA_DIR = "/tmp/finrag_test_data"

    # Fast, small models for tests
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Minimal settings for quick tests
    CHUNK_SIZE = 100
    TOP_K_RESULTS = 2

    # Disable external calls during tests
    MOCK_EXTERNAL_APIS = True
