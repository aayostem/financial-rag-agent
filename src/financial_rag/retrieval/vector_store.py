import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from loguru import logger
from financial_rag.config import config  # Fixed import


class VectorStoreManager:
    def __init__(self):
        self.embedding_model = self._initialize_embeddings()
        self.client = self._initialize_chroma()

    def _initialize_embeddings(self):
        """Initialize the embedding model based on config"""
        try:
            if config.EMBEDDING_MODEL.startswith("text-embedding"):
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(
                    model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY
                )
            else:
                logger.info(f"Using local embeddings: {config.EMBEDDING_MODEL}")
                return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    def _initialize_chroma(self):
        """Initialize ChromaDB client"""
        return chromadb.PersistentClient(
            path=config.VECTOR_STORE_PATH, settings=Settings(anonymized_telemetry=False)
        )

    def create_vector_store(self, documents):
        """Create a new vector store from documents"""
        try:
            logger.info("Creating vector store from documents")

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=config.VECTOR_STORE_PATH,
                client=self.client,
            )

            logger.success(f"Vector store created with {len(documents)} documents")
            return vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            vector_store = Chroma(
                persist_directory=config.VECTOR_STORE_PATH,
                embedding_function=self.embedding_model,
                client=self.client,
            )
            logger.info("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def get_retriever(
        self, vector_store, search_type="similarity", k=config.TOP_K_RESULTS
    ):
        """Create a retriever from vector store"""
        search_kwargs = {"k": k}

        if search_type == "mmr":  # Maximum Marginal Relevance
            search_kwargs["fetch_k"] = k * 2
            search_kwargs["lambda_mult"] = 0.7

        retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        return retriever
