import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
from financial_rag.config import config  # Fixed import


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def process_sec_filing(self, file_path):
        """Process SEC filing text files"""
        try:
            logger.info(f"Processing SEC filing: {file_path}")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Clean SEC filing - remove XML/HTML tags and excessive whitespace
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text()

            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # Extract metadata
            metadata = self._extract_sec_metadata(text, file_path)

            return Document(page_content=text, metadata=metadata)

        except Exception as e:
            logger.error(f"Error processing SEC filing {file_path}: {str(e)}")
            raise

    def _extract_sec_metadata(self, text, file_path):
        """Extract metadata from SEC filing text"""
        metadata = {
            "source": file_path,
            "document_type": "SEC_FILING",
            "source_type": "structured",
        }

        # Extract company name (simplified)
        company_match = re.search(r"COMPANY CONFORMED NAME:\s*([^\n]+)", text)
        if company_match:
            metadata["company"] = company_match.group(1).strip()

        # Extract filing date
        date_match = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
        if date_match:
            metadata["filing_date"] = date_match.group(1)

        # Extract document type
        doc_match = re.search(r"CONFORMED SUBMISSION TYPE:\s*([^\n]+)", text)
        if doc_match:
            metadata["filing_type"] = doc_match.group(1).strip()

        return metadata

    def chunk_documents(self, documents):
        """Split documents into chunks using sophisticated strategy"""
        logger.info(f"Chunking {len(documents)} documents")

        chunks = self.text_splitter.split_documents(documents)

        logger.success(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
