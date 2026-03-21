-- HNSW index for fast approximate nearest neighbor search
-- Run after bulk data load for best performance
-- m=16: max connections per layer (higher = better recall, more memory)
-- ef_construction=64: search width during index build (higher = better quality)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw
ON financial_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
