
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

CREATE TABLE filings (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker           VARCHAR(10) NOT NULL,
    filing_type      VARCHAR(20) NOT NULL,
    fiscal_year      SMALLINT,
    fiscal_quarter   SMALLINT    CHECK (fiscal_quarter BETWEEN 1 AND 4),
    filed_at         DATE,
    source_url       TEXT,
    file_hash        VARCHAR(64) UNIQUE,
    pages            INTEGER,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested_by      VARCHAR(100),
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    CONSTRAINT filings_filing_type_check
        CHECK (filing_type IN ('10-K', '10-Q', '8-K', '20-F', 'DEF 14A', 'S-1'))
);

CREATE INDEX idx_filings_ticker        ON filings (ticker, fiscal_year DESC);
CREATE INDEX idx_filings_type_year     ON filings (filing_type, fiscal_year DESC);
CREATE INDEX idx_filings_active        ON filings (ticker) WHERE is_active = TRUE;

CREATE TABLE financial_chunks (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id       UUID         NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    ticker          VARCHAR(10)  NOT NULL,
    filing_type     VARCHAR(20)  NOT NULL,
    fiscal_year     SMALLINT,
    section         VARCHAR(100),
    chunk_index     INTEGER      NOT NULL,
    chunk_text      TEXT         NOT NULL,
    token_count     INTEGER,
    embedding       vector(3072) NOT NULL,
    metrics         JSONB        NOT NULL DEFAULT '{}',
    entities        JSONB        NOT NULL DEFAULT '{}',
    sentiment_score REAL         CHECK (sentiment_score BETWEEN -1.0 AND 1.0),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(50)  NOT NULL DEFAULT 'text-embedding-3-large',
    CONSTRAINT financial_chunks_chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT financial_chunks_token_count_positive CHECK (token_count IS NULL OR token_count > 0)
);

CREATE INDEX idx_chunks_ticker_year    ON financial_chunks (ticker, fiscal_year DESC);
CREATE INDEX idx_chunks_filing_section ON financial_chunks (filing_id, section);
CREATE INDEX idx_chunks_filing_type    ON financial_chunks (filing_type, ticker);
CREATE INDEX idx_chunks_metrics        ON financial_chunks USING GIN (metrics);
CREATE INDEX idx_chunks_entities       ON financial_chunks USING GIN (entities);
CREATE INDEX idx_chunks_text_trgm      ON financial_chunks USING GIN (chunk_text gin_trgm_ops);

CREATE TABLE analysis_history (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker          VARCHAR(10),
    question        TEXT        NOT NULL,
    answer          TEXT        NOT NULL,
    analysis_style  VARCHAR(20) NOT NULL DEFAULT 'analyst',
    agent_type      VARCHAR(50) NOT NULL,
    search_type     VARCHAR(20) NOT NULL DEFAULT 'similarity',
    latency_ms      INTEGER     NOT NULL,
    source_chunk_ids UUID[]     DEFAULT '{}',
    real_time_used  BOOLEAN     NOT NULL DEFAULT FALSE,
    error           TEXT,
    session_id      UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT analysis_style_check
        CHECK (analysis_style IN ('analyst', 'executive', 'risk')),
    CONSTRAINT search_type_check
        CHECK (search_type IN ('similarity', 'mmr'))
);

CREATE INDEX idx_analysis_ticker_time ON analysis_history (ticker, created_at DESC);
CREATE INDEX idx_analysis_session     ON analysis_history (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_analysis_agent       ON analysis_history (agent_type, created_at DESC);

CREATE TABLE schema_migrations (
    version     VARCHAR(20) PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial schema: filings, financial_chunks, analysis_history');
