#!/bin/bash
# =============================================================================
# Financial RAG Agent — Project Scaffold
# Run from the directory WHERE you want the project created.
# Usage: bash scaffold.sh
# =============================================================================
set -euo pipefail

PROJECT="financial-rag-agent"
SRC="$PROJECT/src/financial_rag"

echo "🏗️  Scaffolding $PROJECT..."

# =============================================================================
# Directory tree
# =============================================================================
dirs=(
    # Infrastructure
    # "$PROJECT/infrastructure/docker/init"

    # Scripts
    # "$PROJECT/scripts"

    # Source — core
    "$SRC/config"
    "$SRC/storage/repositories"
    "$SRC/ingestion/parsers"
    "$SRC/processing"
    "$SRC/retrieval"
    "$SRC/api"
    "$SRC/utils"
    "$SRC/monitoring"

    # Tests
    "$PROJECT/tests/unit"
    "$PROJECT/tests/integration"
)

for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done

echo "  ✔ Directories created"

# # =============================================================================
# # Touch all Python files
# # =============================================================================
py_files=(
#     # Package roots
#     "$SRC/__init__.py"
#     "$SRC/__version__.py"

#     # Config
#     "$SRC/config/__init__.py"
#     "$SRC/config/settings.py"

    # Storage
    "$SRC/storage/__init__.py"
    "$SRC/storage/database.py"
    "$SRC/storage/cache.py"
    "$SRC/storage/vector_store.py"
    "$SRC/storage/repositories/__init__.py"
    "$SRC/storage/repositories/base.py"
    "$SRC/storage/repositories/chunks.py"
    "$SRC/storage/repositories/filings.py"
    "$SRC/storage/repositories/analysis.py"

#     # Ingestion
#     "$SRC/ingestion/__init__.py"
#     "$SRC/ingestion/sec_ingestor.py"
#     "$SRC/ingestion/yfinance_ingestor.py"
#     "$SRC/ingestion/parsers/__init__.py"
#     "$SRC/ingestion/parsers/html_parser.py"
#     "$SRC/ingestion/parsers/text_parser.py"

#     # Processing
#     "$SRC/processing/__init__.py"
#     "$SRC/processing/text_processor.py"

#     # Retrieval
#     "$SRC/retrieval/__init__.py"
#     "$SRC/retrieval/embeddings.py"
#     "$SRC/retrieval/document_retriever.py"
#     "$SRC/retrieval/hybrid_search.py"
#     "$SRC/retrieval/query_engine.py"

#     # API
#     "$SRC/api/__init__.py"
#     "$SRC/api/server.py"
#     "$SRC/api/routes.py"
#     "$SRC/api/dependencies.py"
#     "$SRC/api/middleware.py"
#     "$SRC/api/models.py"

#     # Utils
#     "$SRC/utils/__init__.py"
#     "$SRC/utils/constants.py"
#     "$SRC/utils/exceptions.py"
#     "$SRC/utils/decorators.py"
#     "$SRC/utils/helpers.py"

#     # Monitoring
#     "$SRC/monitoring/__init__.py"
#     "$SRC/monitoring/logging.py"
#     "$SRC/monitoring/metrics.py"
#     "$SRC/monitoring/tracing.py"

#     # Tests
#     "$PROJECT/tests/__init__.py"
#     "$PROJECT/tests/conftest.py"
#     "$PROJECT/tests/unit/__init__.py"
#     "$PROJECT/tests/unit/test_config.py"
#     "$PROJECT/tests/unit/test_storage.py"
#     "$PROJECT/tests/unit/test_processing.py"
#     "$PROJECT/tests/unit/test_retrieval.py"
#     "$PROJECT/tests/integration/__init__.py"
#     "$PROJECT/tests/integration/test_ingestion.py"
#     "$PROJECT/tests/integration/test_pipeline.py"
)

for f in "${py_files[@]}"; do
    touch "$f"
done

echo "  ✔ Python files created"

# # # =============================================================================
# # Touch SQL + shell scripts
# # =============================================================================
# other_files=(
#     "$PROJECT/infrastructure/docker/docker-compose.yml"
#     "$PROJECT/infrastructure/docker/init/create_schema.sql"
#     "$PROJECT/infrastructure/docker/init/create_hnsw_index.sql"
#     "$PROJECT/scripts/start_dev.sh"
#     "$PROJECT/scripts/stop_dev.sh"
#     "$PROJECT/scripts/run_migrations.sh"
#     "$PROJECT/.env.example"
#     "$PROJECT/.gitignore"
#     "$PROJECT/.pre-commit-config.yaml"
#     "$PROJECT/pyproject.toml"
#     "$PROJECT/README.md"
# )

# for f in "${other_files[@]}"; do
#     touch "$f"
# done

# echo "  ✔ Config and script files created"

# # =============================================================================
# # Seed __version__.py
# # =============================================================================
# cat > "$SRC/__version__.py" << 'EOF'
# __version__ = "0.1.0"
# __author__  = "Financial RAG Team"
# EOF

# # =============================================================================
# # Seed .gitignore
# # =============================================================================
# cat > "$PROJECT/.gitignore" << 'EOF'
# # Environment
# .env
# .env.*
# !.env.example

# # Python
# __pycache__/
# *.py[cod]
# *.pyo
# .Python
# *.egg-info/
# dist/
# build/
# .eggs/

# # Virtual environments
# .venv/
# venv/
# env/

# # Testing
# .pytest_cache/
# .coverage
# htmlcov/
# .mypy_cache/
# .ruff_cache/

# # IDE
# .vscode/
# .idea/
# *.swp
# *.swo

# # OS
# .DS_Store
# Thumbs.db

# # Data / models (never commit)
# data/
# *.pkl
# *.bin
# *.pt
# EOF

# # =============================================================================
# # Seed pyproject.toml
# # =============================================================================
# cat > "$PROJECT/pyproject.toml" << 'EOF'
# [build-system]
# requires      = ["hatchling"]
# build-backend = "hatchling.build"

# [project]
# name        = "financial-rag-agent"
# version     = "0.1.0"
# description = "Production-grade financial RAG pipeline"
# readme      = "README.md"
# requires-python = ">=3.11"

# dependencies = [
#     # API
#     "fastapi>=0.111.0",
#     "uvicorn[standard]>=0.30.0",
#     "pydantic>=2.7.0",
#     "pydantic-settings>=2.3.0",

#     # Database
#     "sqlalchemy[asyncio]>=2.0.30",
#     "asyncpg>=0.29.0",
#     "pgvector>=0.3.0",

#     # Cache
#     "redis[hiredis]>=5.0.0",

#     # LLM + Embeddings
#     "openai>=1.35.0",
#     "tiktoken>=0.7.0",

#     # Ingestion
#     "httpx>=0.27.0",
#     "beautifulsoup4>=4.12.0",
#     "lxml>=5.2.0",
#     "yfinance>=0.2.40",

#     # Processing
#     "structlog>=24.2.0",
#     "tenacity>=8.3.0",
# ]

# [project.optional-dependencies]
# dev = [
#     "pytest>=8.2.0",
#     "pytest-asyncio>=0.23.0",
#     "pytest-cov>=5.0.0",
#     "mypy>=1.10.0",
#     "ruff>=0.4.0",
#     "pre-commit>=3.7.0",
# ]

# [tool.hatch.build.targets.wheel]
# packages = ["src/financial_rag"]

# [tool.ruff]
# line-length    = 100
# target-version = "py311"

# [tool.ruff.lint]
# select = ["E", "F", "I", "UP", "B", "SIM", "TCH"]

# [tool.mypy]
# python_version         = "3.11"
# strict                 = true
# ignore_missing_imports = true

# [tool.pytest.ini_options]
# asyncio_mode = "auto"
# testpaths    = ["tests"]
# addopts      = "--cov=src --cov-report=term-missing"

# [tool.coverage.run]
# source = ["src"]
# omit   = ["*/tests/*", "*/__version__.py"]
# EOF

# # =============================================================================
# # Seed .pre-commit-config.yaml
# # =============================================================================
# cat > "$PROJECT/.pre-commit-config.yaml" << 'EOF'
# repos:
#   - repo: https://github.com/astral-sh/ruff-pre-commit
#     rev: v0.4.9
#     hooks:
#       - id: ruff
#         args: [--fix]
#       - id: ruff-format

#   - repo: https://github.com/pre-commit/mirrors-mypy
#     rev: v1.10.0
#     hooks:
#       - id: mypy
#         additional_dependencies: [pydantic, sqlalchemy]
# EOF

# # =============================================================================
# # Make scripts executable
# # =============================================================================
# chmod +x "$PROJECT/scripts/start_dev.sh"
# chmod +x "$PROJECT/scripts/stop_dev.sh"
# chmod +x "$PROJECT/scripts/run_migrations.sh"

# # =============================================================================
# # Done
# # =============================================================================
# echo ""
# echo "✅ Scaffold complete: $PROJECT/"
# echo ""
# echo "Next steps:"
# echo "  cd $PROJECT"
# echo "  python -m venv .venv && source .venv/Scripts/activate  # Windows"
# echo "  pip install -e '.[dev]'"
# echo "  pre-commit install"
# echo "  cp .env.example .env  # then fill in secrets"
# echo ""
# echo "Then copy your infrastructure files:"
# echo "  docker-compose.yml    → infrastructure/docker/"
# echo "  create_schema.sql     → infrastructure/docker/init/"
# echo "  create_hnsw_index.sql → infrastructure/docker/init/"