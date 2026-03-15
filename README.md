<div align="center">

<br />

```
███████╗██╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗██╗ █████╗ ██╗      
██╔════╝██║████╗  ██║██╔══██╗████╗  ██║██╔════╝██║██╔══██╗██║      
█████╗  ██║██╔██╗ ██║███████║██╔██╗ ██║██║     ██║███████║██║      
██╔══╝  ██║██║╚██╗██║██╔══██║██║╚██╗██║██║     ██║██╔══██║██║      
██║     ██║██║ ╚████║██║  ██║██║ ╚████║╚██████╗██║██║  ██║███████╗ 
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝
                                                                      
    ██████╗  █████╗  ██████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
    ██╔══██╗██╔══██╗██╔════╝     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
    ██████╔╝███████║██║  ███╗    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
    ██╔══██╗██╔══██║██║   ██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
    ██║  ██║██║  ██║╚██████╔╝    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
```

**Multi-modal, multi-agent RAG system for financial intelligence — real-time, predictive, production-ready**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-844FBA?style=flat-square&logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-production-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Helm](https://img.shields.io/badge/Helm-packaged-0F1689?style=flat-square&logo=helm&logoColor=white)](https://helm.sh/)
[![Docker](https://img.shields.io/badge/Docker-containerised-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](./LICENSE)

<br />

[**Docs**](./docs/) · [**Helm Guide**](./README-HELM.md) · [**Roadmap**](./ROADMAP.md) · [**Changelog**](./CHANGELOG.md) · [**Contributing**](./CONTRIBUTING.md)

<br />

</div>

---

## What This Is

`financial-rag-agent` is a production-grade Retrieval-Augmented Generation system purpose-built for financial analysis. It goes beyond a basic RAG pipeline — the architecture supports multi-agent orchestration, multi-modal document understanding, real-time data ingestion, and predictive analytics on top of retrieved context.

Built for the kind of financial intelligence work that requires grounding LLM outputs in real documents, filings, earnings transcripts, and live market data — while being fully deployable to Kubernetes via Helm.

---

## Capabilities

| Capability | Description |
|-----------|-------------|
| **Multi-agent orchestration** | Parallel agent execution with task decomposition and result synthesis |
| **Multi-modal understanding** | Process text, tables, charts, and embedded data in financial documents |
| **Real-time ingestion** | Stream live financial data into the retrieval pipeline |
| **Predictive analysis** | Augment RAG outputs with forward-looking quantitative signals |
| **Production serving** | Containerised, Helm-packaged, Kubernetes-native deployment |
| **Foundation validation** | Hardened retrieval quality checks before agent execution |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        financial-rag-agent                               │
├────────────────────────────────────────────────────────────────────────── │
│                                                                          │
│   ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐   │
│   │  Data Ingestion │────▶│  Vector Store /  │────▶│  RAG Pipeline  │   │
│   │                 │     │  Retrieval Layer  │     │                │   │
│   │  • Real-time    │     │                  │     │  • Reranking   │   │
│   │  • Batch docs   │     │  • Embeddings    │     │  • Context     │   │
│   │  • Multi-modal  │     │  • Semantic search│     │    assembly    │   │
│   └─────────────────┘     └──────────────────┘     └───────┬────────┘   │
│                                                            │             │
│                                                   ┌────────▼────────┐   │
│                                                   │  Agent Layer    │   │
│                                                   │                 │   │
│                                                   │  • Foundation   │   │
│                                                   │  • Multi-agency │   │
│                                                   │  • Predictive   │   │
│                                                   │  • Multi-modal  │   │
│                                                   └────────┬────────┘   │
│                                                            │             │
│                                                   ┌────────▼────────┐   │
│                                                   │    Response     │   │
│                                                   │    Synthesis    │   │
│                                                   └─────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
financial-rag-agent/
├── .github/workflows/          # CI/CD pipelines (lint, test, build, deploy)
├── docs/                       # Architecture documentation
├── helm/                       # Helm chart for Kubernetes deployment
├── kubernetes/                 # Raw Kubernetes manifests
├── notebooks/                  # Jupyter notebooks for RAG experimentation
├── scripts/                    # Utility and data processing scripts
├── src/
│   └── financial_rag/          # Core application package
│       ├── agents/             # Agent implementations (foundation, multi-agency, predictive)
│       ├── ingestion/          # Data ingestion pipelines (real-time + batch)
│       ├── retrieval/          # Retrieval logic, reranking, vector store clients
│       ├── models/             # LLM configuration and prompt management
│       └── utils/              # Shared utilities
├── terraform/                  # Infrastructure as code
├── tests/                      # Test suite
├── tutorial/                   # Guided walkthroughs and examples
├── docker-compose.yml          # Standard local stack
├── docker-compose.dev.yml      # Development overrides
├── docker-compose.prod.yml     # Production-like local simulation
├── Makefile                    # Task runner
├── mkdocs.yml                  # Documentation site config
├── pyproject.toml              # Project config and dependencies
└── workflow.sh                 # End-to-end workflow orchestration script
```

---

## Getting Started

### Prerequisites

```
Python >= 3.10
Docker + Docker Compose
kubectl (for Kubernetes deployment)
Helm >= 3.x (for Helm deployment)
Terraform >= 1.5 (for infrastructure)
make
```

### 1. Clone and configure

```bash
git clone https://github.com/aayostem/financial-rag-agent.git
cd financial-rag-agent

cp .env.example .env
# Configure your LLM API keys, vector store credentials, and data source settings
```

### 2. Install dependencies

```bash
make setup
# Installs Python dependencies and pre-commit hooks (gitleaks, ruff, black)
```

Or manually:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development + testing dependencies
pre-commit install
```

### 3. Start the local stack

```bash
# Standard local environment
make dev-up

# Or use docker-compose directly
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 4. Validate the setup

```bash
python test_foundation.py     # Core RAG pipeline validation
python run_test.py            # Full end-to-end smoke test
```

---

## Running Tests

The test suite covers each major agent capability independently:

```bash
# Run all tests
make test

# Individual capability tests
python test_foundation.py           # Core retrieval + generation pipeline
python test_multi_agency.py         # Multi-agent orchestration
python test_multi_modal.py          # Multi-modal document processing
python test_predictive_analysis.py  # Predictive analytics layer
python test_real_time.py            # Real-time data ingestion path
python test_production.py           # Production readiness checks
```

---

## Deployment

### Docker Compose (local/staging)

```bash
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production simulation
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Helm (Kubernetes)

Full Helm deployment guide: [README-HELM.md](./README-HELM.md)

```bash
# Add chart dependencies
helm dependency update ./helm/

# Install to cluster
helm install financial-rag ./helm/ \
  --namespace financial-rag \
  --create-namespace \
  --values custom-values.yaml

# Upgrade
helm upgrade financial-rag ./helm/ \
  --namespace financial-rag \
  --values custom-values.yaml
```

### Infrastructure (Terraform)

```bash
cd terraform/
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

---

## Workflow Orchestration

For running the full end-to-end pipeline:

```bash
./workflow.sh
```

This executes: ingestion → indexing → agent warm-up → retrieval validation → serving.

---

## Documentation Site

Documentation is built with MkDocs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # Local preview at http://localhost:8000
mkdocs build        # Build static site
```

---

## Security

- **Gitleaks** runs on every commit to prevent secret exposure
- All credentials are injected via environment variables — never hardcoded
- See [SECURITY.md](./SECURITY.md) for the vulnerability disclosure process

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for planned features and the development timeline.

---

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history and release notes.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow, branch strategy, commit conventions, and PR requirements.

---

## License

[MIT](./LICENSE) — Copyright © 2025 Ayo
