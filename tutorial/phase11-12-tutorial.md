# Financial RAG Agent — Phase 11 & 12: LGTM Observability + Production Hardening

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase 11:** Full LGTM Observability Stack — Components #151–178  
> **Phase 12:** Production Hardening, FinOps & Capstone — Components #179–203  
> **Time:** Phase 11 ~3 hours | Phase 12 ~2 hours  
> **Prerequisite:** Phases 1–10 complete — cluster running, Falco and ArgoCD healthy

---

## What You Will Build

**Phase 11** — End-to-end observability across logs, metrics, traces, and alerts:
- OpenTelemetry auto-instrumentation for FastAPI, SQLAlchemy, Redis, and httpx
- Custom RAG spans for embedding, search, and LLM calls
- Tail sampling — 100% of error/slow traces, 1% of success traces
- OpenTelemetry Collector with Jaeger (dev) and AWS X-Ray (prod) exporters
- Trace ID injection into structured logs (correlate traces to log lines)
- SLO recording rules and multi-window multi-burn-rate alerting
- Three SLOs: API availability 99.5%, query latency P99 < 120s, ingestion success 99%
- Grafana dashboards: RAG performance, error budgets, LLM cost
- ArgoCD notifications via Slack and PagerDuty
- OPA Rego policies for Helm manifests and Terraform plans

**Phase 12** — Production readiness, cost management, and the capstone:
- Kubecost for per-namespace cost attribution
- VPA recommendations for right-sizing pod resources
- Karpenter node consolidation policy
- Pod Security Admission labels
- OPA Gatekeeper constraints
- Velero backup and restore
- FinOps Grafana dashboard with budget alerts
- Production readiness validation script

---

# PHASE 11 — LGTM Observability Stack

## New Files in This Phase

```
src/financial_rag/telemetry/
├── __init__.py
└── tracing.py                         ← Components #158–165
observability/
├── tracing/
│   ├── tracer_config.py               ← Components #158–164
│   ├── tracer-provider.yaml           ← OTel Collector (K8s)
│   └── jaeger-install.yaml            ← Jaeger dev/staging
├── slo/
│   ├── slo-rules.yaml                 ← Components #172–173
│   └── slo-alerts.yaml                ← Component #173
└── prometheus-rules/
    └── financial-rag-alerts.yaml      ← Component #172 (already in Phase 10)
argocd/
└── notifications/
    └── templates.yaml                 ← ArgoCD Slack + PagerDuty
policies/
├── rego/
│   ├── helm_policy.rego               ← Helm manifest validation
│   ├── k8s_admission.rego             ← Pod admission validation
│   ├── terraform_policy.rego          ← Terraform plan validation
│   └── vault_policy.rego              ← Already in Phase 6
└── tests/
    └── k8s_admission_test.rego        ← OPA unit tests
```

---

## Step 1 — Install the Observability Namespace (Component #151)

```bash
kubectl create namespace observability
kubectl create namespace monitoring   # if not already created by kube-prometheus-stack
```

Install the LGTM stack via Helm:

```bash
# Loki — log aggregation
helm repo add grafana https://grafana.github.io/helm-charts
helm upgrade --install loki grafana/loki \
  --namespace observability \
  --set loki.storage.type=s3 \
  --set loki.storage.s3.bucketNames.chunks=financial-rag-loki-chunks \
  --set loki.storage.s3.bucketNames.ruler=financial-rag-loki-ruler \
  --set loki.storage.s3.region=us-east-1

# Tempo — distributed tracing storage
helm upgrade --install tempo grafana/tempo-distributed \
  --namespace observability \
  --set storage.trace.backend=s3 \
  --set storage.trace.s3.bucket=financial-rag-tempo-traces \
  --set storage.trace.s3.region=us-east-1

# Mimir — long-term metrics
helm upgrade --install mimir grafana/mimir-distributed \
  --namespace observability \
  --set mimir.structuredConfig.common.storage.backend=s3 \
  --set mimir.structuredConfig.common.storage.s3.bucket_name=financial-rag-mimir-metrics \
  --set mimir.structuredConfig.common.storage.s3.region=us-east-1

# Grafana — visualization
helm upgrade --install grafana grafana/grafana \
  --namespace observability \
  --set persistence.enabled=true \
  --set adminPassword="${GRAFANA_PASSWORD}" \
  --set datasources."datasources\.yaml".apiVersion=1 \
  --set-file datasources."datasources\.yaml".datasources[0]=observability/grafana/datasources.yaml
```

> **Why S3 for all three?** Loki, Tempo, and Mimir all use object storage as
> their long-term backend. S3 costs ~$0.023/GB/month vs ~$0.10/GB/month for
> EBS. At production scale with months of logs, traces, and metrics, this
> difference is significant. The LGTM stack is designed for S3 — use it.

---

## Step 2 — OpenTelemetry Dependencies (Component #157)

Add to `pyproject.toml` dependencies:

```toml
# Observability
"opentelemetry-api>=1.24.0",
"opentelemetry-sdk>=1.24.0",
"opentelemetry-exporter-otlp-proto-grpc>=1.24.0",
"opentelemetry-instrumentation-fastapi>=0.45b0",
"opentelemetry-instrumentation-sqlalchemy>=0.45b0",
"opentelemetry-instrumentation-redis>=0.45b0",
"opentelemetry-instrumentation-httpx>=0.45b0",
"opentelemetry-instrumentation-logging>=0.45b0",
```

Install:

```bash
pip install -e ".[dev]"
```

---

## Step 3 — Tracer Provider Configuration (Components #158–165)

Create `src/financial_rag/telemetry/__init__.py`:

```python
from financial_rag.telemetry.tracing import configure_tracing, get_tracer

__all__ = ["configure_tracing", "get_tracer"]
```

Use `src/financial_rag/telemetry/tracing.py`:
# =============================================================================
# observability/tracing/tracer_config.py
# OpenTelemetry tracer provider configuration for financial-rag-agent.
# Import and call configure_tracing() at application startup (main.py).
# Instruments: FastAPI, SQLAlchemy (asyncpg), Redis, httpx, logging.
# =============================================================================
import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

logger = logging.getLogger(__name__)

OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://financial-rag-otel-collector:4317")
SERVICE_NAME  = os.getenv("OTEL_SERVICE_NAME", "financial-rag-api")
ENVIRONMENT   = os.getenv("ENVIRONMENT", "dev")
SAMPLING_RATE = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "0.01"))  # 1% default, 100% for errors via tail sampling


def configure_tracing(app=None, engine=None) -> TracerProvider:
    """
    Configure the OpenTelemetry tracer provider and instrument all frameworks.
    Call once at application startup before the ASGI app is created.

    Args:
        app:    FastAPI application instance (for FastAPI instrumentation)
        engine: SQLAlchemy async engine (for database query tracing)

    Returns:
        TracerProvider — the configured provider (also set as global)
    """
    # Resource: identifies this service in all traces
    resource = Resource.create({
        "service.name":        SERVICE_NAME,
        "service.version":     os.getenv("APP_VERSION", "unknown"),
        "service.namespace":   "financial-rag",
        "deployment.environment": ENVIRONMENT,
        "k8s.namespace.name":  os.getenv("K8S_NAMESPACE", "financial-rag"),
        "k8s.pod.name":        os.getenv("HOSTNAME", "unknown"),
    })

    # Sampler: low rate by default — tail sampling in the collector handles the rest
    sampler = ParentBased(root=TraceIdRatioBased(SAMPLING_RATE))

    # Provider
    provider = TracerProvider(resource=resource, sampler=sampler)

    # OTLP exporter → OpenTelemetry Collector → Jaeger + X-Ray
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTEL_ENDPOINT,
        insecure=ENVIRONMENT != "prod",
    )
    provider.add_span_processor(BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=2048,
        max_export_batch_size=512,
        export_timeout_millis=5000,
    ))

    # Console exporter in dev for local debugging
    if ENVIRONMENT == "dev":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set as global provider
    trace.set_tracer_provider(provider)
    logger.info("TracerProvider configured: service=%s env=%s sampler_rate=%.2f%%",
                SERVICE_NAME, ENVIRONMENT, SAMPLING_RATE * 100)

    # Instrument FastAPI — adds spans for every HTTP request
    if app is not None:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health,/metrics",  # skip health + metrics endpoints
            server_request_hook=_add_request_attributes,
        )

    # Instrument SQLAlchemy — adds spans for every DB query
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(
            engine=engine.sync_engine,
            enable_commenter=True,   # adds traceparent comment to SQL queries
            commenter_options={"db_framework": True},
        )

    # Instrument Redis — adds spans for every cache operation
    RedisInstrumentor().instrument()

    # Instrument httpx — adds spans for LLM API calls (OpenAI, Groq, Azure)
    HTTPXClientInstrumentor().instrument(
        request_hook=_add_llm_attributes,
    )

    # Correlate trace IDs with log records
    LoggingInstrumentor().instrument(set_logging_format=True)

    return provider


def _add_request_attributes(span, scope):
    """Add financial-rag-specific attributes to HTTP request spans."""
    if span and span.is_recording():
        # Tag agent spans for the tail sampler policy
        if "/query" in scope.get("path", ""):
            span.set_attribute("component", "query-endpoint")
            span.set_attribute("financial_rag.query_type", "rag")


def _add_llm_attributes(span, request):
    """Add LLM provider info to outbound httpx spans."""
    if span and span.is_recording():
        url = str(request.url)
        if "openai" in url:
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("component", "llm-client")
        elif "groq" in url:
            span.set_attribute("llm.provider", "groq")
            span.set_attribute("component", "llm-client")
        elif "azure" in url:
            span.set_attribute("llm.provider", "azure-openai")
            span.set_attribute("component", "llm-client")


def get_tracer(name: str):
    """Get a named tracer for manual instrumentation."""
    return trace.get_tracer(name, schema_url="https://opentelemetry.io/schemas/1.24.0")



Five things to teach from this file:

### Component #158 — TracerProvider with tail sampling

```python
sampler = ParentBased(root=TraceIdRatioBased(SAMPLING_RATE))
```

`SAMPLING_RATE` defaults to `0.01` (1%). This means 99% of successful,
fast traces are dropped at the SDK before export. The OpenTelemetry Collector
then applies **tail sampling** — after seeing the full trace, it keeps 100%
of error traces regardless of head sampling.

`ParentBased` respects the sampling decision from upstream services — if the
ALB or a calling service samples a request, your service honours that decision.

### Component #159 — FastAPI auto-instrumentation

```python
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="/health,/metrics",
    server_request_hook=_add_request_attributes,
)
```

`excluded_urls` skips health checks and Prometheus scrapes — these generate
thousands of spans with zero debugging value and would consume your sampling
budget. Always exclude them.

### Component #160 — SQLAlchemy instrumentation

```python
SQLAlchemyInstrumentor().instrument(
    engine=engine.sync_engine,
    enable_commenter=True,
)
```

`enable_commenter=True` adds a `/*traceparent='00-...'*/` SQL comment to
every query. When you look at slow query logs in PostgreSQL, you can extract
the trace ID and find the exact request that caused the slow query in Jaeger.

### Component #165 — Trace ID in logs

```python
LoggingInstrumentor().instrument(set_logging_format=True)
```

This adds `trace_id` and `span_id` to every Python log record. Combined
with Loki's Grafana datasource, you can click a log line and jump directly
to the trace in Tempo — and vice versa.

### Wire it into `server.py`

Add to `src/financial_rag/api/server.py` lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    # Configure tracing BEFORE initialising dependencies
    # (so DB connection spans are captured from the start)
    if settings.APP_ENV != "testing":
        from financial_rag.telemetry.tracing import configure_tracing
        from financial_rag.storage.database import get_db_client
        db = await get_db_client()
        configure_tracing(app=app, engine=db._engine)

    await initialise_dependencies()
    ...
```

---

## Step 4 — Custom RAG Spans (Components #162–164)

Auto-instrumentation covers HTTP, DB, and Redis. The RAG-specific operations
— embedding, vector search, and LLM generation — need manual spans.

Add to `src/financial_rag/retrieval/embeddings.py`:

```python
from financial_rag.telemetry.tracing import get_tracer

_tracer = get_tracer("financial_rag.embeddings")

# In EmbeddingClient.embed_texts():
async def embed_texts(self, texts: list[str]) -> list[list[float]]:
    with _tracer.start_as_current_span("rag.embedding") as span:
        span.set_attribute("embedding.provider", self.provider_name)
        span.set_attribute("embedding.text_count", len(texts))
        span.set_attribute("embedding.dimensions", self.dimensions)
        # ... existing code
```

Add to `src/financial_rag/retrieval/hybrid_search.py`:

```python
from financial_rag.telemetry.tracing import get_tracer

_tracer = get_tracer("financial_rag.search")

# In HybridSearcher.search():
async def search(self, question: str, ...) -> list[RetrievalResult]:
    with _tracer.start_as_current_span("rag.search") as span:
        span.set_attribute("search.type", "hybrid")
        span.set_attribute("search.question_length", len(question))
        span.set_attribute("search.alpha", effective_alpha)
        # ... existing code
        span.set_attribute("search.results_count", len(fused))
```

Add to `src/financial_rag/retrieval/query_engine.py`:

```python
from financial_rag.telemetry.tracing import get_tracer

_tracer = get_tracer("financial_rag.llm")

# In QueryEngine._generate_answer():
async def _generate_answer(self, *, question, context, analysis_style):
    with _tracer.start_as_current_span("rag.llm") as span:
        span.set_attribute("llm.model", self._settings.LLM_MODEL)
        span.set_attribute("llm.provider", self._settings.LLM_PROVIDER)
        span.set_attribute("llm.analysis_style", analysis_style)
        span.set_attribute("llm.context_length", len(context))
        # ... existing code
```

These three spans produce a flame graph in Jaeger that shows exactly where
time is spent in a RAG query:

```
HTTP POST /query (400ms total)
  └── rag.embedding (45ms)     ← embed the question
  └── rag.search (180ms)       ← vector + BM25 + RRF
      └── pgvector query (90ms)  ← auto from SQLAlchemy
      └── redis get (5ms)        ← auto from Redis
  └── rag.llm (170ms)          ← LLM generation
      └── httpx POST (165ms)     ← auto from httpx
```

---

## Step 5 — OpenTelemetry Collector (Component #158)

Use `observability/tracing/tracer-provider.yaml` 
# =============================================================================
# observability/tracing/tracer-provider.yaml
# OpenTelemetry Collector + Tracer Provider configuration.
# Collects traces from: FastAPI (via opentelemetry-instrumentation-fastapi),
# SQLAlchemy (asyncpg), Redis, httpx (LLM API calls).
# Exports to: OTLP → Jaeger (in-cluster) + AWS X-Ray (prod).
# =============================================================================
apiVersion: opentelemetry.io/v1alpha1
kind: OpenTelemetryCollector
metadata:
  name: financial-rag-otel-collector
  namespace: financial-rag
spec:
  mode: Deployment
  replicas: 2
  image: otel/opentelemetry-collector-contrib:0.96.0

  config: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      # Add resource attributes to all spans
      resource:
        attributes:
          - key: service.namespace
            value: financial-rag
            action: upsert
          - key: deployment.environment
            from_attribute: k8s.namespace.name
            action: upsert

      # Batch for performance
      batch:
        timeout: 5s
        send_batch_size: 1024
        send_batch_max_size: 2048

      # Tail sampling — keep 100% of error traces, 1% of success traces
      tail_sampling:
        decision_wait: 10s
        num_traces: 100000
        expected_new_traces_per_sec: 500
        policies:
          - name: errors-policy
            type: status_code
            status_code: {status_codes: [ERROR]}
          - name: slow-traces-policy
            type: latency
            latency: {threshold_ms: 2000}
          - name: llm-traces-policy
            type: string_attribute
            string_attribute:
              key: component
              values: [llm-client, agent]
          - name: probabilistic-policy
            type: probabilistic
            probabilistic: {sampling_percentage: 1}

      # Memory limiter — prevent OOM on trace bursts
      memory_limiter:
        check_interval: 1s
        limit_mib: 512
        spike_limit_mib: 128

    exporters:
      # Jaeger for in-cluster trace visualization
      jaeger:
        endpoint: jaeger-collector.monitoring.svc.cluster.local:14250
        tls:
          insecure: true

      # AWS X-Ray for prod trace storage (30-day retention)
      awsxray:
        region: us-east-1
        no_verify_ssl: false
        local_mode: false

      # Prometheus exporter for trace-based metrics
      prometheus:
        endpoint: "0.0.0.0:8889"
        namespace: financial_rag
        const_labels:
          project: financial-rag-agent

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, resource, tail_sampling, batch]
          exporters: [jaeger, awsxray]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, resource, batch]
          exporters: [prometheus]


The tail sampling policy is the most important part:


**Why tail sampling instead of head sampling?** Head sampling (at the SDK)
decides before the trace is complete. A request that starts fast but then
calls a slow LLM endpoint would be dropped. Tail sampling sees the complete
trace — so all slow or errored traces are kept regardless of how they started.

Apply:

```bash
kubectl apply -f observability/tracing/tracer-provider.yaml
kubectl apply -f observability/tracing/jaeger-install.yaml  # dev/staging only
```

---

## Step 6 — Structured JSON Logging (Component #166)

Add to `src/financial_rag/api/server.py` or a dedicated
`src/financial_rag/api/logging_config.py`:

```python
# src/financial_rag/api/logging_config.py
import logging
import structlog
from financial_rag.config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        # Inject trace_id and span_id from OpenTelemetry
        _add_trace_context,
    ]

    if settings.LOG_FORMAT == "json":
        # Production: JSON for log aggregators (Loki, CloudWatch)
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: human-readable console output
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _add_trace_context(logger, method, event_dict):
    """Inject OpenTelemetry trace context into every log record."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:
        pass
    return event_dict
```

Call `configure_logging()` at the start of the lifespan function — before
any other imports that might call `logging.getLogger()`.

---

## Step 7 — SLO Recording Rules and Alerts (Components #172–173)

Use `observability/slo/slo-rules.yaml` and `observability/slo/slo-alerts.yaml`
# =============================================================================
# observability/slo/slo-rules.yaml
# SLO recording rules and alerting rules using multi-window multi-burn-rate.
# Based on Google SRE workbook Chapter 5 alerting methodology.
#
# SLOs defined:
#   SLO-1: API availability — 99.5% over 30 days
#   SLO-2: Query latency P99 < 120s — 95% of requests
#   SLO-3: Ingestion success rate — 99% over 7 days
# =============================================================================
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: financial-rag-slos
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
    slo: "true"
spec:
  groups:
    # =========================================================================
    # SLO-1: API Availability — 99.5% (error budget: 0.5% = 3.65 hours/month)
    # =========================================================================
    - name: financial-rag.slo.availability.recording
      interval: 30s
      rules:
        # Good requests = non-5xx
        - record: financial_rag:api_request_success:rate5m
          expr: |
            sum(rate(http_requests_total{
              namespace="financial-rag",component="api",status!~"5.."
            }[5m]))
            /
            sum(rate(http_requests_total{
              namespace="financial-rag",component="api"
            }[5m]))

        # Rolling window error rates for burn rate calculation
        - record: financial_rag:api_error_rate:rate1h
          expr: |
            1 - (
              sum(rate(http_requests_total{namespace="financial-rag",component="api",status!~"5.."}[1h]))
              / sum(rate(http_requests_total{namespace="financial-rag",component="api"}[1h]))
            )

        - record: financial_rag:api_error_rate:rate6h
          expr: |
            1 - (
              sum(rate(http_requests_total{namespace="financial-rag",component="api",status!~"5.."}[6h]))
              / sum(rate(http_requests_total{namespace="financial-rag",component="api"}[6h]))
            )

        - record: financial_rag:api_error_rate:rate24h
          expr: |
            1 - (
              sum(rate(http_requests_total{namespace="financial-rag",component="api",status!~"5.."}[24h]))
              / sum(rate(http_requests_total{namespace="financial-rag",component="api"}[24h]))
            )

        - record: financial_rag:api_error_rate:rate72h
          expr: |
            1 - (
              sum(rate(http_requests_total{namespace="financial-rag",component="api",status!~"5.."}[72h]))
              / sum(rate(http_requests_total{namespace="financial-rag",component="api"}[72h]))
            )

    - name: financial-rag.slo.availability.alerts
      rules:
        # Tier 1: Fast burn — 14.4x burn rate — pages immediately
        # Burns 2% of monthly budget in 1 hour
        - alert: SLOAvailabilityFastBurn
          expr: |
            financial_rag:api_error_rate:rate1h > (14.4 * 0.005)
            and
            financial_rag:api_error_rate:rate6h > (14.4 * 0.005)
          labels:
            severity: critical
            slo: availability
            project: financial-rag
          annotations:
            summary: "SLO-1 CRITICAL: API availability fast burn rate"
            description: >
              Error rate: 1h={{ $value | humanizePercentage }}.
              Burn rate: 14.4x. At this rate the monthly error budget
              (3.65 hours) will be exhausted in 1 hour.
              Immediate action required.
            runbook: "https://github.com/aayostem/financial-rag-agent/docs/runbooks/slo-availability.md"

        # Tier 2: Slow burn — 6x burn rate — warning, investigate today
        # Burns 5% of monthly budget in 6 hours
        - alert: SLOAvailabilitySlowBurn
          expr: |
            financial_rag:api_error_rate:rate6h > (6 * 0.005)
            and
            financial_rag:api_error_rate:rate24h > (6 * 0.005)
          labels:
            severity: warning
            slo: availability
          annotations:
            summary: "SLO-1 WARNING: API availability slow burn rate"
            description: >
              Error rate: 6h={{ $value | humanizePercentage }}.
              Burn rate: 6x. Budget will exhaust in ~5 days if sustained.

        # Error budget remaining (recording rule for dashboards)
        - record: financial_rag:slo1:error_budget_remaining
          expr: |
            1 - (financial_rag:api_error_rate:rate72h / 0.005)

    # =========================================================================
    # SLO-2: Query Latency — P99 < 120s for 95% of windows
    # =========================================================================
    - name: financial-rag.slo.latency.recording
      interval: 30s
      rules:
        - record: financial_rag:query_latency_slo:rate5m
          expr: |
            sum(rate(http_request_duration_seconds_bucket{
              namespace="financial-rag",component="api",
              path=~"/query.*",le="120"
            }[5m]))
            /
            sum(rate(http_request_duration_seconds_count{
              namespace="financial-rag",component="api",path=~"/query.*"
            }[5m]))

        - record: financial_rag:query_latency_error_rate:rate1h
          expr: |
            1 - (
              sum(rate(http_request_duration_seconds_bucket{
                namespace="financial-rag",component="api",path=~"/query.*",le="120"
              }[1h]))
              / sum(rate(http_request_duration_seconds_count{
                namespace="financial-rag",component="api",path=~"/query.*"
              }[1h]))
            )

        - record: financial_rag:query_latency_error_rate:rate6h
          expr: |
            1 - (
              sum(rate(http_request_duration_seconds_bucket{
                namespace="financial-rag",component="api",path=~"/query.*",le="120"
              }[6h]))
              / sum(rate(http_request_duration_seconds_count{
                namespace="financial-rag",component="api",path=~"/query.*"
              }[6h]))
            )

    - name: financial-rag.slo.latency.alerts
      rules:
        - alert: SLOLatencyFastBurn
          expr: |
            financial_rag:query_latency_error_rate:rate1h > (14.4 * 0.05)
            and
            financial_rag:query_latency_error_rate:rate6h > (14.4 * 0.05)
          labels:
            severity: critical
            slo: latency
          annotations:
            summary: "SLO-2 CRITICAL: Query latency fast burn rate"
            description: "More than 5% of queries exceeding 120s P99 latency SLO. Burn rate 14.4x."

        - alert: SLOLatencySlowBurn
          expr: |
            financial_rag:query_latency_error_rate:rate6h > (6 * 0.05)
            and
            financial_rag:query_latency_error_rate:rate6h < (14.4 * 0.05)
          labels:
            severity: warning
            slo: latency
          annotations:
            summary: "SLO-2 WARNING: Query latency slow burn rate"
            description: "Latency SLO burn rate 6x — budget will exhaust in ~5 days."

    # =========================================================================
    # SLO-3: Ingestion Success Rate — 99% over 7 days
    # =========================================================================
    - name: financial-rag.slo.ingestion
      interval: 60s
      rules:
        - record: financial_rag:ingestion_success_rate:rate24h
          expr: |
            sum(financial_rag_ingestion_filings_processed_total)
            /
            (sum(financial_rag_ingestion_filings_processed_total)
             + sum(financial_rag_ingestion_filings_failed_total))

        - alert: SLOIngestionFailureRate
          expr: |
            financial_rag:ingestion_success_rate:rate24h < 0.99
          for: 30m
          labels:
            severity: warning
            slo: ingestion
          annotations:
            summary: "SLO-3: Ingestion success rate below 99%"
            description: "Ingestion success rate: {{ $value | humanizePercentage }}. Check EDGAR API connectivity and SHA-256 dedup logic."

    # =========================================================================
    # SLO Dashboard recording rules (used by Grafana)
    # =========================================================================
    - name: financial-rag.slo.dashboard
      interval: 60s
      rules:
        - record: financial_rag:slo_budget_consumed_percent:30d
          expr: |
            (
              sum(financial_rag:api_error_rate:rate72h) * 30
            ) / 0.005 * 100

        - record: financial_rag:cost_per_query_cents:1h
          expr: |
            (financial_rag_llm_cost_usd_total + financial_rag_infra_cost_usd_total)
            / financial_rag_queries_total * 100

# =============================================================================
# SLO Alerting Rules — Multi-window multi-burn-rate (Google SRE Workbook Ch.5)
# Separate file for SLO alerts to keep them distinct from symptom-based alerts
# =============================================================================
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: financial-rag-slo-alerts
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
    slo: "true"
spec:
  groups:
    - name: financial-rag.slo.page
      rules:
        # Page: burns >2% budget in 1h (fast burn 14.4x)
        - alert: SLOErrorBudgetFastBurn
          expr: |
            (
              financial_rag:api_error_rate:rate1h > (14.4 * 0.005)
              and
              financial_rag:api_error_rate:rate6h > (14.4 * 0.005)
            )
            or
            (
              financial_rag:query_latency_error_rate:rate1h > (14.4 * 0.05)
              and
              financial_rag:query_latency_error_rate:rate6h > (14.4 * 0.05)
            )
          labels:
            severity: critical
            slo: "true"
            page: "true"
          annotations:
            summary: "SLO fast burn — error budget draining rapidly"
            description: >
              At current burn rate, monthly error budget exhausts in < 1 hour.
              Immediate investigation and response required.
            runbook: "https://github.com/aayostem/financial-rag-agent/docs/runbooks/slo-burn.md"

        # Page: burns >5% budget in 6h (slow burn 6x)
        - alert: SLOErrorBudgetSlowBurn
          expr: |
            (
              financial_rag:api_error_rate:rate6h > (6 * 0.005)
              and
              financial_rag:api_error_rate:rate24h > (6 * 0.005)
            )
            or
            (
              financial_rag:query_latency_error_rate:rate6h > (6 * 0.05)
              and
              financial_rag:query_latency_error_rate:rate6h < (14.4 * 0.05)
            )
          labels:
            severity: warning
            slo: "true"
          annotations:
            summary: "SLO slow burn — error budget draining"
            description: "Budget will exhaust in ~5 days if burn rate is sustained."

    - name: financial-rag.slo.budget
      interval: 5m
      rules:
        # Error budget remaining as percentage — used in Grafana dashboard
        - record: financial_rag:slo_availability:error_budget_remaining
          expr: |
            (1 - (financial_rag:api_error_rate:rate72h / 0.005)) * 100

        - record: financial_rag:slo_latency:error_budget_remaining
          expr: |
            (1 - (financial_rag:query_latency_error_rate:rate6h / 0.05)) * 100

        # Alert when < 10% budget remains
        - alert: SLOErrorBudgetAlmostExhausted
          expr: financial_rag:slo_availability:error_budget_remaining < 10
          labels:
            severity: warning
            slo: "true"
          annotations:
            summary: "Less than 10% of monthly error budget remaining"
            description: "{{ $value | humanize }}% of availability error budget remains this month."

        - alert: SLOErrorBudgetExhausted
          expr: financial_rag:slo_availability:error_budget_remaining <= 0
          labels:
            severity: critical
            slo: "true"
          annotations:
            summary: "Monthly error budget fully exhausted"
            description: "100% of the monthly availability error budget has been consumed. All remaining errors exceed the SLO."


Three SLOs defined:

| SLO | Target | Error Budget (30 days) |
|---|---|---|
| API availability | 99.5% | 3.65 hours of downtime |
| Query P99 latency < 120s | 95% of requests | 5% of requests can be slow |
| Ingestion success rate | 99% | 1% of filings can fail |

### Multi-window multi-burn-rate alerting

This is the Google SRE Workbook methodology. Two tiers, two time windows each:

```
CRITICAL (page immediately):
  1h error rate  > 14.4 × SLO_error_rate
  AND
  6h error rate  > 14.4 × SLO_error_rate
  → Burns 2% of monthly budget in 1 hour

WARNING (investigate today):
  6h error rate  > 6 × SLO_error_rate
  AND
  24h error rate > 6 × SLO_error_rate
  → Burns 5% of monthly budget in 6 hours
```

Why two windows? A single spike in one window triggers false alerts. Requiring
both windows to be elevated simultaneously means the burn rate is sustained,
not a one-off blip. This dramatically reduces alert fatigue.

For the availability SLO with error_rate = 0.005 (0.5%):
- Fast burn threshold: `14.4 × 0.005 = 0.072` (7.2% error rate)
- Slow burn threshold: `6 × 0.005 = 0.03` (3% error rate)

Apply:

```bash
kubectl apply -f observability/slo/slo-rules.yaml
kubectl apply -f observability/slo/slo-alerts.yaml
```

---

## Step 8 — ArgoCD Notifications (Component #116 wiring)

Use `argocd/notifications/templates.yaml`


The templates use ArgoCD's notification template language. Three events wired:

```yaml
trigger.on-sync-succeeded: ✅ green Slack message
trigger.on-sync-failed:    ❌ red Slack + PagerDuty
trigger.on-health-degraded: ⚠️ yellow Slack + PagerDuty
```

The PagerDuty integration only fires on prod application failures — the
`apps-appset.yaml` annotation `notifications.argoproj.io/subscribe.on-health-degraded.pagerduty`
uses `{{environment}}-oncall` which only resolves in prod.

Apply:

```bash
kubectl apply -f argocd/notifications/templates.yaml
```
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
data:
  service.slack: |
    token: $slack-token
    username: ArgoCD
    icon: ":argo:"
  service.pagerduty: |
    serviceKeys:
      prod-oncall: $pagerduty-prod-key
  template.app-sync-succeeded: |
    slack:
      attachments: |
        [{
          "color": "#18be52",
          "title": "✅ {{.app.metadata.name}} synced",
          "fields": [
            {"title": "Environment", "value": "{{.app.metadata.labels.environment}}", "short": true},
            {"title": "Revision", "value": "{{.app.status.sync.revision}}", "short": true}
          ]
        }]
  template.app-sync-failed: |
    slack:
      attachments: |
        [{
          "color": "#e51b00",
          "title": "❌ {{.app.metadata.name}} sync FAILED",
          "fields": [
            {"title": "Environment", "value": "{{.app.metadata.labels.environment}}", "short": true},
            {"title": "Error", "value": "{{.app.status.operationState.message}}", "short": false}
          ]
        }]
    pagerduty:
      summary: "ArgoCD sync failed: {{.app.metadata.name}}"
      severity: error
  template.app-health-degraded: |
    slack:
      attachments: |
        [{
          "color": "#f4c030",
          "title": "⚠️ {{.app.metadata.name}} health DEGRADED",
          "fields": [
            {"title": "Health", "value": "{{.app.status.health.status}}", "short": true}
          ]
        }]
    pagerduty:
      summary: "App degraded: {{.app.metadata.name}}"
      severity: warning
  trigger.on-sync-succeeded: |
    - when: app.status.operationState.phase in ['Succeeded']
      send: [app-sync-succeeded]
  trigger.on-sync-failed: |
    - when: app.status.operationState.phase in ['Error', 'Failed']
      send: [app-sync-failed]
  trigger.on-health-degraded: |
    - when: app.status.health.status == 'Degraded'
      send: [app-health-degraded]
  defaultTriggers: |
    - on-sync-failed
    - on-health-degraded

---

## Step 9 — OPA Rego Policies (CI enforcement)

Your repo has three Rego policy files and a test file. These run in CI via
`conftest` and optionally at admission via OPA Gatekeeper.

### `k8s_admission.rego` — Pod admission rules
# =============================================================================
# policies/rego/k8s_admission.rego
# OPA Gatekeeper / Conftest admission policies for financial-rag-agent.
# Enforced at:
#   1. CI (conftest) — every PR touching Helm or raw manifests
#   2. Kubernetes admission (Gatekeeper ConstraintTemplate) — every pod create
#
# Rules:
#   - No :latest image tags in production
#   - All containers must declare resource requests AND limits
#   - readOnlyRootFilesystem must be true
#   - allowPrivilegeEscalation must be false
#   - Must not run as root (runAsNonRoot or runAsUser > 0)
#   - No host network, host PID, host IPC
#   - ALL capabilities must be dropped
#   - No privileged containers
#   - Required labels: app.kubernetes.io/name, app.kubernetes.io/component
# =============================================================================

package financial_rag.k8s.admission

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# Deny: :latest image tag in any container
# Mutable tags make deployments non-reproducible and bypass supply chain controls
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf(
        "Container '%v' uses ':latest' tag. Pin to an immutable SHA or semver tag. Image: %v",
        [container.name, container.image]
    )
}

deny contains msg if {
    container := input.review.object.spec.initContainers[_]
    endswith(container.image, ":latest")
    msg := sprintf(
        "InitContainer '%v' uses ':latest' tag. Image: %v",
        [container.name, container.image]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not contains(container.image, ":")
    msg := sprintf(
        "Container '%v' has no image tag at all. All images must use pinned tags. Image: %v",
        [container.name, container.image]
    )
}

# ---------------------------------------------------------------------------
# Deny: Missing resource requests or limits
# HPA requires requests; limits prevent noisy-neighbour OOM on shared nodes
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.requests.cpu
    msg := sprintf(
        "Container '%v' missing resources.requests.cpu. HPA cannot function without CPU requests.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.requests.memory
    msg := sprintf(
        "Container '%v' missing resources.requests.memory.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf(
        "Container '%v' missing resources.limits.cpu. Unbounded CPU causes node starvation.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf(
        "Container '%v' missing resources.limits.memory. Unbounded memory causes OOM evictions.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: readOnlyRootFilesystem not set to true
# Writable root filesystem allows post-exploit persistence
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem == true
    msg := sprintf(
        "Container '%v': securityContext.readOnlyRootFilesystem must be true. Use emptyDir for writable paths.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: allowPrivilegeEscalation not set to false
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.allowPrivilegeEscalation == false
    msg := sprintf(
        "Container '%v': securityContext.allowPrivilegeEscalation must be false.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: Running as root
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    container.securityContext.runAsUser == 0
    msg := sprintf(
        "Container '%v' runs as UID 0 (root). Set runAsUser to a non-zero UID.",
        [container.name]
    )
}

deny contains msg if {
    not input.review.object.spec.securityContext.runAsNonRoot == true
    not_all_containers_have_user := {c.name | c := input.review.object.spec.containers[_]; not c.securityContext.runAsUser}
    count(not_all_containers_have_user) > 0
    msg := "Pod must set securityContext.runAsNonRoot: true at pod level, or runAsUser at container level."
}

# ---------------------------------------------------------------------------
# Deny: Privileged containers
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf(
        "Container '%v' is privileged. Privileged containers have full host access.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: Capabilities not fully dropped
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf(
        "Container '%v': securityContext.capabilities.drop must be set. Require drop: [ALL].",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    caps := container.securityContext.capabilities.drop
    not "ALL" in caps
    msg := sprintf(
        "Container '%v': capabilities.drop must include 'ALL'. Current drop: %v",
        [container.name, caps]
    )
}

# ---------------------------------------------------------------------------
# Deny: Host namespaces
# ---------------------------------------------------------------------------
deny contains msg if {
    input.review.object.spec.hostNetwork == true
    msg := "Pod must not use hostNetwork. This grants access to the node's network namespace."
}

deny contains msg if {
    input.review.object.spec.hostPID == true
    msg := "Pod must not use hostPID. This grants visibility into all host processes."
}

deny contains msg if {
    input.review.object.spec.hostIPC == true
    msg := "Pod must not use hostIPC. This allows IPC namespace sharing with the host."
}

# ---------------------------------------------------------------------------
# Deny: Missing required Kubernetes labels
# Required for: Cilium policy selectors, Istio AuthorizationPolicy, cost attribution
# ---------------------------------------------------------------------------
deny contains msg if {
    not input.review.object.metadata.labels["app.kubernetes.io/name"]
    msg := "Pod missing required label: app.kubernetes.io/name. Required for service mesh routing and cost attribution."
}

deny contains msg if {
    not input.review.object.metadata.labels["app.kubernetes.io/component"]
    msg := "Pod missing required label: app.kubernetes.io/component. Required for Cilium network policy selectors."
}

# ---------------------------------------------------------------------------
# Warn: No liveness or readiness probe
# Not blocking (warn only) — some jobs legitimately skip probes
# ---------------------------------------------------------------------------
warn contains msg if {
    container := input.review.object.spec.containers[_]
    not container.livenessProbe
    msg := sprintf(
        "Container '%v' has no livenessProbe. Unhealthy pods will not be restarted automatically.",
        [container.name]
    )
}

warn contains msg if {
    container := input.review.object.spec.containers[_]
    not container.readinessProbe
    msg := sprintf(
        "Container '%v' has no readinessProbe. Pods will receive traffic before they are ready.",
        [container.name]
    )
}


Ten rules covering the production security baseline:

| Rule | What it catches |
|---|---|
| No `:latest` tag | Mutable image tags |
| Resource requests + limits required | Missing CPU/memory |
| `readOnlyRootFilesystem: true` | Writable container root |
| `allowPrivilegeEscalation: false` | setuid binary escalation |
| No root UID | `runAsUser: 0` |
| No privileged containers | `privileged: true` |
| Capabilities drop ALL | Unrestricted kernel caps |
| No host namespaces | `hostNetwork/hostPID/hostIPC` |
| Required labels | Missing `app.kubernetes.io/name` and `component` |
| Liveness + readiness probes (warn) | Missing health checks |

### `helm_policy.rego` — Helm manifest rules
# =============================================================================
# policies/rego/helm_policy.rego
# Conftest policies for Helm-rendered Kubernetes manifests.
# Run in CI: helm template . -f values.yaml -f values.prod.yaml | conftest test -
# =============================================================================
package financial_rag.helm

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# Deny: Deprecated Kubernetes APIs (EKS 1.29 removals)
# ---------------------------------------------------------------------------
removed_apis := {
    "flowcontrol.apiserver.k8s.io/v1beta1",
    "flowcontrol.apiserver.k8s.io/v1beta2",
    "autoscaling/v2beta1",
    "autoscaling/v2beta2",
    "batch/v1beta1",
}

deny contains msg if {
    input.apiVersion
    removed_apis[input.apiVersion]
    msg := sprintf("Resource '%v' uses removed API version '%v' — not supported in Kubernetes 1.29+.", [input.metadata.name, input.apiVersion])
}

# ---------------------------------------------------------------------------
# Deny: HPA using autoscaling/v1 (must use autoscaling/v2 for multi-metric)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "HorizontalPodAutoscaler"
    input.apiVersion == "autoscaling/v1"
    msg := sprintf("HPA '%v' uses autoscaling/v1 — must use autoscaling/v2 for CPU+memory dual-metric scaling.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: Service type LoadBalancer without annotations (bypasses Istio gateway)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "Service"
    input.spec.type == "LoadBalancer"
    not input.metadata.annotations["service.beta.kubernetes.io/aws-load-balancer-type"]
    not input.metadata.annotations["service.beta.kubernetes.io/aws-load-balancer-nlb-target-type"]
    msg := sprintf("Service '%v' of type LoadBalancer missing ALB/NLB annotations — must route through Istio gateway.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: Ingress without Istio gateway class or ALB class
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "Ingress"
    not input.spec.ingressClassName
    not input.metadata.annotations["kubernetes.io/ingress.class"]
    msg := sprintf("Ingress '%v' has no ingressClassName — must specify 'alb' or 'istio'.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: PodDisruptionBudget with minAvailable=0 (allows full disruption)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "PodDisruptionBudget"
    input.spec.minAvailable == 0
    msg := sprintf("PodDisruptionBudget '%v' has minAvailable=0 — allows complete pod disruption.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: StatefulSet without a serviceName (headless service required)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "StatefulSet"
    not input.spec.serviceName
    msg := sprintf("StatefulSet '%v' missing spec.serviceName — headless service required for stable DNS.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: CronJob without concurrencyPolicy
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "CronJob"
    not input.spec.concurrencyPolicy
    msg := sprintf("CronJob '%v' missing spec.concurrencyPolicy — set to 'Forbid' for ingestion jobs.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Warn: Deployment without podAntiAffinity (single-node HA risk)
# ---------------------------------------------------------------------------
warn contains msg if {
    input.kind == "Deployment"
    input.spec.replicas > 1
    not input.spec.template.spec.affinity.podAntiAffinity
    msg := sprintf("Deployment '%v' has replicas > 1 but no podAntiAffinity — all pods may schedule on one node.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Warn: No topologySpreadConstraints on multi-replica workloads
# ---------------------------------------------------------------------------
warn contains msg if {
    input.kind == "Deployment"
    input.spec.replicas >= 3
    not input.spec.template.spec.topologySpreadConstraints
    msg := sprintf("Deployment '%v' has 3+ replicas but no topologySpreadConstraints — consider spreading across AZs.", [input.metadata.name])
}


Eight rules for Kubernetes resource best practices:

| Rule | What it catches |
|---|---|
| Deprecated API versions | `autoscaling/v2beta1`, `batch/v1beta1` |
| HPA using `autoscaling/v1` | Cannot do CPU+memory dual-metric |
| LoadBalancer without ALB annotations | Bypasses Istio gateway |
| Ingress without class | ALB or Istio required |
| PDB with `minAvailable: 0` | Allows complete disruption |
| StatefulSet without `serviceName` | Breaks stable DNS |
| CronJob without `concurrencyPolicy` | Overlapping runs |
| Deployment without podAntiAffinity (warn) | All pods on one node |

### `terraform_policy.rego` — Infrastructure rules
# =============================================================================
# policies/rego/terraform_policy.rego
# Conftest policies for Terraform plan JSON.
# Run in CI: terraform plan -out plan.out && terraform show -json plan.out | conftest test -
# Catches: public S3, unencrypted RDS, missing KMS, public EKS endpoints in prod.
# =============================================================================
package financial_rag.terraform

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# S3 — no public buckets
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_public_access_block"
    resource.values.block_public_acls != true
    msg := sprintf("S3 bucket '%v' does not block public ACLs — all buckets must block public access.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_public_access_block"
    resource.values.restrict_public_buckets != true
    msg := sprintf("S3 bucket '%v' does not restrict public buckets.", [resource.name])
}

# ---------------------------------------------------------------------------
# S3 — encryption required
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket"
    not bucket_has_encryption(resource.name)
    msg := sprintf("S3 bucket '%v' has no server-side encryption configuration.", [resource.name])
}

bucket_has_encryption(bucket_name) if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_server_side_encryption_configuration"
    contains(resource.values.bucket, bucket_name)
}

# ---------------------------------------------------------------------------
# RDS — encryption, deletion protection, no public access
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    not resource.values.storage_encrypted == true
    msg := sprintf("RDS instance '%v' storage_encrypted must be true.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    resource.values.publicly_accessible == true
    msg := sprintf("RDS instance '%v' must not be publicly accessible.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    resource.values.backup_retention_period == 0
    msg := sprintf("RDS instance '%v' has no backup retention — minimum 1 day required.", [resource.name])
}

# ---------------------------------------------------------------------------
# EKS — private endpoint in prod, secrets encryption, audit logs
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    resource.values.vpc_config[_].endpoint_public_access == true
    contains(resource.name, "prod")
    msg := sprintf("EKS cluster '%v' has public API endpoint enabled in prod — must be private-only.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    not resource.values.encryption_config
    msg := sprintf("EKS cluster '%v' has no encryption_config — Kubernetes Secrets must be encrypted at rest with KMS.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    log_types := resource.values.enabled_cluster_log_types
    required := {"api", "audit", "authenticator", "controllerManager", "scheduler"}
    missing := required - {l | l := log_types[_]}
    count(missing) > 0
    msg := sprintf("EKS cluster '%v' missing control plane log types: %v", [resource.name, missing])
}

# ---------------------------------------------------------------------------
# ElastiCache — encryption in transit and at rest required
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_elasticache_replication_group"
    not resource.values.at_rest_encryption_enabled == true
    msg := sprintf("ElastiCache replication group '%v' at_rest_encryption_enabled must be true.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_elasticache_replication_group"
    not resource.values.transit_encryption_enabled == true
    msg := sprintf("ElastiCache replication group '%v' transit_encryption_enabled must be true.", [resource.name])
}

# ---------------------------------------------------------------------------
# IAM — no wildcard Resource in policies, no * Action
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_iam_role_policy"
    policy := json.unmarshal(resource.values.policy)
    statement := policy.Statement[_]
    statement.Effect == "Allow"
    statement.Resource == "*"
    statement.Action == "*"
    msg := sprintf("IAM policy '%v' has Action=* with Resource=* — overly permissive.", [resource.name])
}

# ---------------------------------------------------------------------------
# Security Groups — no 0.0.0.0/0 inbound on sensitive ports
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_security_group_rule"
    resource.values.type == "ingress"
    resource.values.cidr_blocks[_] == "0.0.0.0/0"
    resource.values.from_port <= 22
    resource.values.to_port >= 22
    msg := sprintf("Security group rule '%v' allows SSH (22) from 0.0.0.0/0.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_security_group_rule"
    resource.values.type == "ingress"
    resource.values.cidr_blocks[_] == "0.0.0.0/0"
    resource.values.from_port <= 5432
    resource.values.to_port >= 5432
    msg := sprintf("Security group rule '%v' allows PostgreSQL (5432) from 0.0.0.0/0.", [resource.name])
}


Catches IaC misconfigurations before they reach AWS:

| Rule | What it catches |
|---|---|
| Public S3 buckets | `block_public_acls: false` |
| Unencrypted S3 | Missing SSE config |
| Unencrypted RDS | `storage_encrypted: false` |
| Public RDS | `publicly_accessible: true` |
| EKS public endpoint in prod | `endpoint_public_access: true` |
| EKS missing secrets encryption | No KMS for K8s secrets |
| EKS missing control plane logs | Missing audit/api logs |
| Unencrypted ElastiCache | `at_rest_encryption_enabled: false` |
| SSH from `0.0.0.0/0` | Open port 22 |
| PostgreSQL from `0.0.0.0/0` | Open port 5432 |

### Running OPA tests

```bash
# Install conftest
brew install conftest  # macOS

# Test the admission policy
conftest verify --policy policies/rego/ --namespace financial_rag

# Test Helm templates
helm template infrastructure/helm/ -f infrastructure/helm/values.yaml \
  | conftest test - --policy policies/rego/

# Test Terraform plan
cd infrastructure/terraform/environments/prod/vpc
terragrunt plan -out plan.out
terraform show -json plan.out | conftest test - --policy policies/rego/
```

The `k8s_admission_test.rego` tests run with `conftest verify` and cover:
- `:latest` tag is denied
- Image with no tag is denied
- Compliant pod passes all 10 checks
- Missing CPU request is denied
- Privileged container is denied

---

## Phase 11 Verification

```bash
# Traces visible in Jaeger
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686 &
# Open http://localhost:16686, search for service: financial-rag-api

# Metrics flowing to Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &
# Query: financial_rag:api_error_rate:rate1h

# SLO error budget visible
# Query: financial_rag:slo1:error_budget_remaining

# Trace-to-log correlation
# In Grafana: open a Loki log entry → click "View Trace" → opens Tempo
```

---

# PHASE 12 — Production Hardening, FinOps & Capstone

## New Files in This Phase

```
infrastructure/
├── kubecost/
│   └── kubecost-cost-allocation.yaml  ← Component #180
├── vpa/
│   └── financial-rag-api.yaml         ← Component #183
├── karpenter/
│   └── nodepool-consolidation.yaml    ← Component #186
└── velero/
    └── backup-schedule.yaml           ← Component #196
observability/
└── grafana/
    └── finops-dashboard.json          ← Component #200
scripts/
└── production-readiness-check.sh      ← Component #203
```

---

## Step 10 — Kubecost (Components #179–181)

```bash
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm upgrade --install kubecost kubecost/cost-analyzer \
  --namespace monitoring \
  --set kubecostToken="your-token" \
  --set prometheus.enabled=false \
  --set prometheus.fqdn=http://kube-prometheus-stack-prometheus.monitoring.svc:9090
```

Create `infrastructure/kubecost/kubecost-cost-allocation.yaml`:

```yaml
# infrastructure/kubecost/kubecost-cost-allocation.yaml
# Cost allocation labels — Kubecost reads these to attribute costs per workload
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubecost-allocation-config
  namespace: monitoring
data:
  allocation-config.yaml: |
    labelConfig:
      # Cost split by component (api, agent, ingestion, pgvector, redis)
      departmentLabel: "app.kubernetes.io/component"
      # Cost split by environment
      environmentLabel: "environment"
      # Cost split by project
      ownerLabel: "app.kubernetes.io/name"
    shareTenancyCosts: true
    sharedNamespaces:
      - monitoring
      - kube-system
      - vault
```

Access the Kubecost dashboard:

```bash
kubectl port-forward -n monitoring svc/kubecost-cost-analyzer 9090:9090 &
# Open http://localhost:9090
```

Key cost reports to review:
- **Namespace costs:** financial-rag vs monitoring vs kube-system
- **Component costs:** API pod hours vs agent pod hours vs pgvector storage
- **Efficiency:** requested vs used CPU and memory (right-sizing signal)

---

## Step 11 — VPA (Components #182–185)

VPA observes actual resource usage and recommends better `requests`/`limits`.

```bash
# Install VPA
helm repo add fairwinds-stable https://charts.fairwinds.com/stable
helm upgrade --install vpa fairwinds-stable/vpa \
  --namespace kube-system \
  --set recommender.enabled=true \
  --set updater.enabled=false \
  --set admissionPlugin.enabled=false
```

Create `infrastructure/vpa/financial-rag-api.yaml`:

```yaml
# infrastructure/vpa/financial-rag-api.yaml
# Component #183 — VPA in Recommend-only mode (never modifies live pods)
# Review recommendations weekly. Apply manually after testing in staging.
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: financial-rag-api
  namespace: financial-rag
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-rag-agent-api
  updatePolicy:
    updateMode: "Off"       # NEVER auto-update — recommendations only
  resourcePolicy:
    containerPolicies:
      - containerName: api
        minAllowed:
          cpu: "100m"
          memory: "256Mi"
        maxAllowed:
          cpu: "4000m"
          memory: "8Gi"
        controlledResources: ["cpu", "memory"]
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: financial-rag-agent
  namespace: financial-rag
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-rag-agent-agent
  updatePolicy:
    updateMode: "Off"
  resourcePolicy:
    containerPolicies:
      - containerName: agent
        minAllowed:
          cpu: "250m"
          memory: "512Mi"
        maxAllowed:
          cpu: "8000m"
          memory: "16Gi"
        controlledResources: ["cpu", "memory"]
```

> **Why `updateMode: "Off"`?** VPA in Auto mode restarts pods to apply new
> resource values. For an LLM agent mid-reasoning-loop, a restart drops the
> request. Use Recommend mode, review weekly, and apply manually during the
> maintenance window.

Apply and read recommendations:

```bash
kubectl apply -f infrastructure/vpa/financial-rag-api.yaml

# Wait 24 hours for VPA to collect data, then review
kubectl describe vpa financial-rag-api -n financial-rag
# Look for: Recommendation: Container: api, Target: cpu/memory
```

---

## Step 12 — Karpenter Consolidation (Components #186–188)

Use the three NodePool files from your repo (`application`, `ingestion`,
`spot-burst`). They already have consolidation policies set:

```yaml
# application NodePool
disruption:
  consolidationPolicy: WhenUnderutilized
  consolidateAfter: 30s     # consolidate quickly when underutilised
  expireAfter: 720h         # replace nodes after 30 days (OS patches)

# ingestion NodePool (spot)
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 5m      # wait 5 min after pods exit before terminating node

# spot-burst NodePool
disruption:
  consolidationPolicy: WhenUnderutilized
  consolidateAfter: 2m
  expireAfter: 24h          # short-lived burst nodes expire daily
```

Apply and verify consolidation:

```bash
kubectl apply -f infrastructure/helm/karpenter/

# Watch consolidation in action — scale down replicas then watch nodes
kubectl scale deployment financial-rag-agent-api \
  --replicas=1 -n financial-rag

# Watch Karpenter logs for consolidation decisions
kubectl logs -n karpenter deployment/karpenter --follow \
  | grep -i "consolidat"

# Check node count before and after
kubectl get nodes -l role=application
```

---

## Step 13 — Pod Security Admission (Components #191–192)

```bash
# Label the namespace with restricted Pod Security Standard
kubectl label namespace financial-rag \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/enforce-version=v1.29 \
  pod-security.kubernetes.io/warn=restricted \
  pod-security.kubernetes.io/audit=restricted
```

Test that privileged pods are rejected:

```bash
# This should FAIL — privileged pod blocked by PSA
kubectl run test-privileged \
  --image=alpine \
  --namespace=financial-rag \
  --overrides='{"spec":{"containers":[{"name":"test","image":"alpine","securityContext":{"privileged":true}}]}}'

# Expected output:
# Error from server (Forbidden): pods "test-privileged" is forbidden:
# violates PodSecurity "restricted:v1.29": privileged
```

---

## Step 14 — OPA Gatekeeper (Components #193–194)

```bash
helm repo add gatekeeper https://open-policy-agent.github.io/gatekeeper/charts
helm upgrade --install gatekeeper gatekeeper/gatekeeper \
  --namespace gatekeeper-system \
  --create-namespace
```

Create a constraint from the `k8s_admission.rego` policies:

```yaml
# infrastructure/gatekeeper/no-latest-tag.yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: nolatestimage
spec:
  crd:
    spec:
      names:
        kind: NoLatestImage
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package nolatestimage
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          endswith(container.image, ":latest")
          msg := sprintf("Container '%v' uses :latest tag.", [container.name])
        }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: NoLatestImage
metadata:
  name: no-latest-image
spec:
  match:
    namespaces: ["financial-rag"]
  enforcementAction: deny
```

```bash
kubectl apply -f infrastructure/gatekeeper/no-latest-tag.yaml
```

---

## Step 15 — Velero Backup (Components #195–199)

```bash
# Install Velero CLI
brew install velero  # macOS

# Install Velero server with S3 backend
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.9.0 \
  --bucket financial-rag-velero-backups \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1 \
  --secret-file ./velero-credentials
```

Create `infrastructure/velero/backup-schedule.yaml`:

```yaml
# infrastructure/velero/backup-schedule.yaml
# Component #196 — daily backup of the financial-rag namespace
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup
  namespace: velero
spec:
  # Run daily at 01:00 UTC — before the ingestion CronJob at 02:00
  schedule: "0 1 * * *"
  template:
    includedNamespaces:
      - financial-rag
      - vault
    includedResources:
      - "*"
    includeClusterResources: true
    # Retain 14 days of backups
    ttl: 336h0m0s
    snapshotVolumes: true
    storageLocation: default
    labelSelector:
      matchExpressions:
        - key: app.kubernetes.io/name
          operator: In
          values: [financial-rag-agent]
```

Apply and test restore:

```bash
# Apply backup schedule
kubectl apply -f infrastructure/velero/backup-schedule.yaml

# Create an on-demand backup before any major upgrade
velero backup create pre-upgrade-backup \
  --include-namespaces financial-rag \
  --wait

# Verify backup
velero backup describe pre-upgrade-backup

# Test restore (to a test namespace — never restore over prod to test)
velero restore create test-restore \
  --from-backup pre-upgrade-backup \
  --namespace-mappings financial-rag:financial-rag-restore \
  --wait

# Verify restore
kubectl get pods -n financial-rag-restore

# Clean up test restore
kubectl delete namespace financial-rag-restore
```

---

## Step 16 — Production Readiness Check (Component #203)

Create `scripts/production-readiness-check.sh`:

```bash
#!/usr/bin/env bash
# =============================================================================
# scripts/production-readiness-check.sh
# Component #203 — Final production readiness validation.
# Runs all checks and produces a pass/fail report.
# Usage: ./scripts/production-readiness-check.sh [--namespace financial-rag]
# =============================================================================

set -euo pipefail

NAMESPACE="${1:-financial-rag}"
PASS=0
FAIL=0
WARN=0

green() { echo "  ✅ $*"; ((PASS++)); }
red()   { echo "  ❌ $*"; ((FAIL++)); }
warn()  { echo "  ⚠️  $*"; ((WARN++)); }
header(){ echo; echo "━━━ $* ━━━"; }

echo "═══════════════════════════════════════════════════"
echo " Financial RAG Agent — Production Readiness Check"
echo " Namespace: ${NAMESPACE}"
echo " Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════"

# =============================================================================
# 1. Infrastructure
# =============================================================================
header "Infrastructure"

# Cluster connectivity
if kubectl cluster-info &>/dev/null; then
  green "Cluster reachable"
else
  red "Cannot reach Kubernetes cluster"
fi

# All nodes ready
NOT_READY=$(kubectl get nodes --no-headers | grep -v " Ready" | wc -l)
if [ "$NOT_READY" -eq 0 ]; then
  green "All nodes Ready"
else
  red "${NOT_READY} node(s) not Ready"
fi

# =============================================================================
# 2. Application Pods
# =============================================================================
header "Application Pods"

for component in api agent; do
  READY=$(kubectl get pods -n "$NAMESPACE" \
    -l "app.kubernetes.io/component=${component}" \
    --field-selector=status.phase=Running \
    --no-headers 2>/dev/null | wc -l)
  DESIRED=$(kubectl get deployment -n "$NAMESPACE" \
    -l "app.kubernetes.io/component=${component}" \
    -o jsonpath='{.items[0].spec.replicas}' 2>/dev/null || echo 0)
  if [ "$READY" -ge "$DESIRED" ] && [ "$DESIRED" -gt 0 ]; then
    green "${component}: ${READY}/${DESIRED} pods running"
  else
    red "${component}: ${READY}/${DESIRED} pods running"
  fi
done

# StatefulSets
for component in pgvector redis; do
  READY=$(kubectl get statefulset -n "$NAMESPACE" \
    -l "app.kubernetes.io/component=${component}" \
    -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo 0)
  if [ "$READY" -ge 1 ]; then
    green "${component}: ${READY} replica(s) ready"
  else
    red "${component}: not ready"
  fi
done

# =============================================================================
# 3. Health Endpoints
# =============================================================================
header "Health Endpoints"

API_HEALTH=$(kubectl exec -n "$NAMESPACE" \
  "$(kubectl get pod -n "$NAMESPACE" -l app.kubernetes.io/component=api \
  -o name | head -1)" \
  -- curl -sf http://localhost:8000/health 2>/dev/null || echo "FAILED")

if echo "$API_HEALTH" | grep -q '"status":"healthy"'; then
  green "API /health → healthy"
elif echo "$API_HEALTH" | grep -q '"status":"degraded"'; then
  warn "API /health → degraded (cache down)"
else
  red "API /health → unhealthy or unreachable"
fi

# =============================================================================
# 4. Security
# =============================================================================
header "Security"

# All containers running as non-root
ROOT_CONTAINERS=$(kubectl get pods -n "$NAMESPACE" \
  -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.securityContext.runAsUser}{"\n"}{end}{end}' \
  2>/dev/null | grep "^0$" | wc -l)
if [ "$ROOT_CONTAINERS" -eq 0 ]; then
  green "No containers running as root (UID 0)"
else
  red "${ROOT_CONTAINERS} container(s) running as root"
fi

# All images have pinned tags (no :latest)
LATEST_IMAGES=$(kubectl get pods -n "$NAMESPACE" \
  -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.image}{"\n"}{end}{end}' \
  2>/dev/null | grep ":latest" | wc -l)
if [ "$LATEST_IMAGES" -eq 0 ]; then
  green "No :latest image tags in production"
else
  red "${LATEST_IMAGES} container(s) using :latest tag"
fi

# Vault agent sidecars running
VAULT_SIDECARS=$(kubectl get pods -n "$NAMESPACE" \
  -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.name}{"\n"}{end}{end}' \
  2>/dev/null | grep "vault-agent" | wc -l)
if [ "$VAULT_SIDECARS" -ge 2 ]; then
  green "Vault Agent sidecars running (${VAULT_SIDECARS})"
else
  warn "Vault Agent sidecars: ${VAULT_SIDECARS} (expected ≥ 2)"
fi

# Falco running
FALCO_PODS=$(kubectl get pods -n kube-system \
  -l app.kubernetes.io/name=falco --no-headers 2>/dev/null | \
  grep Running | wc -l)
if [ "$FALCO_PODS" -ge 1 ]; then
  green "Falco running (${FALCO_PODS} pod(s))"
else
  red "Falco not running"
fi

# =============================================================================
# 5. HPA and Autoscaling
# =============================================================================
header "Autoscaling"

for component in api agent; do
  HPA_STATUS=$(kubectl get hpa -n "$NAMESPACE" \
    -l "app.kubernetes.io/component=${component}" \
    -o jsonpath='{.items[0].status.conditions[?(@.type=="AbleToScale")].status}' \
    2>/dev/null || echo "Unknown")
  if [ "$HPA_STATUS" = "True" ]; then
    green "HPA ${component}: able to scale"
  else
    warn "HPA ${component}: status=${HPA_STATUS}"
  fi
done

# =============================================================================
# 6. Observability
# =============================================================================
header "Observability"

# Prometheus running
PROM_PODS=$(kubectl get pods -n monitoring \
  -l app.kubernetes.io/name=prometheus --no-headers 2>/dev/null | \
  grep Running | wc -l)
if [ "$PROM_PODS" -ge 1 ]; then
  green "Prometheus running"
else
  warn "Prometheus not running"
fi

# Grafana running
GRAFANA_PODS=$(kubectl get pods -n observability \
  -l app.kubernetes.io/name=grafana --no-headers 2>/dev/null | \
  grep Running | wc -l)
if [ "$GRAFANA_PODS" -ge 1 ]; then
  green "Grafana running"
else
  warn "Grafana not running"
fi

# OTel Collector running
OTEL_PODS=$(kubectl get pods -n "$NAMESPACE" \
  -l app.kubernetes.io/component=opentelemetry-collector \
  --no-headers 2>/dev/null | grep Running | wc -l)
if [ "$OTEL_PODS" -ge 1 ]; then
  green "OTel Collector running"
else
  warn "OTel Collector not running"
fi

# =============================================================================
# 7. Backup
# =============================================================================
header "Backup"

LAST_BACKUP=$(velero backup get --output json 2>/dev/null | \
  python3 -c "import sys,json; items=json.load(sys.stdin).get('items',[]); \
  items.sort(key=lambda x: x['metadata']['creationTimestamp']); \
  print(items[-1]['status']['phase'] if items else 'None')" 2>/dev/null || echo "Unknown")
if [ "$LAST_BACKUP" = "Completed" ]; then
  green "Latest Velero backup: Completed"
elif [ "$LAST_BACKUP" = "None" ]; then
  warn "No Velero backups found"
else
  red "Latest Velero backup: ${LAST_BACKUP}"
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo "═══════════════════════════════════════════════════"
echo " Results: ✅ ${PASS} passed  ❌ ${FAIL} failed  ⚠️  ${WARN} warnings"
echo "═══════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
  echo " Status: NOT PRODUCTION READY — resolve failures before deploying"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo " Status: REVIEW WARNINGS before proceeding"
  exit 0
else
  echo " Status: PRODUCTION READY ✅"
  exit 0
fi
```

Make it executable and run it:

```bash
chmod +x scripts/production-readiness-check.sh
./scripts/production-readiness-check.sh financial-rag
```

---

## Step 17 — FinOps Dashboard (Component #200)

Create `observability/grafana/finops-dashboard.json` — a Grafana dashboard
with four panels. Import it via Grafana UI → Dashboards → Import:

```json
{
  "title": "Financial RAG — FinOps",
  "panels": [
    {
      "title": "Monthly Cost by Component",
      "type": "piechart",
      "targets": [{
        "expr": "sum by (component) (kubecost_cluster_costs_total{namespace='financial-rag'})"
      }]
    },
    {
      "title": "Cost per Query (cents)",
      "type": "stat",
      "targets": [{
        "expr": "financial_rag:cost_per_query_cents:1h"
      }]
    },
    {
      "title": "Error Budget Remaining (%)",
      "type": "gauge",
      "fieldConfig": {
        "thresholds": {
          "steps": [
            {"color": "red", "value": 0},
            {"color": "yellow", "value": 25},
            {"color": "green", "value": 50}
          ]
        }
      },
      "targets": [{
        "expr": "financial_rag:slo_availability:error_budget_remaining"
      }]
    },
    {
      "title": "LLM Token Cost ($/hour)",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(financial_rag_llm_cost_usd_total[1h]) * 3600"
      }]
    }
  ]
}
```

Apply the budget alerts (Component #201):

```bash
kubectl apply -f - <<'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: financial-rag-budget-alerts
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
spec:
  groups:
    - name: financial-rag.finops
      rules:
        - alert: MonthlyBudgetExceeded
          expr: |
            sum(kubecost_cluster_costs_total{namespace="financial-rag"}) > 5000
          for: 1h
          labels:
            severity: warning
            team: finops
          annotations:
            summary: "Monthly infrastructure cost exceeds $5,000"
            description: "Current monthly cost: ${{ $value | humanize }}. Review Karpenter consolidation and pod right-sizing."

        - alert: LLMCostSpike
          expr: |
            rate(financial_rag_llm_cost_usd_total[1h]) * 3600 > 50
          for: 15m
          labels:
            severity: warning
            team: finops
          annotations:
            summary: "LLM API cost > $50/hour"
            description: "Current LLM cost rate: ${{ $value | humanize }}/hour. Check for runaway agent loops or prompt injection."
EOF
```

---

## Final Checklist — All 12 Phases

| Phase | Status | Key Deliverable |
|---|---|---|
| 1 | ✅ | PostgreSQL + Redis + Settings |
| 2 | ✅ | SEC EDGAR ingestion + parsers |
| 3 | ✅ | Embeddings + hybrid search + Alembic |
| 4 | ✅ | LLM agent + audit trail |
| 5 | ✅ | FastAPI + auth + metrics |
| 6 | ✅ | Vault dynamic secrets |
| 7 | ✅ | CI/CD pipeline |
| 8 | ✅ | Helm + EKS + Karpenter + Terragrunt |
| 9 | ✅ | ArgoCD + Cilium + Istio |
| 10 | ✅ | Falco runtime security |
| 11 | ✅ | LGTM observability + SLOs + OPA policies |
| 12 | ✅ | Kubecost + VPA + Velero + Production readiness |

---

## Common Errors and Fixes — Phases 11 & 12

| Error | Cause | Fix |
|---|---|---|
| `OTLPSpanExporter: failed to export` | OTel Collector not running | `kubectl get pods -n financial-rag -l component=opentelemetry-collector` |
| Traces not in Jaeger | Sampling rate too low | Set `OTEL_TRACE_SAMPLE_RATE=1.0` temporarily in dev |
| `engine.sync_engine` AttributeError | SQLAlchemy async engine | Call `configure_tracing` after `get_db_client()` connects |
| SLO recording rules not found | Prometheus not loading | Check `release: kube-prometheus-stack` label on PrometheusRule |
| Conftest `deny` on compliant pod | Missing required label | Add `app.kubernetes.io/component` label to pod spec |
| VPA shows no recommendations | Insufficient history | Wait 24–48 hours after applying VPA |
| Velero backup `PartiallyFailed` | PVC snapshot permissions | Verify `velero-credentials` has `ec2:CreateSnapshot` IAM permission |
| Production readiness check fails | Pods using `:latest` | Pin image tags in `values.prod.yaml` to a semver or SHA |
| Grafana `no data` on FinOps panels | Kubecost not scraping | Check Kubecost ServiceMonitor and `release` label |
| `conftest verify` test failures | Policy logic error | Run `opa test policies/rego/ -v` directly |
