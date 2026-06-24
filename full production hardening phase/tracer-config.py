# # =============================================================================
# # observability/tracing/tracer_config.py
# # OpenTelemetry tracer provider configuration for financial-rag-agent.
# # Import and call configure_tracing() at application startup (main.py).
# # Instruments: FastAPI, SQLAlchemy (asyncpg), Redis, httpx, logging.
# # =============================================================================
# import logging
# import os

# from opentelemetry import trace
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
# from opentelemetry.instrumentation.logging import LoggingInstrumentor
# from opentelemetry.instrumentation.redis import RedisInstrumentor
# from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

# logger = logging.getLogger(__name__)

# OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://financial-rag-otel-collector:4317")
# SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "financial-rag-api")
# ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
# SAMPLING_RATE = float(
#     os.getenv("OTEL_TRACE_SAMPLE_RATE", "0.01")
# )  # 1% default, 100% for errors via tail sampling


# def configure_tracing(app=None, engine=None) -> TracerProvider:
#     """
#     Configure the OpenTelemetry tracer provider and instrument all frameworks.
#     Call once at application startup before the ASGI app is created.

#     Args:
#         app:    FastAPI application instance (for FastAPI instrumentation)
#         engine: SQLAlchemy async engine (for database query tracing)

#     Returns:
#         TracerProvider — the configured provider (also set as global)
#     """
#     # Resource: identifies this service in all traces
#     resource = Resource.create(
#         {
#             "service.name": SERVICE_NAME,
#             "service.version": os.getenv("APP_VERSION", "unknown"),
#             "service.namespace": "financial-rag",
#             "deployment.environment": ENVIRONMENT,
#             "k8s.namespace.name": os.getenv("K8S_NAMESPACE", "financial-rag"),
#             "k8s.pod.name": os.getenv("HOSTNAME", "unknown"),
#         }
#     )

#     # Sampler: low rate by default — tail sampling in the collector handles the rest
#     sampler = ParentBased(root=TraceIdRatioBased(SAMPLING_RATE))

#     # Provider
#     provider = TracerProvider(resource=resource, sampler=sampler)

#     # OTLP exporter → OpenTelemetry Collector → Jaeger + X-Ray
#     otlp_exporter = OTLPSpanExporter(
#         endpoint=OTEL_ENDPOINT,
#         insecure=ENVIRONMENT != "prod",
#     )
#     provider.add_span_processor(
#         BatchSpanProcessor(
#             otlp_exporter,
#             max_queue_size=2048,
#             max_export_batch_size=512,
#             export_timeout_millis=5000,
#         )
#     )

#     # Console exporter in dev for local debugging
#     if ENVIRONMENT == "dev":
#         provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

#     # Set as global provider
#     trace.set_tracer_provider(provider)
#     logger.info(
#         "TracerProvider configured: service=%s env=%s sampler_rate=%.2f%%",
#         SERVICE_NAME,
#         ENVIRONMENT,
#         SAMPLING_RATE * 100,
#     )

#     # Instrument FastAPI — adds spans for every HTTP request
#     if app is not None:
#         FastAPIInstrumentor.instrument_app(
#             app,
#             excluded_urls="/health,/metrics",  # skip health + metrics endpoints
#             server_request_hook=_add_request_attributes,
#         )

#     # Instrument SQLAlchemy — adds spans for every DB query
#     if engine is not None:
#         SQLAlchemyInstrumentor().instrument(
#             engine=engine.sync_engine,
#             enable_commenter=True,  # adds traceparent comment to SQL queries
#             commenter_options={"db_framework": True},
#         )

#     # Instrument Redis — adds spans for every cache operation
#     RedisInstrumentor().instrument()

#     # Instrument httpx — adds spans for LLM API calls (OpenAI, Groq, Azure)
#     HTTPXClientInstrumentor().instrument(
#         request_hook=_add_llm_attributes,
#     )

#     # Correlate trace IDs with log records
#     LoggingInstrumentor().instrument(set_logging_format=True)

#     return provider


# def _add_request_attributes(span, scope):
#     """Add financial-rag-specific attributes to HTTP request spans."""
#     if span and span.is_recording():
#         # Tag agent spans for the tail sampler policy
#         if "/query" in scope.get("path", ""):
#             span.set_attribute("component", "query-endpoint")
#             span.set_attribute("financial_rag.query_type", "rag")


# def _add_llm_attributes(span, request):
#     """Add LLM provider info to outbound httpx spans."""
#     if span and span.is_recording():
#         url = str(request.url)
#         if "openai" in url:
#             span.set_attribute("llm.provider", "openai")
#             span.set_attribute("component", "llm-client")
#         elif "groq" in url:
#             span.set_attribute("llm.provider", "groq")
#             span.set_attribute("component", "llm-client")
#         elif "azure" in url:
#             span.set_attribute("llm.provider", "azure-openai")
#             span.set_attribute("component", "llm-client")


# def get_tracer(name: str):
#     """Get a named tracer for manual instrumentation."""
#     return trace.get_tracer(name, schema_url="https://opentelemetry.io/schemas/1.24.0")
# =============================================================================
# observability/tracing/tracer_config.py
# OpenTelemetry tracer provider configuration for financial-rag-agent.
# =============================================================================
import logging
import os
from typing import Any

# Import types for instrumentation
from fastapi import FastAPI
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
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "financial-rag-api")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SAMPLING_RATE = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "0.01"))


def configure_tracing(app: FastAPI | None = None, engine: Any | None = None) -> TracerProvider:
    """
    Configure the OpenTelemetry tracer provider and instrument all frameworks.
    """
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": os.getenv("APP_VERSION", "unknown"),
            "service.namespace": "financial-rag",
            "deployment.environment": ENVIRONMENT,
            "k8s.namespace.name": os.getenv("K8S_NAMESPACE", "financial-rag"),
            "k8s.pod.name": os.getenv("HOSTNAME", "unknown"),
        }
    )

    sampler = ParentBased(root=TraceIdRatioBased(SAMPLING_RATE))
    provider = TracerProvider(resource=resource, sampler=sampler)

    otlp_exporter = OTLPSpanExporter(
        endpoint=OTEL_ENDPOINT,
        insecure=ENVIRONMENT != "prod",
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=5000,
        )
    )

    if ENVIRONMENT == "dev":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    logger.info(
        "TracerProvider configured: service=%s env=%s sampler_rate=%.2f%%",
        SERVICE_NAME,
        ENVIRONMENT,
        SAMPLING_RATE * 100,
    )

    if app is not None:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health,/metrics",
            server_request_hook=_add_request_attributes,
        )

    if engine is not None:
        SQLAlchemyInstrumentor().instrument(
            engine=engine.sync_engine,
            enable_commenter=True,
            commenter_options={"db_framework": True},
        )

    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument(request_hook=_add_llm_attributes)
    LoggingInstrumentor().instrument(set_logging_format=True)

    return provider


def _add_request_attributes(span: Any, scope: dict[str, Any]) -> None:
    """Add financial-rag-specific attributes to HTTP request spans."""
    if span and span.is_recording() and "/query" in scope.get("path", ""):
        span.set_attribute("component", "query-endpoint")
        span.set_attribute("financial_rag.query_type", "rag")


def _add_llm_attributes(span: Any, request: Any) -> None:
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


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer for manual instrumentation."""
    return trace.get_tracer(name, schema_url="https://opentelemetry.io/schemas/1.24.0")
