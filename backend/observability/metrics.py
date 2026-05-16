"""Prometheus metrics."""

from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "rag_http_requests_total",
    "Total HTTP requests served.",
    labelnames=("method", "path", "status"),
)

http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

queries_total = Counter(
    "rag_queries_total",
    "Total /v1/query calls, by variant and cache outcome.",
    labelnames=("variant", "cache_hit"),
)

ingest_total = Counter(
    "rag_ingest_total",
    "Total /v1/ingest calls, by outcome.",
    labelnames=("outcome",),
)

ingest_chunks_total = Counter(
    "rag_ingest_chunks_total",
    "Total chunks stored across all ingests.",
)
