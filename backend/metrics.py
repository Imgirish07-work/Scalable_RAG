"""Prometheus counters, gauges, and histograms exported by the backend."""

from prometheus_client import Counter, Gauge, Histogram

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
    "Total ingest calls, by outcome.",
    labelnames=("outcome",),
)

ingest_chunks_total = Counter(
    "rag_ingest_chunks_total",
    "Total chunks stored across all ingests.",
)

ingest_jobs_queued_total = Counter(
    "rag_ingest_jobs_queued_total",
    "Total ingest jobs accepted onto the Arq queue.",
    labelnames=("origin",),
)

ingest_jobs_failed_total = Counter(
    "rag_ingest_jobs_failed_total",
    "Total ingest jobs that ended in DLQ, by failure reason.",
    labelnames=("reason",),
)

ingest_jobs_inflight = Gauge(
    "rag_ingest_jobs_inflight",
    "Ingest jobs currently being processed by this worker.",
)

ingest_duration_seconds = Histogram(
    "rag_ingest_duration_seconds",
    "End-to-end ingest latency per document, by outcome.",
    labelnames=("outcome",),
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200),
)

ingest_batch_duration_seconds = Histogram(
    "rag_ingest_batch_duration_seconds",
    "Per-batch embed+upsert latency inside add_documents.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "Query latency, by variant and cache outcome.",
    labelnames=("variant", "cache_hit"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
