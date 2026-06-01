"""Soak test for ingestion stability — uploads N docs sequentially and reports drift."""

import argparse
import asyncio
import json
import mimetypes
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from httpx_sse import aconnect_sse


_TERMINAL_PHASES = {"ready", "failed", "duplicate"}
_PER_DOC_TIMEOUT_S = 1800
_UPLOAD_TIMEOUT_S = 300


async def _upload_one(
    client: httpx.AsyncClient,
    api_url: str,
    path: Path,
) -> dict:
    """Run the create→PUT→finalize→SSE flow for one doc; return result dict."""
    file_name = path.name
    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    size_bytes = path.stat().st_size

    start = time.perf_counter()

    session_resp = await client.post(
        f"{api_url}/v1/ingest",
        json={
            "file_name": file_name,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "collection": "default",
        },
        timeout=30,
    )
    session_resp.raise_for_status()
    session = session_resp.json()
    doc_id = session["doc_id"]
    presigned_url = session["presigned_url"]

    with path.open("rb") as fh:
        put_resp = await client.put(
            presigned_url,
            content=fh.read(),
            headers={"Content-Type": mime_type},
            timeout=_UPLOAD_TIMEOUT_S,
        )
    put_resp.raise_for_status()

    finalize_resp = await client.post(
        f"{api_url}/v1/documents/{doc_id}/finalize", timeout=30,
    )
    finalize_resp.raise_for_status()

    terminal_phase, message = await _wait_for_terminal(client, api_url, doc_id)

    elapsed = time.perf_counter() - start
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "size_bytes": size_bytes,
        "elapsed_s": round(elapsed, 2),
        "phase": terminal_phase,
        "message": message,
    }


async def _wait_for_terminal(
    client: httpx.AsyncClient, api_url: str, doc_id: str,
) -> tuple[str, str]:
    """Consume SSE stream until a terminal phase arrives or timeout fires."""
    url = f"{api_url}/v1/documents/{doc_id}/events"
    deadline = time.monotonic() + _PER_DOC_TIMEOUT_S

    async with aconnect_sse(client, "GET", url, timeout=_PER_DOC_TIMEOUT_S) as sse:
        async for event in sse.aiter_sse():
            if time.monotonic() > deadline:
                return "timeout", "SSE wait exceeded per-doc deadline"
            if not event.data:
                continue
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            phase = payload.get("phase")
            if phase in _TERMINAL_PHASES:
                return phase, payload.get("message", "")

    return "stream_closed", "SSE closed without terminal event"


async def _fetch_metrics(client: httpx.AsyncClient, api_url: str) -> str:
    """Pull raw Prometheus exposition text from /metrics."""
    resp = await client.get(f"{api_url}/metrics", timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_histogram_p99(metrics_text: str, metric_name: str) -> Optional[float]:
    """Approximate p99 from a Prometheus histogram by reading _count/_sum/_bucket lines."""
    sum_total = 0.0
    count_total = 0.0
    buckets: list[tuple[float, float]] = []
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(metric_name):
            continue
        if "_bucket{" in line and 'le="' in line:
            try:
                le_str = line.split('le="', 1)[1].split('"', 1)[0]
                count_str = line.rsplit(" ", 1)[1]
                le = float("inf") if le_str == "+Inf" else float(le_str)
                buckets.append((le, float(count_str)))
            except (ValueError, IndexError):
                continue
        elif line.startswith(f"{metric_name}_sum"):
            sum_total = float(line.rsplit(" ", 1)[1])
        elif line.startswith(f"{metric_name}_count"):
            count_total = float(line.rsplit(" ", 1)[1])

    if count_total <= 0 or not buckets:
        return None
    buckets.sort()
    target = 0.99 * count_total
    for le, cumulative in buckets:
        if cumulative >= target:
            return le if le != float("inf") else sum_total / count_total
    return None


def _drift_pct(first_half: list[float], second_half: list[float]) -> float:
    """Mean drift between halves as a percentage of first-half mean."""
    if not first_half or not second_half:
        return 0.0
    a = statistics.mean(first_half)
    b = statistics.mean(second_half)
    if a == 0:
        return 0.0
    return round((b - a) / a * 100, 1)


def _report(results: list[dict], metrics_text: str) -> int:
    """Print stability summary and return exit code (0 = pass, 1 = fail)."""
    durations = [r["elapsed_s"] for r in results if r["phase"] == "ready"]
    failures = [r for r in results if r["phase"] not in ("ready", "duplicate")]
    duplicates = [r for r in results if r["phase"] == "duplicate"]

    if not durations:
        print("\nFAIL — no successful ingestions to report")
        return 1

    p50 = statistics.median(durations)
    p95 = statistics.quantiles(durations, n=20)[-1] if len(durations) >= 20 else max(durations)
    p99 = statistics.quantiles(durations, n=100)[-1] if len(durations) >= 100 else max(durations)

    mid = len(durations) // 2
    drift = _drift_pct(durations[:mid], durations[mid:])

    batch_p99 = _parse_histogram_p99(metrics_text, "rag_ingest_batch_duration_seconds")

    print()
    print("=" * 60)
    print(f"Soak test results — {len(results)} documents")
    print("=" * 60)
    print(f"Ready:      {len(durations)}")
    print(f"Duplicate:  {len(duplicates)}")
    print(f"Failed:     {len(failures)}")
    print()
    print(f"Per-doc latency (ready only):")
    print(f"  p50: {p50:.1f}s   p95: {p95:.1f}s   p99: {p99:.1f}s")
    print(f"  min: {min(durations):.1f}s   max: {max(durations):.1f}s")
    if batch_p99 is not None:
        print(f"  per-batch upsert p99: {batch_p99:.2f}s")
    print()
    print(f"Drift first-half → second-half: {drift:+.1f}%")
    print()

    pass_drift = abs(drift) < 15
    pass_failures = len(failures) == 0
    pass_p99 = p99 < (2.5 * p50) if p50 > 0 else True

    print(f"  drift < 15%:        {'PASS' if pass_drift else 'FAIL'}")
    print(f"  zero failures:      {'PASS' if pass_failures else 'FAIL'}")
    print(f"  p99 < 2.5x p50:     {'PASS' if pass_p99 else 'FAIL'}")
    print("=" * 60)

    return 0 if (pass_drift and pass_failures and pass_p99) else 1


async def _run(
    docs_dir: Path, count: int, api_url: str, rate_delay_s: float,
) -> int:
    candidates = sorted(p for p in docs_dir.iterdir() if p.is_file())
    if not candidates:
        print(f"No files found in {docs_dir}", file=sys.stderr)
        return 1

    selected = (candidates * ((count // len(candidates)) + 1))[:count]
    print(f"Soak test starting | docs={len(selected)} | api={api_url}")

    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for i, path in enumerate(selected, start=1):
            print(f"[{i}/{len(selected)}] {path.name} ...", flush=True)
            try:
                result = await _upload_one(client, api_url, path)
            except Exception as exc:
                result = {
                    "doc_id": "",
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "elapsed_s": 0.0,
                    "phase": "client_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                f"    → {result['phase']} in {result['elapsed_s']}s "
                f"({result['message'] or '-'})",
                flush=True,
            )
            if rate_delay_s > 0 and i < len(selected):
                await asyncio.sleep(rate_delay_s)

        metrics_text = await _fetch_metrics(client, api_url)

    return _report(results, metrics_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir", type=Path, required=True,
        help="Directory containing test documents.",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of ingestions to perform (cycles through docs-dir).",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="Base URL of the backend API.",
    )
    parser.add_argument(
        "--rate-delay", type=float, default=0.0,
        help="Seconds to wait between docs (0 = no wait).",
    )
    args = parser.parse_args()

    if not args.docs_dir.is_dir():
        print(f"docs-dir not found: {args.docs_dir}", file=sys.stderr)
        sys.exit(2)

    exit_code = asyncio.run(
        _run(args.docs_dir, args.count, args.api_url, args.rate_delay),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
