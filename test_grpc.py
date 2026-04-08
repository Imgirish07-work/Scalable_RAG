import time
from qdrant_client import QdrantClient
from config.settings import settings

URL  = settings.qdrant_url
HOST = URL.replace("https://", "").replace("http://", "")
KEY  = settings.qdrant_api_key

print(f"Cluster : {HOST}")
print(f"gRPC port: 6334\n")

# ── gRPC probe ────────────────────────────────────────────────────────────────
grpc_available = False
grpc_client = None

try:
    print("[gRPC] Connecting...")
    grpc_client = QdrantClient(
        host=HOST,
        api_key=KEY,
        prefer_grpc=True,
        port=6334,
        https=True,
    )
    result = grpc_client.get_collections()
    grpc_available = True
    print(f"[gRPC] Connected — collections: {[c.name for c in result.collections]}")
except Exception as e:
    print(f"[gRPC] FAILED — {type(e).__name__}: {e}")
    print("[gRPC] Falling back to HTTP\n")

# ── HTTP probe ────────────────────────────────────────────────────────────────
print("[HTTP] Connecting...")
http_client = QdrantClient(url=URL, api_key=KEY)
result = http_client.get_collections()
print(f"[HTTP] Connected — collections: {[c.name for c in result.collections]}\n")

# ── Benchmark (only if gRPC succeeded) ───────────────────────────────────────
if grpc_available:
    print("Benchmarking 10 calls each (after warmup)...\n")
    for label, client in [("HTTP", http_client), ("gRPC", grpc_client)]:
        t0 = time.perf_counter()
        for _ in range(10):
            client.get_collections()
        avg = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {label}: avg={avg:.1f}ms per call")
else:
    print("Benchmarking skipped — gRPC unavailable, HTTP is the active transport.")
    t0 = time.perf_counter()
    for _ in range(10):
        http_client.get_collections()
    avg = (time.perf_counter() - t0) / 10 * 1000
    print(f"  HTTP: avg={avg:.1f}ms per call")

print("\nResult:", "gRPC supported — consider enabling" if grpc_available else "gRPC NOT supported on this network — stay on HTTP")
