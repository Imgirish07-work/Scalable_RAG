import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"]                   = ""
os.environ["REQUESTS_CA_BUNDLE"]               = ""

from vectorstore.embeddings import get_embeddings, get_embedding_dimension

# Test 1 — Model loads
print("Test 1 — Loading model...")
embeddings = get_embeddings()
print("✅ Model loaded\n")

# Test 2 — Dimension
print("Test 2 — Checking dimension...")
dim = get_embedding_dimension()
print(f"✅ Dimension = {dim}  (expected 384 for bge-small)\n")

# Test 3 — Embed a query
print("Test 3 — Embedding a query...")
vector = embeddings.embed_query("What is RAG?")
print(f"✅ Vector length = {len(vector)}")
print(f"✅ First 5 values = {vector[:5]}\n")

# Test 4 — lru_cache works (same instance returned)
print("Test 4 — Checking lru_cache...")
embeddings2 = get_embeddings()
print(f"✅ Same instance = {embeddings is embeddings2}  (must be True)\n")

print("All tests passed ✅")