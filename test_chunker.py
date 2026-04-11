"""
Unit tests for the Chunker component.

Test scope:
    Unit tests covering token counting, structure-aware splitting (paragraph,
    table, code, list), routing, chunk filtering, deduplication, metadata
    enrichment, context prepending, total-chunk counting, the full pipeline,
    chunk statistics, and character/RLM splitting utilities.

Flow:
    Module-level execution — each Test N section runs sequentially; a failed
    assert exits immediately via sys.exit(1).

Dependencies:
    Chunker (structure-aware splitter), langchain_core Document.
    No external services required.
"""

import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"]                   = ""
os.environ["REQUESTS_CA_BUNDLE"]               = ""

import sys
from langchain_core.documents import Document
from chunking.chunker import Chunker

chunker = Chunker()

# Helpers

def section(title: str) : print(f"\n{'='*60}\n  {title}\n{'='*60}")
def ok(msg: str)        : print(f"  ✅ {msg}")
def fail(msg: str)      : print(f"  ❌ {msg}"); sys.exit(1)

def make_doc(text: str, structure_type: str = "paragraph", page: int = 1) -> Document:
    return Document(
        page_content = text,
        metadata     = {
            "source"         : "test.pdf",
            "page"           : page,
            "section"        : "Introduction",
            "structure_type" : structure_type,
            "has_table"      : structure_type == "table",
            "has_list"       : structure_type == "list",
            "has_code"       : structure_type == "code",
            "heading_level"  : 0,
        }
    )

# Test Data

PARAGRAPH   = "RAG combines retrieval with generation for grounded responses. " * 20
SMALL_PARA  = "RAG combines retrieval with generation."
TABLE_SMALL = "| Name | Score |\n|------|-------|\n| RAG  | 0.95  |\n| LLM  | 0.90  |"
TABLE_LARGE = "| Name | Score | Grade | Year |\n|------|-------|-------|------|\n" + \
              "| Model{i} | 0.9{i} | A | 2024 |\n".join([f"| Model{i} | 0.9 | A | 2024 |" for i in range(200)])
CODE_SMALL  = "```python\ndef embed(text):\n    return model.encode(text)\n```"
CODE_LARGE  = "\n".join([f"def function_{i}(x):\n    return x * {i}\n" for i in range(100)])
LIST_SMALL  = "- Install dependencies\n- Configure settings\n- Run pipeline"
LIST_LARGE  = "\n".join([f"- Item number {i} with some description text here" for i in range(200)])

# Test 1 — Empty Input

section("Test 1 — Empty Input")

result = chunker.split_documents([])
assert result == []
ok("Empty list returned as-is")

# Test 2 — Token Counter

section("Test 2 — Token Counter")

count = chunker._count_tokens("Hello world")
assert count > 0
ok(f"Token count works | 'Hello world' = {count} tokens")

count = chunker._count_tokens("")
assert count == 0
ok("Empty string = 0 tokens")

# Test 3 — Standard Paragraph Split

section("Test 3 — Standard Paragraph Split")

doc    = make_doc(PARAGRAPH, "paragraph")
chunks = chunker._standard_split(doc)

assert len(chunks) > 0
assert all(isinstance(c, Document) for c in chunks)
assert all(chunker._count_tokens(c.page_content) <= 512 for c in chunks)
ok(f"Paragraph split | chunks={len(chunks)} | all ≤ 512 tokens")

# Test 4 — Table Splitting

section("Test 4 — Table Splitting")

# 4a — Small table → single chunk
doc    = make_doc(TABLE_SMALL, "table")
chunks = chunker._split_table(doc)
assert len(chunks) == 1
ok(f"Small table → 1 chunk intact")

# 4b — Large table → multiple chunks, header repeated
doc    = make_doc(TABLE_LARGE, "table")
chunks = chunker._split_table(doc)
assert len(chunks) > 1
header = TABLE_LARGE.split("\n")[0]
assert all(header in c.page_content for c in chunks)
assert all(chunker._count_tokens(c.page_content) <= 512 for c in chunks)
ok(f"Large table → {len(chunks)} chunks | header repeated | all ≤ 512 tokens")

# Test 5 — Code Splitting

section("Test 5 — Code Splitting")

# 5a — Small code → single chunk
doc    = make_doc(CODE_SMALL, "code")
chunks = chunker._split_code(doc)
assert len(chunks) == 1
ok("Small code → 1 chunk intact")

# 5b — Large code → multiple chunks, all within token limit
doc    = make_doc(CODE_LARGE, "code")
chunks = chunker._split_code(doc)
assert len(chunks) > 1
assert all(chunker._count_tokens(c.page_content) <= 512 for c in chunks)
ok(f"Large code → {len(chunks)} chunks | all ≤ 512 tokens")

# Test 6 — List Splitting

section("Test 6 — List Splitting")

# 6a — Small list → single chunk
doc    = make_doc(LIST_SMALL, "list")
chunks = chunker._split_list(doc)
assert len(chunks) == 1
ok("Small list → 1 chunk intact")

# 6b — Large list → multiple chunks
doc    = make_doc(LIST_LARGE, "list")
chunks = chunker._split_list(doc)
assert len(chunks) > 1
assert all(chunker._count_tokens(c.page_content) <= 512 for c in chunks)
ok(f"Large list → {len(chunks)} chunks | all ≤ 512 tokens")

# Test 7 — Structure Routing

section("Test 7 — Structure-Aware Routing")

for structure_type in ["paragraph", "heading", "table", "list", "code"]:
    text = TABLE_SMALL if structure_type == "table" else SMALL_PARA
    doc  = make_doc(text, structure_type)
    chunks = chunker._split_by_structure(doc)
    assert len(chunks) > 0
    ok(f"Routed correctly | structure_type={structure_type} | chunks={len(chunks)}")

# Test 8 — Chunk Filtering

# section("Test 8 — Chunk Filtering")

# valid_chunk   = Document(page_content=SMALL_PARA,  metadata=make_doc(SMALL_PARA,  "paragraph").metadata)
# empty_chunk   = Document(page_content="",           metadata=make_doc("",           "paragraph").metadata)
# short_chunk   = Document(page_content="Hi",         metadata=make_doc("Hi",         "paragraph").metadata)
# bplate_chunk  = Document(page_content="1",          metadata=make_doc("1",          "paragraph").metadata)  # standalone digit → boilerplate

# chunks   = [empty_chunk, short_chunk, bplate_chunk, valid_chunk]
# filtered = chunker._filter_chunks(chunks)

# assert len(filtered) == 1
# assert filtered[0].page_content == SMALL_PARA
# ok(f"Filtered correctly | kept=1/{len(chunks)} | removed empty, short, boilerplate")


# ...existing code...

section("Test 8 — Chunk Filtering")

valid_chunk  = Document(page_content=SMALL_PARA, metadata=make_doc(SMALL_PARA, "paragraph").metadata)
empty_chunk  = Document(page_content="",          metadata=make_doc("",          "paragraph").metadata)
short_chunk  = Document(page_content="Hi",        metadata=make_doc("Hi",        "paragraph").metadata)
bplate_chunk = Document(page_content="1",         metadata=make_doc("1",         "paragraph").metadata)

chunks   = [empty_chunk, short_chunk, bplate_chunk, valid_chunk]
filtered = chunker._filter_chunks(chunks)

# Print what survived and why
for c in filtered:
    token_count = chunker._count_tokens(c.page_content.strip())
    print(f"  SURVIVED | tokens={token_count} | content='{c.page_content[:50]}'")

assert len(filtered) == 1
assert filtered[0].page_content == SMALL_PARA
ok(f"Filtered correctly | kept=1/{len(chunks)}")

# ...existing code...

# Test 9 — Deduplication

section("Test 9 — Deduplication")

doc         = make_doc(SMALL_PARA, "paragraph")
chunks      = [doc, doc, doc]   # same doc 3 times
seen        = set()
deduped     = chunker._deduplicate(chunks, seen)

assert len(deduped) == 1
ok(f"Deduplicated | 3 identical chunks → 1 unique chunk")

# different content → all kept
chunks  = [make_doc(f"Unique content {i} " * 5, "paragraph") for i in range(3)]
seen    = set()
deduped = chunker._deduplicate(chunks, seen)
assert len(deduped) == 3
ok("3 unique chunks → all 3 kept")

# Test 10 — Metadata Enrichment

section("Test 10 — Metadata Enrichment")

source_doc = make_doc(SMALL_PARA, "paragraph")
chunks     = [make_doc(SMALL_PARA, "paragraph")]
enriched   = chunker._enrich_metadata(chunks, source_doc)

assert "chunk_index"  in enriched[0].metadata
assert "word_count"   in enriched[0].metadata
assert "token_count"  in enriched[0].metadata
assert "doc_type"     in enriched[0].metadata
assert enriched[0].metadata["doc_type"]    == "pdf"
assert enriched[0].metadata["chunk_index"] == 0
assert enriched[0].metadata["word_count"]  > 0
assert enriched[0].metadata["token_count"] > 0
ok(f"Metadata enriched | doc_type=pdf | chunk_index=0 | word_count={enriched[0].metadata['word_count']}")

# Test 11 — Context Prepending

section("Test 11 — Context Prepending")

chunks   = [make_doc(SMALL_PARA, "paragraph")]
prepped  = chunker._prepend_context(chunks)

assert "embed_content"          in prepped[0].metadata
assert "Title:"                 in prepped[0].metadata["embed_content"]
assert "Section:"               in prepped[0].metadata["embed_content"]
assert prepped[0].page_content  == SMALL_PARA   # original never modified
ok(f"Context prepended | embed_content present | page_content unchanged")

# Test 12 — Total Chunks Count

section("Test 12 — Total Chunks Count")

chunks = [make_doc(f"Content {i} " * 10, "paragraph") for i in range(5)]
result = chunker._add_total_chunks(chunks)

assert all(c.metadata["total_chunks"] == 5 for c in result)
ok("total_chunks=5 added to all chunks correctly")

# Test 13 — Full Pipeline

section("Test 13 — Full Pipeline (split_documents)")

docs = [
    make_doc(PARAGRAPH,   "paragraph", page=1),
    make_doc(TABLE_SMALL, "table",     page=2),
    make_doc(CODE_SMALL,  "code",      page=3),
    make_doc(LIST_SMALL,  "list",      page=4),
]

chunks = chunker.split_documents(docs)

assert len(chunks) > 0
assert all(isinstance(c, Document)                              for c in chunks)
assert all(chunker._count_tokens(c.page_content) <= 512        for c in chunks)
assert all("embed_content"  in c.metadata                      for c in chunks)
assert all("token_count"    in c.metadata                      for c in chunks)
assert all("chunk_index"    in c.metadata                      for c in chunks)
assert all("total_chunks"   in c.metadata                      for c in chunks)
ok(f"Full pipeline | chunks={len(chunks)} | all metadata present | all ≤ 512 tokens")

# Test 14 — Chunk Stats

section("Test 14 — Chunk Stats")

stats = chunker.chunk_stats(chunks)

assert "total_chunks"    in stats
assert "avg_tokens"      in stats
assert "min_tokens"      in stats
assert "max_tokens"      in stats
assert "structure_types" in stats
assert stats["total_chunks"] == len(chunks)
ok(f"Stats | total={stats['total_chunks']} | avg={stats['avg_tokens']} tokens | min={stats['min_tokens']} | max={stats['max_tokens']}")

# Test 15 — split_by_character and split_for_rlm

section("Test 15 — split_by_character and split_for_rlm")

chunks = chunker.split_by_character(PARAGRAPH)
assert len(chunks) > 0
assert all(isinstance(c, str) for c in chunks)
ok(f"split_by_character | chunks={len(chunks)}")

chunks = chunker.split_for_rlm(PARAGRAPH)
assert len(chunks) > 0
assert all(isinstance(c, str) for c in chunks)
ok(f"split_for_rlm | chunks={len(chunks)}")

chunks = chunker.split_by_character("")
assert chunks == []
ok("split_by_character | empty input → []")

chunks = chunker.split_for_rlm("")
assert chunks == []
ok("split_for_rlm | empty input → []")

# Summary

section("All Tests Passed ✅")
print("  Chunker is working correctly\n")
