"""
Unit tests for the StructurePreserver component.

Test scope:
    Unit tests (no pytest) covering structure-type priority resolution, single
    document tagging, section-label carry-over across pages, section-label
    updates on new headings, original metadata preservation, and a full
    multi-page pipeline with mixed structure types.
    Tests 1-5 (heading/table/list/code detectors) are commented out — they
    tested internal helpers that have since been consolidated.

Flow:
    Module-level execution — each test section runs sequentially; a failed
    assert exits immediately via sys.exit(1).

Dependencies:
    StructurePreserver; langchain_core Document; no external services.
"""

import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"]                   = ""
os.environ["REQUESTS_CA_BUNDLE"]               = ""

import sys
from langchain_core.documents import Document
from chunking.structure_preserver import StructurePreserver

preserver = StructurePreserver()


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ok(msg: str)   : print(f"  ✅ {msg}")
def fail(msg: str) : print(f"  ❌ {msg}"); sys.exit(1)

def make_doc(text: str, page: int = 1) -> Document:
    """Build a minimal Document for use in structure preservation tests."""
    return Document(page_content=text, metadata={"source": "test.pdf", "page": page})


MARKDOWN_HEADING_TEXT  = "# Introduction\nThis section covers RAG basics."
PLAIN_HEADING_TEXT     = "Introduction\nThis section covers RAG basics."
ALLCAPS_HEADING_TEXT   = "CHAPTER ONE\nThis section covers RAG basics."

MARKDOWN_TABLE_TEXT    = "| Name | Score |\n|------|-------|\n| RAG  | 0.95  |"
PLAIN_TABLE_TEXT       = "Name         Score        Grade\nRAG          0.95         A\nLLM          0.90         A"

BULLET_LIST_TEXT       = "- Install dependencies\n- Configure settings\n- Run the pipeline"
NUMBERED_LIST_TEXT     = "1. Install dependencies\n2. Configure settings\n3. Run pipeline"

CODE_FENCED_TEXT       = "```python\ndef embed(text):\n    return model.encode(text)\n```"
CODE_INDENTED_TEXT     = "    def embed(text):\n        return model.encode(text)"

PARAGRAPH_TEXT         = "RAG combines retrieval with generation for grounded responses. " * 5

EMPTY_TEXT             = ""


# # Test 1 — Empty document list

# section("Test 1 — Empty Document List")

# result = preserver.preserve([])
# assert result == []
# ok("Empty list returned as-is")

# # Test 2 — Heading Detection

# section("Test 2 — Heading Detection")

# # 2a — Markdown H1
# heading, level = preserver._detect_heading(MARKDOWN_HEADING_TEXT)
# assert heading == "Introduction" and level == 1
# ok(f"Markdown H1 | heading='{heading}' | level={level}")

# # 2b — Markdown H2
# heading, level = preserver._detect_heading("## Methods\nContent here.")
# assert level == 2
# ok(f"Markdown H2 | heading='{heading}' | level={level}")

# # 2c — Markdown H3
# heading, level = preserver._detect_heading("### Results\nContent here.")
# assert level == 3
# ok(f"Markdown H3 | heading='{heading}' | level={level}")

# # 2d — Plain title case heading
# heading, level = preserver._detect_heading(PLAIN_HEADING_TEXT)
# assert heading != "" and level == 2
# ok(f"Plain heading | heading='{heading}' | level={level}")

# # 2e — No heading → returns empty string and level 0
# heading, level = preserver._detect_heading(PARAGRAPH_TEXT)
# assert heading == "" and level == 0
# ok(f"No heading | heading='' | level=0")

# # Test 3 — Table Detection

# section("Test 3 — Table Detection")

# assert preserver._detect_table(MARKDOWN_TABLE_TEXT) == True
# ok("Markdown table detected")

# assert preserver._detect_table(PLAIN_TABLE_TEXT) == True
# ok("Plain table detected")

# assert preserver._detect_table(PARAGRAPH_TEXT) == False
# ok("No false positive on paragraph text")

# # Test 4 — List Detection

# section("Test 4 — List Detection")

# assert preserver._detect_list(BULLET_LIST_TEXT) == True
# ok("Bullet list detected")

# assert preserver._detect_list(NUMBERED_LIST_TEXT) == True
# ok("Numbered list detected")

# assert preserver._detect_list(PARAGRAPH_TEXT) == False
# ok("No false positive on paragraph text")

# # Test 5 — Code Detection

# section("Test 5 — Code Detection")

# assert preserver._detect_code(CODE_FENCED_TEXT) == True
# ok("Fenced code block detected")

# assert preserver._detect_code(CODE_INDENTED_TEXT) == True
# ok("Indented code block detected")

# assert preserver._detect_code(PARAGRAPH_TEXT) == False
# ok("No false positive on paragraph text")

# Test 6 — Structure Type Priority

section("Test 6 — Structure Type Priority (table > code > list > heading > paragraph)")

assert preserver._resolve_structure_type("Intro", True,  True,  True)  == "table"
ok("table wins over all")

assert preserver._resolve_structure_type("Intro", False, True,  True)  == "code"
ok("code wins over list and heading")

assert preserver._resolve_structure_type("Intro", False, True , False)  == "list"
ok("list wins over heading")

assert preserver._resolve_structure_type("Intro", False, False, False) == "heading"
ok("heading wins over paragraph")

assert preserver._resolve_structure_type("",      False, False, False) == "paragraph"
ok("paragraph is default")

# Test 7 — Single Document Tagging

section("Test 7 — Single Document Tagging")

doc = make_doc(MARKDOWN_HEADING_TEXT)
tagged, section_out = preserver._tag_document(doc, "unknown")

assert tagged.metadata["section"]        == "Introduction"
assert tagged.metadata["heading_level"]  == 1
assert tagged.metadata["structure_type"] == "heading"
assert tagged.metadata["has_table"]      == False
assert tagged.metadata["has_list"]       == False
assert tagged.metadata["has_code"]       == False
ok(f"Tagged correctly | section='{tagged.metadata['section']}' | type={tagged.metadata['structure_type']}")

# Test 8 — Section Carries Over Across Pages

section("Test 8 — Section Carries Over Across Pages")

docs = [
    make_doc("# Introduction\nRAG combines retrieval with generation.", page=1),
    make_doc(PARAGRAPH_TEXT,                                             page=2),
    make_doc(PARAGRAPH_TEXT,                                             page=3),
]

result = preserver.preserve(docs)

assert result[0].metadata["section"] == "Introduction"
assert result[1].metadata["section"] == "Introduction"  # carried over
assert result[2].metadata["section"] == "Introduction"  # carried over
ok("Section 'Introduction' carried over across pages 1, 2, 3")

# Test 9 — Section Updates On New Heading

section("Test 9 — Section Updates On New Heading")

docs = [
    make_doc("# Introduction\nRAG overview.",  page=1),
    make_doc(PARAGRAPH_TEXT,                   page=2),
    make_doc("# Methods\nHow RAG works.",      page=3),
    make_doc(PARAGRAPH_TEXT,                   page=4),
]

result = preserver.preserve(docs)

assert result[0].metadata["section"] == "Introduction"
assert result[1].metadata["section"] == "Introduction"
assert result[2].metadata["section"] == "Methods"
assert result[3].metadata["section"] == "Methods"
ok("Section updated to 'Methods' at page 3, carried to page 4")

# Test 10 — Metadata Fully Preserved

section("Test 10 — Original Metadata Preserved")

doc = Document(
    page_content=PARAGRAPH_TEXT,
    metadata={"source": "report.pdf", "page": 5, "author": "test_user", "doc_id": "abc123"}
)
result = preserver.preserve([doc])

assert result[0].metadata["source"]  == "report.pdf"
assert result[0].metadata["page"]    == 5
assert result[0].metadata["author"]  == "test_user"
assert result[0].metadata["doc_id"]  == "abc123"
ok("All original metadata preserved after tagging")

# Test 11 — Full Pipeline (multiple structure types)

section("Test 11 — Full Pipeline (mixed structure types)")

docs = [
    make_doc(MARKDOWN_HEADING_TEXT, page=1),
    make_doc(PARAGRAPH_TEXT,        page=2),
    make_doc(MARKDOWN_TABLE_TEXT,   page=3),
    make_doc(BULLET_LIST_TEXT,      page=4),
    make_doc(CODE_FENCED_TEXT,      page=5),
]

result = preserver.preserve(docs)

assert len(result)                              == 5
assert result[0].metadata["structure_type"]    == "heading"
assert result[1].metadata["structure_type"]    == "paragraph"
assert result[2].metadata["structure_type"]    == "table"
assert result[3].metadata["structure_type"]    == "list"
assert result[4].metadata["structure_type"]    == "code"
assert result[2].metadata["has_table"]         == True
assert result[3].metadata["has_list"]          == True
assert result[4].metadata["has_code"]          == True

for doc in result:
    assert "section"        in doc.metadata
    assert "heading_level"  in doc.metadata
    assert "structure_type" in doc.metadata
    assert "has_table"      in doc.metadata
    assert "has_list"       in doc.metadata
    assert "has_code"       in doc.metadata

ok("All 5 pages tagged with correct structure types")
ok("All required metadata keys present in every page")

section("All Tests Passed ✅")
print("  StructurePreserver is working correctly\n")