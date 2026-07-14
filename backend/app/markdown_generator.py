# markdown_generator.py — Converts loaded documents into markdown with page markers,
# and implements smart chunking that preserves tables as atomic units.
#
# NEW CONCEPTS for students:
#
#   1. PAGE MARKERS — we embed <!-- page:N --> comments in the markdown so every
#      chunk we create can tell the frontend exactly which page it came from.
#
#   2. SMART CHUNKING — standard RecursiveCharacterTextSplitter splits blindly.
#      A table cut in half produces broken columns and missing headers.
#      Smart chunking detects table blocks (lines starting with |) and keeps
#      them as ONE atomic chunk — never split across two chunks.
#
# Pipeline:
#   List[Document]  (from file_loader)
#     ↓  generate_markdown()
#   Single markdown string with <!-- page:N --> markers
#     ↓  smart_chunk()
#   List[{"text": str, "page_number": int, "is_table": bool}]

import re
import logging
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 500   # Max characters per prose chunk
CHUNK_OVERLAP = 50    # Characters shared between consecutive prose chunks

# Matches <!-- page:1 --> or <!-- page:12 --> etc.
PAGE_MARKER_RE = re.compile(r'<!--\s*page:(\d+)\s*-->')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate markdown with page markers
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown(docs: List[Document], file_name: str) -> str:
    """
    Convert a list of LangChain Documents into a single markdown string.

    Each document (page/row/section) gets a page marker at the top:
        <!-- page:1 -->

        Content of page 1...

        ---

        <!-- page:2 -->

        Content of page 2...

    The <!-- page:N --> markers survive all the way into the Qdrant payload,
    so the frontend always knows which page a chunk came from.
    """
    parts = []

    for doc in docs:
        page_num = doc.metadata.get("page_number", 1)   # 1-indexed from file_loader
        text     = doc.page_content.strip()

        if not text:
            continue   # Skip blank pages

        # Prepend the page marker — this is what enables per-chunk page tracking
        parts.append(f"<!-- page:{page_num} -->\n\n{text}")

    # Join pages with a visible --- separator (renders as <hr> in markdown)
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Smart chunking
# ─────────────────────────────────────────────────────────────────────────────

def smart_chunk(markdown_text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split markdown into chunks with two key guarantees:

    GUARANTEE 1 — Tables are ATOMIC:
        Any block of consecutive lines starting with | is one chunk.
        Example:
            | Name | Score |    ← these three lines
            |------|-------|    ← stay together
            | Alice | 95  |    ← as ONE chunk

    GUARANTEE 2 — Page numbers are TRACKED:
        <!-- page:N --> markers tell us which page each chunk came from.
        A chunk split from page 3 always carries page_number=3.

    Returns:
        [{"text": str, "page_number": int, "is_table": bool}, ...]
    """
    chunks: List[Dict] = []

    # Split the full markdown by page markers.
    # re.split() with a capturing group returns:
    #   [pre_content, "1", content_of_page_1, "2", content_of_page_2, ...]
    segments = PAGE_MARKER_RE.split(markdown_text)

    # segments[0] = content before the very first page marker (usually empty)
    if segments[0].strip():
        _chunk_section(segments[0], page_number=1, chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap, result=chunks)

    # Process (page_number, content) pairs from index 1 onward
    i = 1
    while i + 1 <= len(segments) - 1:
        try:
            page_number = int(segments[i])   # The captured page number
        except (ValueError, IndexError):
            page_number = 1
        content = segments[i + 1]            # The content following that page marker
        if content.strip():
            _chunk_section(content, page_number=page_number, chunk_size=chunk_size,
                           chunk_overlap=chunk_overlap, result=chunks)
        i += 2

    return [c for c in chunks if c["text"].strip()]


def _chunk_section(text: str, page_number: int, chunk_size: int, chunk_overlap: int, result: list):
    """
    Process one page section:
    - Lines starting with | are collected into table blocks → kept as ONE chunk
    - All other lines go into prose buffers → split by RecursiveCharacterTextSplitter

    This is the heart of smart chunking: a simple state machine that switches
    between "in table" and "in prose" mode as it scans line by line.
    """
    lines    = text.split('\n')
    prose_buf: List[str] = []   # Lines of current prose section
    table_buf: List[str] = []   # Lines of current table section
    in_table = False

    def flush_prose():
        """Split accumulated prose lines and add to result."""
        if not prose_buf:
            return
        prose_text = '\n'.join(prose_buf).strip()
        prose_buf.clear()
        if not prose_text:
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            separators    = ['\n\n', '\n', '. ', ' ', ''],
        )
        for chunk_text in splitter.split_text(prose_text):
            if chunk_text.strip():
                result.append({
                    "text":        chunk_text.strip(),
                    "page_number": page_number,
                    "is_table":    False,
                })

    def flush_table():
        """Keep entire table as one atomic chunk."""
        if not table_buf:
            return
        table_text = '\n'.join(table_buf).strip()
        table_buf.clear()
        if table_text:
            result.append({
                "text":        table_text,
                "page_number": page_number,
                "is_table":    True,   # Flag so frontend can style tables differently
            })

    for line in lines:
        # A table row starts with | OR is a separator row like |---|---|
        is_table_row = bool(re.match(r'\s*\|', line)) or (
            bool(re.match(r'^\s*[-:| ]+$', line)) and '|' in line
        )

        if is_table_row:
            if not in_table:
                flush_prose()       # Close the current prose section
                in_table = True
            table_buf.append(line)
        else:
            if in_table:
                flush_table()       # Close the current table section
                in_table = False
            prose_buf.append(line)

    # Flush whatever is left in the buffers
    if in_table:
        flush_table()
    else:
        flush_prose()
