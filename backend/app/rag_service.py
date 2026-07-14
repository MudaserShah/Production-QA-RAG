# rag_service.py — Core RAG logic for Document Butler.
#
# NEW vs Lecture 18:
#   1. All file types supported (PDF, DOCX, MD, CSV, TXT) via file_loader
#   2. Every document → markdown with page markers via markdown_generator
#   3. Smart chunking preserves tables as atomic units
#   4. Original file saved permanently so frontend can serve it
#   5. Citations include doc_id so frontend can open the file viewer
#
# Indexing pipeline:
#   Any file
#     ↓ file_loader.load_file()          load pages/rows into Documents
#     ↓ markdown_generator.generate_markdown()   convert to markdown + page markers
#     ↓ markdown_generator.smart_chunk()         split (tables kept intact)
#     ↓ embeddings.embed_query()         convert each chunk to vector
#     ↓ qdrant_client.upsert()           store vector + full payload in Qdrant

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.file_loader import load_file
from app.markdown_generator import generate_markdown, smart_chunk
from app.models import Citation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_doc_id(file_name: str) -> str:
    """Deterministic ID from filename — same file always gets the same doc_id."""
    return hashlib.sha256(file_name.encode()).hexdigest()[:16]


def get_embeddings(provider: str):
    """Return the embedding model based on the configured provider."""
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")   # 1536-dim, paid
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dim, free


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return collection_name in [c.name for c in client.get_collections().collections]


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create the Qdrant collection + doc_id payload index if they don't exist."""
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config  = VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' (dim={vector_size})")

    # Always ensure the doc_id index exists — required for DELETE filtering.
    # Qdrant silently ignores this if the index already exists.
    client.create_payload_index(
        collection_name = collection_name,
        field_name      = "doc_id",
        field_schema    = PayloadSchemaType.KEYWORD,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDEXING
# ─────────────────────────────────────────────────────────────────────────────

def index_document(
    file_path       : str,       # Temp path of the uploaded file
    file_name       : str,       # Original filename (e.g. "report.pdf")
    client          : QdrantClient,
    embeddings,
    collection_name : str,
    uploads_dir     : str,       # Permanent storage for the original file
    markdown_dir    : str,       # Where to save the generated markdown
) -> dict:
    """
    Full indexing pipeline for one file:

    1. Load file with file_loader (all types supported)
    2. Save original file permanently (for frontend "Open file" feature)
    3. Generate markdown with page markers
    4. Save markdown to disk (for frontend file viewer)
    5. Smart chunk the markdown (tables kept atomic)
    6. Embed each chunk and upsert to Qdrant

    Every Qdrant point stores:
        file_name, file_type, page_number, chunk_index, doc_id,
        chunk_text, is_table, markdown_path, original_path, upload_timestamp
    """
    logger.info(f"Indexing: {file_name}")

    doc_id    = compute_doc_id(file_name)
    ext       = Path(file_name).suffix.lower()
    upload_ts = datetime.now(timezone.utc).isoformat()

    # Step 1: Load all pages/rows into LangChain Documents
    docs = load_file(file_path, file_name)
    logger.info(f"  Loaded {len(docs)} document(s)")

    # Strip lone surrogate characters that some PDFs produce — they are
    # not valid UTF-8 and will crash the markdown write step.
    for doc in docs:
        doc.page_content = doc.page_content.encode("utf-8", errors="replace").decode("utf-8")

    # Step 2: Save original file permanently so the frontend can serve it
    os.makedirs(uploads_dir, exist_ok=True)
    original_path = os.path.join(uploads_dir, f"{doc_id}{ext}")
    shutil.copy2(file_path, original_path)   # copy2 preserves metadata

    # Step 3: Generate markdown with <!-- page:N --> markers
    markdown_text = generate_markdown(docs, file_name)

    # Step 4: Save markdown to disk for the file viewer endpoint
    os.makedirs(markdown_dir, exist_ok=True)
    md_path = os.path.join(markdown_dir, f"{doc_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    logger.info(f"  Markdown saved → {md_path}")

    # Step 5: Smart chunk — tables kept as single atomic chunks
    chunks = smart_chunk(markdown_text)
    logger.info(f"  {len(chunks)} chunks created ({sum(1 for c in chunks if c['is_table'])} tables)")

    if not chunks:
        logger.warning(f"  No chunks produced for '{file_name}'")
        return {"doc_id": doc_id, "file_name": file_name, "total_chunks": 0,
                "total_pages": len(docs), "upload_timestamp": upload_ts}

    # Step 6: Embed each chunk and build Qdrant points
    total_pages = max((c["page_number"] for c in chunks), default=0)
    points: list = []
    collection_ready = False

    for idx, chunk in enumerate(chunks):
        text   = chunk["text"]
        vector = embeddings.embed_query(text)   # Convert chunk to float vector

        # Ensure collection exists on first vector (we need the dimension)
        if not collection_ready:
            ensure_collection(client, collection_name, len(vector))
            collection_ready = True

        points.append(PointStruct(
            id     = str(uuid.uuid4()),
            vector = vector,
            payload = {
                "file_name":         file_name,
                "file_type":         ext.lstrip("."),
                "page_number":       chunk["page_number"],
                "chunk_index":       idx,
                "doc_id":            doc_id,
                "chunk_text":        text,
                "is_table":          chunk["is_table"],  # Smart chunking result
                "markdown_path":     md_path,            # Path for file viewer
                "original_path":     original_path,      # Path for download
                "total_pages":       total_pages,
                "upload_timestamp":  upload_ts,
            },
        ))

    # Batch upsert — faster than inserting one by one
    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"  Indexed {len(points)} chunks into '{collection_name}'")

    return {
        "doc_id":           doc_id,
        "file_name":        file_name,
        "file_type":        ext.lstrip("."),
        "total_chunks":     len(points),
        "total_pages":      total_pages,
        "upload_timestamp": upload_ts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY WITH CITATIONS
# ─────────────────────────────────────────────────────────────────────────────

def query_with_citations(
    question        : str,
    top_k           : int,
    client          : QdrantClient,
    embeddings,
    llm             : ChatOpenAI,
    collection_name : str,
) -> dict:
    """
    Semantic search → extract citation payload → generate answer.

    KEY LESSON: query_points() returns ScoredPoint objects, each with:
      .score   → cosine similarity
      .payload → our full metadata dict (file_name, page_number, chunk_text, doc_id, ...)

    The doc_id in each citation lets the frontend fetch
    GET /files/{doc_id}/markdown to open the file viewer.
    """
    if not collection_exists(client, collection_name):
        return {
            "answer":     "No documents have been uploaded yet. Please upload a file first.",
            "references": [],
        }

    # Step 1: Embed the question
    query_vector = embeddings.embed_query(question)

    # Step 2: Semantic search — top_k most similar chunks
    results = client.query_points(
        collection_name = collection_name,
        query           = query_vector,
        limit           = top_k,
        with_payload    = True,
        with_vectors    = False,
    )

    if not results.points:
        return {
            "answer":     "No documents have been uploaded yet. Please upload a file first.",
            "references": [],
        }

    # Step 3: Build citations and context string
    citations     : List[Citation] = []
    context_parts : List[str]      = []

    for hit in results.points:
        p = hit.payload   # Full metadata dict from indexing

        citations.append(Citation(
            file_name       = p["file_name"],
            page_number     = p["page_number"],
            chunk_index     = p["chunk_index"],
            chunk_text      = p["chunk_text"],
            relevance_score = round(hit.score, 4),
            doc_id          = p["doc_id"],      # Frontend uses this to open the file viewer
            is_table        = p.get("is_table", False),
        ))

        context_parts.append(
            f"[Source: {p['file_name']}, page {p['page_number']}]\n{p['chunk_text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer using LCEL — same | operator from Lecture 10
    prompt = ChatPromptTemplate.from_template(
        """You are a precise document analyst. Answer the question using ONLY the context below.
Do NOT mention file names or page numbers in your answer — those are shown separately as citations.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    )

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "references": citations}


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

async def query_stream(
    question        : str,
    top_k           : int,
    client          : QdrantClient,
    embeddings,
    llm             : ChatOpenAI,
    collection_name : str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings:
      1. {"type": "citations", "data": [...]}   — search results, sent immediately
      2. {"type": "token",     "content": "…"}  — one per LLM token (streaming)
      3. {"type": "done"}                        — signals end of stream

    This lets the frontend show citations while the answer is still being written,
    giving a much snappier feel compared to waiting for the full response.
    """
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    if not collection_exists(client, collection_name):
        yield _sse({"type": "error", "message": "No documents uploaded yet. Please upload a file first."})
        return

    # Embed query in a thread so the event loop stays unblocked
    query_vector = await asyncio.to_thread(embeddings.embed_query, question)

    results = await asyncio.to_thread(
        lambda: client.query_points(
            collection_name = collection_name,
            query           = query_vector,
            limit           = top_k,
            with_payload    = True,
            with_vectors    = False,
        )
    )

    if not results.points:
        yield _sse({"type": "error", "message": "No documents uploaded yet. Please upload a file first."})
        return

    citations     : list = []
    context_parts : list = []

    for hit in results.points:
        p = hit.payload
        citations.append({
            "file_name":       p["file_name"],
            "page_number":     p["page_number"],
            "chunk_index":     p["chunk_index"],
            "chunk_text":      p["chunk_text"],
            "relevance_score": round(hit.score, 4),
            "doc_id":          p["doc_id"],
            "is_table":        p.get("is_table", False),
        })
        context_parts.append(
            f"[Source: {p['file_name']}, page {p['page_number']}]\n{p['chunk_text']}"
        )

    # Send citations immediately — frontend renders them before the answer starts
    yield _sse({"type": "citations", "data": citations})

    context = "\n\n---\n\n".join(context_parts)
    prompt  = ChatPromptTemplate.from_template(
        """You are a precise document analyst. Answer the question using ONLY the context below.
Do NOT mention file names or page numbers in your answer — those are shown separately as citations.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | llm | StrOutputParser()

    async for token in chain.astream({"context": context, "question": question}):
        if token:
            yield _sse({"type": "token", "content": token})

    yield _sse({"type": "done"})


def list_documents(client: QdrantClient, collection_name: str) -> list:
    """Scroll Qdrant and return one entry per document (not per chunk)."""
    # Guard: if collection doesn't exist, return empty list
    if not collection_exists(client, collection_name):
        return []

    seen   = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name = collection_name,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )

        for record in records:
            p      = record.payload
            doc_id = p.get("doc_id", "unknown")

            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id":           doc_id,
                    "file_name":        p.get("file_name", "unknown"),
                    "file_type":        p.get("file_type", "unknown"),
                    "total_pages":      p.get("total_pages", 0),
                    "upload_timestamp": p.get("upload_timestamp", ""),
                    "total_chunks":     0,
                    "status":           "ready",
                }
            seen[doc_id]["total_chunks"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return list(seen.values())


def delete_document(doc_id: str, client: QdrantClient, collection_name: str) -> int:
    """Delete all Qdrant chunks for a doc_id. Returns number of points deleted."""
    if not collection_exists(client, collection_name):
        return 0

    filt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

    count_result  = client.count(collection_name=collection_name, count_filter=filt)
    deleted_count = count_result.count

    if deleted_count > 0:
        client.delete(collection_name=collection_name, points_selector=filt)

    return deleted_count
