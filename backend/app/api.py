# api.py — Actual logic for every API endpoint.
# routes.py imports these functions and registers them on the router.

import logging
import os
import re
from pathlib import Path
from typing import List

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.config import settings
from app.file_loader import is_supported
from app.models import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    FileContentResponse,
    FileUploadResult,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.rag_service import (
    compute_doc_id,
    delete_document,
    index_document,
    list_documents,
    query_stream,
    query_with_citations,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

app_state: dict = {}   # Holds embeddings, qdrant_client, llm — populated by lifespan


def get_qdrant_client() -> QdrantClient:
    return app_state["qdrant_client"]


def get_embeddings_dep():
    return app_state["embeddings"]


def get_llm() -> ChatOpenAI:
    return app_state["llm"]


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def health() -> dict:
    return {
        "status":             "ok",
        "embedding_provider": settings.embedding_provider,
        "collection":         settings.qdrant_collection,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD — supports PDF, DOCX, MD, CSV, TXT (multi-file)
# ─────────────────────────────────────────────────────────────────────────────

async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF, DOCX, MD, CSV, or TXT files"),
) -> UploadResponse:
    """
    Upload one or more files and index them immediately.
    The response is returned once all files are fully indexed.

    Supported: PDF, DOCX, MD, CSV, TXT
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[FileUploadResult] = []

    for upload in files:
        file_name = upload.filename or "unknown"
        ext       = Path(file_name).suffix.lower().lstrip(".")

        # Reject unsupported file types up front
        if not is_supported(file_name):
            results.append(FileUploadResult(
                file_name = file_name,
                doc_id    = compute_doc_id(file_name),
                file_type = ext,
                status    = "error",
                message   = "Unsupported file type",
                error     = f"Accepted: PDF, DOCX, MD, CSV, TXT. Got: .{ext}",
            ))
            continue

        # Read file bytes and check size limit
        content = await upload.read()
        size_mb  = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            results.append(FileUploadResult(
                file_name = file_name,
                doc_id    = compute_doc_id(file_name),
                file_type = ext,
                status    = "error",
                message   = "File too large",
                error     = f"File is {size_mb:.1f} MB. Max: {settings.max_upload_size_mb} MB",
            ))
            continue

        # Write to a temp path — loaders (PyPDF, Docx2txt, etc.) need a real file on disk
        doc_id   = compute_doc_id(file_name)
        os.makedirs(settings.upload_dir, exist_ok=True)
        tmp_path = os.path.join(settings.upload_dir, f"{doc_id}_{file_name}")

        with open(tmp_path, "wb") as f:
            f.write(content)

        # Index the file — runs here, response returned when done
        try:
            logger.info(f"Indexing: {file_name}")
            index_document(
                file_path       = tmp_path,
                file_name       = file_name,
                client          = app_state["qdrant_client"],
                embeddings      = app_state["embeddings"],
                collection_name = settings.qdrant_collection,
                uploads_dir     = settings.uploads_dir,
                markdown_dir    = settings.markdown_dir,
            )
            logger.info(f"Indexed:  {file_name}")

            results.append(FileUploadResult(
                file_name = file_name,
                doc_id    = doc_id,
                file_type = ext,
                status    = "ready",
                message   = f"'{file_name}' indexed successfully.",
            ))

        except Exception as e:
            logger.error(f"Indexing failed for {file_name}: {e}")
            results.append(FileUploadResult(
                file_name = file_name,
                doc_id    = doc_id,
                file_type = ext,
                status    = "error",
                message   = "Indexing failed",
                error     = str(e),
            ))

        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return UploadResponse(
        total_files = len(files),
        successful  = sum(1 for r in results if r.status == "ready"),
        failed      = sum(1 for r in results if r.status == "error"),
        results     = results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────────────────────

def query(
    request       : QueryRequest,
    qdrant_client : QdrantClient = Depends(get_qdrant_client),
    embeddings                   = Depends(get_embeddings_dep),
    llm           : ChatOpenAI   = Depends(get_llm),
) -> QueryResponse:
    """Ask a question — returns answer + citations with doc_id for file viewer."""
    try:
        result = query_with_citations(
            question        = request.question,
            top_k           = request.top_k,
            client          = qdrant_client,
            embeddings      = embeddings,
            llm             = llm,
            collection_name = settings.qdrant_collection,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# LIST DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def get_documents(
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> DocumentListResponse:
    """List all indexed documents."""
    try:
        docs = list_documents(
            client          = qdrant_client,
            collection_name = settings.qdrant_collection,
        )
        return DocumentListResponse(
            documents = [DocumentInfo(**d) for d in docs],
            total     = len(docs),
        )
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# DELETE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

def remove_document(
    doc_id        : str,
    qdrant_client : QdrantClient = Depends(get_qdrant_client),
) -> DeleteResponse:
    """Remove a document and all its chunks from Qdrant."""
    try:
        deleted_count = delete_document(
            doc_id          = doc_id,
            client          = qdrant_client,
            collection_name = settings.qdrant_collection,
        )

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

        return DeleteResponse(
            status  = "deleted",
            doc_id  = doc_id,
            message = f"Deleted {deleted_count} chunks for document '{doc_id}'",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def query_stream_endpoint(
    request       : QueryRequest,
    qdrant_client : QdrantClient = Depends(get_qdrant_client),
    embeddings                   = Depends(get_embeddings_dep),
    llm           : ChatOpenAI   = Depends(get_llm),
):
    """
    Streaming query endpoint — returns SSE events so the UI can show
    citations instantly and stream the LLM answer token-by-token.
    """
    return StreamingResponse(
        query_stream(
            question        = request.question,
            top_k           = request.top_k,
            client          = qdrant_client,
            embeddings      = embeddings,
            llm             = llm,
            collection_name = settings.qdrant_collection,
        ),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILE VIEWER — serve markdown content and original file
# ─────────────────────────────────────────────────────────────────────────────

def _find_original_file(doc_id: str) -> str | None:
    """Return the filename (e.g. 'abc123.pdf') for a doc_id, or None."""
    if not os.path.exists(settings.uploads_dir):
        return None
    for fname in os.listdir(settings.uploads_dir):
        if fname.startswith(doc_id):
            return fname
    return None


def get_file_markdown(doc_id: str) -> FileContentResponse:
    """Return the generated markdown for a document so the frontend can render and highlight it."""
    md_path = os.path.join(settings.markdown_dir, f"{doc_id}.md")
    if not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail=f"Markdown not found for doc_id='{doc_id}'")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Resolve original file name from the uploads folder
    original_file = _find_original_file(doc_id)
    file_name     = original_file or doc_id
    ext           = Path(file_name).suffix.lstrip(".")

    page_numbers = [int(m) for m in re.findall(r'<!--\s*page:(\d+)\s*-->', content)]
    total_pages  = max(page_numbers) if page_numbers else 1

    return FileContentResponse(
        doc_id      = doc_id,
        file_name   = file_name,
        file_type   = ext,
        content     = content,
        total_pages = total_pages,
    )


MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md":   "text/markdown",
    ".csv":  "text/csv",
    ".txt":  "text/plain",
}

def get_original_file(doc_id: str):
    """Serve the original uploaded file inline (no download prompt)."""
    fname = _find_original_file(doc_id)
    if not fname:
        raise HTTPException(status_code=404, detail=f"Original file not found for doc_id='{doc_id}'")

    ext        = Path(fname).suffix.lower()
    media_type = MIME_TYPES.get(ext, "application/octet-stream")

    return FileResponse(
        path       = os.path.join(settings.uploads_dir, fname),
        media_type = media_type,
    )

    raise HTTPException(status_code=404, detail=f"Original file not found for doc_id='{doc_id}'")
