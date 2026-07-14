# models.py — All Pydantic request/response models for Document Butler.
# FastAPI auto-generates Swagger UI from these models.

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """One source chunk used to generate the answer. Contains everything needed
    for the frontend to open the exact location in the original document."""
    file_name: str               # Original filename
    page_number: int             # 1-indexed page number
    chunk_index: int             # Position of this chunk within the document
    chunk_text: str              # The exact text excerpt that was used
    relevance_score: float       # Cosine similarity score (0.0 – 1.0)
    doc_id: str                  # Used by frontend to fetch /files/{doc_id}/markdown
    is_table: bool = False       # True if this chunk is a table (smart chunking result)


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    references: List[Citation]


# ── Upload ────────────────────────────────────────────────────────────────────

class FileUploadResult(BaseModel):
    """Status for a single file in a batch upload."""
    file_name: str
    doc_id: str
    file_type: str
    status: str                  # "processing" | "error"
    message: str
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Summary for a batch upload — one result per file."""
    total_files: int
    successful: int
    failed: int
    results: List[FileUploadResult]


# ── Document listing ──────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    total_chunks: int
    total_pages: int
    upload_timestamp: str
    status: str = Field(description="'processing' | 'ready' | 'error'")


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


# ── Delete ────────────────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    status: str
    doc_id: str
    message: str


# ── File viewer ───────────────────────────────────────────────────────────────

class FileContentResponse(BaseModel):
    """Response for GET /files/{doc_id}/markdown — used by the file viewer."""
    doc_id: str
    file_name: str
    file_type: str
    content: str          # Full markdown content with <!-- page:N --> markers
    total_pages: int
