# routes.py — Route declarations only.
# Maps every URL path + HTTP method to its handler in api.py.

from fastapi import APIRouter

from app.api import (
    get_documents,
    get_file_markdown,
    get_original_file,
    health,
    query,
    query_stream_endpoint,
    remove_document,
    upload_documents,
)
from app.models import (
    DeleteResponse,
    DocumentListResponse,
    FileContentResponse,
    QueryResponse,
    UploadResponse,
)

router = APIRouter()

# path                          method      handler              response shape
router.add_api_route("/health",                    health,           methods=["GET"],                                          tags=["Ops"])
router.add_api_route("/upload",                    upload_documents, methods=["POST"],   response_model=UploadResponse,        tags=["Documents"])
router.add_api_route("/query",                     query,                 methods=["POST"],   response_model=QueryResponse, tags=["RAG"])
router.add_api_route("/query/stream",              query_stream_endpoint, methods=["POST"],                                 tags=["RAG"])
router.add_api_route("/documents",                 get_documents,    methods=["GET"],    response_model=DocumentListResponse,  tags=["Documents"])
router.add_api_route("/documents/{doc_id}",        remove_document,  methods=["DELETE"], response_model=DeleteResponse,        tags=["Documents"])
router.add_api_route("/files/{doc_id}/markdown",   get_file_markdown,methods=["GET"],    response_model=FileContentResponse,   tags=["Files"])
router.add_api_route("/files/{doc_id}/original",   get_original_file,methods=["GET"],                                         tags=["Files"])
