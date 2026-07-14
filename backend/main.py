# main.py — Entry point for Document Butler API.
# Creates the FastAPI app, handles startup/shutdown, includes all routes.
#
# Run:
#   cd lecture_18_document_butler/backend
#   uvicorn main:app --reload --port 8001
#
# Frontend:
#   Open lecture_18_document_butler/frontend/index.html in your browser

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.api import app_state
from app.config import settings
from app.rag_service import get_embeddings
from app.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s pip install --upgrade pip| %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all shared resources once at startup."""
    for d in [settings.upload_dir, settings.uploads_dir, settings.markdown_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model...")
    app_state["embeddings"] = get_embeddings(settings.embedding_provider)

    logger.info("Connecting to Qdrant...")
    app_state["qdrant_client"] = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=300)

    logger.info("Connecting to OpenAI LLM...")
    app_state["llm"] = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)

    logger.info("Document Butler ready.")
    yield

    app_state.clear()
    logger.info("Server shut down.")


app = FastAPI(
    title       = "Document Butler — Lecture 18",
    description = "Upload files → Ask questions → See exact citations → Open file viewer",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
