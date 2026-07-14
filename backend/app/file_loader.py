# file_loader.py — Loads any supported file type into LangChain Documents.
# Identical to Lecture 17 — supports PDF, DOCX, MD, CSV, TXT.
#
# A LangChain Document has two fields:
#   doc.page_content  → the text of this page/row/section
#   doc.metadata      → dict with file_name, file_type, page_number

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
# UnstructuredMarkdownLoader is imported lazily inside _load_by_extension
# to avoid a startup crash if the 'unstructured' package is not installed.
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".csv", ".txt"}


def is_supported(file_name: str) -> bool:
    """Return True if we have a loader for this file extension."""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(file_path: str, file_name: str) -> List[Document]:
    """
    Load a file from disk and return a list of LangChain Documents.
    Every Document gets file_name, file_type and page_number in its metadata.
    """
    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    logger.info(f"Loading '{file_name}' (type: {ext})")
    docs = _load_by_extension(file_path, ext)

    # Stamp every Document with where it came from
    for doc in docs:
        doc.metadata["file_name"] = file_name
        doc.metadata["file_type"] = ext.lstrip(".")

    logger.info(f"  → {len(docs)} document(s) loaded")
    return docs


def _load_by_extension(file_path: str, ext: str) -> List[Document]:
    """Pick the correct LangChain loader and return raw Documents."""

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs   = loader.load()
        for doc in docs:
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page_number"] = raw_page + 1   # 0-indexed → 1-indexed
        return docs

    if ext == ".docx":
        loader = Docx2txtLoader(file_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    if ext in (".md", ".markdown"):
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
            docs   = loader.load()
        except (ImportError, ModuleNotFoundError):
            loader = TextLoader(file_path, encoding="utf-8")   # Fallback: 'unstructured' not installed
            docs   = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    if ext == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
        docs   = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["page_number"] = i + 1   # Row number as page
        return docs

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs   = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    raise ValueError(f"No loader implemented for extension: {ext}")
