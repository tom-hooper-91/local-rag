"""Ingest markdown documents into a Chroma vector store."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_data")
COLLECTION_NAME = "local-docs"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest() -> None:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("**/*.md")):
        print(f"No markdown files found in {DOCS_DIR}/")
        sys.exit(1)

    print(f"Loading documents from {DOCS_DIR}/...")
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} file(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunk(s)")

    print("Generating embeddings with nomic-embed-text...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL,
    )

    print(f"Storing in Chroma at {CHROMA_DIR}/...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Done — {len(documents)} file(s), {len(chunks)} chunk(s) indexed.")


if __name__ == "__main__":
    ingest()
