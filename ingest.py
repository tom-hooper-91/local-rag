"""Ingest markdown documents into a Chroma vector store."""

import argparse
import os
import shutil
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest markdown docs into vector store")
    parser.add_argument("--chunk-size", type=int, default=500, help="chunk size in characters (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="chunk overlap in characters (default: 50)")
    parser.add_argument("--reindex", action="store_true", help="delete existing vector store before ingesting")
    return parser.parse_args()


def ingest(chunk_size: int, chunk_overlap: int, reindex: bool) -> None:
    if reindex and CHROMA_DIR.exists():
        print(f"Removing existing vector store at {CHROMA_DIR}/...")
        shutil.rmtree(CHROMA_DIR)

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

    print(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
    args = parse_args()
    ingest(chunk_size=args.chunk_size, chunk_overlap=args.overlap, reindex=args.reindex)
