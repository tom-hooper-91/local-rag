# Local Documentation RAG

A local-first RAG (Retrieval-Augmented Generation) application that indexes personal markdown notes into a vector store and answers natural-language questions with cited sources.

Built with **LangChain + Ollama + Chroma**. Fully isolated, localhost-only.

## Prerequisites

- Docker with NVIDIA GPU support
- Python 3.12+

## Setup

```bash
# Start Ollama
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Pull required models
docker exec local-rag-ollama ollama pull nomic-embed-text
docker exec local-rag-ollama ollama pull qwen3.5:2b
```

## Usage

### Ingest documents

Place your `.md` files in the `docs/` directory, then run:

```bash
python ingest.py
```

Options:
- `--chunk-size 500` — chunk size in characters (default: 500)
- `--overlap 50` — chunk overlap in characters (default: 50)
- `--reindex` — delete existing vector store before re-ingesting

### Query (single-shot)

```bash
python query.py "What is the purpose of X?"
```

### Chat (multi-turn with follow-ups)

```bash
python query.py --chat
```

Options for both query modes:
- `--top-k 4` — number of chunks to retrieve (default: 4)

In chat mode, follow-up questions like "How do I create one?" will be understood in context of the conversation.

## Architecture

```
[Markdown Files] → [Loader + Splitter] → [Embeddings (Ollama)] → [Chroma Vector DB]
                                                                         ↓
[User Question] → [Embedding] → [Similarity Search] → [Retrieved Chunks + Question] → [LLM (Ollama)] → [Answer + Sources]
```

All local. No network calls. Ollama runs in Docker with GPU passthrough. Chroma runs in-process.

## VRAM Budget (6GB)

| Model | Size | When Used |
|-------|------|-----------|
| `nomic-embed-text` | ~274MB | Ingest + query (embedding) |
| `qwen3.5:2b` | ~2.7GB | Query (generation) |

No contention — embedding and generation don't run simultaneously during ingest. During query, both fit comfortably.
