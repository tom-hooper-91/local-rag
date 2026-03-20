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
- `--provider github` — use GitHub Models (GPT-4.1) instead of local Ollama

In chat mode, follow-up questions like "How do I create one?" will be understood in context of the conversation.

### Using GitHub Models

To use cloud models via [GitHub Models](https://github.com/marketplace/models) instead of local Ollama:

1. Create a [Personal Access Token](https://github.com/settings/tokens) with the `models` scope
2. Add it to your `.env` file: `GITHUB_TOKEN=ghp_...`
3. Run with `--provider github`:

```bash
python query.py --provider github "What is git bisect?"
python query.py --provider github --chat
```

This swaps the LLM only (embeddings stay local via Ollama). No need to re-ingest documents.

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

## How RAG Works

RAG (Retrieval-Augmented Generation) solves a fundamental problem: language models can only work with text they've seen during training, plus whatever you put in the prompt. They can't read your personal notes, internal docs, or anything written after their training cutoff. RAG bridges this gap by finding relevant text from your documents and injecting it into the prompt before the model generates an answer.

The pipeline has two phases:

### Phase 1: Ingestion (done once, ahead of time)

**Step 1 — Load documents.** Raw text is read from your files. This could be markdown, PDFs, web pages — anything that can be converted to plain text.

**Step 2 — Split into chunks.** Documents are broken into smaller overlapping pieces (typically a few hundred characters each). This is necessary because embedding models and LLMs have limited context windows, and smaller chunks produce more precise search results. The overlap ensures that sentences at the boundary between two chunks aren't lost.

**Step 3 — Generate embeddings.** Each chunk is passed through an *embedding model* — a neural network that converts text into a high-dimensional numerical vector (a long list of numbers, typically 768 or more). These vectors capture the *meaning* of the text, not just the words. Chunks about similar topics will have vectors that point in similar directions, even if they use completely different words.

**Step 4 — Store in a vector database.** The vectors and their original text are saved to a vector database. This database is optimised for a single operation: given a new vector, find the stored vectors most similar to it.

### Phase 2: Query (every time a question is asked)

**Step 1 — Embed the question.** The user's question is passed through the same embedding model used during ingestion. This produces a vector in the same mathematical space as the document chunks.

**Step 2 — Similarity search.** The question vector is compared against all stored chunk vectors using a distance metric (typically cosine similarity). The database returns the top-k most similar chunks — the ones whose meaning is closest to the question.

**Step 3 — Build the prompt.** The retrieved chunks are assembled into a prompt alongside the user's question. This is the "retrieval-augmented" part — the model's generation is *augmented* by retrieved context it wouldn't otherwise have access to.

**Step 4 — Generate the answer.** The assembled prompt is sent to a language model, which reads the provided context and generates a natural-language answer. Because the relevant information is right there in the prompt, the model can answer accurately about content it was never trained on.

### Why this works

The key insight is that *meaning* can be represented as geometry. When an embedding model converts "How do I persist data between container restarts?" into a vector, that vector lands near other vectors about Docker volumes, named storage, and data persistence — even if none of those chunks contain the exact words from the question. This semantic search is far more powerful than keyword matching, which would miss relevant results that use different terminology.

The language model never needs to "know" your documents. It just needs to be good at reading provided text and synthesising a clear answer — which is exactly what LLMs excel at.
