"""Query the Chroma vector store and answer questions with cited sources."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

CHROMA_DIR = Path("chroma_data")
COLLECTION_NAME = "local-docs"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")
TOP_K = 4
HISTORY_WINDOW = 5

# Rewrites a follow-up question into a standalone question using chat history.
# e.g. "What about volumes?" after discussing Docker → "What are Docker volumes?"
# /no_think disables qwen3.5's internal reasoning to avoid 30-60s hidden latency.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the question "
     "to be a standalone question that doesn't require the chat history to understand. "
     "If the question is already standalone, return it unchanged. "
     "Do NOT answer the question — only rewrite it. /no_think"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_template("""\
Answer the question based only on the following context. If the context doesn't \
contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    """Format retrieved documents into a numbered context block for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def print_sources(docs):
    """Print deduplicated source citations after the answer."""
    seen = set()
    print("\n\n--- Sources ---")
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            print(f"  • {source}")


class RAGPipeline:
    """RAG pipeline with optional conversational memory."""

    def __init__(self):
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
        )
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        self.chat_history: list = []

    def _contextualize_question(self, question: str) -> str:
        """Rewrite a follow-up question as standalone using chat history."""
        if not self.chat_history:
            return question

        print("  (rewriting question with context...)", flush=True)
        chain = CONTEXTUALIZE_PROMPT | self.llm | StrOutputParser()
        # Only send the last N turns to avoid overflowing context
        history = self.chat_history[-(HISTORY_WINDOW * 2):]
        return chain.invoke({"question": question, "chat_history": history})

    def ask(self, question: str) -> str:
        """Ask a question, retrieve context, stream the answer, return it."""
        standalone = self._contextualize_question(question)

        retrieved_docs = self.retriever.invoke(standalone)
        if not retrieved_docs:
            print("No relevant documents found.")
            return ""

        chain = (
            {"context": lambda _: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # Stream answer token-by-token
        print()
        answer_parts = []
        for chunk in chain.stream(standalone):
            print(chunk, end="", flush=True)
            answer_parts.append(chunk)

        print_sources(retrieved_docs)
        answer = "".join(answer_parts)

        # Record in chat history for follow-up questions
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def chat(self):
        """Interactive conversation loop with memory."""
        print("Chat mode — ask follow-up questions. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            self.ask(question)


def main():
    if not CHROMA_DIR.exists():
        print("No vector store found. Run ingest.py first.")
        sys.exit(1)

    pipeline = RAGPipeline()

    if "--chat" in sys.argv:
        pipeline.chat()
    elif len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        pipeline.ask(sys.argv[1])
    else:
        print('Usage: python query.py "your question"')
        print('       python query.py --chat')
        sys.exit(1)


if __name__ == "__main__":
    main()
