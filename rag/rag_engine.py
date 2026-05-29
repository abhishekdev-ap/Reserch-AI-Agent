"""
RAG (Retrieval-Augmented Generation) module
Handles document ingestion, vector storage, and retrieval.
Supports both cloud (Gemini) and local (sentence-transformers) embeddings.
"""
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm, get_embeddings, get_mode_info

# ─── Embeddings & Vector Store ──────────────────────────────────────────────────

class DynamicEmbeddingsProxy:
    def embed_documents(self, texts):
        return get_embeddings().embed_documents(texts)

    def embed_query(self, text):
        return get_embeddings().embed_query(text)

embeddings = DynamicEmbeddingsProxy()
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_PATH, exist_ok=True)

_vectorstores = {}

def get_vectorstore() -> Chroma:
    global _vectorstores
    mode_name = get_mode_info()["mode"]
    if mode_name not in _vectorstores:
        mode_chroma_path = os.path.join(CHROMA_PATH, mode_name)
        os.makedirs(mode_chroma_path, exist_ok=True)
        _vectorstores[mode_name] = Chroma(
            collection_name=f"research_docs_{mode_name}",
            embedding_function=embeddings,
            persist_directory=mode_chroma_path,
        )
    return _vectorstores[mode_name]


# ─── Document Ingestion ────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


def ingest_text(text: str, source: str = "manual", metadata: dict = None) -> int:
    """Ingest plain text into the vector store. Returns number of chunks added."""
    docs = [Document(page_content=text, metadata={"source": source, **(metadata or {})})]
    chunks = text_splitter.split_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def ingest_search_results(results: List[dict]) -> int:
    """Ingest Tavily search results into vector store."""
    docs = []
    for r in results:
        if r.get("content"):
            docs.append(Document(
                page_content=r["content"],
                metadata={
                    "source": r.get("url", "unknown"),
                    "title": r.get("title", ""),
                    "type": "web_search",
                }
            ))
    chunks = text_splitter.split_documents(docs)
    if chunks:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
    return len(chunks)


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = 5) -> List[Document]:
    """Retrieve the most relevant documents for a query."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def retrieve_with_scores(query: str, k: int = 5) -> List[tuple]:
    """Retrieve documents with relevance scores."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


# ─── RAG Query ────────────────────────────────────────────────────────────────

class DynamicLLMProxy:
    def __init__(self, temperature=0.05):
        self.temperature = temperature

    def invoke(self, *args, **kwargs):
        return get_llm(temperature=self.temperature).invoke(*args, **kwargs)

llm = DynamicLLMProxy(temperature=0.05)


import time

def _retry_on_error(func, *args, max_retries=3, **kwargs):
    """Retry a function call with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if any(x in error_str for x in ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE', 'high demand']):
                wait_time = (2 ** attempt) * 10
                print(f"⏳ RAG API Busy ({error_str[:50]}...). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return func(*args, **kwargs)

def rag_query(question: str, k: int = 5) -> dict:
    """
    Answer a question using RAG: retrieve relevant docs + generate answer with citations.
    Returns dict with 'answer', 'sources', and 'docs'.
    """
    docs = retrieve_with_scores(question, k=k)

    if not docs:
        return {
            "answer": "No relevant documents found in the knowledge base. Please run a research query first.",
            "sources": [],
            "docs": [],
        }

    # Filter out low-relevance documents (score below 0.3)
    filtered_docs = [(doc, score) for doc, score in docs if score >= 0.3]
    if not filtered_docs:
        filtered_docs = docs[:2]  # Keep at least top 2 if all are low-scoring

    # Build context with citations
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(filtered_docs):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "")
        sources.append({"index": i + 1, "source": source, "title": title, "score": round(score, 3)})
        context_parts.append(f"[{i+1}] {doc.page_content}\n(Source: {source})")

    context = "\n\n---\n\n".join(context_parts)

    response = _retry_on_error(llm.invoke, [
        SystemMessage(content="""You are a knowledgeable research assistant. Answer questions using ONLY the provided context documents.

STRICT RULES — FOLLOW WITHOUT EXCEPTION:
1. ONLY use information that is EXPLICITLY stated in the context documents below.
2. ALWAYS cite your sources using [number] notation for every factual claim.
3. If the context doesn't contain enough information to answer the question, clearly state: "The available knowledge base does not contain sufficient information to answer this question fully."
4. NEVER fabricate facts, statistics, dates, names, or any other specific information.
5. Do NOT use your general knowledge to supplement the context — ONLY use what is provided.
6. If multiple sources contradict each other, note the contradiction rather than choosing one.
7. Format your answer in clear markdown."""),
        HumanMessage(content=f"Context (USE ONLY THIS — no external knowledge):\n{context}\n\nQuestion: {question}")])

    return {
        "answer": response.content,
        "sources": sources,
        "docs": [doc for doc, _ in docs],
    }


def get_collection_stats() -> dict:
    """Get stats about the current vector store collection."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    count = collection.count()
    return {"total_chunks": count, "collection": "research_docs", "path": CHROMA_PATH}


def clear_collection() -> bool:
    """Clear all documents from the vector store."""
    global _vectorstore
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"source": {"$ne": ""}})
    _vectorstore = None
    return True
