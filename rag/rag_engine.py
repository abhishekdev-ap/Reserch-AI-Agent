"""
RAG (Retrieval-Augmented Generation) module
Handles document ingestion, vector storage, and retrieval.
Supports both cloud (Gemini) and local (sentence-transformers) embeddings.
"""
import os
import sys
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
import io

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

_chroma_clients = {}
_vectorstores = {}

def get_vectorstore() -> Chroma:
    global _vectorstores, _chroma_clients
    mode_name = get_mode_info()["mode"]
    if mode_name not in _vectorstores:
        mode_chroma_path = os.path.join(CHROMA_PATH, mode_name)
        os.makedirs(mode_chroma_path, exist_ok=True)
        try:
            if mode_name not in _chroma_clients:
                _chroma_clients[mode_name] = chromadb.PersistentClient(
                    path=mode_chroma_path,
                    settings=Settings(anonymized_telemetry=False)
                )
            _vectorstores[mode_name] = Chroma(
                collection_name=f"research_docs_{mode_name}",
                embedding_function=embeddings,
                client=_chroma_clients[mode_name],
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vectorstore: {e}")
            # Retry with a fresh directory if corrupted
            _vectorstores.pop(mode_name, None)
            _chroma_clients.pop(mode_name, None)
            raise RuntimeError(
                f"ChromaDB initialization failed for mode '{mode_name}': {e}. "
                f"Try deleting '{mode_chroma_path}' and restarting."
            ) from e
    return _vectorstores[mode_name]



# ─── Document Ingestion ────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    length_function=len,
)


def ingest_text(text: str, source: str = "manual", metadata: dict = None) -> int:
    """Ingest plain text into the vector store. Returns number of chunks added."""
    if not text or not text.strip():
        logger.warning("ingest_text called with empty text, skipping.")
        return 0
    try:
        docs = [Document(page_content=text, metadata={"source": source, **(metadata or {})})]
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Ingesting {len(chunks)} chunks from source '{source}'")
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully ingested {len(chunks)} chunks.")
        return len(chunks)
    except Exception as e:
        logger.error(f"Ingestion failed for source '{source}': {e}", exc_info=True)
        raise RuntimeError(f"Document ingestion failed: {e}") from e


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


def retrieve_with_scores(query: str, k: int = 5, filter: dict = None) -> List[tuple]:
    """Retrieve documents with relevance scores and optional metadata filtering."""
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        # Clamp negative scores to 0 (ChromaDB cosine distance can produce negatives)
        return [(doc, max(0.0, score)) for doc, score in results]
    except Exception as e:
        logger.error(f"Retrieval failed for query '{query[:50]}': {e}", exc_info=True)
        return []


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
    # Step 1: Retrieve documents
    logger.info(f"RAG query: '{question[:80]}' (k={k})")
    docs = retrieve_with_scores(question, k=k)

    if not docs:
        logger.warning("No documents found in knowledge base for RAG query.")
        return {
            "answer": "No relevant documents found in the knowledge base. Please run a research query first or ingest documents in the 'Ingest Documents' tab.",
            "sources": [],
            "docs": [],
        }

    # Step 2: Filter low-relevance documents (score below 0.3)
    filtered_docs = [(doc, score) for doc, score in docs if score >= 0.3]
    if not filtered_docs:
        filtered_docs = docs[:2]  # Keep at least top 2 if all are low-scoring

    logger.info(f"Retrieved {len(filtered_docs)} relevant docs (from {len(docs)} total).")

    # Step 3: Build context with citations
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(filtered_docs):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "")
        sources.append({"index": i + 1, "source": source, "title": title, "score": round(score, 3)})
        context_parts.append(f"[{i+1}] {doc.page_content}\n(Source: {source})")

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer via LLM
    try:
        import datetime
        current_date_str = datetime.date.today().strftime("%B %d, %Y")
        response = _retry_on_error(llm.invoke, [
            SystemMessage(content=f"""You are a knowledgeable research assistant. Answer questions using ONLY the provided context documents.

TEMPORAL GROUNDING:
- TODAY'S DATE IS {current_date_str}. We are currently in the year 2026.
- Financial data, stock prices, and events dated 2025 or 2026 are current, valid, and real-time facts.

STRICT RULES — FOLLOW WITHOUT EXCEPTION:
1. ONLY use information that is EXPLICITLY stated in the context documents below.
2. ALWAYS cite your sources using [number] notation for every factual claim.
3. If the context doesn't contain enough information to answer the question, clearly state: "The available knowledge base does not contain sufficient information to answer this question fully."
4. NEVER fabricate facts, statistics, dates, names, or any other specific information.
5. Do NOT use your general knowledge to supplement the context — ONLY use what is provided.
6. If multiple sources contradict each other, note the contradiction rather than choosing one.
7. Format your answer in clear markdown."""),
            HumanMessage(content=f"Context (USE ONLY THIS — no external knowledge):\n{context}\n\nQuestion: {question}")])

        answer_content = response.content
        if isinstance(answer_content, list):
            answer_content = '\n'.join(
                b.get('text', '') if isinstance(b, dict) else str(b) for b in answer_content
            )
        logger.info(f"RAG answer generated successfully ({len(str(answer_content))} chars).")
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}", exc_info=True)
        answer_content = (
            f"⚠️ Answer generation failed: {str(e)}\n\n"
            f"**Retrieved context is available below.** The LLM could not generate a response. "
            f"Please check that Ollama is running (local mode) or your API key is valid (cloud mode).\n\n"
            f"---\n\n{context}"
        )

    return {
        "answer": str(answer_content),
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
    global _vectorstores, _chroma_clients
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"source": {"$ne": ""}})
    mode_name = get_mode_info()["mode"]
    _vectorstores.pop(mode_name, None)
    _chroma_clients.pop(mode_name, None)
    return True


# ─── PDF Ingestion & Document-Aware Local RAG ───────────────────────────────────

import re

def clean_extracted_text(text: str) -> str:
    """Clean extracted PDF text to remove duplicate whitespace and preserve structure."""
    if not text:
        return ""
    # Standardize spaces and tabs without breaking newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple consecutive newlines (3 or more) with double newlines to maintain paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing/leading spaces on each individual line
    lines = [line.strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines)
    return cleaned.strip()

def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """Parse and ingest a PDF directly from a bytes stream into ChromaDB with structural text cleaning."""
    if not pdf_bytes:
        logger.warning("ingest_pdf_bytes called with empty bytes.")
        return 0
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            cleaned_text = clean_extracted_text(text)
            if cleaned_text:
                docs.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "title": filename,
                        "page": i + 1,
                        "type": "pdf_upload"
                    }
                ))
                
        if not docs:
            logger.warning(f"No readable text extracted from PDF '{filename}'")
            return 0
            
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Extracted {len(chunks)} chunks from PDF '{filename}'")
        
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully ingested PDF '{filename}' into ChromaDB.")
        return len(chunks)
    except Exception as e:
        logger.error(f"PDF ingestion failed for '{filename}': {e}", exc_info=True)
        raise RuntimeError(f"PDF Ingestion failed: {e}") from e


def get_uploaded_documents() -> list:
    """Query ChromaDB for a list of unique uploaded PDF source names and their metadata."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(include=['metadatas'])
        metadatas = results.get("metadatas", [])
        doc_dict = {}
        for meta in metadatas:
            if meta and meta.get("type") == "pdf_upload":
                src = meta.get("source")
                if src not in doc_dict:
                    doc_dict[src] = {"pages": set(), "chunks": 0}
                doc_dict[src]["pages"].add(meta.get("page", 1))
                doc_dict[src]["chunks"] += 1
        
        doc_list = []
        for src, stats in doc_dict.items():
            doc_list.append({
                "name": src,
                "pages": len(stats["pages"]),
                "chunks": stats["chunks"]
            })
        return sorted(doc_list, key=lambda x: x["name"])
    except Exception as e:
        logger.error(f"Failed to query uploaded documents: {e}", exc_info=True)
        return []


def delete_document(source_name: str) -> bool:
    """Delete all chunks belonging to a specific document source from ChromaDB."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        collection.delete(where={"source": source_name})
        # Invalidate local caches to trigger fresh state
        mode_name = get_mode_info()["mode"]
        global _vectorstores, _chroma_clients
        _vectorstores.pop(mode_name, None)
        _chroma_clients.pop(mode_name, None)
        logger.info(f"Successfully deleted document '{source_name}' from vector database.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete document '{source_name}': {e}", exc_info=True)
        return False


def document_qa_query(question: str, k: int = 5) -> dict:
    """
    Strict RAG Q&A limited exclusively to uploaded local PDFs.
    If the documents don't contain enough information, says so cleanly.
    """
    logger.info(f"Document Q&A query: '{question[:80]}' (k={k})")
    # Retrieve ONLY from pdf_upload sources
    docs = retrieve_with_scores(question, k=k, filter={"type": "pdf_upload"})
    
    if not docs:
        return {
            "answer": "No uploaded documents found. Please upload one or more PDFs in the 'Document Q&A' console.",
            "sources": [],
            "docs": [],
        }
    
    # Filter low-relevance documents (score below 0.25)
    filtered_docs = [(doc, score) for doc, score in docs if score >= 0.25]
    if not filtered_docs:
        # Keep the top result to inspect
        filtered_docs = docs[:1]
        
    # Sort filtered documents by source name and page number to preserve original structure/reading order
    def get_sort_key(doc_score):
        doc, _ = doc_score
        return (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
    
    filtered_docs = sorted(filtered_docs, key=get_sort_key)
        
    # Build context and extract citations
    context_parts = []
    sources = []
    for i, (doc, score) in enumerate(filtered_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        title = doc.metadata.get("title", source)
        sources.append({
            "index": i + 1,
            "source": source,
            "title": title,
            "page": page,
            "snippet": doc.page_content[:200] + "...",
            "score": round(score, 3)
        })
        context_parts.append(f"[{i+1}] (Document: {source}, Page: {page})\n{doc.page_content}")
        
    context = "\n\n---\n\n".join(context_parts)
    
    try:
        response = _retry_on_error(llm.invoke, [
            SystemMessage(content="""You are a precise local Document Q&A assistant.
You will be asked questions about the uploaded PDF documents.

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the provided document excerpts in the Context.
2. For EVERY factual claim or answer detail you write, ALWAYS cite the index number, e.g. [1], [2], corresponding to the source document excerpt.
3. If the context does not contain sufficient or clear information to answer the question, state exactly and only:
"The uploaded documents do not contain sufficient information to answer this question."
4. Absolutely DO NOT use your pre-trained general knowledge, external facts, or assumptions.
5. Pay close attention to headings, lists, numbers, and tabular data presented in the text chunks. Chunks have been ordered logically by page numbers to preserve paragraph transitions, page changes, and relational context.
6. Keep your tone neutral, professional, and entirely grounded in the text."""),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ])
        
        answer_content = response.content
        if isinstance(answer_content, list):
            answer_content = '\n'.join(
                b.get('text', '') if isinstance(b, dict) else str(b) for b in answer_content
            )
        logger.info(f"Local Document QA answer generated successfully ({len(str(answer_content))} chars).")
    except Exception as e:
        logger.error(f"Ollama local answer generation failed: {e}", exc_info=True)
        answer_content = (
            f"⚠️ Local Answer Generation Failed: {str(e)}\n\n"
            f"**Retrieved context is available below.** Please check that your local Ollama server is running.\n\n"
            f"---\n\n{context}"
        )
        
    return {
        "answer": str(answer_content),
        "sources": sources,
        "docs": [doc for doc, _ in docs],
    }


