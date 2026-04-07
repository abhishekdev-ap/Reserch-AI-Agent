"""
RAG (Retrieval-Augmented Generation) module
Handles document ingestion, vector storage, and retrieval
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ─── Embeddings & Vector Store ──────────────────────────────────────────────────

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_PATH, exist_ok=True)

_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="research_docs",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return _vectorstore


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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.05,
)


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

    response = llm.invoke([
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
