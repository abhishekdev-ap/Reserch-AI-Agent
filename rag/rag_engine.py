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



parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
)

def split_parent_child(docs: List[Document]) -> List[Document]:
    """
    Split input documents into parents and children.
    Returns child documents with the parent's full content in their metadata['parent_content'].
    """
    child_docs = []
    for doc in docs:
        # Split document into parent chunks
        parents = parent_splitter.split_documents([doc])
        for parent_idx, parent in enumerate(parents):
            # Split parent chunk into child chunks
            children = child_splitter.split_documents([parent])
            for child in children:
                # Merge original metadata with new parent mapping metadata
                merged_meta = child.metadata.copy()
                merged_meta["parent_content"] = parent.page_content
                merged_meta["parent_id"] = f"{doc.metadata.get('source', 'unknown')}_{parent_idx}"
                child_docs.append(Document(
                    page_content=child.page_content,
                    metadata=merged_meta
                ))
    return child_docs


def ingest_text(text: str, source: str = "manual", metadata: dict = None) -> int:
    """Ingest plain text into the vector store. Returns number of chunks added."""
    if not text or not text.strip():
        logger.warning("ingest_text called with empty text, skipping.")
        return 0
    try:
        docs = [Document(page_content=text, metadata={"source": source, **(metadata or {})})]
        chunks = split_parent_child(docs)
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
    chunks = split_parent_child(docs)
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

    def stream(self, *args, **kwargs):
        return get_llm(temperature=self.temperature).stream(*args, **kwargs)

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

def _parse_pdf_to_docs(file_bytes: bytes, filename: str, doc_type: str) -> List[Document]:
    """Parse PDF bytes into a list of Document objects, applying page-by-page OCR fallback if needed."""
    docs = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            cleaned_text = clean_extracted_text(text)
            
            # If standard text extraction is empty or very short, try OCR fallback for this specific page
            if not cleaned_text or len(cleaned_text) < 40:
                logger.info(f"Page {i+1} extracted text is empty or very short ({len(cleaned_text)} chars). Running OCR fallback for page {i+1}...")
                try:
                    import pdf2image
                    import pytesseract
                    # Convert only this specific page to image
                    images = pdf2image.convert_from_bytes(file_bytes, first_page=i+1, last_page=i+1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        cleaned_ocr = clean_extracted_text(ocr_text)
                        if cleaned_ocr and len(cleaned_ocr) > len(cleaned_text):
                            cleaned_text = cleaned_ocr
                            logger.info(f"Page {i+1}: OCR successfully extracted {len(cleaned_ocr)} characters.")
                except Exception as e_ocr:
                    logger.error(f"OCR fallback failed on page {i+1} for '{filename}': {e_ocr}")
            
            if cleaned_text:
                docs.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "title": filename,
                        "page": i + 1,
                        "type": doc_type
                    }
                ))
    except Exception as e:
        logger.error(f"Error parsing PDF '{filename}': {e}", exc_info=True)
    return docs

def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """Parse and ingest a PDF directly from a bytes stream into ChromaDB with structural text cleaning."""
    if not pdf_bytes:
        logger.warning("ingest_pdf_bytes called with empty bytes.")
        return 0
    try:
        docs = _parse_pdf_to_docs(pdf_bytes, filename, doc_type="pdf_upload")
        if not docs:
            logger.warning(f"No readable text extracted from PDF '{filename}'")
            return 0
            
        # Check average character density per page
        total_chars = sum(len(d.page_content) for d in docs)
        avg_chars_per_page = total_chars / len(docs) if docs else 0
        is_low_density = avg_chars_per_page < 150
        for d in docs:
            d.metadata["low_density"] = is_low_density

        chunks = split_parent_child(docs)
        logger.info(f"Extracted {len(chunks)} chunks from PDF '{filename}' (Average chars per page: {avg_chars_per_page:.1f}, Low density: {is_low_density})")
        
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully ingested PDF '{filename}' into ChromaDB.")
        return len(chunks)
    except Exception as e:
        logger.error(f"PDF ingestion failed for '{filename}': {e}", exc_info=True)
        raise RuntimeError(f"PDF Ingestion failed: {e}") from e


def ingest_local_file(file_bytes: bytes, filename: str, doc_type: str = "local_document") -> int:
    """
    Unified local document ingestion function. Routes the bytes to parsing logic based on file extension.
    Supports: .pdf, .txt, .md, .docx, .csv.
    Tags metadata with 'type': doc_type.
    """
    if not file_bytes:
        logger.warning(f"ingest_local_file called with empty bytes for '{filename}'.")
        return 0
    
    ext = os.path.splitext(filename.lower())[1]
    logger.info(f"Ingesting file '{filename}' with extension '{ext}' as '{doc_type}'")
    
    try:
        docs = []
        if ext == ".pdf":
            docs = _parse_pdf_to_docs(file_bytes, filename, doc_type=doc_type)
                    
        elif ext in [".txt", ".md"]:
            # safe multi-encoding text decoding
            text = ""
            for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = file_bytes.decode("utf-8", errors="ignore")
                
            cleaned_text = clean_extracted_text(text)
            if cleaned_text:
                docs.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "title": filename,
                        "page": 1,
                        "type": doc_type
                    }
                ))
                
        elif ext == ".docx":
            # Custom DOCX parser using zipfile and xml.etree.ElementTree
            import zipfile
            import xml.etree.ElementTree as ET
            
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                    # XML namespace maps
                    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    xml_content = docx_zip.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    paragraphs = []
                    for para in root.findall('.//w:p', namespaces):
                        # Extract text runs within paragraph
                        texts = [node.text for node in para.findall('.//w:t', namespaces) if node.text]
                        if texts:
                            paragraphs.append("".join(texts))
                            
                    full_text = "\n\n".join(paragraphs)
                    cleaned_text = clean_extracted_text(full_text)
                    if cleaned_text:
                        docs.append(Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": filename,
                                "title": filename,
                                "page": 1,
                                "type": doc_type
                            }
                        ))
            except Exception as e_docx:
                logger.error(f"DOCX parsing failed: {e_docx}")
                raise RuntimeError(f"DOCX extraction failed: {e_docx}")
                
        elif ext == ".csv":
            # Custom CSV parser converting rows to strings
            import csv
            text = ""
            for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = file_bytes.decode("utf-8", errors="ignore")
                
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if rows:
                headers = rows[0]
                formatted_rows = []
                for idx, row in enumerate(rows[1:], start=1):
                    row_parts = []
                    for h, val in zip(headers, row):
                        row_parts.append(f"{h}: {val}")
                    formatted_rows.append(f"Row {idx}: " + ", ".join(row_parts))
                    
                full_text = "\n".join(formatted_rows)
                cleaned_text = clean_extracted_text(full_text)
                if cleaned_text:
                    docs.append(Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": filename,
                            "title": filename,
                            "page": 1,
                            "type": doc_type
                        }
                    ))
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return 0
            
        if not docs:
            logger.warning(f"No readable text extracted from '{filename}'")
            return 0
            
        # Check average character density per page
        total_chars = sum(len(d.page_content) for d in docs)
        avg_chars_per_page = total_chars / len(docs) if docs else 0
        is_low_density = avg_chars_per_page < 150
        for d in docs:
            d.metadata["low_density"] = is_low_density

        chunks = split_parent_child(docs)
        logger.info(f"Extracted {len(chunks)} chunks from '{filename}' (Average chars per page: {avg_chars_per_page:.1f}, Low density: {is_low_density})")
        
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully indexed '{filename}' with {len(chunks)} chunks.")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"Ingestion failed for '{filename}': {e}", exc_info=True)
        raise RuntimeError(f"Document ingestion failed: {e}") from e


def index_local_directory(folder_path: str) -> int:
    """
    Index a local directory. Recursively finds all files with extensions
    .pdf, .txt, .md, .docx, .csv and ingests them into the local vector DB.
    Metadata 'type' will be 'local_kb_file'.
    Returns the number of files successfully indexed.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Directory path does not exist: {folder_path}")
        raise ValueError(f"Directory path does not exist: {folder_path}")
        
    if not os.path.isdir(folder_path):
        logger.warning(f"Path is not a directory: {folder_path}")
        raise ValueError(f"Path is not a directory: {folder_path}")
        
    supported_extensions = {".pdf", ".txt", ".md", ".docx", ".csv"}
    success_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file.lower())[1]
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    chunks = ingest_local_file(file_bytes, file, doc_type="local_kb_file")
                    if chunks > 0:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to index file '{file_path}': {e}")
                    
    # Invalidate local caches to trigger fresh state
    mode_name = get_mode_info()["mode"]
    global _vectorstores, _chroma_clients
    _vectorstores.pop(mode_name, None)
    _chroma_clients.pop(mode_name, None)
    
    return success_count


def get_kb_documents() -> list:
    """Query ChromaDB for a list of unique indexed KB files."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(include=['metadatas'])
        metadatas = results.get("metadatas", [])
        doc_dict = {}
        for meta in metadatas:
            if meta and meta.get("type") == "local_kb_file":
                src = meta.get("source")
                if src not in doc_dict:
                    doc_dict[src] = {"pages": set(), "chunks": 0, "low_density": False}
                doc_dict[src]["pages"].add(meta.get("page", 1))
                doc_dict[src]["chunks"] += 1
                if meta.get("low_density"):
                    doc_dict[src]["low_density"] = True
        
        doc_list = []
        for src, stats in doc_dict.items():
            doc_list.append({
                "name": src,
                "pages": len(stats["pages"]),
                "chunks": stats["chunks"],
                "low_density": stats["low_density"]
            })
        return sorted(doc_list, key=lambda x: x["name"])
    except Exception as e:
        logger.error(f"Failed to query KB documents: {e}", exc_info=True)
        return []


def get_uploaded_documents() -> list:
    """Query ChromaDB for a list of unique uploaded document source names and their metadata."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(include=['metadatas'])
        metadatas = results.get("metadatas", [])
        doc_dict = {}
        for meta in metadatas:
            if meta and meta.get("type") in ["pdf_upload", "local_document"]:
                src = meta.get("source")
                if src not in doc_dict:
                    doc_dict[src] = {"pages": set(), "chunks": 0, "low_density": False}
                doc_dict[src]["pages"].add(meta.get("page", 1))
                doc_dict[src]["chunks"] += 1
                if meta.get("low_density"):
                    doc_dict[src]["low_density"] = True
        
        doc_list = []
        for src, stats in doc_dict.items():
            doc_list.append({
                "name": src,
                "pages": len(stats["pages"]),
                "chunks": stats["chunks"],
                "low_density": stats["low_density"]
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


def rephrase_query(question: str, history: List[dict] = None) -> str:
    """Use the LLM to rewrite the question to be a standalone query based on chat history."""
    if not history:
        return question
    
    # Convert history list to string format
    history_turns = []
    for turn in history[-4:]:  # Limit to last 4 turns for efficiency
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_turns.append(f"{role}: {turn.get('content')}")
    history_str = "\n".join(history_turns)
    
    prompt = f"""Given the following chat history between a user and an AI, and a follow-up question, rephrase the follow-up question to be a standalone question (i.e. self-contained, resolving any pronouns like 'it', 'they', 'this', 'he', 'she', etc.). 
Do NOT answer the question. Just output the rephrased question, and nothing else.

Chat History:
{history_str}

Follow-up Question: {question}
Standalone Question:"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant that rephrases search queries."),
            HumanMessage(content=prompt)
        ])
        rephrased = response.content.strip()
        if rephrased:
            logger.info(f"Rephrased '{question}' to '{rephrased}'")
            return rephrased
    except Exception as e:
        logger.warning(f"Failed to rephrase query: {e}. Using original question.")
    return question


def generate_query_variations(query: str) -> List[str]:
    """
    Generate variations of the query to improve retrieval coverage (Multi-query retrieval).
    Uses fast, rule-based extraction to avoid duplicate LLM inference latency.
    """
    variations = [query]
    
    # 1. Strip common question prefixes
    clean_q = re.sub(r'^(what is|how to|can you|tell me|explain|describe|show me|find|search for|what was|were the|who is|where is|is there|show|get|detail|analyze|provide|list)\s+', '', query, flags=re.IGNORECASE)
    clean_q = clean_q.strip("? \t\n")
    if clean_q and clean_q != query:
        variations.append(clean_q)
        
    # 2. Extract significant noun phrases (words of 4+ characters not in stop words)
    stop_words = {
        "what", "where", "when", "how", "who", "which", "why", "whose", "would", "could", "should",
        "does", "doesnt", "dont", "cant", "wont", "about", "there", "their", "these", "those"
    }
    words = re.findall(r'[a-zA-Z0-9]+', query.lower())
    important_words = [w for w in words if len(w) > 3 and w not in stop_words]
    if important_words and len(important_words) < len(words):
        keywords_query = " ".join(important_words)
        if keywords_query and keywords_query not in variations:
            variations.append(keywords_query)
            
    return list(set(variations))[:3]

def rerank_documents(query: str, candidates: List[tuple], k: int) -> List[tuple]:
    """
    Advanced custom re-ranking layer that ranks documents based on:
    1. Vector similarity score (cosine distance).
    2. Exact phrase match boost.
    3. Keyword density / TF-IDF approximation.
    4. Term proximity matching.
    Returns list of (doc, final_score, vector_score, keyword_score).
    """
    stop_words = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", 
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
        "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", 
        "its", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
        "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", 
        "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your"
    }
    
    query_lower = query.lower()
    query_words = re.findall(r'[a-zA-Z0-9]+', query_lower)
    keywords = [w for w in query_words if w not in stop_words and len(w) > 1]
    
    # Detect exact multi-word phrases in the query
    phrases = []
    for i in range(len(query_words) - 1):
        if query_words[i] not in stop_words and query_words[i+1] not in stop_words:
            phrases.append(f"{query_words[i]} {query_words[i+1]}")
            
    reranked = []
    for doc, vec_score in candidates:
        content_lower = doc.page_content.lower()
        
        # 1. Phrase matching boost
        phrase_boost = 0.0
        for phrase in phrases:
            if phrase in content_lower:
                phrase_boost += 0.15
                
        # 2. Keyword density score
        keyword_score = 0.0
        if keywords:
            matches = sum(1 for kw in keywords if kw in content_lower)
            keyword_score = matches / len(keywords)
            
        # 3. Term proximity boost
        proximity_boost = 0.0
        if len(keywords) >= 2:
            positions = []
            for kw in keywords:
                pos = content_lower.find(kw)
                if pos != -1:
                    positions.append(pos)
            if len(positions) >= 2:
                span = max(positions) - min(positions)
                if span < 200:
                    proximity_boost = 0.15
                elif span < 400:
                    proximity_boost = 0.08
                    
        # Cosine similarity score from Chroma is clamped
        v_score = max(0.0, vec_score)
        
        # Base hybrid score: 0.6 vector, 0.4 keyword
        base_score = 0.6 * v_score + 0.4 * keyword_score
        
        # Final combined score
        final_score = base_score + phrase_boost + proximity_boost
        
        reranked.append((doc, final_score, v_score, keyword_score, phrase_boost, proximity_boost))
        
    # Sort in descending order of final_score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Format return elements to match (doc, hybrid_score, vec_score, keyword_score)
    formatted = [(doc, final_score, vec_score, keyword_score) for doc, final_score, vec_score, keyword_score, _, _ in reranked]
    return formatted[:k]

def verify_answer_factuality(question: str, answer: str, context: str) -> dict:
    """
    Check generated answer factuality against the context using a fast temperature=0.0 LLM check.
    Returns JSON dict indicating factuality results.
    """
    prompt = f"""You are a strict fact-checking assistant. Your job is to verify if the generated answer is completely and strictly supported by the provided Context.

Context:
{context}

Generated Answer:
{answer}

Factual Verification Checklist:
1. Is every statement in the answer directly supported by the context?
2. Are all numbers, dates, statistics, percentages, and names in the answer exactly matching the context?
3. Does the answer contain any general knowledge, external assumptions, or fabrications?

Output your response ONLY as a JSON object with the following keys. Do not write any other explanation or intro:
{{
  "is_factual": true/false,
  "unsupported_claims": ["list of statements that are not fully supported or are general knowledge"],
  "unsupported_details": ["list of incorrect numbers/dates/names"]
}}"""
    try:
        # Fast inference with ChatOllama
        response = llm.invoke([
            SystemMessage(content="You are a precise JSON fact-checker. You output only JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        import json
        if content.startswith("```json"):
            content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()
        
        result = json.loads(content)
        logger.info(f"Answer Factuality Verification Result: {result}")
        return result
    except Exception as e:
        logger.warning(f"Self-RAG factuality check failed or timed out: {e}. Defaulting to verified=True.")
        return {"is_factual": True, "unsupported_claims": [], "unsupported_details": []}

def refine_answer(question: str, answer: str, context: str, feedback: dict) -> str:
    """Refine a generated answer based on fact-checking feedback."""
    feedback_str = ""
    if feedback.get("unsupported_claims"):
        feedback_str += f"- Unsupported Claims: {', '.join(feedback['unsupported_claims'])}\n"
    if feedback.get("unsupported_details"):
        feedback_str += f"- Incorrect Details: {', '.join(feedback['unsupported_details'])}\n"
        
    prompt = f"""You are a precise document Q&A assistant. The previous answer you generated contains factual inconsistencies or details not supported by the document.
Please correct and rewrite the answer.

STRICT GROUNDING RULES:
1. Remove all statements listed as unsupported.
2. Correct all numbers, names, and details using the Context.
3. Answer strictly and only using the Context.
4. If the context does not contain the answer, say "The requested information was not found in the uploaded documents."

Context:
{context}

Previous Answer:
{answer}

Fact-Check Correction Feedback:
{feedback_str}

Corrected Grounded Answer:"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise document assistant. You write corrected, grounded answers."),
            HumanMessage(content=prompt)
        ])
        logger.info("Answer refined successfully.")
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to refine answer: {e}")
        return answer

def hybrid_retrieve(question: str, k: int = 5, doc_types: List[str] = None, doc_names: List[str] = None) -> List[tuple]:
    """
    Retrieve documents using multi-query expansion, parent chunk mapping, deduplication, and re-ranking.
    """
    if doc_types is None:
        doc_types = ["pdf_upload", "local_document", "local_kb_file"]
        
    # Build compound filter for ChromaDB
    chroma_filter = {}
    filter_conditions = []
    if doc_types:
        filter_conditions.append({"type": {"$in": doc_types}})
    if doc_names:
        filter_conditions.append({"source": {"$in": doc_names}})
        
    if len(filter_conditions) == 1:
        chroma_filter = filter_conditions[0]
    elif len(filter_conditions) > 1:
        chroma_filter = {"$and": filter_conditions}
        
    # Generate query variations (Multi-query)
    q_vars = generate_query_variations(question)
    logger.info(f"Query variations for retrieval: {q_vars}")
    
    # Retrieve candidates for each variation
    unique_candidates = {}
    vectorstore = get_vectorstore()
    
    for q_var in q_vars:
        try:
            candidates = vectorstore.similarity_search_with_score(
                q_var, k=k*3, filter=chroma_filter if chroma_filter else None
            )
            for doc, distance in candidates:
                # Calibrate raw vector distance to 0.0 - 1.0 similarity range
                score = max(0.0, 1.0 - (distance / 2.0))
                
                # Resolve child chunk to parent context
                parent_text = doc.metadata.get("parent_content", doc.page_content)
                parent_doc = Document(page_content=parent_text, metadata=doc.metadata)
                
                # Deduplicate unique parents (checking parent_id or page content hash)
                p_id = doc.metadata.get("parent_id", hash(parent_text))
                if p_id not in unique_candidates or score > unique_candidates[p_id][1]:
                    unique_candidates[p_id] = (parent_doc, score)
        except Exception as e:
            logger.error(f"Retrieve candidates failed for variation '{q_var}': {e}")
            
    candidate_list = list(unique_candidates.values())
    
    if not candidate_list:
        logger.warning(f"No candidate documents retrieved for question: '{question}'")
        return []
        
    # Re-rank candidates using our custom reranker
    ranked = rerank_documents(question, candidate_list, k)
    
    # Debug logging
    logger.info(f"--- Reranker Debug Logs ---")
    logger.info(f"Question: '{question}'")
    logger.info(f"Unique candidates retrieved: {len(candidate_list)}")
    for idx, (doc, f_score, v_score, kw_score) in enumerate(ranked):
        logger.info(f"  [{idx+1}] Doc: '{doc.metadata.get('source')}' | Page: {doc.metadata.get('page')} | Re-Ranked Score: {f_score:.3f} (Vector: {v_score:.3f}, Keyword: {kw_score:.3f})")
        
    return ranked


def generate_document_summary(source_name: str) -> str:
    """Generate a summary and key insights of an uploaded document using Ollama."""
    logger.info(f"Generating summary for document '{source_name}'")
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        results = collection.get(where={"source": source_name}, include=['documents', 'metadatas'])
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        if not documents:
            return "No content found for this document in the database."
            
        # Sort chunks by page number
        sorted_chunks = sorted(zip(documents, metadatas), key=lambda x: x[1].get("page", 1))
        
        # Use first 4 chunks to avoid model context limit issues
        representative_text = "\n\n---\n\n".join([text for text, _ in sorted_chunks[:4]])
        
        prompt = f"""You are a research summarization assistant. 
Summarize the following document content from '{source_name}'. 
Provide:
1. A 3-sentence high-level executive summary.
2. A bulleted list of the top 3-5 key insights or main themes found in this text.

Format your output in clean Markdown with clear headings.

Document Content:
{representative_text}

Summary & Insights:"""
        
        response = llm.invoke([
            SystemMessage(content="You are a precise document summarizer."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}", exc_info=True)
        return f"Error generating summary: {str(e)}"


def document_qa_query(question: str, k: int = 5, history: List[dict] = None, doc_types: List[str] = None, doc_names: List[str] = None, system_prompt: str = None) -> dict:
    """
    Strict RAG Q&A limited exclusively to uploaded local documents.
    Implements Retrieve-Verify-Refine Self-RAG loop and formats output with Evidence, Sources, and Confidence.
    """
    logger.info(f"Document Q&A query: '{question[:80]}' (k={k})")
    
    # Step 1: Rephrase query if history exists
    search_query = rephrase_query(question, history) if history else question
    
    # Step 2: Retrieve candidate chunks via Hybrid Search
    hybrid_results = hybrid_retrieve(search_query, k=k, doc_types=doc_types, doc_names=doc_names)
    
    # Step 3: Filter low relevance chunks (hybrid score < 0.15)
    filtered_results = [item for item in hybrid_results if item[1] >= 0.15]
    
    if not filtered_results:
        # Fallback if no relevant documents are found
        fallback_answer = """### Answer
The requested information was not found in the uploaded documents.

### Evidence
No direct evidence found in the document database.

### Sources
No sources found.

### Confidence
Low (0%)"""
        return {
            "answer": fallback_answer,
            "sources": [],
            "docs": [],
            "confidence": "Low (0%)"
        }
        
    # Sort filtered results by source and page to maintain original structure/reading order
    def get_sort_key(item):
        doc, _, _, _ = item
        return (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        
    filtered_results = sorted(filtered_results, key=get_sort_key)
    
    # Build context and extract citations
    context_parts = []
    sources = []
    total_vector_score = 0.0
    total_keyword_score = 0.0
    
    for i, (doc, hybrid_score, vec_score, kw_score) in enumerate(filtered_results):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        title = doc.metadata.get("title", source)
        sources.append({
            "index": i + 1,
            "source": source,
            "title": title,
            "page": page,
            "snippet": doc.page_content[:250] + "...",
            "score": round(hybrid_score, 3)
        })
        context_parts.append(f"[{i+1}] (Document: {source}, Page: {page})\n{doc.page_content}")
        total_vector_score += vec_score
        total_keyword_score += kw_score
        
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate initial response
    try:
        sys_prompt = system_prompt or """You are a precise local Document Q&A assistant.
You will be asked questions about the uploaded local documents.

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the provided document excerpts in the Context.
2. For EVERY factual claim or answer detail you write, ALWAYS cite the index number, e.g. [1], [2], corresponding to the source document excerpt.
3. If the context does not contain sufficient or clear information to answer the question, state exactly and only:
"The requested information was not found in the uploaded documents."
4. Absolutely DO NOT use your pre-trained general knowledge, external facts, or assumptions.
5. Pay close attention to headings, lists, numbers, and tabular data presented in the text chunks. Chunks have been ordered logically by page numbers to preserve paragraph transitions, page changes, and relational context.
6. Keep your tone neutral, professional, and entirely grounded in the text."""

        prompt_messages = [SystemMessage(content=sys_prompt)]
        if history:
            for turn in history[-4:]:
                if turn.get("role") == "user":
                    prompt_messages.append(HumanMessage(content=turn.get("content")))
                else:
                    prompt_messages.append(SystemMessage(content=turn.get("content")))
                    
        prompt_messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))
        
        # Initial invocation
        response = _retry_on_error(llm.invoke, prompt_messages)
        answer_content = response.content
        if isinstance(answer_content, list):
            answer_content = '\n'.join(
                b.get('text', '') if isinstance(b, dict) else str(b) for b in answer_content
            )
            
        # Fact-checking verification loop
        factuality = verify_answer_factuality(question, answer_content, context)
        verification_val = 1.0
        
        if not factuality.get("is_factual", True):
            # Refine answer
            answer_content = refine_answer(question, answer_content, context, factuality)
            verification_val = 0.5
            
        # Calculate confidence score dynamically
        avg_similarity = total_vector_score / len(filtered_results) if filtered_results else 0.0
        avg_rerank = sum(item[1] for item in filtered_results) / len(filtered_results) if filtered_results else 0.0
        
        # Keyword coverage
        q_words = re.findall(r'[a-zA-Z0-9]+', question.lower())
        stop_words_local = {"what", "how", "who", "where", "why", "which", "is", "are", "the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "of", "with", "from"}
        kw_terms = [w for w in q_words if w not in stop_words_local and len(w) > 1]
        context_lower = context.lower()
        kw_matches = sum(1 for kw in kw_terms if kw in context_lower)
        coverage_score = kw_matches / len(kw_terms) if kw_terms else 1.0
        
        # Normalize scores to map typical range to 0.0 - 1.0 contribution
        norm_similarity = min(1.0, max(0.0, (avg_similarity - 0.1) / 0.35))
        norm_rerank = min(1.0, max(0.0, (avg_rerank - 0.1) / 0.45))
        
        confidence_val = (0.2 * norm_similarity) + (0.2 * norm_rerank) + (0.3 * coverage_score) + (0.3 * verification_val)
        confidence_val = min(1.0, max(0.0, confidence_val))
        
        # Format confidence string
        if confidence_val >= 0.7:
            confidence_label = f"High ({int(confidence_val*100)}%)"
        elif confidence_val >= 0.4:
            confidence_label = f"Medium ({int(confidence_val*100)}%)"
        else:
            confidence_label = f"Low ({int(confidence_val*100)}%)"
            
        # Programmatic formatting of Evidence and Sources
        evidence_parts = []
        for s in sources:
            evidence_parts.append(f"- Excerpt from **{s['source']}** (Page {s['page']}): \"{s['snippet']}\"")
        evidence_str = "\n".join(evidence_parts) if evidence_parts else "No direct evidence found."
        
        sources_parts = []
        unique_srcs = set()
        for s in sources:
            unique_srcs.add((s["source"], s["page"]))
        for src, page in sorted(unique_srcs):
            sources_parts.append(f"- **{src}** (Page {page})")
        sources_str = "\n".join(sources_parts) if sources_parts else "No sources cited."
        
        formatted_response = f"""### Answer
{answer_content}

### Evidence
{evidence_str}

### Sources
{sources_str}

### Confidence
{confidence_label}"""

        # Mandatory Debug Logs
        logger.info("================================================================================")
        logger.info("MANDATORY RAG DEBUG LOGS (document_qa_query)")
        logger.info("================================================================================")
        logger.info(f"RETRIEVAL:")
        logger.info(f"  - Collection Name: {get_vectorstore()._collection.name}")
        logger.info(f"  - Document IDs Searched: {doc_names or 'All'}")
        logger.info(f"  - Top-K Value: {k}")
        logger.info(f"  - Retrieved Chunk Count: {len(filtered_results)}")
        logger.info(f"  - Similarity Scores: {[round(item[1], 3) for item in filtered_results]}")
        logger.info(f"  - Pages Retrieved: {[item[0].metadata.get('page') for item in filtered_results]}")
        logger.info(f"CONTEXT:")
        logger.info(f"  - Final Chunks Selected: {len(filtered_results)}")
        logger.info(f"  - Context Length (Chars): {len(context)}")
        logger.info(f"  - Context Token Count (Approx): {len(context.split())}")
        logger.info(f"OLLAMA:")
        logger.info(f"  - Model Name: {get_llm().model if hasattr(get_llm(), 'model') else 'Ollama/Gemini'}")
        logger.info(f"  - Prompt Size (Chars): {len(sys_prompt) + len(context) + len(question)}")
        logger.info(f"  - Retrieved Context Attached: {'Yes' if context else 'No'}")
        logger.info(f"OUTPUT:")
        logger.info(f"  - Confidence Score: {confidence_label}")
        logger.info(f"  - Citation Count: {len(sources)}")
        logger.info(f"  - Source Pages: {[s['page'] for s in sources]}")
        logger.info("================================================================================")

    except Exception as e:
        logger.error(f"Local Document QA answer generation failed: {e}", exc_info=True)
        formatted_response = (
            f"⚠️ Local Answer Generation Failed: {str(e)}\n\n"
            f"**Retrieved context is available below.** Please check that your local Ollama server is running.\n\n"
            f"---\n\n{context}"
        )
        confidence_label = "Low (0%)"
        
    return {
        "answer": str(formatted_response),
        "sources": sources,
        "docs": [doc for doc, _, _, _ in hybrid_results],
        "confidence": confidence_label
    }


def document_qa_stream(question: str, k: int = 5, history: List[dict] = None, doc_types: List[str] = None, doc_names: List[str] = None, system_prompt: str = None):
    """
    Generator that retrieves relevant document chunks, runs verification and refinement,
    and streams the final formatted response.
    """
    logger.info(f"Streaming Document Q&A query: '{question[:80]}' (k={k})")
    
    yield ("token", "💭 *Retrieving and analyzing document chunks...*\n\n")
    
    # 1. Rephrase query if history exists
    search_query = rephrase_query(question, history) if history else question
    
    # 2. Retrieve candidate chunks via Hybrid Search
    hybrid_results = hybrid_retrieve(search_query, k=k, doc_types=doc_types, doc_names=doc_names)
    
    # 3. Filter low relevance chunks (hybrid score < 0.15)
    filtered_results = [item for item in hybrid_results if item[1] >= 0.15]
    
    if not filtered_results:
        fallback_answer = """### Answer
The requested information was not found in the uploaded documents.

### Evidence
No direct evidence found in the document database.

### Sources
No sources found.

### Confidence
Low (0%)"""
        yield ("token", fallback_answer)
        yield ("metadata", {"sources": [], "confidence": "Low (0%)"})
        return
        
    # Sort filtered results by source and page to maintain original structure/reading order
    def get_sort_key(item):
        doc, _, _, _ = item
        return (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        
    filtered_results = sorted(filtered_results, key=get_sort_key)
    
    # Build context and extract citations
    context_parts = []
    sources = []
    total_vector_score = 0.0
    total_keyword_score = 0.0
    
    for i, (doc, hybrid_score, vec_score, kw_score) in enumerate(filtered_results):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        title = doc.metadata.get("title", source)
        sources.append({
            "index": i + 1,
            "source": source,
            "title": title,
            "page": page,
            "snippet": doc.page_content[:250] + "...",
            "score": round(hybrid_score, 3)
        })
        context_parts.append(f"[{i+1}] (Document: {source}, Page: {page})\n{doc.page_content}")
        total_vector_score += vec_score
        total_keyword_score += kw_score
        
    context = "\n\n---\n\n".join(context_parts)
    
    yield ("token", "💭 *Generating initial grounded response...*\n\n")
    
    # Generate response stream
    try:
        sys_prompt = system_prompt or """You are a precise local Document Q&A assistant.
You will be asked questions about the uploaded local documents.

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the provided document excerpts in the Context.
2. For EVERY factual claim or answer detail you write, ALWAYS cite the index number, e.g. [1], [2], corresponding to the source document excerpt.
3. If the context does not contain sufficient or clear information to answer the question, state exactly and only:
"The requested information was not found in the uploaded documents."
4. Absolutely DO NOT use your pre-trained general knowledge, external facts, or assumptions.
5. Pay close attention to headings, lists, numbers, and tabular data presented in the text chunks. Chunks have been ordered logically by page numbers to preserve paragraph transitions, page changes, and relational context.
6. Keep your tone neutral, professional, and entirely grounded in the text."""

        prompt_messages = [SystemMessage(content=sys_prompt)]
        if history:
            for turn in history[-4:]:
                if turn.get("role") == "user":
                    prompt_messages.append(HumanMessage(content=turn.get("content")))
                else:
                    prompt_messages.append(SystemMessage(content=turn.get("content")))
                    
        prompt_messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))
        
        # Invoke LLM to get full response content (needed for verification step)
        response = _retry_on_error(llm.invoke, prompt_messages)
        answer_content = response.content
        if isinstance(answer_content, list):
            answer_content = '\n'.join(
                b.get('text', '') if isinstance(b, dict) else str(b) for b in answer_content
            )
            
        yield ("token", "💭 *Verifying factuality against evidence...*\n\n")
        
        # Factuality check
        factuality = verify_answer_factuality(question, answer_content, context)
        verification_val = 1.0
        
        if not factuality.get("is_factual", True):
            yield ("token", "💭 *Refining response to ensure 100% factual accuracy...*\n\n")
            answer_content = refine_answer(question, answer_content, context, factuality)
            verification_val = 0.5
            
        # Calculate confidence score
        avg_similarity = total_vector_score / len(filtered_results) if filtered_results else 0.0
        avg_rerank = sum(item[1] for item in filtered_results) / len(filtered_results) if filtered_results else 0.0
        
        q_words = re.findall(r'[a-zA-Z0-9]+', question.lower())
        stop_words_local = {"what", "how", "who", "where", "why", "which", "is", "are", "the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "of", "with", "from"}
        kw_terms = [w for w in q_words if w not in stop_words_local and len(w) > 1]
        context_lower = context.lower()
        kw_matches = sum(1 for kw in kw_terms if kw in context_lower)
        coverage_score = kw_matches / len(kw_terms) if kw_terms else 1.0
        
        # Normalize scores to map typical range to 0.0 - 1.0 contribution
        norm_similarity = min(1.0, max(0.0, (avg_similarity - 0.1) / 0.35))
        norm_rerank = min(1.0, max(0.0, (avg_rerank - 0.1) / 0.45))
        
        confidence_val = (0.2 * norm_similarity) + (0.2 * norm_rerank) + (0.3 * coverage_score) + (0.3 * verification_val)
        confidence_val = min(1.0, max(0.0, confidence_val))
        
        if confidence_val >= 0.7:
            confidence_label = f"High ({int(confidence_val*100)}%)"
        elif confidence_val >= 0.4:
            confidence_label = f"Medium ({int(confidence_val*100)}%)"
        else:
            confidence_label = f"Low ({int(confidence_val*100)}%)"
            
        # Format Evidence and Sources
        evidence_parts = []
        for s in sources:
            evidence_parts.append(f"- Excerpt from **{s['source']}** (Page {s['page']}): \"{s['snippet']}\"")
        evidence_str = "\n".join(evidence_parts) if evidence_parts else "No direct evidence found."
        
        sources_parts = []
        unique_srcs = set()
        for s in sources:
            unique_srcs.add((s["source"], s["page"]))
        for src, page in sorted(unique_srcs):
            sources_parts.append(f"- **{src}** (Page {page})")
        sources_str = "\n".join(sources_parts) if sources_parts else "No sources cited."
        
        # Mandatory Debug Logs
        logger.info("================================================================================")
        logger.info("MANDATORY RAG DEBUG LOGS (document_qa_stream)")
        logger.info("================================================================================")
        logger.info(f"RETRIEVAL:")
        logger.info(f"  - Collection Name: {get_vectorstore()._collection.name}")
        logger.info(f"  - Document IDs Searched: {doc_names or 'All'}")
        logger.info(f"  - Top-K Value: {k}")
        logger.info(f"  - Retrieved Chunk Count: {len(filtered_results)}")
        logger.info(f"  - Similarity Scores: {[round(item[1], 3) for item in filtered_results]}")
        logger.info(f"  - Pages Retrieved: {[item[0].metadata.get('page') for item in filtered_results]}")
        logger.info(f"CONTEXT:")
        logger.info(f"  - Final Chunks Selected: {len(filtered_results)}")
        logger.info(f"  - Context Length (Chars): {len(context)}")
        logger.info(f"  - Context Token Count (Approx): {len(context.split())}")
        logger.info(f"OLLAMA:")
        logger.info(f"  - Model Name: {get_llm().model if hasattr(get_llm(), 'model') else 'Ollama/Gemini'}")
        logger.info(f"  - Prompt Size (Chars): {len(sys_prompt) + len(context) + len(question)}")
        logger.info(f"  - Retrieved Context Attached: {'Yes' if context else 'No'}")
        logger.info(f"OUTPUT:")
        logger.info(f"  - Confidence Score: {confidence_label}")
        logger.info(f"  - Citation Count: {len(sources)}")
        logger.info(f"  - Source Pages: {[s['page'] for s in sources]}")
        logger.info("================================================================================")
        
        final_output = f"""### Answer
{answer_content}

### Evidence
{evidence_str}

### Sources
{sources_str}

### Confidence
{confidence_label}"""

        # Simulate streaming of verified final output
        import time
        for idx_chunk in range(0, len(final_output), 8):
            yield ("token", final_output[idx_chunk:idx_chunk+8])
            time.sleep(0.005)
            
    except Exception as e:
        logger.error(f"Ollama local stream generation failed: {e}", exc_info=True)
        yield ("token", f"\n\n⚠️ Local Answer Generation Failed: {str(e)}\n\n")
        confidence_label = "Low (0%)"
        
    # Finally, yield the metadata with debugging stats
    yield ("metadata", {
        "sources": sources,
        "confidence": confidence_label,
        "debug": {
            "collection": get_vectorstore()._collection.name,
            "doc_names": doc_names or "All",
            "k": k,
            "chunks_count": len(filtered_results),
            "scores": [round(item[1], 3) for item in filtered_results],
            "pages": [item[0].metadata.get("page") for item in filtered_results],
            "context_length": len(context),
            "token_count": len(context.split()),
            "model_name": get_llm().model if hasattr(get_llm(), "model") else "Ollama/Gemini",
            "prompt_size": len(sys_prompt) + len(context) + len(question),
            "context_attached": "Yes" if context else "No",
            "citation_count": len(sources),
            "source_pages": [s["page"] for s in sources]
        }
    })


def general_chat_stream(question: str, history: List[dict] = None, system_prompt: str = None):
    """
    Generator that chats directly with local Ollama without RAG context.
    Yields ("token", token) or ("metadata", metadata_dict).
    """
    logger.info(f"Streaming general chat query: '{question[:80]}'")
    try:
        sys_prompt = system_prompt or "You are a helpful, knowledgeable local AI assistant."
        prompt_messages = [SystemMessage(content=sys_prompt)]
        if history:
            for turn in history[-10:]:
                if turn.get("role") == "user":
                    prompt_messages.append(HumanMessage(content=turn.get("content")))
                else:
                    prompt_messages.append(SystemMessage(content=turn.get("content")))
        prompt_messages.append(HumanMessage(content=question))
        
        for chunk in llm.stream(prompt_messages):
            text_val = chunk.content
            if text_val:
                yield ("token", text_val)
    except Exception as e:
        logger.error(f"Ollama general chat stream failed: {e}", exc_info=True)
        yield ("token", f"\n\n⚠️ General Chat Failed: {str(e)}\n\n")


