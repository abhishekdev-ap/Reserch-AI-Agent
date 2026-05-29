"""
Centralized configuration for Multi-Agent Research Assistant.
Detects LLM_MODE (cloud / local) and provides the correct LLM,
embeddings, and search tool for the rest of the application.
Supports dynamic mode switching at runtime.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Dynamic Mode Support ───────────────────────────────────────────────────

_mode_override = None

def set_mode(mode: str):
    """Dynamically set the LLM mode (cloud / local) at runtime."""
    global _mode_override, LLM_MODE, IS_LOCAL
    _mode_override = mode.strip().lower()
    LLM_MODE = _mode_override
    IS_LOCAL = _mode_override == "local"
    os.environ["LLM_MODE"] = _mode_override
    if IS_LOCAL:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_OFFLINE", None)

def get_current_mode() -> str:
    """Get the currently active mode string."""
    if _mode_override is not None:
        return _mode_override
    return os.getenv("LLM_MODE", "cloud").strip().lower()

def is_local_active() -> bool:
    """Check if the currently active mode is local."""
    return get_current_mode() == "local"

# Maintain module-level compatibility variables
LLM_MODE = get_current_mode()
IS_LOCAL = is_local_active()

if IS_LOCAL:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_OFFLINE"] = "1"

# ─── Cache Containers ────────────────────────────────────────────────────────
_llm_cache = {}
_embeddings_cache = {}

# ─── LLM Provider ────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    """Return the appropriate LLM based on dynamic LLM_MODE with caching."""
    global _llm_cache
    mode = get_current_mode()
    cache_key = (mode, temperature)
    if cache_key not in _llm_cache:
        if mode == "local":
            from langchain_ollama import ChatOllama
            _llm_cache[cache_key] = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temperature,
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=temperature,
            )
    return _llm_cache[cache_key]


# ─── Embeddings Provider ─────────────────────────────────────────────────────

def get_embeddings():
    """Return the appropriate embeddings model based on dynamic LLM_MODE with caching."""
    global _embeddings_cache
    mode = get_current_mode()
    if mode not in _embeddings_cache:
        if mode == "local":
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_OFFLINE"] = "1"
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings_cache[mode] = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _embeddings_cache[mode] = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
    return _embeddings_cache[mode]


# ─── Search Provider ─────────────────────────────────────────────────────────

def do_web_search(query: str, max_results: int = 7) -> list:
    """
    Perform a web search. Uses Tavily in cloud mode, DuckDuckGo in local mode.
    If offline, API fails, or returns zero results, gracefully falls back to local vector database search.
    """
    if is_local_active():
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "url": r.get("href", "N/A"),
                        "title": r.get("title", "N/A"),
                        "content": r.get("body", "N/A"),
                        "published_date": "N/A",
                    })
            if results:
                return results
            print("⚠️ DuckDuckGo search returned 0 results. Falling back to local knowledge base...")
        except Exception as e:
            print(f"⚠️ DuckDuckGo web search failed (offline): {e}. Falling back to local knowledge base...")
        
        # Local vector database fallback block
        try:
            from rag.rag_engine import get_vectorstore
            vs = get_vectorstore()
            docs = vs.similarity_search(query, k=max_results)
            results = []
            for doc in docs:
                results.append({
                    "url": doc.metadata.get("source", "local_kb"),
                    "title": doc.metadata.get("title", "Local KB"),
                    "content": doc.page_content,
                    "published_date": "N/A",
                    "score": 1.0
                })
            if results:
                print(f"✅ Offline Mode: Retrieved {len(results)} chunks from local vector store.")
                return results
        except Exception as le:
            print(f"❌ Local vector database retrieval failed: {le}")
        
        # Final fallback notice chunk
        return [{
            "url": "offline_kb_fallback",
            "title": "Offline Status",
            "content": f"The system is offline and no matching data was found in the local knowledge base for the search query: '{query}'. Please upload relevant documents in the knowledge base.",
            "published_date": "N/A"
        }]
    else:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
            )
            results = response.get("results", [])
            if results:
                return results
            print("⚠️ Tavily search returned 0 results. Falling back to local knowledge base...")
        except Exception as e:
            print(f"⚠️ Cloud Tavily search failed: {e}. Falling back to local knowledge base...")
        
        # Local vector database fallback block
        try:
            from rag.rag_engine import get_vectorstore
            vs = get_vectorstore()
            docs = vs.similarity_search(query, k=max_results)
            results = []
            for doc in docs:
                results.append({
                    "url": doc.metadata.get("source", "local_kb"),
                    "title": doc.metadata.get("title", "Local KB"),
                    "content": doc.page_content,
                    "published_date": "N/A",
                    "score": 1.0
                })
            if results:
                print(f"✅ Cloud Fallback: Retrieved {len(results)} chunks from local vector store.")
                return results
        except Exception as le:
            print(f"❌ Local vector database retrieval failed: {le}")
        
        return [{
            "url": "offline_kb_fallback",
            "title": "Offline / API Error Status",
            "content": f"API search request failed and no matching data was found in the local knowledge base for: '{query}'. Please check your internet connection or Tavily API key.",
            "published_date": "N/A"
        }]


# ─── Mode Info (for UI) ──────────────────────────────────────────────────────

def get_mode_info() -> dict:
    """Return current mode details for display in the UI."""
    if is_local_active():
        return {
            "mode": "local",
            "label": "🔒 Local Mode",
            "llm": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "search": "DuckDuckGo",
            "embeddings": "all-MiniLM-L6-v2",
            "description": "100% offline — no data leaves your machine",
            "color": "#22c55e",
        }
    else:
        return {
            "mode": "cloud",
            "label": "☁️ Cloud Mode",
            "llm": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            "search": "Tavily",
            "embeddings": "Gemini Embeddings",
            "description": "Uses Google Gemini API for best quality",
            "color": "#6366f1",
        }
