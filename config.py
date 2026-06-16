"""
Centralized configuration for Multi-Agent Research Assistant.
Detects LLM_MODE (cloud / local) and provides the correct LLM,
embeddings, and search tool for the rest of the application.
Supports dynamic mode switching at runtime.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Dynamic Mode Support (Refreshed Linter Check) ───────────────────────────

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
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["DO_NOT_TRACK"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("ANONYMIZED_TELEMETRY", None)
        os.environ.pop("DO_NOT_TRACK", None)

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
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DO_NOT_TRACK"] = "1"

# ─── Cache Containers ────────────────────────────────────────────────────────
_llm_cache = {}
_embeddings_cache = {}

# ─── LLM Provider ────────────────────────────────────────────────────────────

def get_local_ollama_models(base_url: str = None) -> list:
    """Fetch the list of installed Ollama models from the local server."""
    import httpx
    if not base_url:
        try:
            import streamlit as st
            if hasattr(st, "session_state") and "ollama_base_url" in st.session_state:
                base_url = st.session_state.ollama_base_url
        except Exception:
            pass
    if not base_url:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if models:
                return sorted(models)
    except Exception:
        pass
    # Default fallback list if Ollama is unreachable/offline
    return ["llama3.2:latest", "llama3.2", "mistral", "gemma2", "qwen2.5", "deepseek-r1:1.5b", "deepseek-r1:8b"]


def get_llm(temperature: float = 0.1):
    """Return the appropriate LLM based on dynamic LLM_MODE with caching."""
    global _llm_cache
    mode = get_current_mode()
    
    selected_model = None
    if mode == "local":
        try:
            import streamlit as st
            if hasattr(st, "session_state") and "selected_ollama_model" in st.session_state:
                selected_model = st.session_state.selected_ollama_model
        except Exception:
            pass
        if not selected_model:
            selected_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            
        cache_key = (mode, temperature, selected_model)
        if cache_key not in _llm_cache:
            from langchain_ollama import ChatOllama
            base_url = None
            try:
                import streamlit as st
                if hasattr(st, "session_state") and "ollama_base_url" in st.session_state:
                    base_url = st.session_state.ollama_base_url
            except Exception:
                pass
            if not base_url:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                
            _llm_cache[cache_key] = ChatOllama(
                model=selected_model,
                base_url=base_url,
                temperature=temperature,
            )
    else:
        selected_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        cache_key = (mode, temperature, selected_model)
        if cache_key not in _llm_cache:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=selected_model,
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
            
            # Detect real-time financial or news queries (stock, share price, today, latest, etc.)
            q_lower = query.lower()
            realtime_keywords = [
                "today", "price", "share", "stock", "latest", "current", "now", "news", "rate", 
                "market", "valuation", "ticker", "ipo", "earnings", "dividend", "quarterly", "nse", "bse"
            ]
            is_realtime = any(kw in q_lower for kw in realtime_keywords)
            
            news_results = []
            gen_results = []
            if is_realtime:
                # 1. Fetch from General (for stable financial portals like Yahoo Finance, Screener.in, BSE, etc.)
                try:
                    gen_kwargs = {
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_raw_content": False,
                        "topic": "general"
                    }
                    gen_response = tavily_client.search(**gen_kwargs)
                    gen_results = gen_response.get("results", [])
                except Exception as ge:
                    print(f"⚠️ Tavily general search failed: {ge}")

                # 2. Fetch from News (week-scoped for hot-off-the-press updates)
                try:
                    news_kwargs = {
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_raw_content": False,
                        "topic": "news",
                        "time_range": "week",
                    }
                    news_response = tavily_client.search(**news_kwargs)
                    news_results = news_response.get("results", [])
                except Exception as ne:
                    print(f"⚠️ Tavily news search failed: {ne}")
            else:
                # Standard non-realtime search using general topic
                try:
                    search_kwargs = {
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_raw_content": False,
                        "topic": "general",
                    }
                    response = tavily_client.search(**search_kwargs)
                    gen_results = response.get("results", [])
                except Exception as se:
                    print(f"⚠️ Tavily standard search failed: {se}")
            
            # Prioritize results that contain the specific subject terms of the query (e.g., 'nsdl')
            stop_words = {
                "today", "price", "share", "stock", "latest", "current", "now", "news", "rate", 
                "market", "valuation", "ticker", "ipo", "earnings", "dividend", "quarterly", "nse", "bse",
                "of", "and", "the", "in", "is", "a", "for", "on", "what", "tell", "me", "show", "get", "about"
            }
            query_words = [w.strip("?,.+-()\"'") for w in q_lower.split()]
            core_terms = [w for w in query_words if w and w not in stop_words and len(w) > 1]
            
            # Combine: prioritize general search, then news search
            combined_results = gen_results + news_results
            
            seen_urls = set()
            relevant_results = []
            other_results = []
            
            for r in combined_results:
                url = r.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Check if it contains at least one core search term in title or content
                title_lower = r.get("title", "").lower()
                content_lower = r.get("content", "").lower()
                
                has_core_term = any(term in title_lower or term in content_lower for term in core_terms) if core_terms else True
                if has_core_term:
                    relevant_results.append(r)
                else:
                    other_results.append(r)
            
            # Put relevant results first, followed by others, and slice to max_results
            results = (relevant_results + other_results)[:max_results]
            
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
