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

# ─── LLM Provider ────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    """Return the appropriate LLM based on dynamic LLM_MODE."""
    if is_local_active():
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )


# ─── Embeddings Provider ─────────────────────────────────────────────────────

def get_embeddings():
    """Return the appropriate embeddings model based on dynamic LLM_MODE."""
    if is_local_active():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )


# ─── Search Provider ─────────────────────────────────────────────────────────

def do_web_search(query: str, max_results: int = 7) -> list:
    """
    Perform a web search. Uses Tavily in cloud mode, DuckDuckGo in local mode.
    Returns a list of dicts with keys: url, title, content.
    """
    if is_local_active():
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
        return results
    else:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=False,
        )
        return response.get("results", [])


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
