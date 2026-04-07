"""
FastAPI Backend for Multi-Agent Research Assistant
Provides REST API endpoints for the Streamlit frontend
"""
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))

from agents.research_graph import run_research
from rag.rag_engine import rag_query, ingest_text, get_collection_stats, clear_collection

# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Production-grade agentic research system powered by LangGraph + Gemini + Tavily",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response Models ───────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = 2


class RAGRequest(BaseModel):
    question: str
    k: int = 5


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Multi-Agent Research Assistant"}


@app.post("/research")
def research(req: ResearchRequest):
    """Run the full multi-agent research pipeline."""
    try:
        state = run_research(req.query, max_iterations=req.max_iterations)
        messages = state.get("messages", [])
        
        # Extract final report content
        final_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and len(msg.content) > 100:
                final_content = msg.content
                break

        return {
            "query": req.query,
            "status": state.get("status", "complete"),
            "final_report": final_content,
            "message_count": len(messages),
            "iterations": state.get("iteration", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
def query_rag(req: RAGRequest):
    """Query the RAG knowledge base with citation-aware answers."""
    try:
        result = rag_query(req.question, k=req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ingest")
def ingest(req: IngestRequest):
    """Ingest custom text into the knowledge base."""
    try:
        n_chunks = ingest_text(req.text, source=req.source)
        return {"message": f"Successfully ingested {n_chunks} chunks", "source": req.source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
def rag_stats():
    """Get stats about the knowledge base."""
    try:
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/clear")
def clear_rag():
    """Clear the knowledge base."""
    try:
        clear_collection()
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
