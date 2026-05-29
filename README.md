# 🔬 Multi-Agent Research Assistant & RAG Engine

A production-grade, autonomous research system powered by **LangGraph**, **FastAPI**, and **Streamlit**. It orchestrates specialized AI agents to search, analyze, critique, and synthesize comprehensive, citation-aware research reports.

The system features a **Dual-Mode Execution Engine**—allowing you to run **100% locally & offline** (using Ollama, DuckDuckGo, and local embeddings) or in the **Cloud** (using Google Gemini and Tavily Search).

---

## 🌐 Dynamic Dual-Mode Architecture

The system is architected with a hot-swappable dynamic proxy system. You can switch between **Cloud Mode** and **Local Mode** instantly via the frontend UI without editing files or restarting any servers.

```mermaid
graph TD
    User([User Query]) --> UI[Streamlit UI / FastAPI]
    UI --> Router{Dynamic Mode Selector}
    
    subgraph Cloud Mode (☁️ Cloud)
        Router -->|Cloud Mode| Gemini[Gemini 2.5 Flash]
        Gemini --> Tavily[Tavily Advanced Search]
        Tavily --> GeminiEmbed[Gemini Embeddings]
        GeminiEmbed --> ChromaCloud[(ChromaDB: cloud collection)]
    end

    subgraph Local Mode (🔒 100% Offline)
        Router -->|Local Mode| Ollama[Ollama: llama3.2]
        Ollama --> DDG[DuckDuckGo Search]
        DDG --> LocalEmbed[sentence-transformers / MiniLM]
        LocalEmbed --> ChromaLocal[(ChromaDB: local collection)]
    end

    ChromaCloud --> Report[Final Synthesized Report with Citations]
    ChromaLocal --> Report
```

---

## ✨ Features

* **Multi-Agent Orchestration**: Built on **LangGraph** to model agents (Search, Critique, Synthesis) as a stateful, cyclic graph with automated fallback tools.
* **Semantic RAG Q&A**: Ingests synthesized reports or custom texts into a local **ChromaDB** vector store to answer follow-up questions with precise citations.
* **Database Isolation**: Dynamically routes local and cloud vector embeddings into separate databases to prevent dimension conflicts and database crashes.
* **Figma-Inspired UI/UX**: Ultra-premium glassmorphic interface featuring a breathing neon background, custom sliding segmented controllers, and interactive chat bubble dialogue threads.
* **100% Privacy-Preserved Local Execution**: Runs fully offline with zero data leaving your machine, bypassing all public APIs and internet connections.

---

## 🛠️ Technology Stack

| Component | Cloud Mode | Local Mode |
|---|---|---|
| **Core LLM** | Google Gemini API (`gemini-1.5-flash`) | Ollama (`llama3.2`) |
| **Web Search** | Tavily Search API (Advanced Depth) | DuckDuckGo Search (Rate-limit safe) |
| **Embeddings** | Google Generative AI Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Local CPU) |
| **Vector DB** | ChromaDB (`chroma_db/cloud`) | ChromaDB (`chroma_db/local`) |
| **Orchestration** | LangGraph & LangChain | LangGraph & LangChain |
| **Backend** | FastAPI & Uvicorn | FastAPI & Uvicorn |
| **Frontend** | Streamlit (Custom Glassmorphism) | Streamlit (Custom Glassmorphism) |

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your machine.

If you plan to run in **Local Mode**, install Ollama and pull the lightweight model:
1. Download Ollama from [ollama.com](https://ollama.com) or run:
   ```bash
   brew install ollama
   ```
2. Pull the default `llama3.2` model:
   ```bash
   ollama pull llama3.2
   ```
3. Make sure the Ollama application is active and running.

---

### 2. Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd "Multi Agent research agent"
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory (refer to `.env.example` if available) and add your keys:
   ```env
   # Mode Selection: "cloud" or "local"
   LLM_MODE=local

   # Cloud Credentials (required for Cloud Mode)
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_MODEL=gemini-1.5-flash

   # Local Settings
   OLLAMA_MODEL=llama3.2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

---

## ⚡ Running the Application

To run the full suite, you will start the FastAPI backend server and the Streamlit frontend UI.

### Step 1: Start the Backend API
The FastAPI backend serves the multi-agent endpoints and custom ingestion routes:
```bash
python api.py
```
*The API will be available at `http://localhost:8000`. You can view the interactive OpenAPI documentation at `http://localhost:8000/docs`.*

### Step 2: Start the Frontend UI
In a separate terminal window (with your virtual environment activated), start the UI:
```bash
streamlit run app.py
```
*The UI will launch automatically in your browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```text
├── agents/
│   └── research_graph.py  # LangGraph research workflow & agent nodes
├── rag/
│   ├── chroma_db/         # Isolated local Chroma DB directories
│   └── rag_engine.py      # Vector store retrieval & QA citation system
├── .env                   # Environment configurations (ignored by git)
├── .gitignore             # Standard project git ignore rules
├── api.py                 # FastAPI backend entry point
├── app.py                 # Premium Streamlit UI & conversation canvas
├── config.py              # Central dynamic mode resolver
├── requirements.txt       # Project python dependencies
└── README.md              # Project documentation
```

---

## 🛡️ Security Best Practices

* **API Key Protection**: Never commit your `.env` file. The `.gitignore` file is pre-configured to ensure no secrets or local database folders are pushed to public repositories.
* **Offline Execution**: Switching to **Local Mode** completely stops all public network calls. All documents, queries, and search actions are handled 100% locally on your computer.
