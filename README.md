# 🔬 Multi-Agent Research Assistant & RAG Engine

A production-grade, autonomous research system powered by **LangGraph**, **FastAPI**, and **Streamlit**. It orchestrates specialized AI agents to search, analyze, critique, and compile comprehensive, citation-aware research reports.

The system features a **Dual-Mode Execution Engine**—allowing you to run **100% locally & offline** (using Ollama, DuckDuckGo, and local embeddings) or in the **Cloud** (using Google Gemini and Tavily Search).

---

## 🌐 Dynamic Dual-Mode Architecture

The system is architected with a hot-swappable dynamic proxy system. You can switch between **Cloud Mode** and **Local Mode** instantly via the frontend UI without editing files or restarting any servers.

```mermaid
graph TD
    User([User Query]) --> UI[Streamlit UI / FastAPI]
    UI --> Router{Dynamic Mode Selector}
    
    subgraph "Cloud Mode (☁️ Cloud)"
        Router -->|Cloud Mode| Gemini[Gemini 2.5 Flash]
        Gemini --> Tavily[Tavily Advanced Search]
        Tavily --> GeminiEmbed[Gemini Embeddings]
        GeminiEmbed --> ChromaCloud[(ChromaDB: cloud collection)]
    end

    subgraph "Local Mode (🔒 100% Offline)"
        Router -->|Local Mode| Ollama[Ollama: llama3.2]
        Ollama --> DDG[DuckDuckGo Search]
        DDG --> LocalEmbed[sentence-transformers / MiniLM]
        LocalEmbed --> ChromaLocal[(ChromaDB: local collection)]
    end

    ChromaCloud --> Report[Final Synthesized Report with Citations]
    ChromaLocal --> Report
```

---

## 🧠 Core Architectural Breakdown

The system is built on **LangGraph**, which structures the agent interaction as a stateful, cyclic state machine rather than a simple sequential pipeline.

### 1. The Multi-Agent Workflow
* **🔍 Search & Research Agent**: Takes the user's initial query, plans a search strategy, and dynamically executes parallel search queries. It retrieves raw search results and summarizes them, extracting factual contents and source URLs.
* **🧠 Critique & Analysis Agent**: Acts as an active peer reviewer and fact-checker. It critically evaluates the gathered information, flags unverified or biased claims, identifies logical contradictions, and specifies gaps that require further search iterations.
* **📝 Synthesis & Report Agent**: Compiles all verified research summaries, resolves conflicting views, and synthesizes a beautifully formatted Markdown report complete with inline number citations (e.g., `[1]`).
* **🔄 Cyclic Feedback Loop**: The Critique Agent can trigger the Search Agent to run additional targeted searches up to your chosen `Research Depth` to fill missing information before finalizing the report.

### 2. The Semantic RAG Engine
* **Dynamic Chunking**: The generated report is automatically chunked into overlapping passages (1000 characters with 200 character overlap) and ingested into your local **ChromaDB**.
* **Citation-Aware Q&A**: When asking follow-up questions, the system performs a semantic similarity search in ChromaDB, retrieves the most relevant document chunks, and compiles a comprehensive answer citing exact source materials.

---

## 🔌 Dynamic Proxy & Database Isolation

A crucial engineering highlight of this project is its robust runtime flexibility and database integrity:

### 1. Dynamic LLM & Embeddings Proxies
Rather than importing static model configurations at startup (which would require restarting the server to switch models), the code utilizes lazy-loaded dynamic proxies (`DynamicLLMProxy` and `DynamicEmbeddingsProxy`). These classes intercept standard model calls (`invoke`, `embed_documents`, `embed_query`) and resolve them *on-the-fly* using the currently active environment mode selected in the UI.

### 2. Isolated Vector Storage
Local embeddings (`all-MiniLM-L6-v2`) output vectors with **384 dimensions**, whereas Google Gemini embeddings output **768 dimensions**. Storing both inside the same vector database collection would cause instant coordinate crashes. 
The database manager dynamically isolates storage into separate physical folders:
* **Local Collection**: `chroma_db/local`
* **Cloud Collection**: `chroma_db/cloud`

---

## ✨ Features

* **Multi-Agent Orchestration**: Stateful, cyclic graph modeling built on **LangGraph** with automated fallback search tools.
* **Segmented Q&A Conversations**: Beautiful, high-fidelity chat dialogue threads (with custom avatar glows and animation states) for follow-up Q&A.
* **Premium Figma-Inspired UI/UX**: Ultra-premium Streamlit canvas featuring a breathing neon background, custom sliding segmented controllers, and clean, collapsible references.
* **100% Privacy-Preserved Local Mode**: Bypasses all public APIs and internet connections for full offline data safety.

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
