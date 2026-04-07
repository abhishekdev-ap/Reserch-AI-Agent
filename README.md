# Research-AI-Agent

A Multi-Agent Research Assistant powered by LangGraph and RAG (Retrieval-Augmented Generation). This system uses multiple AI agents working together to perform comprehensive research, critique findings, and synthesize accurate, well-grounded results.

## Features

- **Multi-Agent Architecture** — Specialized agents for research, critique, and synthesis
- **RAG-Powered Knowledge Base** — Retrieval-Augmented Generation for grounded, accurate responses
- **Anti-Hallucination Guardrails** — Strict grounding and verification to minimize false information
- **Premium Web Interface** — Modern, responsive UI with light/dark mode support

## Tech Stack

- **Backend**: Python, LangGraph, LangChain
- **Frontend**: Streamlit
- **LLM**: Google Gemini
- **Vector Store**: FAISS / ChromaDB

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/abhishekdev-ap/Reserch-AI-Agent.git
   cd Reserch-AI-Agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
├── agents/          # Multi-agent definitions and workflows
├── rag/             # RAG engine and knowledge base
├── api.py           # API endpoints
├── app.py           # Streamlit frontend
├── requirements.txt # Python dependencies
└── test_setup.py    # Setup verification tests
```

## License

MIT
