"""
Multi-Agent Research Assistant
Core agent definitions using LangGraph + Gemini/Ollama + Tavily/DuckDuckGo
Supports both cloud (Gemini API) and local (Ollama) modes.
"""
import os
import sys
import time
import datetime
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_llm, do_web_search, get_mode_info


def _retry_on_rate_limit(func, *args, max_retries=3, **kwargs):
    """Retry a function call with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if any(x in error_str for x in ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE', 'high demand']):
                wait_time = (2 ** attempt) * 15  # 15s, 30s, 60s
                print(f"⏳ API Busy/Rate limit hit ({error_str[:50]}...). Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    return func(*args, **kwargs)  # Final attempt


def _safe_content(msg) -> str:
    """Safely extract text content from a message, handling list-type content blocks."""
    content = getattr(msg, 'content', None) or msg
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get('text', '') or str(block))
            else:
                parts.append(str(block))
        return '\n'.join(parts)
    return str(content) if content is not None else ''

# ─── LLM & Search (auto-detects cloud/local from config dynamically) ───────────

class DynamicLLMProxy:
    def __init__(self, temperature=0.1, bind_tools_list=None):
        self.temperature = temperature
        self.bind_tools_list = bind_tools_list

    def _get_underlying_llm(self):
        llm_instance = get_llm(temperature=self.temperature)
        if self.bind_tools_list:
            llm_instance = llm_instance.bind_tools(self.bind_tools_list)
        return llm_instance

    def invoke(self, *args, **kwargs):
        return self._get_underlying_llm().invoke(*args, **kwargs)

    def bind_tools(self, tools_list):
        return DynamicLLMProxy(temperature=self.temperature, bind_tools_list=tools_list)

llm = DynamicLLMProxy(temperature=0.1)

# ─── Tools ─────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for recent, factual information on any topic."""
    results = do_web_search(query, max_results=7)
    formatted = []
    for r in results:
        formatted.append(
            f"**Source**: {r.get('url', 'N/A')}\n"
            f"**Title**: {r.get('title', 'N/A')}\n"
            f"**Content**: {r.get('content', 'N/A')}\n"
            f"**Published Date**: {r.get('published_date', 'N/A')}\n"
        )
    return "\n---\n".join(formatted) if formatted else "No results found."


@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into key bullet points."""
    response = llm.invoke([
        SystemMessage(content="""You are an expert summarizer. Extract the most important facts and insights as concise bullet points.
CRITICAL RULES:
- ONLY include facts that are explicitly stated in the provided text.
- Do NOT add any information, statistics, dates, or claims that are not directly present in the source text.
- If the text is vague or incomplete, reflect that in your summary rather than filling in gaps.
- Preserve source attributions (URLs, titles) when summarizing."""),
        HumanMessage(content=f"Summarize this:\n\n{text}")
    ])
    return _safe_content(response)


@tool
def analyze_and_critique(text: str) -> str:
    """Critically analyze a piece of text, identifying strengths, weaknesses, gaps, and potential biases."""
    response = llm.invoke([
        SystemMessage(content="""You are a critical research analyst. Analyze the provided text for accuracy, logical gaps, potential biases, and missing perspectives.
CRITICAL RULES:
- Flag any claims that lack source citations as UNVERIFIED.
- Identify any statements that appear fabricated or unsupported by the provided evidence.
- Check for logical inconsistencies between different parts of the text.
- Do NOT introduce new factual claims in your critique — only assess what is provided."""),
        HumanMessage(content=f"Critically analyze:\n\n{text}")
    ])
    return _safe_content(response)


@tool
def generate_report(topic: str, research_data: str) -> str:
    """Generate a comprehensive, well-structured research report from gathered data."""
    response = llm.invoke([
        SystemMessage(content="""You are a professional research report writer.
        Create a comprehensive, well-structured report with:
        - Executive Summary
        - Key Findings (with citations where possible)
        - Detailed Analysis
        - Conclusions & Recommendations
        Use markdown formatting.

        CRITICAL ACCURACY RULES:
        - ONLY include facts, statistics, dates, and claims that are EXPLICITLY present in the research data.
        - ALWAYS cite the source URL for every factual claim.
        - NEVER fabricate statistics, percentages, dates, quotes, or study results.
        - If data is insufficient for a section, state "Insufficient data available" rather than inventing content.
        - Clearly distinguish between facts (from sources) and your analytical observations.
        - Use phrases like "According to [source]..." to attribute information."""),
        HumanMessage(content=f"Topic: {topic}\n\nResearch Data:\n{research_data}")
    ])
    return _safe_content(response)


# ─── Agent State ───────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_results: List[str]
    summary: str
    critique: str
    final_report: str
    iteration: int
    max_iterations: int
    agent_scratchpad: str
    status: str


# ─── Agent Nodes ───────────────────────────────────────────────────────────────

tools = [web_search, summarize_text, analyze_and_critique, generate_report]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def research_agent(state: ResearchState) -> ResearchState:
    """Primary research agent: decides what to search and gather."""
    current_date = datetime.date.today().strftime("%B %d, %Y")
    system_prompt = f"""You are an expert research agent. 
    
TEMPORAL GROUNDING:
- TODAY'S DATE IS {current_date}. We are currently in the year 2026.
- Any search results containing financial, market, or news data from 2025 or 2026 are current, valid, and real-time facts (NOT future-dated or fabricated).

YOUR JOB:
1. Search for comprehensive, highly detailed, and exhaustive information on the given topic.
2. Gather multiple perspectives, authoritative sources, and precise numbers.
3. Call web_search multiple times for different aspects of the topic.
4. **Precise Stock / Ticker Searching**: If the topic asks for a stock price, financial metric, market update, or current valuation (e.g. NSDL share price), you must formulate precise search queries containing terms like 'share price BSE', 'stock price ticker', 'valuation Screener.in', 'closing price 2026', or the exact exchange code to directly fetch official figures.
5. **Cross-Check Sources**: Diversify your search queries to gather findings from multiple distinct, highly reliable financial and news portals (e.g., Economic Times, Moneycontrol, Bombay Stock Exchange, Screener.in, Yahoo Finance, Reuters, Bloomberg) to ensure high-accuracy cross-checking of values and stock rates.
6. **Timeline Accuracy**: Ensure you check for the most recent trading days (such as Friday, May 29, 2026 or today) to fetch the actual latest closing rates rather than months-old values.

Always be extremely thorough and call at least 2-3 searches for different angles of the topic.
After searching, summarize your findings using the summarize_text tool.

CRITICAL ACCURACY RULES:
- Your goal is to find FACTUAL, VERIFIABLE information from reliable sources.
- When searching, use specific and precise queries to get authoritative results.
- Do NOT make up or assume any facts — only report what the search results explicitly state.
- Always preserve source URLs so they can be cited in the final report."""

    messages = state["messages"] + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research this topic thoroughly: {state['query']}")
    ]

    response = llm_with_tools.invoke(messages)
    return {**state, "messages": [response], "status": "researching"}


def critique_agent(state: ResearchState) -> ResearchState:
    """Critique agent: reviews and challenges the research findings."""
    current_date = datetime.date.today().strftime("%B %d, %Y")
    system_prompt = f"""You are a critical analysis agent specializing in FACT-CHECKING. 

TEMPORAL GROUNDING:
- TODAY'S DATE IS {current_date}. We are currently in the year 2026.
- Information dated up to May 2026 is current, valid, and real-time. Do NOT flag 2025 or 2026 dates as "future-dated", "unreliable", or "fabricated". They are present/past facts.

Review the research gathered so far and:
1. Identify any claims that lack proper source citations — flag them as UNVERIFIED.
2. Check for potential biases or one-sided perspectives.
3. **Cross-Check Factual Data**: Specifically cross-check all numbers, valuations, and stock rates across the different search results. If source A says ₹808 and source B says ₹814, compare their timestamps and document the discrepancy or state the most recent one.
4. **Spot Conflicts & Gaps**: Identify if the research has relied on outdated data (e.g., share price from the 2024 IPO prospectus instead of today's 2026 trading value).
5. Identify any statistics, dates, or specific claims that seem fabricated, mismatched, or unsupported by the raw search results.
6. Suggest what additional research might be needed to fill factual gaps.
7. Use analyze_and_critique on key findings.

CRITICAL: If you find any claims that appear to be hallucinated (not backed by any search result), 
explicitly flag them for removal. Do NOT introduce new unsourced claims yourself."""

    # Use more messages for better context (up to last 10 instead of 3)
    last_messages = state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
    context = "\n".join([_safe_content(m) for m in last_messages if hasattr(m, 'content')])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Fact-check and critique this research on '{state['query']}':\n\n{context}")
    ]
    response = llm_with_tools.invoke(messages)
    return {**state, "messages": [response], "status": "critiquing", "iteration": state.get("iteration", 0) + 1}


def synthesis_agent(state: ResearchState) -> ResearchState:
    """Synthesis agent: compiles everything into a final report using plain LLM (no tools)."""
    current_date = datetime.date.today().strftime("%B %d, %Y")
    system_prompt = f"""You are a professional research report writer.

TEMPORAL GROUNDING:
- TODAY'S DATE IS {current_date}. We are currently in the year 2026.
- Information, stock prices, and events dated 2025 or 2026 are current, factual, and accurate (NOT future-dated or fabricated).

Compile all the research, summaries, and critiques into a comprehensive, highly detailed, and exhaustive step-by-step report.

Your report MUST be structured in the following exact step-by-step format with these clear, numbered headers:
## 1. Introduction and Background
- Provide a clear overview of the topic and the necessary background context. Write at least two detailed paragraphs.

## 2. Data Collection and Sources
- Transparently list all retrieved sources and verified websites, explaining what datasets were gathered. Cite the source URLs and timestamps.

## 3. Key Findings
- Outline the main discoveries, absolute values, precise timelines, and latest live facts from the web. Write in clear, structured lists and paragraphs.

## 4. Detailed Analysis
- Provide a highly comprehensive, multi-paragraph, analytical breakdown with context, examples, comparisons, reasoning, and market implications. Do NOT write a brief summary; show deep analytical reasoning.

## 5. Supporting Evidence
- Detail the empirical data, statistics, cross-referenced findings, and supporting facts.

## 6. Risks and Limitations
- Outline any gaps in data, risks, market volatility, and limitations of the current study.
- Under dedicated bulleted lists or subheadings, you MUST clearly and explicitly categorize and distinguish between:
  - **Confirmed Facts**: Authoritative details verified directly from reliable sources (e.g. stock exchange closing prices, official listings).
  - **Estimates & Projections**: Projections, growth forecasts, consensus figures, or forward-looking statements.
  - **Opinions & Sentiments**: Analyst consensus, subjective sentiments, and qualitative market views.
- Detail any discrepancies or conflicting values between retrieved sources.

## 7. Conclusion and Recommendations
- Summarize the final outcomes and formulate action-oriented, professional, and strategic recommendations.

CRITICAL DRAFTING & ACCURACY RULES:
- **Exhaustive Detail**: Do NOT write brief summaries. Every single section must be highly detailed, comprehensive, and rich in context, using multiple paragraphs.
- **Fact vs Opinion**: Clearly and explicitly label facts, estimates, and opinions inside Section 6 (Risks and Limitations).
- **Real-Time Cross-Checking**: Cross-check information from all retrieved search results. If different sources report different share prices, compare them and mention any discrepancies or date differences.
- **Financial/Stock Queries**: Always prioritize and cite the most recent available information. Present historical listings and IPO statistics alongside today's current valuation, ensuring high accuracy.
- **Source Grounds**: ONLY include facts, statistics, dates, numbers, and claims that are EXPLICITLY present in the research data below.
- **Attributions**: EVERY factual claim MUST be attributed to a specific source URL from the research data.
- **No Hallucinations**: NEVER fabricate or invent statistics, percentages, dates, quotes, or names.
- Use markdown formatting with headers, bullet points, and bold text."""

    # Collect all tool results and text content from the pipeline
    all_content_parts = []
    for m in state["messages"]:
        text = _safe_content(m)
        if text and len(text) > 30 and text not in ['', 'None']:
            all_content_parts.append(text)

    all_content = "\n\n---\n\n".join(all_content_parts)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic: {state['query']}\n\nResearch gathered (USE ONLY THIS DATA — do not add external knowledge):\n\n{all_content[:30000]}")
    ]
    # Use plain LLM (no tools) so it outputs text directly
    response = llm.invoke(messages)
    report_text = _safe_content(response)
    return {**state, "messages": [response], "status": "synthesizing", "final_report": report_text}


def should_continue_research(state: ResearchState) -> str:
    """Router: decides whether to keep researching or move to synthesis."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)
    last_message = state["messages"][-1] if state["messages"] else None

    # Check if tools were called
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    if iteration >= max_iter:
        return "synthesize"
    return "critique"


def should_use_tools_or_continue(state: ResearchState) -> str:
    """After tool execution, routes back to the appropriate agent."""
    status = state.get("status", "researching")
    if status == "researching":
        return "critique"
    elif status == "critiquing":
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 2)
        if iteration >= max_iter:
            return "synthesize"
        return "research"
    return "synthesize"


# ─── Graph Construction ────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("research", research_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("critique", critique_agent)
    graph.add_node("synthesize", synthesis_agent)
    graph.add_node("synthesis_tools", tool_node)

    graph.set_entry_point("research")

    graph.add_conditional_edges(
        "research",
        should_continue_research,
        {"tools": "tools", "critique": "critique", "synthesize": "synthesize"}
    )

    graph.add_conditional_edges(
        "tools",
        should_use_tools_or_continue,
        {"research": "research", "critique": "critique", "synthesize": "synthesize"}
    )

    graph.add_conditional_edges(
        "critique",
        should_continue_research,
        {"tools": "synthesis_tools", "critique": "critique", "synthesize": "synthesize"}
    )

    graph.add_edge("synthesis_tools", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ─── Public API ────────────────────────────────────────────────────────────────

def run_research(query: str, max_iterations: int = 2) -> dict:
    """Run the full multi-agent research pipeline on a query."""
    graph = build_research_graph()

    initial_state: ResearchState = {
        "messages": [],
        "query": query,
        "search_results": [],
        "summary": "",
        "critique": "",
        "final_report": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "agent_scratchpad": "",
        "status": "researching",
    }

    final_state = graph.invoke(initial_state)
    return final_state
