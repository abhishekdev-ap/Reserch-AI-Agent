"""
Quick test script to verify API keys and core functionality
"""
import os
from dotenv import load_dotenv
load_dotenv()

def test_gemini():
    print("Testing Gemini API...")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    response = llm.invoke([HumanMessage(content="Say 'Gemini OK' and nothing else.")])
    print(f"  ✅ Gemini: {response.content.strip()}")
    return True

def test_tavily():
    print("Testing Tavily API...")
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search("LangGraph multi-agent systems", max_results=2)
    print(f"  ✅ Tavily: Found {len(results.get('results', []))} results")
    return True

def test_embeddings():
    print("Testing Gemini Embeddings...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    vec = embeddings.embed_query("test query")
    print(f"  ✅ Embeddings: Vector dim = {len(vec)}")
    return True

def test_chromadb():
    print("Testing ChromaDB...")
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("test")
    col.add(documents=["test doc"], ids=["1"])
    result = col.query(query_texts=["test"], n_results=1)
    print(f"  ✅ ChromaDB: Found {len(result['documents'][0])} result(s)")
    return True

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Multi-Agent Research Assistant — Sanity Check")
    print("="*50 + "\n")
    
    tests = [test_gemini, test_tavily, test_embeddings, test_chromadb]
    all_pass = True
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            all_pass = False
    
    print("\n" + ("✅ All checks passed! Run: streamlit run app.py" if all_pass else "❌ Some checks failed."))
    print("="*50 + "\n")
