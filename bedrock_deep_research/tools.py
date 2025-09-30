import asyncio
from typing import List, Dict, Any
from tavily import TavilyClient

def tavily_web_search(queries: List[str]) -> List[Dict[str, Any]]:
    """
    A tool to perform web searches using the Tavily API.
    This function takes a list of search queries and returns the search results.
    """
    if not queries:
        return []
        
    tavily_client = TavilyClient()
    try:
        # Using the synchronous client for simplicity in AutoGen tool functions
        results = tavily_client.search(query=str(queries), max_results=5, include_raw_content=False)
        return results.get("results", [])
    except Exception as e:
        print(f"Error performing Tavily search: {e}")
        return []