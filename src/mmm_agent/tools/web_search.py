"""
Web Search Tool for MMM Agent

Provides web search capabilities for research during planning and interpretation.
Uses DuckDuckGo for simplicity (no API key required).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, Field


@dataclass
class SearchResult:
    """A web search result."""
    title: str
    url: str
    snippet: str
    source: str = ""


class WebSearcher:
    """
    Web search using DuckDuckGo.
    
    For production, consider using:
    - Tavily API (better for AI agents)
    - SerpAPI (Google results)
    - Bing Search API
    """
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
    
    async def search(
        self,
        query: str,
        max_results: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[SearchResult]:
        """
        Search the web.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            on_progress: Optional progress callback
        
        Returns:
            List of search results
        """
        max_results = max_results or self.max_results
        
        if on_progress:
            on_progress(f"Searching: {query}")
        
        try:
            from duckduckgo_search import DDGS
            
            results = []
            
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo",
                    ))
            
            if on_progress:
                on_progress(f"Found {len(results)} results")
            
            logger.info(f"Web search '{query}': {len(results)} results")
            return results
            
        except ImportError:
            logger.warning("duckduckgo_search not installed, using mock results")
            return self._mock_search(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_sync(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Synchronous search wrapper."""
        return asyncio.run(self.search(query, max_results))
    
    def _mock_search(self, query: str) -> list[SearchResult]:
        """Return mock results for testing."""
        return [
            SearchResult(
                title=f"Mock Result for: {query}",
                url="https://example.com",
                snippet="This is a mock search result for testing purposes.",
                source="mock",
            )
        ]


# =============================================================================
# Research Query Generator
# =============================================================================

class ResearchQueries(BaseModel):
    """Generated research queries."""
    queries: list[str] = Field(description="List of search queries")
    reasoning: str = Field(description="Why these queries")


def generate_research_queries(
    user_query: str,
    phase: str,
    context: str = "",
    llm = None,
) -> list[str]:
    """
    Generate research queries for a given user question.
    
    Uses LLM to create targeted search queries based on:
    - The user's question
    - The current workflow phase
    - Available context
    """
    if llm is None:
        # Default queries based on phase
        phase_queries = {
            "planning": [
                f"{user_query} marketing mix model",
                f"{user_query} media effectiveness measurement",
                f"MMM variable selection {user_query}",
            ],
            "eda": [
                f"{user_query} data quality checks",
                f"feature engineering {user_query}",
            ],
            "modeling": [
                f"bayesian MMM {user_query}",
                f"pymc marketing {user_query}",
            ],
            "interpretation": [
                f"marketing ROI interpretation {user_query}",
                f"media budget optimization {user_query}",
            ],
        }
        return phase_queries.get(phase, [user_query])[:3]
    
    # Use LLM to generate queries
    from langchain_core.messages import HumanMessage, SystemMessage
    
    prompt = f"""Generate 2-4 web search queries to research this marketing analytics question.

User Question: {user_query}

Workflow Phase: {phase}

Context:
{context[:500]}

Focus queries on:
- Planning: Variable selection, confounders, data requirements
- EDA: Data quality, feature engineering, transformations
- Modeling: Bayesian methods, model specification
- Interpretation: ROI calculation, optimization, recommendations

Return queries as a numbered list."""

    try:
        structured_llm = llm.with_structured_output(ResearchQueries)
        result = structured_llm.invoke([
            SystemMessage(content="Generate targeted web search queries for marketing analytics research."),
            HumanMessage(content=prompt),
        ])
        return result.queries
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        return [user_query]


# =============================================================================
# Research Tool
# =============================================================================

class ResearchTool:
    """
    Combines query generation and web search for research.
    """
    
    def __init__(
        self,
        searcher: WebSearcher | None = None,
        llm = None,
    ):
        self.searcher = searcher or WebSearcher()
        self.llm = llm
    
    async def research(
        self,
        question: str,
        phase: str,
        context: str = "",
        max_results_per_query: int = 5,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """
        Conduct research on a question.
        
        Args:
            question: Research question
            phase: Workflow phase
            context: Additional context
            max_results_per_query: Results per query
            on_progress: Progress callback
        
        Returns:
            List of search results as dicts
        """
        # Generate queries
        queries = generate_research_queries(question, phase, context, self.llm)
        
        if on_progress:
            on_progress(f"Generated {len(queries)} research queries")
        
        # Search each query
        all_results = []
        seen_urls = set()
        
        for query in queries:
            results = await self.searcher.search(query, max_results_per_query, on_progress)
            
            for r in results:
                if r.url not in seen_urls:
                    all_results.append({
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "query": query,
                    })
                    seen_urls.add(r.url)
        
        if on_progress:
            on_progress(f"Found {len(all_results)} unique results")
        
        return all_results
    
    def research_sync(
        self,
        question: str,
        phase: str,
        context: str = "",
    ) -> list[dict]:
        """Synchronous research wrapper."""
        return asyncio.run(self.research(question, phase, context))


# =============================================================================
# LangChain Tool Integration
# =============================================================================

def create_web_search_tool(searcher: WebSearcher | None = None):
    """Create a LangChain tool for web search."""
    from langchain_core.tools import tool
    
    searcher = searcher or WebSearcher()
    
    @tool
    async def web_search(query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web for information.
        
        Use this tool to find:
        - Industry best practices
        - Technical documentation
        - Research papers and articles
        - Case studies
        
        Args:
            query: Search query (be specific)
            max_results: Number of results (default 5)
        
        Returns:
            List of search results with title, url, and snippet
        """
        results = await searcher.search(query, max_results)
        return [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            }
            for r in results
        ]
    
    return web_search


def create_research_tool(llm = None):
    """Create a LangChain tool for comprehensive research."""
    from langchain_core.tools import tool
    
    research_tool = ResearchTool(llm=llm)
    
    @tool
    async def conduct_research(
        question: str,
        phase: str = "planning",
        context: str = "",
    ) -> list[dict]:
        """
        Conduct research on a marketing analytics question.
        
        This tool generates multiple search queries and aggregates results.
        
        Args:
            question: Research question
            phase: Workflow phase (planning, eda, modeling, interpretation)
            context: Additional context
        
        Returns:
            List of research results with deduplication
        """
        return await research_tool.research(question, phase, context)
    
    return conduct_research


# =============================================================================
# Factory
# =============================================================================

_searcher: WebSearcher | None = None


def get_web_searcher() -> WebSearcher:
    """Get or create global web searcher."""
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher
