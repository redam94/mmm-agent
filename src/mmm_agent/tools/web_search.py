"""
Web Search Tools for MMM Research Agent

Provides:
- DuckDuckGo search integration
- Optional crawl4ai for deep content extraction
- LLM-powered query optimization
- Result synthesis
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, Field

# Check for optional dependencies
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("DuckDuckGo search not available")

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.debug("crawl4ai not available - using basic search")


# =============================================================================
# Models
# =============================================================================

class SearchResult(BaseModel):
    """A web search result."""
    title: str
    url: str
    content: str  # Snippet or full content
    score: float = 0.5
    source: str = "duckduckgo"
    query_used: str | None = None
    crawled: bool = False


class SearchQueries(BaseModel):
    """Generated search queries."""
    queries: list[str] = Field(description="Search queries to execute")
    reasoning: str = Field(description="Why these queries were chosen")


class SearchSynthesis(BaseModel):
    """Synthesized search results."""
    summary: str = Field(description="Concise summary of findings")
    key_insights: list[str] = Field(description="Key insights from search")
    relevance_score: float = Field(description="Relevance to original query (0-1)")
    sources_used: int = 0


class WebSearchContext(BaseModel):
    """Complete search context."""
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    queries_used: list[str] = Field(default_factory=list)
    synthesis: SearchSynthesis | None = None
    error: str | None = None


# =============================================================================
# DuckDuckGo Search
# =============================================================================

async def search_duckduckgo(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
) -> list[SearchResult]:
    """
    Search DuckDuckGo for results.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        region: Search region
    
    Returns:
        List of search results
    """
    if not DDGS_AVAILABLE:
        logger.warning("DuckDuckGo not available")
        return []
    
    results = []
    
    try:
        # Run sync search in thread pool
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(
                    query,
                    region=region,
                    max_results=max_results,
                ))
        
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, _search)
        
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", r.get("link", "")),
                content=r.get("body", r.get("snippet", "")),
                source="duckduckgo",
                query_used=query,
            ))
        
        logger.debug(f"DuckDuckGo search '{query}': {len(results)} results")
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
    
    return results


# =============================================================================
# URL Crawling
# =============================================================================

async def crawl_url(
    url: str,
    query: str | None = None,
    max_chars: int = 5000,
) -> SearchResult | None:
    """
    Crawl a URL for full content using crawl4ai.
    
    Args:
        url: URL to crawl
        query: Original query for context
        max_chars: Maximum content characters
    
    Returns:
        SearchResult with crawled content or None
    """
    if not CRAWL4AI_AVAILABLE:
        return None
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=50,
                excluded_tags=["nav", "footer", "header", "aside"],
            )
            
            if result and result.markdown:
                content = result.markdown[:max_chars]
                return SearchResult(
                    title=result.metadata.get("title", url),
                    url=url,
                    content=content,
                    source="crawl4ai",
                    query_used=query,
                    crawled=True,
                )
    
    except Exception as e:
        logger.warning(f"Failed to crawl {url}: {e}")
    
    return None


# =============================================================================
# Search Agent with LLM
# =============================================================================

class SearchAgent:
    """
    Intelligent web search agent using LLM for query optimization.
    
    Features:
    - LLM-generated search queries
    - Iterative refinement based on results
    - Optional deep crawling of top results
    - Result synthesis
    """
    
    def __init__(
        self,
        llm=None,
        max_iterations: int = 2,
        results_per_query: int = 3,
        enable_crawl: bool = True,
        max_crawl_urls: int = 3,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.results_per_query = results_per_query
        self.enable_crawl = enable_crawl and CRAWL4AI_AVAILABLE
        self.max_crawl_urls = max_crawl_urls
    
    async def generate_queries(
        self,
        user_query: str,
        context: str = "",
        on_progress: Callable[[str], None] | None = None,
    ) -> SearchQueries:
        """Generate optimized search queries."""
        if not self.llm:
            # Fallback without LLM
            return SearchQueries(
                queries=[
                    f"{user_query} marketing mix model",
                    f"{user_query} MMM best practices",
                ],
                reasoning="Using default query patterns (LLM not available)"
            )
        
        if on_progress:
            on_progress("🧠 Generating search queries...")
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            structured_llm = self.llm.with_structured_output(SearchQueries)
            
            prompt = f"""Generate 2-4 optimized web search queries for MMM research.

User's Question: {user_query}

Context:
{context[:500] if context else "None provided"}

Guidelines:
- Create specific, targeted queries
- Include relevant technical terms
- One query should be broad, others more specific
- Focus on Marketing Mix Modeling, attribution, media effectiveness
"""
            
            result = structured_llm.invoke([
                SystemMessage(content="Generate effective web search queries for marketing analytics research."),
                HumanMessage(content=prompt)
            ])
            
            if on_progress:
                on_progress(f"📝 Generated {len(result.queries)} queries")
            
            return result
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return SearchQueries(
                queries=[user_query],
                reasoning=f"Fallback due to error: {str(e)[:100]}"
            )
    
    async def synthesize_results(
        self,
        query: str,
        results: list[SearchResult],
        on_progress: Callable[[str], None] | None = None,
    ) -> SearchSynthesis:
        """Synthesize search results into insights."""
        if not self.llm or not results:
            return SearchSynthesis(
                summary="No results to synthesize",
                key_insights=[],
                relevance_score=0.0,
                sources_used=0,
            )
        
        if on_progress:
            on_progress("✨ Synthesizing results...")
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            structured_llm = self.llm.with_structured_output(SearchSynthesis)
            
            results_text = "\n\n".join([
                f"[{r.title}]\n{r.content[:500]}..."
                for r in results[:10]
            ])
            
            prompt = f"""Synthesize these search results for the query: {query}

Results:
{results_text}

Provide:
1. A concise summary of key findings
2. 3-5 actionable insights for MMM analysis
3. Relevance score (0-1) for how well results answer the query
"""
            
            result = structured_llm.invoke([
                SystemMessage(content="Synthesize web research into actionable marketing analytics insights."),
                HumanMessage(content=prompt)
            ])
            result.sources_used = len(results)
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SearchSynthesis(
                summary=f"Error during synthesis: {str(e)[:100]}",
                key_insights=[],
                relevance_score=0.0,
                sources_used=len(results),
            )
    
    async def search(
        self,
        query: str,
        context: str = "",
        on_progress: Callable[[str], None] | None = None,
    ) -> WebSearchContext:
        """
        Perform intelligent web search.
        
        Args:
            query: User's research question
            context: Additional context
            on_progress: Progress callback
        
        Returns:
            WebSearchContext with results and synthesis
        """
        ctx = WebSearchContext(query=query)
        
        try:
            # Generate queries
            search_queries = await self.generate_queries(query, context, on_progress)
            ctx.queries_used = search_queries.queries
            
            # Execute searches
            if on_progress:
                on_progress(f"🔍 Executing {len(search_queries.queries)} searches...")
            
            for q in search_queries.queries:
                results = await search_duckduckgo(q, max_results=self.results_per_query)
                ctx.results.extend(results)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_results = []
            for r in ctx.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)
            ctx.results = unique_results
            
            if on_progress:
                on_progress(f"📄 Found {len(ctx.results)} unique results")
            
            # Crawl top results for more content
            if self.enable_crawl and ctx.results:
                if on_progress:
                    on_progress(f"🕷️ Crawling top {self.max_crawl_urls} results...")
                
                for i, result in enumerate(ctx.results[:self.max_crawl_urls]):
                    crawled = await crawl_url(result.url, query)
                    if crawled and len(crawled.content) > len(result.content):
                        ctx.results[i] = crawled
                        if on_progress:
                            on_progress(f"   ✅ Enriched: {result.title[:40]}")
            
            # Synthesize
            ctx.synthesis = await self.synthesize_results(query, ctx.results, on_progress)
            
            if on_progress:
                on_progress("✅ Search complete")
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            ctx.error = str(e)
        
        return ctx


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_search(
    query: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """Quick DuckDuckGo search without LLM."""
    return await search_duckduckgo(query, max_results)


async def research_topic(
    query: str,
    llm=None,
    context: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> WebSearchContext:
    """
    Research a topic with intelligent search.
    
    Args:
        query: Research question
        llm: Optional LLM for query optimization
        context: Additional context
        on_progress: Progress callback
    
    Returns:
        WebSearchContext with synthesized results
    """
    agent = SearchAgent(
        llm=llm,
        max_iterations=2,
        results_per_query=3,
        enable_crawl=True,
    )
    return await agent.search(query, context, on_progress)


def format_search_results(
    results: list[SearchResult],
    max_chars: int = 3000,
) -> str:
    """Format search results for LLM context."""
    if not results:
        return "No search results found."
    
    parts = ["=== Web Search Results ===\n"]
    total_len = len(parts[0])
    
    for i, r in enumerate(results, 1):
        entry = f"\n[{i}] {r.title}\nURL: {r.url}\n{r.content[:400]}...\n"
        
        if total_len + len(entry) > max_chars:
            parts.append(f"\n... and {len(results) - i + 1} more results")
            break
        
        parts.append(entry)
        total_len += len(entry)
    
    return "".join(parts)
