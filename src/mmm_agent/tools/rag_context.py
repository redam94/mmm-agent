"""
RAG Context Manager for MMM Agent

Provides context retrieval for each workflow phase using:
- Vector similarity search for relevant documents
- Workflow history for phase decisions
- Domain knowledge for MMM best practices
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field


# =============================================================================
# Document Models
# =============================================================================

class Document(BaseModel):
    """A document in the RAG store."""
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """A search result from RAG."""
    document: Document
    score: float
    highlights: list[str] = Field(default_factory=list)


# =============================================================================
# Simple In-Memory RAG (for POC)
# =============================================================================

class SimpleRAG:
    """
    Simple in-memory RAG for POC.
    
    For production, replace with ChromaDB or Neo4j vector store.
    """
    
    def __init__(self, persist_dir: str | None = None):
        self.documents: dict[str, Document] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        
        # Load default MMM domain knowledge
        self._load_domain_knowledge()
    
    def _load(self):
        """Load persisted documents."""
        if not self.persist_dir:
            return
        
        store_file = self.persist_dir / "documents.json"
        if store_file.exists():
            data = json.loads(store_file.read_text())
            for doc_data in data:
                doc = Document(**doc_data)
                self.documents[doc.id] = doc
            logger.info(f"Loaded {len(self.documents)} documents from store")
    
    def _save(self):
        """Persist documents."""
        if not self.persist_dir:
            return
        
        store_file = self.persist_dir / "documents.json"
        data = [doc.model_dump() for doc in self.documents.values()]
        store_file.write_text(json.dumps(data, indent=2))
    
    def _load_domain_knowledge(self):
        """Load default MMM domain knowledge."""
        domain_docs = [
            Document(
                id="mmm_adstock",
                content="""
                Adstock Transformations for Marketing Mix Models:
                
                Geometric Adstock: Models carryover effect with exponential decay.
                - Parameter: decay rate (alpha) between 0 and 1
                - Higher alpha = longer carryover
                - Typical values: 0.5-0.9 for TV, 0.1-0.5 for digital
                
                Weibull Adstock: More flexible, can model delayed peak effects.
                - Parameters: shape (k), scale (lambda)
                - Can model advertising that takes time to build awareness
                
                Best practice: Start with geometric adstock and l_max=8 weeks.
                """,
                metadata={"type": "domain_knowledge", "topic": "adstock"},
            ),
            Document(
                id="mmm_saturation",
                content="""
                Saturation Functions for Media Response:
                
                Hill Function (recommended): y = (x^n) / (K^n + x^n)
                - n: steepness of curve
                - K: half-saturation point (spend level at 50% effect)
                
                Logistic: y = 1 / (1 + exp(-k*(x-x0)))
                - More numerically stable
                - Good for standardized data
                
                Key insight: All media channels show diminishing returns.
                The saturation point helps identify optimal spend levels.
                """,
                metadata={"type": "domain_knowledge", "topic": "saturation"},
            ),
            Document(
                id="mmm_confounders",
                content="""
                Common Confounders in MMM:
                
                1. Seasonality: Affects both spend (advertisers spend more in Q4) 
                   and sales (holiday shopping). Use Fourier terms.
                
                2. Economic conditions: GDP, unemployment affect sales and 
                   marketing budgets simultaneously.
                
                3. Competitor activity: May cause spend increases and affect sales.
                
                4. Distribution/availability: Product availability drives both
                   marketing decisions and sales potential.
                
                5. Pricing/promotions: Often correlated with advertising campaigns.
                
                Best practice: Include controls for these factors to avoid
                biased media effect estimates.
                """,
                metadata={"type": "domain_knowledge", "topic": "confounders"},
            ),
            Document(
                id="mmm_diagnostics",
                content="""
                Model Diagnostics for Bayesian MMM:
                
                Convergence checks:
                - R-hat < 1.05 for all parameters
                - ESS (bulk and tail) > 400
                - No divergences
                
                Model fit:
                - Posterior predictive checks
                - MAPE < 15% typically acceptable
                - Check residual patterns
                
                If diagnostics fail:
                - Simplify model (fewer channels, simpler priors)
                - Increase tune/draws
                - Use non-centered parameterization
                - Consider time-varying parameters
                """,
                metadata={"type": "domain_knowledge", "topic": "diagnostics"},
            ),
            Document(
                id="mmm_interpretation",
                content="""
                Interpreting MMM Results:
                
                Channel Contributions:
                - Sum of fitted effects over time
                - Should sum to total explained variance
                - Compare to baseline (no marketing)
                
                ROI Calculation:
                - ROI = (Revenue Contribution - Spend) / Spend
                - Or: Incremental Revenue per $ Spent
                - Always report with uncertainty intervals
                
                Budget Optimization:
                - Use response curves to find optimal allocation
                - Consider constraints (min/max spend, contracts)
                - Scenario analysis for different budgets
                
                Cautions:
                - Short-term vs long-term effects
                - Brand building effects may not show in model
                - External validity across time periods
                """,
                metadata={"type": "domain_knowledge", "topic": "interpretation"},
            ),
        ]
        
        for doc in domain_docs:
            self.documents[doc.id] = doc
    
    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document to the store."""
        doc_id = doc_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.documents)}"
        
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )
        
        self.documents[doc_id] = doc
        self._save()
        
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for relevant documents.
        
        Uses simple keyword matching for POC.
        Replace with vector similarity for production.
        """
        query_terms = set(query.lower().split())
        
        results = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            # Simple keyword scoring
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            
            if score > 0:
                results.append(SearchResult(
                    document=doc,
                    score=score,
                ))
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:top_k]
    
    def get_phase_context(
        self,
        phase: Literal["planning", "eda", "modeling", "interpretation"],
        query: str,
        max_chars: int = 4000,
    ) -> str:
        """
        Get relevant context for a workflow phase.
        
        Combines domain knowledge with search results.
        """
        # Phase-specific topic filters
        phase_topics = {
            "planning": ["confounders", "variables"],
            "eda": ["adstock", "saturation", "data_quality"],
            "modeling": ["diagnostics", "priors"],
            "interpretation": ["interpretation", "roi"],
        }
        
        topics = phase_topics.get(phase, [])
        
        # Get domain knowledge for phase
        domain_docs = [
            doc for doc in self.documents.values()
            if doc.metadata.get("type") == "domain_knowledge"
            and any(t in doc.metadata.get("topic", "") for t in topics)
        ]
        
        # Search for query-relevant docs
        search_results = self.search(query, top_k=3)
        
        # Combine context
        context_parts = []
        
        # Add domain knowledge
        for doc in domain_docs[:2]:
            context_parts.append(f"## {doc.metadata.get('topic', 'Knowledge')}\n{doc.content[:800]}")
        
        # Add search results
        for result in search_results:
            if result.document.id not in [d.id for d in domain_docs]:
                context_parts.append(f"## Relevant Info\n{result.document.content[:600]}")
        
        context = "\n\n".join(context_parts)
        
        # Truncate if needed
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[truncated]"
        
        return context


# =============================================================================
# Workflow History Manager
# =============================================================================

class WorkflowHistory:
    """
    Manages workflow history for context retrieval.
    
    Stores decisions, artifacts, and outcomes from past workflows.
    """
    
    def __init__(self, persist_dir: str | None = None):
        self.workflows: dict[str, list[dict]] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()
    
    def _load(self):
        """Load persisted history."""
        if not self.persist_dir:
            return
        
        history_file = self.persist_dir / "workflow_history.json"
        if history_file.exists():
            self.workflows = json.loads(history_file.read_text())
            logger.info(f"Loaded {len(self.workflows)} workflow histories")
    
    def _save(self):
        """Persist history."""
        if not self.persist_dir:
            return
        
        history_file = self.persist_dir / "workflow_history.json"
        history_file.write_text(json.dumps(self.workflows, indent=2, default=str))
    
    def add_event(
        self,
        workflow_id: str,
        phase: str,
        event_type: str,
        content: dict[str, Any],
    ):
        """Add an event to workflow history."""
        if workflow_id not in self.workflows:
            self.workflows[workflow_id] = []
        
        self.workflows[workflow_id].append({
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "event_type": event_type,
            "content": content,
        })
        
        self._save()
    
    def get_workflow_context(
        self,
        workflow_id: str,
        current_phase: str,
    ) -> str:
        """Get context from current workflow's history."""
        if workflow_id not in self.workflows:
            return ""
        
        events = self.workflows[workflow_id]
        
        # Format recent events
        context_parts = [f"## Workflow History ({len(events)} events)"]
        
        for event in events[-10:]:  # Last 10 events
            context_parts.append(
                f"- [{event['phase']}] {event['event_type']}: "
                f"{json.dumps(event['content'])[:200]}"
            )
        
        return "\n".join(context_parts)
    
    def get_similar_workflows(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:
        """Find similar past workflows (simple matching for POC)."""
        # Simple implementation - would use embeddings in production
        similar = []
        
        for wf_id, events in self.workflows.items():
            # Check if any event content matches query terms
            for event in events:
                content_str = json.dumps(event.get("content", {})).lower()
                if any(term in content_str for term in query.lower().split()):
                    similar.append({
                        "workflow_id": wf_id,
                        "events": len(events),
                        "last_phase": events[-1]["phase"] if events else "unknown",
                    })
                    break
        
        return similar[:limit]


# =============================================================================
# Unified Context Manager
# =============================================================================

class ContextManager:
    """
    Unified context manager for MMM Agent.
    
    Combines RAG, workflow history, and web search results.
    """
    
    def __init__(
        self,
        persist_dir: str | None = None,
    ):
        base_dir = Path(persist_dir) if persist_dir else Path("./context_store")
        
        self.rag = SimpleRAG(str(base_dir / "rag"))
        self.history = WorkflowHistory(str(base_dir / "history"))
    
    def get_context(
        self,
        workflow_id: str,
        phase: str,
        query: str,
        web_results: list[dict] | None = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get comprehensive context for agent.
        
        Combines:
        - Domain knowledge from RAG
        - Workflow history
        - Web search results
        """
        # Allocate token budget
        budgets = {
            "domain": int(max_tokens * 0.4),
            "history": int(max_tokens * 0.3),
            "web": int(max_tokens * 0.3),
        }
        
        parts = []
        
        # Domain knowledge
        domain_ctx = self.rag.get_phase_context(phase, query, budgets["domain"])
        if domain_ctx:
            parts.append(f"# Domain Knowledge\n{domain_ctx}")
        
        # Workflow history
        history_ctx = self.history.get_workflow_context(workflow_id, phase)
        if history_ctx:
            parts.append(f"# Workflow History\n{history_ctx}")
        
        # Web search results
        if web_results:
            web_parts = ["# Web Search Results"]
            chars_used = 0
            for result in web_results:
                snippet = f"- {result.get('title', 'N/A')}: {result.get('snippet', '')[:200]}"
                if chars_used + len(snippet) < budgets["web"]:
                    web_parts.append(snippet)
                    chars_used += len(snippet)
            parts.append("\n".join(web_parts))
        
        return "\n\n".join(parts)
    
    def add_decision(
        self,
        workflow_id: str,
        phase: str,
        decision_type: str,
        content: dict[str, Any],
    ):
        """Record a decision in workflow history."""
        self.history.add_event(workflow_id, phase, decision_type, content)
    
    def add_knowledge(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add knowledge to RAG store."""
        return self.rag.add_document(content, metadata)


# =============================================================================
# Factory
# =============================================================================

_context_manager: ContextManager | None = None


def get_context_manager(persist_dir: str | None = None) -> ContextManager:
    """Get or create global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(persist_dir)
    return _context_manager
