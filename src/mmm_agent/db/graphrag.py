"""
GraphRAG System for MMM Workflows

Combines:
- Neo4j for relationship-aware retrieval
- ChromaDB for vector similarity search
- PostgreSQL for session state persistence

Provides context-aware retrieval for each workflow phase.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available - vector search disabled")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("SentenceTransformers not available")

from ..config import Settings, get_settings
from .neo4j_client import Neo4jClient, get_neo4j_client


# =============================================================================
# Models
# =============================================================================

class Document(BaseModel):
    """A document in the vector store."""
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_type: Literal[
        "research", "decision", "analysis", "domain_knowledge", "user_feedback"
    ] = "research"


class SearchResult(BaseModel):
    """A search result from GraphRAG."""
    document: Document
    score: float
    source: Literal["vector", "graph", "hybrid"] = "vector"


class PhaseContext(BaseModel):
    """Context retrieved for a workflow phase."""
    phase: str
    graph_context: list[dict] = Field(default_factory=list)
    vector_results: list[SearchResult] = Field(default_factory=list)
    combined_summary: str = ""
    token_budget_used: int = 0


# =============================================================================
# GraphRAG Manager
# =============================================================================

class GraphRAGManager:
    """
    GraphRAG system combining graph and vector retrieval.
    
    Architecture:
    - Neo4j stores relationships, causal structures, and lineage
    - ChromaDB stores document embeddings for similarity search
    - Context budget management prevents LLM context overflow
    """
    
    # Context budget allocation by content type
    CONTEXT_BUDGET = {
        "graph_relationships": 0.4,
        "vector_chunks": 0.3,
        "phase_history": 0.2,
        "system": 0.1,
    }
    
    def __init__(
        self,
        settings: Settings | None = None,
        neo4j_client: Neo4jClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.neo4j = neo4j_client or get_neo4j_client(self.settings)
        
        # Initialize ChromaDB
        self._init_chroma()
        
        # Initialize embedding model
        self._init_embeddings()
    
    def _init_chroma(self):
        """Initialize ChromaDB vector store."""
        if not CHROMA_AVAILABLE:
            self.chroma_client = None
            self.collection = None
            return
        
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="mmm_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized: {self.collection.count()} documents")
    
    def _init_embeddings(self):
        """Initialize embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            self.embedder = None
            return
        
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        logger.info(f"Embedding model loaded: {self.settings.embedding_model}")
    
    def _generate_id(self, content: str, prefix: str = "doc") -> str:
        """Generate unique document ID."""
        hash_input = f"{prefix}:{content[:200]}:{datetime.now().isoformat()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"
    
    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if self.embedder is None:
            return []
        return self.embedder.encode(text).tolist()
    
    # =========================================================================
    # Document Storage
    # =========================================================================
    
    def add_document(
        self,
        content: str,
        doc_type: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            content: Document content
            doc_type: Type of document (research, decision, etc.)
            metadata: Additional metadata
            doc_id: Optional specific ID
        
        Returns:
            Document ID
        """
        if self.collection is None:
            logger.warning("ChromaDB not available")
            return ""
        
        doc_id = doc_id or self._generate_id(content, doc_type)
        metadata = metadata or {}
        metadata["doc_type"] = doc_type
        metadata["created_at"] = datetime.now().isoformat()
        
        # Generate embedding
        embedding = self._embed(content)
        
        if embedding:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            )
        else:
            # Fallback without embeddings
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
            )
        
        logger.debug(f"Added document: {doc_id} ({doc_type})")
        return doc_id
    
    def add_research_result(
        self,
        analysis_id: str,
        query: str,
        summary: str,
        sources: list[dict],
        insights: list[str],
    ) -> str:
        """
        Store web research results in both graph and vector store.
        
        Args:
            analysis_id: Parent analysis ID
            query: Research query
            summary: Synthesized summary
            sources: List of source URLs and snippets
            insights: Key insights extracted
        
        Returns:
            Research result ID
        """
        research_id = self._generate_id(query, "research")
        
        # Store in Neo4j for relationship tracking
        self.neo4j.store_research_result(
            research_id=research_id,
            analysis_id=analysis_id,
            query=query,
            summary=summary,
            sources=sources,
            insights=insights,
        )
        
        # Store full content in vector store
        full_content = f"""
Query: {query}

Summary: {summary}

Key Insights:
{chr(10).join(f"- {i}" for i in insights)}

Sources:
{chr(10).join(f"- {s.get('title', 'Unknown')}: {s.get('url', '')}" for s in sources)}
"""
        
        self.add_document(
            content=full_content,
            doc_type="research",
            metadata={
                "analysis_id": analysis_id,
                "query": query,
                "source_count": len(sources),
            },
            doc_id=research_id,
        )
        
        return research_id
    
    def add_decision(
        self,
        analysis_id: str,
        phase: str,
        decision_type: str,
        content: dict,
        rationale: str,
    ) -> str:
        """
        Store a workflow decision in both graph and vector store.
        """
        from .neo4j_client import Decision
        
        decision_id = self._generate_id(rationale, f"decision_{phase}")
        
        # Store in Neo4j for lineage
        decision = Decision(
            id=decision_id,
            analysis_id=analysis_id,
            phase=phase,
            decision_type=decision_type,
            content=content,
            rationale=rationale,
        )
        self.neo4j.create_decision(decision)
        
        # Store in vector store for retrieval
        decision_text = f"""
Phase: {phase}
Type: {decision_type}

Decision:
{json.dumps(content, indent=2)}

Rationale: {rationale}
"""
        
        self.add_document(
            content=decision_text,
            doc_type="decision",
            metadata={
                "analysis_id": analysis_id,
                "phase": phase,
                "decision_type": decision_type,
            },
            doc_id=decision_id,
        )
        
        return decision_id
    
    def add_user_feedback(
        self,
        analysis_id: str,
        feedback: str,
        context: str,
        rating: int | None = None,
    ) -> str:
        """Store user feedback for learning."""
        feedback_id = self._generate_id(feedback, "feedback")
        
        feedback_text = f"""
User Feedback for Analysis: {analysis_id}

Feedback: {feedback}

Context: {context}
{f"Rating: {rating}/5" if rating else ""}
"""
        
        self.add_document(
            content=feedback_text,
            doc_type="user_feedback",
            metadata={
                "analysis_id": analysis_id,
                "rating": rating,
            },
            doc_id=feedback_id,
        )
        
        return feedback_id
    
    # =========================================================================
    # Retrieval
    # =========================================================================
    
    def search_vectors(
        self,
        query: str,
        n_results: int = 5,
        doc_types: list[str] | None = None,
        analysis_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search vector store for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results
            doc_types: Filter by document types
            analysis_id: Filter by analysis
        
        Returns:
            List of search results
        """
        if self.collection is None:
            return []
        
        # Build filter
        where_filter = {}
        if doc_types:
            where_filter["doc_type"] = {"$in": doc_types}
        if analysis_id:
            where_filter["analysis_id"] = analysis_id
        
        # Generate query embedding
        query_embedding = self._embed(query)
        
        try:
            if query_embedding:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter if where_filter else None,
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter if where_filter else None,
                )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.5
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                doc_id = results["ids"][0][i] if results["ids"] else f"doc_{i}"
                
                search_results.append(SearchResult(
                    document=Document(
                        id=doc_id,
                        content=doc,
                        metadata=metadata,
                        doc_type=metadata.get("doc_type", "research"),
                    ),
                    score=1.0 - distance,  # Convert distance to similarity
                    source="vector",
                ))
        
        return search_results
    
    def get_phase_context(
        self,
        phase: str,
        state: dict,
        max_tokens: int = 4000,
    ) -> PhaseContext:
        """
        Get comprehensive context for a workflow phase.
        
        Combines graph and vector retrieval with context budget management.
        
        Args:
            phase: Workflow phase (planning, eda, modeling, interpretation)
            state: Current workflow state
            max_tokens: Maximum context tokens
        
        Returns:
            PhaseContext with combined retrieval results
        """
        context = PhaseContext(phase=phase)
        
        # Calculate token budgets
        graph_budget = int(max_tokens * self.CONTEXT_BUDGET["graph_relationships"])
        vector_budget = int(max_tokens * self.CONTEXT_BUDGET["vector_chunks"])
        
        # Phase-specific retrieval
        if phase == "planning":
            context = self._get_planning_context(state, graph_budget, vector_budget)
        elif phase == "eda":
            context = self._get_eda_context(state, graph_budget, vector_budget)
        elif phase == "modeling":
            context = self._get_modeling_context(state, graph_budget, vector_budget)
        elif phase == "interpretation":
            context = self._get_interpretation_context(state, graph_budget, vector_budget)
        else:
            # Generic retrieval
            query = state.get("user_query", state.get("query", ""))
            if query:
                context.vector_results = self.search_vectors(query, n_results=5)
        
        # Generate combined summary
        context.combined_summary = self._summarize_context(context)
        context.token_budget_used = len(context.combined_summary) // 4  # Rough estimate
        
        return context
    
    def _get_planning_context(
        self,
        state: dict,
        graph_budget: int,
        vector_budget: int,
    ) -> PhaseContext:
        """Get context for planning phase."""
        context = PhaseContext(phase="planning")
        query = state.get("user_query", state.get("business_context", ""))
        
        # Graph: Similar successful analyses
        patterns = self.neo4j.get_successful_mmm_patterns(
            query=query,
            max_mape=0.15,
            limit=5,
        )
        context.graph_context = patterns
        
        # Vector: Related research and decisions
        context.vector_results = self.search_vectors(
            query=query,
            n_results=5,
            doc_types=["research", "decision", "domain_knowledge"],
        )
        
        return context
    
    def _get_eda_context(
        self,
        state: dict,
        graph_budget: int,
        vector_budget: int,
    ) -> PhaseContext:
        """Get context for EDA phase."""
        context = PhaseContext(phase="eda")
        
        # Get channels from planning
        channels = state.get("media_channels", [])
        
        if channels:
            # Graph: Feature engineering patterns
            patterns = self.neo4j.get_feature_engineering_patterns(
                channel_ids=channels,
                limit=3,
            )
            context.graph_context = patterns
        
        # Vector: EDA best practices
        context.vector_results = self.search_vectors(
            query="EDA feature engineering adstock saturation",
            n_results=3,
            doc_types=["domain_knowledge", "decision"],
        )
        
        return context
    
    def _get_modeling_context(
        self,
        state: dict,
        graph_budget: int,
        vector_budget: int,
    ) -> PhaseContext:
        """Get context for modeling phase."""
        context = PhaseContext(phase="modeling")
        
        channels = state.get("media_channels", [])
        target = state.get("target_variable", "revenue")
        
        if channels:
            # Graph: Model configs and causal structure
            model_context = self.neo4j.get_modeling_context(
                channel_ids=channels,
                target_id=target,
            )
            context.graph_context = [model_context]
        
        # Vector: Modeling decisions
        context.vector_results = self.search_vectors(
            query=f"Bayesian MMM model configuration {' '.join(channels)}",
            n_results=3,
            doc_types=["decision", "domain_knowledge"],
        )
        
        return context
    
    def _get_interpretation_context(
        self,
        state: dict,
        graph_budget: int,
        vector_budget: int,
    ) -> PhaseContext:
        """Get context for interpretation phase."""
        context = PhaseContext(phase="interpretation")
        
        analysis_id = state.get("analysis_id")
        
        if analysis_id:
            # Get all decisions from this analysis
            decisions = self.neo4j.get_analysis_decisions(analysis_id)
            context.graph_context = [
                {"phase": d.phase, "type": d.decision_type, "rationale": d.rationale}
                for d in decisions
            ]
        
        # Vector: Interpretation guidance
        context.vector_results = self.search_vectors(
            query="MMM ROI interpretation budget optimization recommendations",
            n_results=3,
            doc_types=["domain_knowledge", "research"],
        )
        
        return context
    
    def _summarize_context(self, context: PhaseContext) -> str:
        """Generate a summary string from context."""
        parts = []
        
        if context.graph_context:
            parts.append("=== Related Patterns ===")
            for item in context.graph_context[:3]:
                if isinstance(item, dict):
                    parts.append(json.dumps(item, indent=2)[:500])
        
        if context.vector_results:
            parts.append("\n=== Relevant Knowledge ===")
            for result in context.vector_results[:3]:
                parts.append(f"[{result.document.doc_type}] {result.document.content[:300]}...")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Domain Knowledge
    # =========================================================================
    
    def load_domain_knowledge(self):
        """Load default MMM domain knowledge into vector store."""
        domain_docs = [
            {
                "content": """
Adstock Transformations for Marketing Mix Models:

Geometric Adstock: Models carryover effect with exponential decay.
- Formula: adstocked[t] = spend[t] + alpha * adstocked[t-1]
- Alpha parameter (0-1) controls decay rate
- Higher alpha = longer carryover effect
- Typical values: TV (0.5-0.8), Digital (0.2-0.5)

Weibull Adstock: More flexible with shape and scale parameters.
- Can model delayed peak effects
- Better for channels with non-immediate response
""",
                "doc_type": "domain_knowledge",
                "metadata": {"topic": "adstock", "domain": "feature_engineering"},
            },
            {
                "content": """
Saturation Functions for Media Effects:

Hill Function (Most Common):
- Formula: response = beta * x^alpha / (lambda^alpha + x^alpha)
- Beta: Maximum effect (asymptote)
- Alpha: Steepness of curve
- Lambda: Half-saturation point

Logistic Saturation:
- S-shaped curve with inflection point
- Good for channels with threshold effects

Key Insights:
- Channels typically saturate at different rates
- Digital tends to saturate faster than TV
- Understanding saturation crucial for budget optimization
""",
                "doc_type": "domain_knowledge",
                "metadata": {"topic": "saturation", "domain": "feature_engineering"},
            },
            {
                "content": """
MMM Best Practices for Variable Selection:

Target Variable (KPI):
- Use revenue or sales value, not units
- Weekly aggregation recommended
- At least 2 years of data preferred

Media Channels:
- Include all significant paid media
- Group similar channels if spend is low
- Consider organic channels (search, social)

Control Variables:
- Price/promotions (critical confounder)
- Distribution/availability
- Seasonality (holidays, events)
- Economic indicators
- Competitive activity
- Weather (if relevant)
""",
                "doc_type": "domain_knowledge",
                "metadata": {"topic": "variable_selection", "domain": "planning"},
            },
            {
                "content": """
Model Diagnostics for Bayesian MMM:

Convergence Checks:
- R-hat < 1.05 for all parameters
- ESS bulk > 400 per chain
- ESS tail > 400 per chain
- No divergences

Fit Assessment:
- MAPE < 15% on holdout
- R² > 0.80 for good fit
- Residuals should be random

If Convergence Issues:
1. Increase tune/draws
2. Simplify model (fewer channels)
3. Add stronger priors
4. Check data quality
5. Try different sampler (nutpie, numpyro)
""",
                "doc_type": "domain_knowledge",
                "metadata": {"topic": "diagnostics", "domain": "modeling"},
            },
        ]
        
        for doc in domain_docs:
            self.add_document(
                content=doc["content"],
                doc_type=doc["doc_type"],
                metadata=doc["metadata"],
            )
        
        logger.info(f"Loaded {len(domain_docs)} domain knowledge documents")


# =============================================================================
# Singleton Factory
# =============================================================================

_graphrag_manager: GraphRAGManager | None = None


def get_graphrag_manager(settings: Settings | None = None) -> GraphRAGManager:
    """Get or create singleton GraphRAG manager."""
    global _graphrag_manager
    if _graphrag_manager is None:
        _graphrag_manager = GraphRAGManager(settings)
    return _graphrag_manager
