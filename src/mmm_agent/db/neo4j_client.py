"""
Neo4j Graph Database Client for MMM GraphRAG

Handles:
- Knowledge graph storage and retrieval
- Causal relationship management
- Model artifact lineage
- Vector similarity search combined with graph traversal
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from contextlib import contextmanager

from loguru import logger
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel, Field

from ..config import Settings, get_settings


# =============================================================================
# Schema Models
# =============================================================================

class Variable(BaseModel):
    """A variable in the knowledge graph."""
    id: str
    name: str
    type: Literal["continuous", "categorical", "binary"] = "continuous"
    domain: Literal["media", "control", "kpi", "auxiliary"] = "media"
    unit: str | None = None
    typical_lag: list[int] = Field(default_factory=list)
    saturation_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalRelationship(BaseModel):
    """A causal relationship between variables."""
    source_id: str
    target_id: str
    mechanism: str
    lag_periods: list[int] = Field(default_factory=list)
    confidence: float = 0.5
    evidence_source: str | None = None
    adjustment_set: list[str] = Field(default_factory=list)


class Analysis(BaseModel):
    """An MMM analysis session."""
    id: str
    query: str
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: str | None = None
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """A decision made during workflow."""
    id: str
    analysis_id: str
    phase: Literal["planning", "eda", "modeling", "interpretation"]
    decision_type: str
    content: dict[str, Any]
    rationale: str
    created_at: datetime = Field(default_factory=datetime.now)


class ModelArtifact(BaseModel):
    """A trained model artifact."""
    id: str
    analysis_id: str
    framework: str = "pymc_marketing"
    artifact_path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    training_period: tuple[str, str] | None = None
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Neo4j Client
# =============================================================================

class Neo4jClient:
    """
    Neo4j client for MMM knowledge graph operations.
    
    Provides:
    - Variable and relationship management
    - Analysis session tracking
    - Decision and artifact lineage
    - GraphRAG retrieval patterns
    """
    
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver: Driver | None = None
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
        return self._driver
    
    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        """Get a session context manager."""
        session = self.driver.session(database=self.settings.neo4j_database)
        try:
            yield session
        finally:
            session.close()
    
    # =========================================================================
    # Schema Initialization
    # =========================================================================
    
    def init_schema(self):
        """Initialize the graph schema with constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT variable_id IF NOT EXISTS FOR (v:Variable) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT analysis_id IF NOT EXISTS FOR (a:Analysis) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT research_id IF NOT EXISTS FOR (r:ResearchResult) REQUIRE r.id IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX variable_domain IF NOT EXISTS FOR (v:Variable) ON (v.domain)",
            "CREATE INDEX analysis_status IF NOT EXISTS FOR (a:Analysis) ON (a.status)",
            "CREATE INDEX decision_phase IF NOT EXISTS FOR (d:Decision) ON (d.phase)",
            "CREATE FULLTEXT INDEX variable_search IF NOT EXISTS FOR (v:Variable) ON EACH [v.name, v.id]",
        ]
        
        with self.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
        
        logger.info("Neo4j schema initialized")
    
    # =========================================================================
    # Variable Operations
    # =========================================================================
    
    def create_variable(self, variable: Variable) -> str:
        """Create or update a variable node."""
        query = """
        MERGE (v:Variable {id: $id})
        SET v.name = $name,
            v.type = $type,
            v.domain = $domain,
            v.unit = $unit,
            v.typical_lag = $typical_lag,
            v.saturation_type = $saturation_type,
            v.metadata = $metadata,
            v.updated_at = datetime()
        RETURN v.id as id
        """
        with self.session() as session:
            result = session.run(
                query,
                id=variable.id,
                name=variable.name,
                type=variable.type,
                domain=variable.domain,
                unit=variable.unit,
                typical_lag=variable.typical_lag,
                saturation_type=variable.saturation_type,
                metadata=json.dumps(variable.metadata),
            )
            return result.single()["id"]
    
    def get_variable(self, variable_id: str) -> Variable | None:
        """Get a variable by ID."""
        query = "MATCH (v:Variable {id: $id}) RETURN v"
        with self.session() as session:
            result = session.run(query, id=variable_id)
            record = result.single()
            if record:
                v = record["v"]
                return Variable(
                    id=v["id"],
                    name=v["name"],
                    type=v.get("type", "continuous"),
                    domain=v.get("domain", "media"),
                    unit=v.get("unit"),
                    typical_lag=v.get("typical_lag", []),
                    saturation_type=v.get("saturation_type"),
                    metadata=json.loads(v.get("metadata", "{}")),
                )
        return None
    
    def get_variables_by_domain(self, domain: str) -> list[Variable]:
        """Get all variables in a domain."""
        query = "MATCH (v:Variable {domain: $domain}) RETURN v"
        variables = []
        with self.session() as session:
            result = session.run(query, domain=domain)
            for record in result:
                v = record["v"]
                variables.append(Variable(
                    id=v["id"],
                    name=v["name"],
                    type=v.get("type", "continuous"),
                    domain=v.get("domain", "media"),
                    unit=v.get("unit"),
                    typical_lag=v.get("typical_lag", []),
                    saturation_type=v.get("saturation_type"),
                    metadata=json.loads(v.get("metadata", "{}")),
                ))
        return variables
    
    # =========================================================================
    # Causal Relationship Operations
    # =========================================================================
    
    def create_causal_relationship(self, rel: CausalRelationship) -> bool:
        """Create a causal relationship between variables."""
        query = """
        MATCH (source:Variable {id: $source_id})
        MATCH (target:Variable {id: $target_id})
        MERGE (source)-[r:CAUSES]->(target)
        SET r.mechanism = $mechanism,
            r.lag_periods = $lag_periods,
            r.confidence = $confidence,
            r.evidence_source = $evidence_source,
            r.adjustment_set = $adjustment_set,
            r.updated_at = datetime()
        RETURN true as success
        """
        with self.session() as session:
            result = session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                mechanism=rel.mechanism,
                lag_periods=rel.lag_periods,
                confidence=rel.confidence,
                evidence_source=rel.evidence_source,
                adjustment_set=rel.adjustment_set,
            )
            return result.single() is not None
    
    def get_causal_paths(
        self,
        source_ids: list[str],
        target_id: str,
        max_depth: int = 2
    ) -> list[dict]:
        """Get causal paths from sources to target."""
        query = """
        MATCH path = (source:Variable)-[:CAUSES*1..$max_depth]->(target:Variable {id: $target_id})
        WHERE source.id IN $source_ids
        RETURN path,
               [r IN relationships(path) | r.confidence] as confidences,
               [r IN relationships(path) | r.mechanism] as mechanisms
        """
        paths = []
        with self.session() as session:
            result = session.run(
                query,
                source_ids=source_ids,
                target_id=target_id,
                max_depth=max_depth,
            )
            for record in result:
                path = record["path"]
                paths.append({
                    "nodes": [n["id"] for n in path.nodes],
                    "confidences": record["confidences"],
                    "mechanisms": record["mechanisms"],
                })
        return paths
    
    # =========================================================================
    # Analysis Session Operations
    # =========================================================================
    
    def create_analysis(self, analysis: Analysis) -> str:
        """Create a new analysis session."""
        query = """
        CREATE (a:Analysis {
            id: $id,
            query: $query,
            created_at: datetime(),
            user_id: $user_id,
            status: $status,
            metadata: $metadata
        })
        RETURN a.id as id
        """
        with self.session() as session:
            result = session.run(
                query,
                id=analysis.id,
                query=analysis.query,
                user_id=analysis.user_id,
                status=analysis.status,
                metadata=json.dumps(analysis.metadata),
            )
            return result.single()["id"]
    
    def update_analysis_status(self, analysis_id: str, status: str):
        """Update analysis status."""
        query = """
        MATCH (a:Analysis {id: $id})
        SET a.status = $status, a.updated_at = datetime()
        """
        with self.session() as session:
            session.run(query, id=analysis_id, status=status)
    
    # =========================================================================
    # Decision Operations
    # =========================================================================
    
    def create_decision(self, decision: Decision) -> str:
        """Create a decision and link to analysis."""
        query = """
        MATCH (a:Analysis {id: $analysis_id})
        CREATE (d:Decision {
            id: $id,
            phase: $phase,
            decision_type: $decision_type,
            content: $content,
            rationale: $rationale,
            created_at: datetime()
        })
        CREATE (a)-[:MADE_DECISION]->(d)
        RETURN d.id as id
        """
        with self.session() as session:
            result = session.run(
                query,
                id=decision.id,
                analysis_id=decision.analysis_id,
                phase=decision.phase,
                decision_type=decision.decision_type,
                content=json.dumps(decision.content),
                rationale=decision.rationale,
            )
            return result.single()["id"]
    
    def get_analysis_decisions(
        self,
        analysis_id: str,
        phase: str | None = None
    ) -> list[Decision]:
        """Get all decisions for an analysis."""
        query = """
        MATCH (a:Analysis {id: $analysis_id})-[:MADE_DECISION]->(d:Decision)
        WHERE $phase IS NULL OR d.phase = $phase
        RETURN d
        ORDER BY d.created_at
        """
        decisions = []
        with self.session() as session:
            result = session.run(query, analysis_id=analysis_id, phase=phase)
            for record in result:
                d = record["d"]
                decisions.append(Decision(
                    id=d["id"],
                    analysis_id=analysis_id,
                    phase=d["phase"],
                    decision_type=d["decision_type"],
                    content=json.loads(d["content"]),
                    rationale=d["rationale"],
                ))
        return decisions
    
    # =========================================================================
    # Model Artifact Operations
    # =========================================================================
    
    def create_model_artifact(self, model: ModelArtifact) -> str:
        """Create a model artifact and link to analysis."""
        query = """
        MATCH (a:Analysis {id: $analysis_id})
        CREATE (m:Model {
            id: $id,
            framework: $framework,
            artifact_path: $artifact_path,
            metrics: $metrics,
            training_start: $training_start,
            training_end: $training_end,
            created_at: datetime()
        })
        CREATE (a)-[:PRODUCED]->(m)
        RETURN m.id as id
        """
        with self.session() as session:
            result = session.run(
                query,
                id=model.id,
                analysis_id=model.analysis_id,
                framework=model.framework,
                artifact_path=model.artifact_path,
                metrics=json.dumps(model.metrics),
                training_start=model.training_period[0] if model.training_period else None,
                training_end=model.training_period[1] if model.training_period else None,
            )
            return result.single()["id"]
    
    def link_model_to_variables(self, model_id: str, variable_ids: list[str]):
        """Link a model to the variables it uses."""
        query = """
        MATCH (m:Model {id: $model_id})
        MATCH (v:Variable {id: $var_id})
        MERGE (m)-[:USES_VARIABLE]->(v)
        """
        with self.session() as session:
            for var_id in variable_ids:
                session.run(query, model_id=model_id, var_id=var_id)
    
    # =========================================================================
    # Research Results Operations
    # =========================================================================
    
    def store_research_result(
        self,
        research_id: str,
        analysis_id: str,
        query: str,
        summary: str,
        sources: list[dict],
        insights: list[str],
    ) -> str:
        """Store web research results."""
        query_str = """
        MATCH (a:Analysis {id: $analysis_id})
        CREATE (r:ResearchResult {
            id: $research_id,
            query: $query,
            summary: $summary,
            sources: $sources,
            insights: $insights,
            created_at: datetime()
        })
        CREATE (a)-[:RESEARCHED]->(r)
        RETURN r.id as id
        """
        with self.session() as session:
            result = session.run(
                query_str,
                research_id=research_id,
                analysis_id=analysis_id,
                query=query,
                summary=summary,
                sources=json.dumps(sources),
                insights=insights,
            )
            return result.single()["id"]
    
    # =========================================================================
    # GraphRAG Retrieval Patterns
    # =========================================================================
    
    def get_successful_mmm_patterns(
        self,
        query: str,
        max_mape: float = 0.15,
        limit: int = 5
    ) -> list[dict]:
        """
        Get successful MMM patterns for planning context.
        
        Returns analyses with good model performance and their decisions.
        """
        cypher = """
        MATCH (a:Analysis)-[:PRODUCED]->(m:Model)
        WHERE m.metrics IS NOT NULL
        WITH a, m, apoc.convert.fromJsonMap(m.metrics) as metrics
        WHERE metrics.mape < $max_mape
        MATCH (a)-[:MADE_DECISION]->(d:Decision {phase: 'planning'})
        RETURN a.query as query,
               d.rationale as rationale,
               metrics as model_metrics
        ORDER BY metrics.mape
        LIMIT $limit
        """
        # Fallback without APOC
        fallback_cypher = """
        MATCH (a:Analysis)-[:PRODUCED]->(m:Model)
        MATCH (a)-[:MADE_DECISION]->(d:Decision {phase: 'planning'})
        RETURN a.query as query,
               d.rationale as rationale,
               m.metrics as model_metrics
        LIMIT $limit
        """
        
        patterns = []
        with self.session() as session:
            try:
                result = session.run(
                    cypher,
                    max_mape=max_mape,
                    limit=limit,
                )
            except Exception:
                result = session.run(fallback_cypher, limit=limit)
            
            for record in result:
                patterns.append({
                    "query": record["query"],
                    "rationale": record["rationale"],
                    "metrics": record["model_metrics"],
                })
        
        return patterns
    
    def get_feature_engineering_patterns(
        self,
        channel_ids: list[str],
        limit: int = 3
    ) -> list[dict]:
        """
        Get feature engineering patterns for channels.
        
        Returns successful transformation configurations.
        """
        query = """
        MATCH (m:Model)-[:USES_VARIABLE]->(v:Variable)
        WHERE v.id IN $channel_ids
        WITH m, v
        WHERE m.metrics IS NOT NULL
        RETURN v.id as variable,
               v.saturation_type as saturation,
               v.typical_lag as lag_config,
               m.metrics as model_metrics
        LIMIT $limit
        """
        patterns = []
        with self.session() as session:
            result = session.run(query, channel_ids=channel_ids, limit=limit)
            for record in result:
                patterns.append({
                    "variable": record["variable"],
                    "saturation": record["saturation"],
                    "lag_config": record["lag_config"],
                    "metrics": record["model_metrics"],
                })
        return patterns
    
    def get_modeling_context(
        self,
        channel_ids: list[str],
        target_id: str = "revenue"
    ) -> dict:
        """
        Get comprehensive modeling context including:
        - Successful model configurations
        - Causal structure priors
        """
        # Get model patterns
        model_query = """
        MATCH (m:Model)-[:USES_VARIABLE]->(v:Variable)
        WHERE v.id IN $channel_ids
        RETURN m.framework as framework,
               m.metrics as metrics
        ORDER BY m.created_at DESC
        LIMIT 3
        """
        
        # Get causal paths
        causal_query = """
        MATCH path = (v1:Variable)-[:CAUSES*1..2]->(v2:Variable {id: $target_id})
        WHERE v1.id IN $channel_ids
        RETURN v1.id as source,
               [r IN relationships(path) | r.confidence] as confidences,
               [n IN nodes(path) | n.id] as path_nodes
        """
        
        context = {"model_patterns": [], "causal_structure": []}
        
        with self.session() as session:
            # Model patterns
            result = session.run(model_query, channel_ids=channel_ids)
            for record in result:
                context["model_patterns"].append({
                    "framework": record["framework"],
                    "metrics": record["metrics"],
                })
            
            # Causal structure
            result = session.run(
                causal_query,
                channel_ids=channel_ids,
                target_id=target_id,
            )
            for record in result:
                context["causal_structure"].append({
                    "source": record["source"],
                    "path": record["path_nodes"],
                    "confidences": record["confidences"],
                })
        
        return context


# =============================================================================
# Singleton Factory
# =============================================================================

_neo4j_client: Neo4jClient | None = None


def get_neo4j_client(settings: Settings | None = None) -> Neo4jClient:
    """Get or create singleton Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(settings)
    return _neo4j_client
