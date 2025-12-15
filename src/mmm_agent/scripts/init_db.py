#!/usr/bin/env python
"""
Database Initialization Script

Sets up:
1. Neo4j schema (constraints and indexes)
2. PostgreSQL tables for LangGraph checkpointing
3. ChromaDB collection initialization
4. Domain knowledge seeding
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmm_agent.config import settings
from mmm_agent.db.neo4j_client import Neo4jClient
from mmm_agent.db.graphrag import GraphRAGManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Neo4j schema queries
NEO4J_CONSTRAINTS = [
    # Variable constraints
    "CREATE CONSTRAINT variable_id IF NOT EXISTS FOR (v:Variable) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT variable_name IF NOT EXISTS FOR (v:Variable) REQUIRE v.name IS UNIQUE",
    
    # Analysis constraints
    "CREATE CONSTRAINT analysis_id IF NOT EXISTS FOR (a:Analysis) REQUIRE a.analysis_id IS UNIQUE",
    
    # Decision constraints
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE",
    
    # ModelArtifact constraints
    "CREATE CONSTRAINT artifact_id IF NOT EXISTS FOR (m:ModelArtifact) REQUIRE m.artifact_id IS UNIQUE",
    
    # ResearchResult constraints
    "CREATE CONSTRAINT research_id IF NOT EXISTS FOR (r:ResearchResult) REQUIRE r.research_id IS UNIQUE",
]

NEO4J_INDEXES = [
    # Variable indexes
    "CREATE INDEX variable_domain IF NOT EXISTS FOR (v:Variable) ON (v.domain)",
    "CREATE INDEX variable_type IF NOT EXISTS FOR (v:Variable) ON (v.type)",
    
    # Analysis indexes
    "CREATE INDEX analysis_status IF NOT EXISTS FOR (a:Analysis) ON (a.status)",
    "CREATE INDEX analysis_session IF NOT EXISTS FOR (a:Analysis) ON (a.session_id)",
    
    # Decision indexes
    "CREATE INDEX decision_phase IF NOT EXISTS FOR (d:Decision) ON (d.phase)",
    "CREATE INDEX decision_type IF NOT EXISTS FOR (d:Decision) ON (d.decision_type)",
    
    # ModelArtifact indexes
    "CREATE INDEX artifact_type IF NOT EXISTS FOR (m:ModelArtifact) ON (m.artifact_type)",
    
    # Full-text search indexes
    """
    CREATE FULLTEXT INDEX variable_search IF NOT EXISTS 
    FOR (v:Variable) ON EACH [v.name, v.description]
    """,
    """
    CREATE FULLTEXT INDEX research_search IF NOT EXISTS 
    FOR (r:ResearchResult) ON EACH [r.query, r.summary]
    """,
]

# Domain knowledge for MMM
DOMAIN_KNOWLEDGE = {
    "variables": [
        {
            "id": "tv_spend",
            "name": "TV Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "2-4 weeks",
            "saturation_type": "hill",
            "description": "Television advertising spend with longer carryover effects",
        },
        {
            "id": "radio_spend",
            "name": "Radio Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "1-2 weeks",
            "saturation_type": "hill",
            "description": "Radio advertising spend with medium carryover",
        },
        {
            "id": "digital_spend",
            "name": "Digital Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "0-1 weeks",
            "saturation_type": "hill",
            "description": "Digital advertising (display, programmatic) with short decay",
        },
        {
            "id": "social_spend",
            "name": "Social Media Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "0-1 weeks",
            "saturation_type": "hill",
            "description": "Social media advertising with immediate impact",
        },
        {
            "id": "search_spend",
            "name": "Paid Search Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "0 weeks",
            "saturation_type": "hill",
            "description": "Paid search with immediate conversion impact",
        },
        {
            "id": "print_spend",
            "name": "Print Spend",
            "type": "media",
            "domain": "advertising",
            "typical_lag": "2-6 weeks",
            "saturation_type": "hill",
            "description": "Print advertising with long-lasting brand effects",
        },
        {
            "id": "price",
            "name": "Price",
            "type": "control",
            "domain": "marketing",
            "description": "Product price - typically negative effect on sales",
        },
        {
            "id": "distribution",
            "name": "Distribution",
            "type": "control",
            "domain": "marketing",
            "description": "Distribution/availability metric - positive effect on sales",
        },
        {
            "id": "promotion",
            "name": "Promotion",
            "type": "control",
            "domain": "marketing",
            "description": "Promotional activity indicator",
        },
        {
            "id": "seasonality",
            "name": "Seasonality",
            "type": "control",
            "domain": "time",
            "description": "Seasonal patterns (holidays, weather, etc.)",
        },
        {
            "id": "competitor_spend",
            "name": "Competitor Spend",
            "type": "control",
            "domain": "competitive",
            "description": "Competitor advertising activity",
        },
    ],
    "causal_relationships": [
        {
            "source": "tv_spend",
            "target": "brand_awareness",
            "mechanism": "Brand building through reach and frequency",
            "confidence": 0.9,
            "lag_periods": 4,
        },
        {
            "source": "brand_awareness",
            "target": "sales",
            "mechanism": "Higher awareness drives consideration and purchase",
            "confidence": 0.85,
            "lag_periods": 2,
        },
        {
            "source": "search_spend",
            "target": "sales",
            "mechanism": "Direct response capturing existing demand",
            "confidence": 0.95,
            "lag_periods": 0,
        },
        {
            "source": "price",
            "target": "sales",
            "mechanism": "Price elasticity - higher price reduces demand",
            "confidence": 0.9,
            "lag_periods": 0,
        },
        {
            "source": "promotion",
            "target": "sales",
            "mechanism": "Short-term demand lift from promotional activity",
            "confidence": 0.85,
            "lag_periods": 0,
        },
    ],
}


async def setup_neo4j(uri: str, user: str, password: str):
    """Set up Neo4j schema and domain knowledge."""
    logger.info("Setting up Neo4j...")
    
    neo4j = Neo4jClient(uri, user, password)
    
    # Create constraints
    logger.info("Creating constraints...")
    for constraint in NEO4J_CONSTRAINTS:
        try:
            async with neo4j.driver.session() as session:
                await session.run(constraint)
            logger.info(f"  Created: {constraint[:50]}...")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"  Exists: {constraint[:50]}...")
            else:
                logger.warning(f"  Failed: {e}")
    
    # Create indexes
    logger.info("Creating indexes...")
    for index in NEO4J_INDEXES:
        try:
            async with neo4j.driver.session() as session:
                await session.run(index)
            logger.info(f"  Created: {index[:50]}...")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"  Exists: {index[:50]}...")
            else:
                logger.warning(f"  Failed: {e}")
    
    # Seed domain knowledge
    logger.info("Seeding domain knowledge...")
    
    for var in DOMAIN_KNOWLEDGE["variables"]:
        try:
            await neo4j.create_variable(
                variable_id=var["id"],
                name=var["name"],
                var_type=var["type"],
                domain=var["domain"],
                typical_lag=var.get("typical_lag"),
                saturation_type=var.get("saturation_type"),
            )
            logger.info(f"  Created variable: {var['name']}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"  Exists: {var['name']}")
            else:
                logger.warning(f"  Failed to create {var['name']}: {e}")
    
    for rel in DOMAIN_KNOWLEDGE["causal_relationships"]:
        try:
            await neo4j.create_causal_relationship(
                source_id=rel["source"],
                target_id=rel["target"],
                mechanism=rel["mechanism"],
                confidence=rel["confidence"],
                lag_periods=rel["lag_periods"],
            )
            logger.info(f"  Created relationship: {rel['source']} -> {rel['target']}")
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    logger.info("Neo4j setup complete!")


async def setup_postgres(url: str):
    """Set up PostgreSQL tables for checkpointing."""
    logger.info("Setting up PostgreSQL...")
    
    try:
        import asyncpg
        
        conn = await asyncpg.connect(url)
        
        # LangGraph checkpoint tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_id TEXT,
                checkpoint BYTEA NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread 
            ON checkpoints(thread_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
            ON checkpoints(created_at)
        """)
        
        # Workflow state table (optional, for persistence)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                workflow_id TEXT PRIMARY KEY,
                workflow_type TEXT NOT NULL,
                status TEXT NOT NULL,
                analysis_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                state JSONB,
                error TEXT
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_status 
            ON workflow_runs(status)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_analysis 
            ON workflow_runs(analysis_id)
        """)
        
        await conn.close()
        logger.info("PostgreSQL setup complete!")
        
    except ImportError:
        logger.warning("asyncpg not installed, skipping PostgreSQL setup")
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")


async def setup_chromadb():
    """Initialize ChromaDB collection."""
    logger.info("Setting up ChromaDB...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize client
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=settings.CHROMA_PERSIST_DIR,
        ))
        
        # Create collection if not exists
        collection = client.get_or_create_collection(
            name="mmm_knowledge",
            metadata={"description": "MMM domain knowledge and decisions"}
        )
        
        logger.info(f"ChromaDB collection ready: {collection.count()} documents")
        logger.info("ChromaDB setup complete!")
        
    except ImportError:
        logger.warning("chromadb not installed, skipping ChromaDB setup")
    except Exception as e:
        logger.error(f"ChromaDB setup failed: {e}")


async def setup_graphrag():
    """Initialize GraphRAG with domain knowledge."""
    logger.info("Setting up GraphRAG...")
    
    try:
        neo4j = Neo4jClient(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
        )
        graphrag = GraphRAGManager(neo4j)
        
        # Load domain knowledge
        await graphrag.load_domain_knowledge()
        
        logger.info("GraphRAG setup complete!")
        
    except Exception as e:
        logger.error(f"GraphRAG setup failed: {e}")


async def main():
    """Run all setup tasks."""
    logger.info("=" * 60)
    logger.info("MMM Workflows Database Initialization")
    logger.info("=" * 60)
    
    # Neo4j
    try:
        await setup_neo4j(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
        )
    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
    
    # PostgreSQL
    if settings.ASYNC_DATABASE_URL:
        try:
            await setup_postgres(settings.ASYNC_DATABASE_URL)
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
    else:
        logger.info("Skipping PostgreSQL (no URL configured)")
    
    # ChromaDB
    try:
        await setup_chromadb()
    except Exception as e:
        logger.error(f"ChromaDB setup failed: {e}")
    
    # GraphRAG
    try:
        await setup_graphrag()
    except Exception as e:
        logger.error(f"GraphRAG setup failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Initialization complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
