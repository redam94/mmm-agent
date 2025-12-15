"""
MMM Workflows Configuration

Centralized configuration for all four workflows using:
- Ollama with qwen3:30b / qwen3-coder:30b
- Neo4j for graph storage
- PostgreSQL for checkpointing
- Redis for pub/sub
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMTask(str, Enum):
    """Task types for LLM model selection."""
    REASONING = "reasoning"
    CODE = "code"
    PLANNING = "planning"
    ANALYSIS = "analysis"


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # ==========================================================================
    # Ollama Configuration
    # ==========================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_reasoning_model: str = Field(
        default="qwen3:30b",
        description="Model for reasoning/planning tasks"
    )
    ollama_code_model: str = Field(
        default="qwen3-coder:30b", 
        description="Model for code generation tasks"
    )
    ollama_timeout: int = Field(
        default=600,
        description="Timeout for Ollama requests in seconds"
    )
    
    # ==========================================================================
    # Neo4j Configuration
    # ==========================================================================
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password"
    )
    neo4j_database: str = Field(
        default="mmm",
        description="Neo4j database name"
    )
    
    # ==========================================================================
    # PostgreSQL Configuration
    # ==========================================================================
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="mmm")
    postgres_password: str = Field(default="mmm_password")
    postgres_database: str = Field(default="mmm_workflows")
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )
    
    @property
    def postgres_async_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )
    
    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str | None = Field(default=None)
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    # ==========================================================================
    # ChromaDB / Vector Store
    # ==========================================================================
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # ==========================================================================
    # Storage Paths
    # ==========================================================================
    data_dir: Path = Field(default=Path("./data"))
    uploads_dir: Path = Field(default=Path("./data/uploads"))
    models_dir: Path = Field(default=Path("./data/models"))
    outputs_dir: Path = Field(default=Path("./data/outputs"))
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )
    
    # ==========================================================================
    # Code Execution
    # ==========================================================================
    code_execution_timeout: int = Field(
        default=300,
        description="Timeout for code execution in seconds"
    )
    code_working_dir: Path = Field(
        default=Path("./data/sandbox"),
        description="Working directory for code execution"
    )
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "MMM_",
        "extra": "ignore",
    }
    
    def ensure_dirs(self):
        """Create required directories."""
        for dir_path in [
            self.data_dir,
            self.uploads_dir,
            self.models_dir,
            self.outputs_dir,
            self.code_working_dir,
            Path(self.chroma_persist_dir),
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_for_task(self, task: LLMTask | str) -> str:
        """Get the appropriate model for a task type."""
        task = LLMTask(task) if isinstance(task, str) else task
        
        if task in [LLMTask.CODE, LLMTask.ANALYSIS]:
            return self.ollama_code_model
        return self.ollama_reasoning_model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


# =============================================================================
# LLM Factory
# =============================================================================

def create_ollama_llm(
    task: LLMTask | str = LLMTask.REASONING,
    temperature: float = 0.0,
    settings: Settings | None = None,
):
    """
    Create an Ollama LLM instance for the given task.
    
    Args:
        task: Task type to select model
        temperature: Model temperature
        settings: Optional settings override
    
    Returns:
        ChatOllama instance
    """
    from langchain_ollama import ChatOllama
    
    settings = settings or get_settings()
    model = settings.get_model_for_task(task)
    
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=32768,  # Large context window
        timeout=settings.ollama_timeout,
    )


def create_structured_llm(
    output_schema,
    task: LLMTask | str = LLMTask.REASONING,
    settings: Settings | None = None,
):
    """
    Create an Ollama LLM with structured output.
    
    Args:
        output_schema: Pydantic model for output
        task: Task type to select model
        settings: Optional settings override
    
    Returns:
        LLM with structured output
    """
    llm = create_ollama_llm(task=task, temperature=0, settings=settings)
    return llm.with_structured_output(output_schema)

settings = get_settings()