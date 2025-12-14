"""
MMM Agent Configuration

Provider-agnostic LLM configuration supporting Ollama, OpenAI, Anthropic, and Gemini.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "google_genai"


class LLMConfig(BaseModel):
    """Configuration for a specific LLM."""
    provider: LLMProvider
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str | None = None  # For Ollama
    api_key: str | None = None


class Settings(BaseSettings):
    """Application settings."""
    
    # App settings
    app_name: str = "MMM Agent POC"
    app_version: str = "0.1.0"
    debug: bool = True
    
    # Storage
    data_dir: str = "./data"
    models_dir: str = "./models"
    outputs_dir: str = "./outputs"
    
    # LLM Configuration - defaults to Ollama for local development
    default_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://100.91.155.118:11434"
    ollama_model: str = "qwen3:30b"
    
    # Optional cloud providers
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    
    google_api_key: str | None = None
    google_model: str = "gemini-1.5-pro"
    
    # Code execution
    code_timeout: int = 300
    max_retries: int = 3
    
    # RAG settings
    rag_persist_dir: str = "./rag_store"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    
    # Redis (for job queue)
    redis_url: str = "redis://localhost:6379"
    
    provider: LLMProvider = LLMProvider.OLLAMA
    default_model: str = "llama3.2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_llm_config(
    settings: Settings,
    provider: LLMProvider | None = None,
    task_type: Literal["reasoning", "code", "structured", "fast"] = "reasoning"
) -> LLMConfig:
    """
    Get LLM configuration based on provider and task type.
    
    Task types:
    - reasoning: Complex causal reasoning (use Claude)
    - code: Code generation (use GPT-4o or local)
    - structured: Structured outputs (use GPT-4o)
    - fast: Quick responses (use local Ollama or GPT-4o-mini)
    """
    provider = provider or settings.default_provider
    
    # Task-based routing
    if task_type == "reasoning" and settings.anthropic_api_key:
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model=settings.anthropic_model,
            temperature=0.0,
            api_key=settings.anthropic_api_key,
        )
    elif task_type == "structured" and settings.openai_api_key:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model=settings.openai_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
    elif task_type == "fast":
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=settings.ollama_model,
            temperature=0.0,
            base_url=settings.ollama_base_url,
        )
    
    # Default to configured provider
    if provider == LLMProvider.OLLAMA:
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    elif provider == LLMProvider.OPENAI:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    elif provider == LLMProvider.ANTHROPIC:
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        )
    elif provider == LLMProvider.GEMINI:
        return LLMConfig(
            provider=LLMProvider.GEMINI,
            model=settings.google_model,
            api_key=settings.google_api_key,
        )
    
    raise ValueError(f"Unknown provider: {provider}")


def create_chat_model(config: LLMConfig):
    """
    Create a LangChain chat model from configuration.
    
    Uses init_chat_model for provider-agnostic instantiation.
    """
    from langchain.chat_models import init_chat_model
    
    kwargs = {
        "model": config.model,
        "model_provider": config.provider.value,
        "temperature": config.temperature,
    }
    
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    
    return init_chat_model(**kwargs)


class MFFDimensions(BaseModel):
    """Standard MFF dimensions for data alignment."""
    period_column: str = "Period"
    geography_column: str | None = "Geography"
    product_column: str | None = "Product"
    variable_column: str = "Variable"
    value_column: str = "Value"
    
    # Date format for period parsing
    date_format: str = "%Y-%m-%d"
    frequency: Literal["D", "W", "M"] = "W"


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""
    name: str
    path: str
    source_type: Literal["csv", "excel", "parquet", "json"] = "csv"
    
    # Column mappings to standard MFF dimensions
    period_column: str
    geography_column: str | None = None
    product_column: str | None = None
    
    # Variable columns (will be melted)
    variable_columns: list[str] = Field(default_factory=list)
    
    # Value scaling/transformation
    value_scale: float = 1.0
    
    # Date parsing
    date_format: str = "%Y-%m-%d"
    
    # Metadata
    data_domain: Literal["media", "sales", "control", "external"] = "media"


class WorkflowConfig(BaseModel):
    """Configuration for the MMM workflow."""
    workflow_id: str
    name: str = "MMM Analysis"
    
    # Data sources
    data_sources: list[DataSourceConfig] = Field(default_factory=list)
    
    # MFF configuration
    mff_dimensions: MFFDimensions = Field(default_factory=MFFDimensions)
    
    # Model settings
    kpi_variable: str = "Revenue"
    media_channels: list[str] = Field(default_factory=list)
    control_variables: list[str] = Field(default_factory=list)
    
    # Fitting parameters
    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 500
