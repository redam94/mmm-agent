"""
MMM Workflows - AI-Powered Marketing Mix Modeling

A comprehensive agentic system for Marketing Mix Modeling using:
- LangGraph for workflow orchestration
- Ollama (qwen3:30b/qwen3-coder:30b) for LLM inference
- Neo4j + PostgreSQL for GraphRAG knowledge persistence
- PyMC-Marketing for Bayesian MMM fitting

Four main workflows:
1. Research Agent - Web research and planning
2. EDA Agent - Data cleaning and transformation to MFF format
3. Modeling Agent - Bayesian MMM fitting and interpretation
4. What-If Agent - Scenario analysis and optimization
"""

__version__ = "0.1.0"

from .config import settings, LLMTask, create_ollama_llm

__all__ = [
    "__version__",
    "settings",
    "LLMTask",
    "create_ollama_llm",
]
