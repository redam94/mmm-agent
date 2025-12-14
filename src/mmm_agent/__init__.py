"""
MMM Agent POC - Agentic Marketing Mix Modeling Framework

A proof-of-concept implementation of an AI-powered MMM workflow using:
- LangGraph for multi-phase orchestration
- Multi-LLM support (Ollama, OpenAI, Anthropic, Gemini)
- Local code execution for data analysis
- RAG + Web Search for domain knowledge
- MFF format output for BayesianMMM integration

Workflow Phases:
1. Planning: Research questions, variable selection, causal hypotheses
2. EDA: Data quality, transformations, correlation analysis
3. Modeling: Bayesian MMM fitting with adstock/saturation
4. Interpretation: ROI estimates, budget optimization, recommendations
"""

__version__ = "0.1.0"

from .config import Settings, get_settings, LLMProvider, create_chat_model
from .state import MMMWorkflowState, WorkflowPhase

__all__ = [
    "Settings",
    "get_settings", 
    "LLMProvider",
    "create_chat_model",
    "MMMWorkflowState",
    "WorkflowPhase",
]
