"""
MMM Agent Package - Phase-specific agents for the MMM workflow.

This package contains the four main agents that drive the MMM workflow:
- PlanningAgent: Research and hypothesis formation
- EDAAgent: Data quality analysis and feature engineering
- ModelingAgent: Bayesian MMM fitting and diagnostics
- InterpretationAgent: ROI analysis and recommendations
"""

from .planning import PlanningAgent, planning_node
from .eda import EDAAgent, eda_node
from .modeling import ModelingAgent, modeling_node
from .interpretation import InterpretationAgent, interpretation_node

__all__ = [
    "PlanningAgent",
    "EDAAgent", 
    "ModelingAgent",
    "InterpretationAgent",
    "planning_node",
    "eda_node",
    "modeling_node",
    "interpretation_node",
]
