"""
Shared Workflow State Definitions

Defines the state schemas for all four MMM workflows:
1. Research Agent - Planning and data collection
2. EDA Agent - Data processing and cleaning  
3. Modeling Agent - MMM fitting and interpretation
4. What-If Agent - Scenario analysis

Uses TypedDict for LangGraph compatibility with PostgreSQL checkpointing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# =============================================================================
# Workflow Phases
# =============================================================================

class WorkflowPhase(str, Enum):
    """Workflow execution phases."""
    # Research workflow
    RESEARCH_INIT = "research_init"
    RESEARCH_SEARCH = "research_search"
    RESEARCH_PLAN = "research_plan"
    RESEARCH_COMPLETE = "research_complete"
    
    # EDA workflow
    EDA_INIT = "eda_init"
    EDA_LOAD = "eda_load"
    EDA_QUALITY = "eda_quality"
    EDA_TRANSFORM = "eda_transform"
    EDA_OUTPUT = "eda_output"
    EDA_COMPLETE = "eda_complete"
    
    # Modeling workflow
    MODEL_INIT = "model_init"
    MODEL_CONFIG = "model_config"
    MODEL_FIT = "model_fit"
    MODEL_DIAGNOSE = "model_diagnose"
    MODEL_INTERPRET = "model_interpret"
    MODEL_COMPLETE = "model_complete"
    
    # What-If workflow
    WHATIF_INIT = "whatif_init"
    WHATIF_LOAD = "whatif_load"
    WHATIF_SCENARIO = "whatif_scenario"
    WHATIF_ANALYZE = "whatif_analyze"
    WHATIF_COMPLETE = "whatif_complete"
    
    # Shared
    ERROR = "error"
    HUMAN_INPUT = "human_input"


# =============================================================================
# Pydantic Models for Structured Data
# =============================================================================

class CausalHypothesis(BaseModel):
    """A causal hypothesis for MMM."""
    treatment: str = Field(description="Treatment variable (e.g., TV spend)")
    outcome: str = Field(description="Outcome variable (e.g., Revenue)")
    mechanism: str = Field(description="Hypothesized causal mechanism")
    expected_lag: list[int] = Field(default_factory=list, description="Expected lag periods")
    confounders: list[str] = Field(default_factory=list, description="Potential confounders")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Confidence level")


class DataRequirements(BaseModel):
    """Data requirements for MMM."""
    minimum_periods: int = Field(default=104, description="Minimum time periods needed")
    required_granularity: str = Field(default="weekly", description="Required time granularity")
    required_variables: list[str] = Field(default_factory=list)
    nice_to_have_variables: list[str] = Field(default_factory=list)
    dimension_requirements: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Complete research and planning output."""
    research_questions: list[str] = Field(default_factory=list)
    target_variable: str = Field(default="revenue")
    media_channels: list[str] = Field(default_factory=list)
    control_variables: list[str] = Field(default_factory=list)
    causal_hypotheses: list[CausalHypothesis] = Field(default_factory=list)
    data_requirements: DataRequirements = Field(default_factory=DataRequirements)
    summary: str = ""


class DataQualityIssue(BaseModel):
    """A data quality issue."""
    severity: str = Field(description="high, medium, low")
    variable: str
    issue_type: str
    description: str
    recommendation: str = ""


class FeatureTransformation(BaseModel):
    """A feature transformation configuration."""
    variable: str
    transformation: str  # geometric_adstock, hill_saturation, etc.
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ChannelContribution(BaseModel):
    """Channel contribution estimate."""
    channel: str
    contribution_mean: float
    contribution_lower: float
    contribution_upper: float
    roi_mean: float
    roi_lower: float
    roi_upper: float
    spend_total: float = 0.0
    share_of_contribution: float = 0.0


class ModelDiagnostics(BaseModel):
    """Model convergence diagnostics."""
    rhat_max: float = 0.0
    ess_bulk_min: float = 0.0
    ess_tail_min: float = 0.0
    divergences: int = 0
    convergence_status: str = "unknown"
    recommendations: list[str] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    """Result from a what-if scenario."""
    scenario_name: str
    scenario_description: str
    changes: dict[str, Any] = Field(default_factory=dict)
    predicted_outcome: float = 0.0
    baseline_outcome: float = 0.0
    incremental_impact: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)


# =============================================================================
# Workflow States (TypedDict for LangGraph)
# =============================================================================

class ResearchWorkflowState(TypedDict, total=False):
    """State for Research Agent workflow."""
    # Session
    session_id: str
    analysis_id: str
    user_id: str | None
    
    # Input
    user_query: str
    business_context: str
    
    # Phase tracking
    current_phase: WorkflowPhase
    messages: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    
    # Research outputs
    web_search_results: list[dict]
    research_summary: str
    domain_insights: list[str]
    
    # Planning outputs
    research_plan: dict  # Serialized ResearchPlan
    target_variable: str
    media_channels: list[str]
    control_variables: list[str]
    causal_hypotheses: list[dict]
    data_requirements: dict
    
    # User feedback
    user_feedback: str
    plan_approved: bool
    
    # Context
    kg_context: str
    prior_patterns: list[dict]


class EDAWorkflowState(TypedDict, total=False):
    """State for EDA Agent workflow."""
    # Session
    session_id: str
    analysis_id: str
    
    # Input
    data_sources: list[dict]  # File paths and metadata
    research_plan: dict  # From Research workflow
    
    # Phase tracking
    current_phase: WorkflowPhase
    messages: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    
    # EDA outputs
    data_quality_report: dict
    quality_issues: list[dict]
    statistics_summary: dict
    correlation_matrix: dict
    
    # Feature engineering
    feature_transformations: list[dict]
    adstock_configs: dict
    saturation_configs: dict
    
    # Output data
    mff_data_path: str
    cleaned_data_path: str
    
    # Visualizations
    generated_plots: list[str]
    
    # Recommendations
    modeling_recommendations: list[str]


class ModelingWorkflowState(TypedDict, total=False):
    """State for Modeling Agent workflow."""
    # Session
    session_id: str
    analysis_id: str
    
    # Input
    mff_data_path: str
    research_plan: dict
    feature_transformations: list[dict]
    
    # Phase tracking
    current_phase: WorkflowPhase
    messages: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    
    # Model configuration
    model_config: dict
    target_variable: str
    media_channels: list[str]
    control_variables: list[str]
    
    # Fitting
    model_artifact_path: str
    inference_data_path: str
    fit_duration: float
    
    # Diagnostics
    convergence_diagnostics: dict
    convergence_status: str
    
    # Results
    channel_contributions: list[dict]
    roi_estimates: dict
    total_media_contribution: float
    model_r2: float
    
    # Interpretation
    interpretation_summary: str
    recommendations: list[str]
    
    # Visualizations
    generated_plots: list[str]


class WhatIfWorkflowState(TypedDict, total=False):
    """State for What-If Scenario Agent workflow."""
    # Session
    session_id: str
    analysis_id: str
    
    # Input
    model_artifact_path: str
    user_query: str
    
    # Phase tracking
    current_phase: WorkflowPhase
    messages: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    
    # Model context
    channel_contributions: list[dict]
    baseline_metrics: dict
    
    # Scenarios
    scenarios: list[dict]
    scenario_results: list[dict]
    
    # Analysis
    optimization_suggestions: list[dict]
    sensitivity_analysis: dict
    
    # Response
    response_summary: str
    
    # Visualizations
    generated_plots: list[str]


# =============================================================================
# Combined State for Full Pipeline
# =============================================================================

class MMMPipelineState(TypedDict, total=False):
    """Combined state for full MMM pipeline."""
    # Session
    session_id: str
    analysis_id: str
    user_id: str | None
    
    # Current phase
    current_phase: WorkflowPhase
    current_workflow: str  # research, eda, modeling, whatif
    
    # Messages
    messages: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    
    # Research outputs
    user_query: str
    business_context: str
    research_plan: dict
    
    # EDA outputs
    data_sources: list[dict]
    mff_data_path: str
    data_quality_report: dict
    feature_transformations: list[dict]
    
    # Modeling outputs
    model_artifact_path: str
    channel_contributions: list[dict]
    roi_estimates: dict
    convergence_diagnostics: dict
    
    # What-If outputs
    scenarios: list[dict]
    scenario_results: list[dict]
    
    # Final outputs
    final_report: str
    generated_plots: list[str]
