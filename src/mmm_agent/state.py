"""
MMM Workflow State Management

Defines the state schema for the four-phase MMM workflow:
Planning → EDA → Modeling → Interpretation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class WorkflowPhase(str, Enum):
    """Workflow phases."""
    PLANNING = "planning"
    EDA = "eda"
    MODELING = "modeling"
    INTERPRETATION = "interpretation"
    COMPLETE = "complete"
    ERROR = "error"


class DataQualityIssue(BaseModel):
    """A data quality issue detected during EDA."""
    severity: str  # "high", "medium", "low"
    column: str
    issue_type: str  # "missing", "outlier", "inconsistent", etc.
    description: str
    recommendation: str


class FeatureTransformation(BaseModel):
    """A feature transformation decision."""
    variable: str
    transformation: str  # "geometric_adstock", "hill_saturation", etc.
    parameters: dict[str, Any]
    rationale: str


class CausalHypothesis(BaseModel):
    """A causal hypothesis from planning."""
    treatment: str
    outcome: str
    mechanism: str
    expected_lag: list[int]
    confounders: list[str]
    confidence: float


class ChannelContribution(BaseModel):
    """Channel contribution estimate."""
    channel: str
    contribution_mean: float
    contribution_lower: float
    contribution_upper: float
    roi_mean: float
    roi_lower: float
    roi_upper: float
    spend_total: float


class ModelDiagnostics(BaseModel):
    """Model convergence diagnostics."""
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int
    convergence_status: str  # "good", "acceptable", "poor"
    recommendations: list[str]


class Decision(BaseModel):
    """A decision made during the workflow."""
    phase: WorkflowPhase
    decision_type: str
    content: dict[str, Any]
    rationale: str
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# LangGraph State Schema
# =============================================================================

class MMMWorkflowState(TypedDict, total=False):
    """
    Complete state for the MMM workflow.
    
    This state flows through all four phases, accumulating context and artifacts.
    Uses Annotated[list, add] for lists that should be appended to.
    """
    
    # === Control Flow ===
    messages: Annotated[list, add]  # Conversation history
    current_phase: WorkflowPhase
    error: str | None
    
    # === User Input ===
    user_query: str
    data_paths: list[str]
    workflow_id: str
    
    # === Planning Phase Outputs ===
    research_questions: list[str]
    target_variable: str
    media_channels: list[str]
    control_variables: list[str]
    causal_hypotheses: list[dict]  # List of CausalHypothesis as dicts
    data_requirements: dict[str, Any]
    planning_summary: str
    
    # === EDA Phase Outputs ===
    data_loaded: bool
    raw_data_summary: dict[str, Any]
    data_quality_report: dict[str, Any]
    data_quality_issues: list[dict]  # List of DataQualityIssue as dicts
    correlation_matrix: dict[str, dict]  # Variable -> Variable -> correlation
    feature_transformations: list[dict]  # List of FeatureTransformation as dicts
    mff_data_path: str  # Path to processed MFF data
    eda_summary: str
    eda_plots: list[str]  # Paths to generated plots
    
    # === Modeling Phase Outputs ===
    model_config: dict[str, Any]
    model_job_id: str
    model_status: str
    model_artifact_path: str  # Path to fitted model (.nc)
    inference_data_path: str  # Path to ArviZ InferenceData
    convergence_diagnostics: dict[str, Any]
    channel_contributions: list[dict]  # List of ChannelContribution as dicts
    model_summary: str
    model_plots: list[str]
    
    # === Interpretation Phase Outputs ===
    roi_estimates: dict[str, dict]  # Channel -> ROI stats
    budget_allocation: dict[str, float]  # Optimal spend by channel
    what_if_scenarios: list[dict]
    key_insights: list[str]
    recommendations: list[dict]
    interpretation_summary: str
    interpretation_plots: list[str]
    
    # === Shared Context ===
    rag_context: str  # Retrieved context from RAG
    web_search_results: list[dict]  # Results from web search
    prior_decisions: Annotated[list[dict], add]  # All decisions made
    
    # === Code Execution ===
    generated_code: str
    code_output: str
    code_error: str | None
    execution_artifacts: list[str]  # Paths to generated files


def create_initial_state(
    user_query: str,
    workflow_id: str,
    data_paths: list[str] | None = None,
) -> MMMWorkflowState:
    """Create initial workflow state."""
    return MMMWorkflowState(
        messages=[],
        current_phase=WorkflowPhase.PLANNING,
        error=None,
        user_query=user_query,
        data_paths=data_paths or [],
        workflow_id=workflow_id,
        research_questions=[],
        target_variable="",
        media_channels=[],
        control_variables=[],
        causal_hypotheses=[],
        data_requirements={},
        planning_summary="",
        data_loaded=False,
        raw_data_summary={},
        data_quality_report={},
        data_quality_issues=[],
        correlation_matrix={},
        feature_transformations=[],
        mff_data_path="",
        eda_summary="",
        eda_plots=[],
        model_config={},
        model_job_id="",
        model_status="pending",
        model_artifact_path="",
        inference_data_path="",
        convergence_diagnostics={},
        channel_contributions=[],
        model_summary="",
        model_plots=[],
        roi_estimates={},
        budget_allocation={},
        what_if_scenarios=[],
        key_insights=[],
        recommendations=[],
        interpretation_summary="",
        interpretation_plots=[],
        rag_context="",
        web_search_results=[],
        prior_decisions=[],
        generated_code="",
        code_output="",
        code_error=None,
        execution_artifacts=[],
    )


# =============================================================================
# State Serialization Helpers
# =============================================================================

def state_to_context(state: MMMWorkflowState, max_tokens: int = 4000) -> str:
    """
    Convert state to a context string for LLM prompts.
    
    Focuses on the most relevant information based on current phase.
    """
    phase = state.get("current_phase", WorkflowPhase.PLANNING)
    
    context_parts = [
        f"## Current Phase: {phase.value}",
        f"## User Query: {state.get('user_query', 'N/A')}",
    ]
    
    if phase == WorkflowPhase.PLANNING:
        # Planning needs data overview
        if state.get("data_paths"):
            context_parts.append(f"## Data Files: {', '.join(state['data_paths'])}")
    
    elif phase == WorkflowPhase.EDA:
        # EDA needs planning decisions
        context_parts.append(f"## Target Variable: {state.get('target_variable', 'N/A')}")
        context_parts.append(f"## Media Channels: {', '.join(state.get('media_channels', []))}")
        context_parts.append(f"## Control Variables: {', '.join(state.get('control_variables', []))}")
        if state.get("causal_hypotheses"):
            context_parts.append(f"## Causal Hypotheses: {len(state['causal_hypotheses'])} defined")
    
    elif phase == WorkflowPhase.MODELING:
        # Modeling needs EDA outputs
        context_parts.append(f"## MFF Data: {state.get('mff_data_path', 'N/A')}")
        if state.get("feature_transformations"):
            context_parts.append(f"## Transformations: {len(state['feature_transformations'])} defined")
        if state.get("data_quality_issues"):
            high_issues = [i for i in state["data_quality_issues"] if i.get("severity") == "high"]
            context_parts.append(f"## Data Quality: {len(high_issues)} high-severity issues")
    
    elif phase == WorkflowPhase.INTERPRETATION:
        # Interpretation needs model results
        context_parts.append(f"## Model: {state.get('model_artifact_path', 'N/A')}")
        if state.get("convergence_diagnostics"):
            diag = state["convergence_diagnostics"]
            context_parts.append(f"## Convergence: {diag.get('convergence_status', 'N/A')}")
        if state.get("channel_contributions"):
            context_parts.append(f"## Channels: {len(state['channel_contributions'])} analyzed")
    
    # Add recent decisions
    if state.get("prior_decisions"):
        recent = state["prior_decisions"][-5:]
        context_parts.append("## Recent Decisions:")
        for d in recent:
            context_parts.append(f"  - [{d.get('phase', 'N/A')}] {d.get('decision_type', 'N/A')}")
    
    return "\n".join(context_parts)


def summarize_phase_output(state: MMMWorkflowState) -> str:
    """Create a summary of the current phase's output."""
    phase = state.get("current_phase", WorkflowPhase.PLANNING)
    
    if phase == WorkflowPhase.PLANNING:
        return state.get("planning_summary", "Planning phase pending.")
    elif phase == WorkflowPhase.EDA:
        return state.get("eda_summary", "EDA phase pending.")
    elif phase == WorkflowPhase.MODELING:
        return state.get("model_summary", "Modeling phase pending.")
    elif phase == WorkflowPhase.INTERPRETATION:
        return state.get("interpretation_summary", "Interpretation phase pending.")
    
    return "Workflow complete."
