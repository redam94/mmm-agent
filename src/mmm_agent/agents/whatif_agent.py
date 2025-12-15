"""
What-If Agent Workflow

Workflow 4: Scenario analysis using fitted MMM model artifacts.

This workflow:
1. Loads fitted model artifacts from the modeling workflow
2. Parses user scenario queries (budget changes, channel shifts)
3. Generates counterfactual predictions
4. Provides budget optimization suggestions
5. Runs sensitivity analysis
6. Creates comparison visualizations

Uses:
- qwen3:30b for reasoning and optimization suggestions
- qwen3-coder:30b for scenario computation code
- Neo4j for retrieving model artifacts and prior scenarios
- GraphRAG for optimization patterns and best practices
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from ..config import (
    LLMTask,
    create_ollama_llm,
    create_structured_llm,
    settings,
)
from ..db.neo4j_client import Neo4jClient
from ..db.graphrag import GraphRAGManager
from ..tools.code_executor import CodeExecutor
from .state import (
    WhatIfWorkflowState,
    WorkflowPhase,
    ScenarioResult,
)


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts
# =============================================================================

WHATIF_SYSTEM_PROMPT = """You are an expert Marketing Mix Modeling (MMM) analyst specializing in scenario planning and budget optimization.

Your expertise includes:
- Translating business questions into quantifiable scenarios
- Budget allocation optimization across channels
- Sensitivity analysis and uncertainty quantification
- Communicating trade-offs clearly to business stakeholders

When analyzing scenarios:
- Always quantify the expected impact with uncertainty bounds
- Consider diminishing returns (saturation effects)
- Account for carryover effects (adstock)
- Compare scenarios on a level playing field

When making recommendations:
- Be specific about expected ROI changes
- Highlight risks and assumptions
- Provide actionable next steps
- Consider practical constraints (minimum spend, channel capacity)
"""

SCENARIO_PARSING_PROMPT = """Parse the user's scenario query into structured changes.

Extract:
1. Which channels are being changed
2. The type of change (absolute, percentage, reallocation)
3. The magnitude of change
4. Any constraints mentioned

If the query is ambiguous, make reasonable assumptions and document them.
"""

OPTIMIZATION_PROMPT = """Based on the model's response curves and current budget allocation, provide optimization recommendations.

Consider:
1. Current ROI by channel
2. Marginal ROI at current spend levels
3. Saturation levels by channel
4. Practical constraints (minimum spend, seasonality)

Provide specific, actionable recommendations with expected impact.
"""


# =============================================================================
# Structured Output Models
# =============================================================================

class BudgetChange(BaseModel):
    """A single budget change specification."""
    channel: str = Field(description="Channel name to modify")
    change_type: str = Field(description="absolute, percentage, or multiply")
    value: float = Field(description="Change value (amount, percentage, or multiplier)")
    rationale: str = Field(default="", description="Reason for this change")


class ScenarioDefinition(BaseModel):
    """Complete scenario specification."""
    name: str = Field(description="Scenario name for reference")
    description: str = Field(description="What this scenario represents")
    budget_changes: list[BudgetChange] = Field(default_factory=list)
    time_period: str = Field(default="all", description="Time period to apply changes")
    hold_total_budget: bool = Field(default=False, description="If true, reallocate within total")
    assumptions: list[str] = Field(default_factory=list)


class ParsedScenarios(BaseModel):
    """Parsed scenarios from user query."""
    scenarios: list[ScenarioDefinition]
    baseline_description: str = Field(default="Current budget allocation")
    comparison_metric: str = Field(default="total_revenue", description="Primary metric to compare")


class OptimizationRecommendation(BaseModel):
    """Budget optimization recommendation."""
    channel: str
    current_spend: float
    recommended_spend: float
    expected_roi_change: float
    confidence: str = Field(description="high, medium, low")
    rationale: str


class OptimizationSummary(BaseModel):
    """Complete optimization summary."""
    recommendations: list[OptimizationRecommendation]
    total_budget_change: float
    expected_incremental_outcome: float
    expected_incremental_outcome_lower: float
    expected_incremental_outcome_upper: float
    key_insights: list[str]
    caveats: list[str]


# =============================================================================
# Code Templates
# =============================================================================

LOAD_MODEL_TEMPLATE = '''
"""
Load fitted MMM model artifacts
Session: {session_id}
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Load model results
model_path = Path("{model_artifact_path}")
print(f"Loading model from: {{model_path}}")

# Load model results JSON
results_path = model_path.parent / "model_results.json"
if results_path.exists():
    with open(results_path, "r") as f:
        model_results = json.load(f)
    print("Loaded model results")
else:
    raise FileNotFoundError(f"Model results not found at {{results_path}}")

# Load posterior parameters if available
posterior_path = model_path.parent / "posterior_params.pkl"
if posterior_path.exists():
    with open(posterior_path, "rb") as f:
        posterior_params = pickle.load(f)
    print("Loaded posterior parameters")
else:
    posterior_params = None
    print("No posterior parameters found")

# Extract key information
channels = model_results.get("channels", [])
contributions = model_results.get("contributions", {{}})
model_config = model_results.get("config", {{}})

print(f"Channels: {{channels}}")
print(f"Model configuration loaded")

# Save for use in scenarios
output = {{
    "channels": channels,
    "contributions": contributions,
    "config": model_config,
    "has_posterior": posterior_params is not None,
}}

with open("model_info.json", "w") as f:
    json.dump(output, f, indent=2)

print("Model information saved to model_info.json")
'''

SCENARIO_ANALYSIS_TEMPLATE = '''
"""
Scenario Analysis Script
Session: {session_id}
Scenario: {scenario_name}
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model info
with open("model_info.json", "r") as f:
    model_info = json.load(f)

# Load original data
data = pd.read_csv("{mff_data_path}")
print(f"Data shape: {{data.shape}}")

# Scenario configuration
SCENARIO = {scenario_json}

# Get baseline values
channels = model_info["channels"]
contributions = model_info["contributions"]

# Calculate baseline outcome
baseline_outcome = sum(contributions.get(ch, {{}}).get("contribution_mean", 0) for ch in channels)
baseline_by_channel = {{ch: contributions.get(ch, {{}}).get("contribution_mean", 0) for ch in channels}}

print(f"Baseline total media contribution: {{baseline_outcome:,.2f}}")
print("Baseline by channel:")
for ch, val in baseline_by_channel.items():
    print(f"  {{ch}}: {{val:,.2f}}")

# Apply scenario changes
scenario_by_channel = baseline_by_channel.copy()
changes_applied = {{}}

for change in SCENARIO.get("budget_changes", []):
    channel = change["channel"]
    if channel not in scenario_by_channel:
        print(f"Warning: Channel {{channel}} not found in model")
        continue
    
    current = scenario_by_channel[channel]
    change_type = change["change_type"]
    value = change["value"]
    
    if change_type == "percentage":
        new_value = current * (1 + value / 100)
    elif change_type == "multiply":
        new_value = current * value
    elif change_type == "absolute":
        new_value = current + value
    else:
        new_value = current
    
    # Apply simple saturation adjustment (diminishing returns)
    # This is a simplified model - actual should use fitted curves
    if new_value > current:
        # Diminishing returns factor
        increase_ratio = new_value / current if current > 0 else 1
        saturation_factor = 1 - 0.1 * np.log(increase_ratio)  # Simplified
        saturation_factor = max(0.5, saturation_factor)  # Floor at 50%
        adjusted_value = current + (new_value - current) * saturation_factor
    else:
        adjusted_value = new_value
    
    changes_applied[channel] = {{
        "original": current,
        "requested": new_value,
        "adjusted": adjusted_value,
        "change_pct": (adjusted_value - current) / current * 100 if current > 0 else 0,
    }}
    scenario_by_channel[channel] = adjusted_value

# Calculate scenario outcome
scenario_outcome = sum(scenario_by_channel.values())
incremental_impact = scenario_outcome - baseline_outcome

print(f"\\nScenario: {{SCENARIO['name']}}")
print(f"Scenario total: {{scenario_outcome:,.2f}}")
print(f"Incremental impact: {{incremental_impact:,.2f}}")
print(f"Percentage change: {{incremental_impact/baseline_outcome*100:.2f}}%" if baseline_outcome > 0 else "N/A")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Channel comparison
ax1 = axes[0]
x = np.arange(len(channels))
width = 0.35

baseline_vals = [baseline_by_channel.get(ch, 0) for ch in channels]
scenario_vals = [scenario_by_channel.get(ch, 0) for ch in channels]

bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue')
bars2 = ax1.bar(x + width/2, scenario_vals, width, label=SCENARIO['name'], color='coral')

ax1.set_xlabel('Channel')
ax1.set_ylabel('Contribution')
ax1.set_title('Channel Contribution Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(channels, rotation=45, ha='right')
ax1.legend()

# Total comparison
ax2 = axes[1]
totals = [baseline_outcome, scenario_outcome]
labels = ['Baseline', SCENARIO['name']]
colors = ['steelblue', 'coral']
bars = ax2.bar(labels, totals, color=colors)
ax2.set_ylabel('Total Media Contribution')
ax2.set_title('Total Contribution Comparison')

# Add value labels
for bar, val in zip(bars, totals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{{val:,.0f}}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("scenario_comparison.png", dpi=150, bbox_inches='tight')
print("\\nSaved scenario_comparison.png")

# Save results
result = {{
    "scenario_name": SCENARIO["name"],
    "baseline_outcome": baseline_outcome,
    "scenario_outcome": scenario_outcome,
    "incremental_impact": incremental_impact,
    "incremental_pct": incremental_impact / baseline_outcome * 100 if baseline_outcome > 0 else 0,
    "changes_applied": changes_applied,
    "baseline_by_channel": baseline_by_channel,
    "scenario_by_channel": scenario_by_channel,
}}

with open("scenario_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\\nResults saved to scenario_result.json")
'''

OPTIMIZATION_TEMPLATE = '''
"""
Budget Optimization Analysis
Session: {session_id}
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model info
with open("model_info.json", "r") as f:
    model_info = json.load(f)

# Load original data for current spend
data = pd.read_csv("{mff_data_path}")

channels = model_info["channels"]
contributions = model_info["contributions"]

# Calculate current metrics
current_metrics = {{}}
for ch in channels:
    ch_contrib = contributions.get(ch, {{}})
    contribution = ch_contrib.get("contribution_mean", 0)
    
    # Get actual spend from data
    if ch in data.columns:
        total_spend = data[ch].sum()
    else:
        total_spend = 0
    
    roi = contribution / total_spend if total_spend > 0 else 0
    
    current_metrics[ch] = {{
        "contribution": contribution,
        "spend": total_spend,
        "roi": roi,
        "contribution_lower": ch_contrib.get("contribution_lower", contribution * 0.8),
        "contribution_upper": ch_contrib.get("contribution_upper", contribution * 1.2),
    }}

print("Current Metrics by Channel:")
print("-" * 60)
for ch, metrics in current_metrics.items():
    print(f"{{ch}}:")
    print(f"  Contribution: {{metrics['contribution']:,.2f}}")
    print(f"  Spend: {{metrics['spend']:,.2f}}")
    print(f"  ROI: {{metrics['roi']:.4f}}")

# Simple optimization: shift budget from low ROI to high ROI channels
sorted_by_roi = sorted(current_metrics.items(), key=lambda x: x[1]['roi'], reverse=True)

print("\\nChannels ranked by ROI:")
for i, (ch, metrics) in enumerate(sorted_by_roi, 1):
    print(f"  {{i}}. {{ch}}: {{metrics['roi']:.4f}}")

# Generate recommendations
recommendations = []
total_spend = sum(m['spend'] for m in current_metrics.values())

# Identify underperformers (bottom quartile ROI)
n_channels = len(sorted_by_roi)
underperformers = [ch for ch, _ in sorted_by_roi[-(n_channels//4+1):]]
top_performers = [ch for ch, _ in sorted_by_roi[:max(1, n_channels//3)]]

print(f"\\nUnderperforming channels: {{underperformers}}")
print(f"Top performing channels: {{top_performers}}")

# Calculate reallocation
reallocation_pct = 0.15  # Suggest 15% reallocation from underperformers

for ch in underperformers:
    if current_metrics[ch]['spend'] > 0:
        recommendations.append({{
            "channel": ch,
            "current_spend": current_metrics[ch]['spend'],
            "recommended_change_pct": -reallocation_pct * 100,
            "rationale": f"Below-average ROI ({{current_metrics[ch]['roi']:.4f}})",
        }})

for ch in top_performers:
    if current_metrics[ch]['spend'] > 0:
        increase_amount = sum(
            current_metrics[uc]['spend'] * reallocation_pct 
            for uc in underperformers if current_metrics[uc]['spend'] > 0
        ) / len(top_performers)
        increase_pct = increase_amount / current_metrics[ch]['spend'] * 100 if current_metrics[ch]['spend'] > 0 else 0
        recommendations.append({{
            "channel": ch,
            "current_spend": current_metrics[ch]['spend'],
            "recommended_change_pct": increase_pct,
            "rationale": f"Above-average ROI ({{current_metrics[ch]['roi']:.4f}}), room for growth",
        }})

# Create optimization visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Current allocation pie chart
ax1 = axes[0]
spends = [current_metrics[ch]['spend'] for ch in channels]
ax1.pie(spends, labels=channels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Current Budget Allocation')

# ROI comparison bar chart
ax2 = axes[1]
rois = [current_metrics[ch]['roi'] for ch in channels]
colors = ['green' if ch in top_performers else 'red' if ch in underperformers else 'gray' 
          for ch in channels]
bars = ax2.bar(channels, rois, color=colors)
ax2.set_xlabel('Channel')
ax2.set_ylabel('ROI')
ax2.set_title('ROI by Channel (Green=Top, Red=Underperforming)')
ax2.set_xticklabels(channels, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("optimization_analysis.png", dpi=150, bbox_inches='tight')
print("\\nSaved optimization_analysis.png")

# Save optimization results
output = {{
    "current_metrics": current_metrics,
    "recommendations": recommendations,
    "total_budget": total_spend,
    "top_performers": top_performers,
    "underperformers": underperformers,
}}

with open("optimization_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\\nOptimization results saved to optimization_results.json")
'''

SENSITIVITY_TEMPLATE = '''
"""
Sensitivity Analysis
Session: {session_id}
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model info
with open("model_info.json", "r") as f:
    model_info = json.load(f)

channels = model_info["channels"]
contributions = model_info["contributions"]

# Sensitivity analysis: vary each channel ±50% and measure impact
sensitivity_range = np.linspace(-0.5, 0.5, 11)  # -50% to +50% in 10% increments

baseline_total = sum(contributions.get(ch, {{}}).get("contribution_mean", 0) for ch in channels)

sensitivity_results = {{}}

for channel in channels:
    channel_contrib = contributions.get(channel, {{}}).get("contribution_mean", 0)
    impacts = []
    
    for pct_change in sensitivity_range:
        # Simple linear sensitivity (actual should use response curves)
        # Apply diminishing returns for increases
        if pct_change > 0:
            saturation_factor = 1 - 0.2 * pct_change  # Simplified
            adjusted_change = pct_change * saturation_factor
        else:
            adjusted_change = pct_change
        
        new_contrib = channel_contrib * (1 + adjusted_change)
        new_total = baseline_total - channel_contrib + new_contrib
        impact_pct = (new_total - baseline_total) / baseline_total * 100
        impacts.append(impact_pct)
    
    sensitivity_results[channel] = impacts

# Create sensitivity plot
fig, ax = plt.subplots(figsize=(12, 8))

for channel in channels:
    ax.plot(sensitivity_range * 100, sensitivity_results[channel], 
            marker='o', label=channel, linewidth=2)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Budget Change (%)')
ax.set_ylabel('Impact on Total Outcome (%)')
ax.set_title('Sensitivity Analysis: Impact of Budget Changes by Channel')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensitivity_analysis.png", dpi=150, bbox_inches='tight')
print("Saved sensitivity_analysis.png")

# Calculate elasticities (at midpoint)
elasticities = {{}}
for channel in channels:
    # Approximate elasticity from +10% change
    idx_base = 5  # 0% change
    idx_plus10 = 6  # +10% change
    
    if len(sensitivity_results[channel]) > idx_plus10:
        impact = sensitivity_results[channel][idx_plus10]
        elasticity = impact / 10  # Impact % per 10% spend change
        elasticities[channel] = elasticity

print("\\nChannel Elasticities (% outcome change per 10% spend change):")
for ch, e in sorted(elasticities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {{ch}}: {{e:.3f}}")

# Save results
output = {{
    "sensitivity_range_pct": (sensitivity_range * 100).tolist(),
    "sensitivity_by_channel": sensitivity_results,
    "elasticities": elasticities,
    "baseline_total": baseline_total,
}}

with open("sensitivity_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\\nSensitivity results saved to sensitivity_results.json")
'''


# =============================================================================
# Workflow Nodes
# =============================================================================

async def initialize_whatif(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
) -> WhatIfWorkflowState:
    """Initialize what-if analysis workflow."""
    logger.info("Initializing what-if analysis workflow")
    
    session_id = state.get("session_id") or str(uuid.uuid4())
    analysis_id = state.get("analysis_id")
    
    # Verify model artifact exists
    model_path = state.get("model_artifact_path")
    if not model_path:
        return {
            **state,
            "session_id": session_id,
            "current_phase": WorkflowPhase.ERROR,
            "errors": ["No model artifact path provided"],
        }
    
    if not os.path.exists(model_path):
        return {
            **state,
            "session_id": session_id,
            "current_phase": WorkflowPhase.ERROR,
            "errors": [f"Model artifact not found: {model_path}"],
        }
    
    # Create analysis record if not exists
    if not analysis_id:
        analysis_id = str(uuid.uuid4())
        await neo4j.create_analysis(
            analysis_id=analysis_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            status="whatif_init",
        )
    else:
        await neo4j.update_analysis_status(analysis_id, "whatif_init")
    
    # Retrieve relevant context
    kg_context = await graphrag.get_phase_context(
        analysis_id=analysis_id,
        phase="interpretation",
        max_tokens=2000,
    )
    
    return {
        **state,
        "session_id": session_id,
        "analysis_id": analysis_id,
        "current_phase": WorkflowPhase.WHATIF_INIT,
        "kg_context": kg_context,
        "messages": ["What-if analysis initialized"],
    }


async def load_model_artifacts(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    code_executor: CodeExecutor,
) -> WhatIfWorkflowState:
    """Load model artifacts for scenario analysis."""
    logger.info("Loading model artifacts")
    
    session_id = state["session_id"]
    model_path = state["model_artifact_path"]
    
    # Generate load code
    load_code = LOAD_MODEL_TEMPLATE.format(
        session_id=session_id,
        model_artifact_path=model_path,
    )
    
    # Execute
    result = await code_executor.execute(load_code, session_id)
    
    if not result.success:
        return {
            **state,
            "current_phase": WorkflowPhase.ERROR,
            "errors": [f"Failed to load model: {result.error}"],
        }
    
    # Read model info
    try:
        model_info = code_executor.read_session_file(session_id, "model_info.json")
        model_info_dict = json.loads(model_info)
    except Exception as e:
        return {
            **state,
            "current_phase": WorkflowPhase.ERROR,
            "errors": [f"Failed to parse model info: {str(e)}"],
        }
    
    return {
        **state,
        "current_phase": WorkflowPhase.WHATIF_LOAD,
        "model_info": model_info_dict,
        "channels": model_info_dict.get("channels", []),
        "messages": ["Model artifacts loaded successfully"],
    }


async def parse_scenarios(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
) -> WhatIfWorkflowState:
    """Parse user scenario query into structured scenarios."""
    logger.info("Parsing scenario query")
    
    user_query = state.get("user_query", "")
    channels = state.get("channels", [])
    
    if not user_query:
        # No user query - generate default scenarios
        scenarios = [
            {
                "name": "10% Overall Increase",
                "description": "Increase all channel budgets by 10%",
                "budget_changes": [
                    {"channel": ch, "change_type": "percentage", "value": 10, "rationale": "Uniform increase"}
                    for ch in channels
                ],
                "assumptions": ["Linear response within 10% range"],
            },
            {
                "name": "10% Overall Decrease",
                "description": "Decrease all channel budgets by 10%",
                "budget_changes": [
                    {"channel": ch, "change_type": "percentage", "value": -10, "rationale": "Uniform decrease"}
                    for ch in channels
                ],
                "assumptions": ["Linear response within 10% range"],
            },
        ]
    else:
        # Use LLM to parse user query
        llm = create_ollama_llm(LLMTask.REASONING)
        
        context = f"""Available channels: {channels}
        
User query: {user_query}

Parse this into specific scenario definitions."""
        
        try:
            structured_llm = create_structured_llm(ParsedScenarios)
            response = await structured_llm.ainvoke(
                f"{SCENARIO_PARSING_PROMPT}\n\n{context}"
            )
            scenarios = [s.model_dump() for s in response.scenarios]
        except Exception as e:
            logger.warning(f"Failed to parse scenarios with LLM: {e}")
            # Fallback to default scenarios
            scenarios = [
                {
                    "name": "Custom Scenario",
                    "description": user_query,
                    "budget_changes": [],
                    "assumptions": ["Could not parse specific changes"],
                }
            ]
    
    return {
        **state,
        "current_phase": WorkflowPhase.WHATIF_SCENARIO,
        "scenarios": scenarios,
        "messages": [f"Parsed {len(scenarios)} scenarios"],
    }


async def run_scenario_analysis(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    code_executor: CodeExecutor,
) -> WhatIfWorkflowState:
    """Run analysis for each scenario."""
    logger.info("Running scenario analysis")
    
    session_id = state["session_id"]
    scenarios = state.get("scenarios", [])
    mff_data_path = state.get("mff_data_path", "")
    
    scenario_results = []
    generated_plots = state.get("generated_plots", [])
    
    for scenario in scenarios:
        scenario_name = scenario.get("name", "Unnamed")
        logger.info(f"Analyzing scenario: {scenario_name}")
        
        # Generate analysis code
        analysis_code = SCENARIO_ANALYSIS_TEMPLATE.format(
            session_id=session_id,
            scenario_name=scenario_name,
            scenario_json=json.dumps(scenario, indent=2),
            mff_data_path=mff_data_path,
        )
        
        # Execute
        result = await code_executor.execute(analysis_code, session_id)
        
        if result.success:
            # Read results
            try:
                result_json = code_executor.read_session_file(session_id, "scenario_result.json")
                result_dict = json.loads(result_json)
                
                scenario_results.append(ScenarioResult(
                    scenario_name=result_dict.get("scenario_name", scenario_name),
                    changes=result_dict.get("changes_applied", {}),
                    predicted_outcome=result_dict.get("scenario_outcome", 0),
                    incremental_impact=result_dict.get("incremental_impact", 0),
                    baseline_outcome=result_dict.get("baseline_outcome", 0),
                ))
                
                # Track plots
                if result.plots:
                    generated_plots.extend(result.plots)
                    
            except Exception as e:
                logger.warning(f"Failed to parse scenario result: {e}")
        else:
            logger.warning(f"Scenario analysis failed: {result.error}")
    
    return {
        **state,
        "current_phase": WorkflowPhase.WHATIF_ANALYZE,
        "scenario_results": scenario_results,
        "generated_plots": generated_plots,
        "messages": [f"Completed analysis for {len(scenario_results)} scenarios"],
    }


async def run_optimization_analysis(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
    code_executor: CodeExecutor,
) -> WhatIfWorkflowState:
    """Run budget optimization analysis."""
    logger.info("Running optimization analysis")
    
    session_id = state["session_id"]
    mff_data_path = state.get("mff_data_path", "")
    
    # Generate optimization code
    opt_code = OPTIMIZATION_TEMPLATE.format(
        session_id=session_id,
        mff_data_path=mff_data_path,
    )
    
    result = await code_executor.execute(opt_code, session_id)
    
    optimization_suggestions = []
    generated_plots = state.get("generated_plots", [])
    
    if result.success:
        try:
            opt_json = code_executor.read_session_file(session_id, "optimization_results.json")
            opt_dict = json.loads(opt_json)
            
            optimization_suggestions = opt_dict.get("recommendations", [])
            
            if result.plots:
                generated_plots.extend(result.plots)
                
        except Exception as e:
            logger.warning(f"Failed to parse optimization results: {e}")
    
    # Run sensitivity analysis
    sens_code = SENSITIVITY_TEMPLATE.format(
        session_id=session_id,
    )
    
    sens_result = await code_executor.execute(sens_code, session_id)
    
    sensitivity_results = {}
    if sens_result.success:
        try:
            sens_json = code_executor.read_session_file(session_id, "sensitivity_results.json")
            sensitivity_results = json.loads(sens_json)
            
            if sens_result.plots:
                generated_plots.extend(sens_result.plots)
                
        except Exception as e:
            logger.warning(f"Failed to parse sensitivity results: {e}")
    
    # Use LLM to generate insights
    llm = create_ollama_llm(LLMTask.REASONING)
    
    context = f"""Optimization Results:
{json.dumps(optimization_suggestions, indent=2)}

Sensitivity Analysis:
{json.dumps(sensitivity_results.get('elasticities', {}), indent=2)}
"""
    
    try:
        structured_llm = create_structured_llm(OptimizationSummary)
        summary = await structured_llm.ainvoke(
            f"{OPTIMIZATION_PROMPT}\n\n{context}"
        )
        optimization_summary = summary.model_dump()
    except Exception as e:
        logger.warning(f"Failed to generate optimization summary: {e}")
        optimization_summary = {
            "recommendations": optimization_suggestions,
            "key_insights": [],
            "caveats": ["Automated summary generation failed"],
        }
    
    return {
        **state,
        "optimization_suggestions": optimization_suggestions,
        "optimization_summary": optimization_summary,
        "sensitivity_results": sensitivity_results,
        "generated_plots": generated_plots,
        "messages": ["Optimization analysis complete"],
    }


async def generate_whatif_summary(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
) -> WhatIfWorkflowState:
    """Generate comprehensive what-if analysis summary."""
    logger.info("Generating what-if summary")
    
    scenario_results = state.get("scenario_results", [])
    optimization_summary = state.get("optimization_summary", {})
    sensitivity_results = state.get("sensitivity_results", {})
    
    # Build summary
    llm = create_ollama_llm(LLMTask.REASONING)
    
    context = f"""## Scenario Analysis Results

{json.dumps([s.model_dump() if hasattr(s, 'model_dump') else s for s in scenario_results], indent=2)}

## Optimization Recommendations

{json.dumps(optimization_summary, indent=2)}

## Sensitivity Analysis

Elasticities: {json.dumps(sensitivity_results.get('elasticities', {}), indent=2)}
"""
    
    prompt = f"""{WHATIF_SYSTEM_PROMPT}

Based on the following analysis results, provide a comprehensive executive summary with actionable recommendations:

{context}

Include:
1. Key findings from scenario analysis
2. Budget optimization recommendations
3. Risk assessment based on sensitivity
4. Specific next steps
"""
    
    try:
        response = await llm.ainvoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.warning(f"Failed to generate summary: {e}")
        summary = "Summary generation failed. Please review the individual analysis results."
    
    # Store in GraphRAG
    await graphrag.add_decision(
        analysis_id=state["analysis_id"],
        phase="whatif",
        decision_type="summary",
        decision="What-if analysis complete",
        rationale=summary[:500],
    )
    
    return {
        **state,
        "summary": summary,
        "messages": ["What-if analysis summary generated"],
    }


async def finalize_whatif(
    state: WhatIfWorkflowState,
    neo4j: Neo4jClient,
) -> WhatIfWorkflowState:
    """Finalize what-if workflow."""
    logger.info("Finalizing what-if workflow")
    
    await neo4j.update_analysis_status(
        state["analysis_id"],
        "whatif_complete"
    )
    
    return {
        **state,
        "current_phase": WorkflowPhase.WHATIF_COMPLETE,
        "messages": ["What-if analysis workflow complete"],
    }


# =============================================================================
# Workflow Graph Builder
# =============================================================================

def route_after_init(state: WhatIfWorkflowState) -> str:
    """Route after initialization."""
    if state.get("errors"):
        return "error"
    return "load"


def route_after_load(state: WhatIfWorkflowState) -> str:
    """Route after loading model."""
    if state.get("errors"):
        return "error"
    return "parse"


def route_after_parse(state: WhatIfWorkflowState) -> str:
    """Route after parsing scenarios."""
    if state.get("errors"):
        return "error"
    scenarios = state.get("scenarios", [])
    if not scenarios:
        return "optimize"  # Skip to optimization if no scenarios
    return "analyze"


def route_after_analyze(state: WhatIfWorkflowState) -> str:
    """Route after scenario analysis."""
    return "optimize"


def route_after_optimize(state: WhatIfWorkflowState) -> str:
    """Route after optimization."""
    return "summarize"


def route_after_summarize(state: WhatIfWorkflowState) -> str:
    """Route after summary generation."""
    return "complete"


class WhatIfWorkflow:
    """What-If Agent workflow for scenario analysis."""
    
    def __init__(
        self,
        neo4j: Neo4jClient,
        graphrag: GraphRAGManager,
        code_executor: CodeExecutor | None = None,
        checkpointer: AsyncPostgresSaver | None = None,
    ):
        self.neo4j = neo4j
        self.graphrag = graphrag
        self.code_executor = code_executor or CodeExecutor()
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow state graph."""
        workflow = StateGraph(WhatIfWorkflowState)
        
        # Add nodes
        workflow.add_node(
            "init",
            lambda s: asyncio.get_event_loop().run_until_complete(
                initialize_whatif(s, self.neo4j, self.graphrag)
            )
        )
        workflow.add_node(
            "load",
            lambda s: asyncio.get_event_loop().run_until_complete(
                load_model_artifacts(s, self.neo4j, self.code_executor)
            )
        )
        workflow.add_node(
            "parse",
            lambda s: asyncio.get_event_loop().run_until_complete(
                parse_scenarios(s, self.neo4j, self.graphrag)
            )
        )
        workflow.add_node(
            "analyze",
            lambda s: asyncio.get_event_loop().run_until_complete(
                run_scenario_analysis(s, self.neo4j, self.code_executor)
            )
        )
        workflow.add_node(
            "optimize",
            lambda s: asyncio.get_event_loop().run_until_complete(
                run_optimization_analysis(s, self.neo4j, self.graphrag, self.code_executor)
            )
        )
        workflow.add_node(
            "summarize",
            lambda s: asyncio.get_event_loop().run_until_complete(
                generate_whatif_summary(s, self.neo4j, self.graphrag)
            )
        )
        workflow.add_node(
            "complete",
            lambda s: asyncio.get_event_loop().run_until_complete(
                finalize_whatif(s, self.neo4j)
            )
        )
        workflow.add_node(
            "error",
            lambda s: {**s, "current_phase": WorkflowPhase.ERROR}
        )
        
        # Set entry point
        workflow.set_entry_point("init")
        
        # Add edges
        workflow.add_conditional_edges("init", route_after_init)
        workflow.add_conditional_edges("load", route_after_load)
        workflow.add_conditional_edges("parse", route_after_parse)
        workflow.add_conditional_edges("analyze", route_after_analyze)
        workflow.add_conditional_edges("optimize", route_after_optimize)
        workflow.add_conditional_edges("summarize", route_after_summarize)
        workflow.add_edge("complete", END)
        workflow.add_edge("error", END)
        
        return workflow
    
    def compile(self):
        """Compile the workflow graph."""
        if self.checkpointer:
            return self.graph.compile(checkpointer=self.checkpointer)
        return self.graph.compile()
    
    async def run(
        self,
        model_artifact_path: str,
        mff_data_path: str,
        user_query: str | None = None,
        analysis_id: str | None = None,
    ) -> WhatIfWorkflowState:
        """Run the what-if analysis workflow."""
        initial_state: WhatIfWorkflowState = {
            "model_artifact_path": model_artifact_path,
            "mff_data_path": mff_data_path,
            "user_query": user_query or "",
            "analysis_id": analysis_id,
            "scenarios": [],
            "scenario_results": [],
            "optimization_suggestions": [],
            "messages": [],
            "errors": [],
        }
        
        app = self.compile()
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        final_state = await app.ainvoke(initial_state, config)
        
        return final_state


async def create_whatif_workflow(
    neo4j_uri: str = settings.NEO4J_URI,
    neo4j_user: str = settings.NEO4J_USER,
    neo4j_password: str = settings.NEO4J_PASSWORD,
    postgres_url: str | None = settings.ASYNC_DATABASE_URL,
) -> WhatIfWorkflow:
    """Factory function to create what-if workflow with dependencies."""
    neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    graphrag = GraphRAGManager(neo4j)
    code_executor = CodeExecutor()
    
    checkpointer = None
    if postgres_url:
        checkpointer = AsyncPostgresSaver.from_conn_string(postgres_url)
        await checkpointer.setup()
    
    return WhatIfWorkflow(
        neo4j=neo4j,
        graphrag=graphrag,
        code_executor=code_executor,
        checkpointer=checkpointer,
    )
