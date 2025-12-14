"""
Interpretation Agent for MMM Workflow

Handles the fourth phase of the MMM workflow:
- ROI estimation and confidence intervals
- Budget optimization
- What-if scenario analysis
- Actionable recommendations
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from ..state import MMMWorkflowState, WorkflowPhase, Decision


# =============================================================================
# Structured Outputs
# =============================================================================

class ChannelROI(BaseModel):
    """ROI estimate for a channel."""
    channel: str
    roi_mean: float
    roi_lower: float
    roi_upper: float
    spend_total: float
    contribution_total: float
    efficiency_rank: int


class OptimalAllocation(BaseModel):
    """Optimal budget allocation."""
    channel: str
    current_spend: float
    optimal_spend: float
    change_pct: float
    expected_lift: float


class WhatIfScenario(BaseModel):
    """A what-if scenario result."""
    scenario_name: str
    description: str
    spend_changes: dict[str, float]
    expected_outcome_change: float
    expected_outcome_change_pct: float


class Recommendation(BaseModel):
    """An actionable recommendation."""
    priority: int = Field(ge=1, le=5)
    category: str = Field(description="budget, channel, timing, etc.")
    recommendation: str
    expected_impact: str
    confidence: str = Field(description="high, medium, low")


class InterpretationOutput(BaseModel):
    """Complete output from interpretation phase."""
    channel_rois: list[ChannelROI] = Field(description="ROI by channel")
    optimal_allocation: list[OptimalAllocation] = Field(description="Optimal budget allocation")
    what_if_scenarios: list[WhatIfScenario] = Field(description="Scenario analyses")
    key_insights: list[str] = Field(description="Key business insights")
    recommendations: list[Recommendation] = Field(description="Actionable recommendations")
    executive_summary: str = Field(description="Executive summary for stakeholders")


# =============================================================================
# Interpretation Code Templates
# =============================================================================

ROI_ANALYSIS_CODE = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("{data_path}")

kpi_col = "{kpi_column}"
media_cols = {media_columns}

print("=== ROI ANALYSIS ===")
print()

# Calculate simple ROI metrics
# Note: In a real implementation, this would use the model posterior

roi_results = []
spend_totals = {{}}
contribution_estimates = {{}}

for col in media_cols:
    if col in df.columns:
        spend = df[col].sum()
        spend_totals[col] = spend
        
        # Simple correlation-based contribution estimate
        # In real implementation: use posterior channel contributions
        corr = df[col].corr(df[kpi_col])
        contribution_estimate = abs(corr) * df[kpi_col].sum() * 0.1  # Rough estimate
        contribution_estimates[col] = contribution_estimate
        
        roi = (contribution_estimate - spend) / spend if spend > 0 else 0
        
        roi_results.append({{
            'channel': col,
            'spend': spend,
            'estimated_contribution': contribution_estimate,
            'roi': roi
        }})
        
        print(f"{{col}}:")
        print(f"  Total Spend: ${{spend:,.0f}}")
        print(f"  Est. Contribution: ${{contribution_estimate:,.0f}}")
        print(f"  ROI: {{roi:.2f}}")
        print()

# Create ROI comparison chart
roi_df = pd.DataFrame(roi_results).sort_values('roi', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(roi_df['channel'], roi_df['roi'], color=['green' if x > 0 else 'red' for x in roi_df['roi']])
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('ROI')
ax.set_title('Channel ROI Comparison')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, roi_df['roi']):
    ax.text(val + 0.05 if val >= 0 else val - 0.15, bar.get_y() + bar.get_height()/2, 
            f'{{val:.2f}}', va='center')

plt.tight_layout()
save_figure('roi_comparison')
plt.close()

print("ROI chart saved")
'''


OPTIMIZATION_CODE = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data and configuration
df = pd.read_csv("{data_path}")
media_cols = {media_columns}
total_budget = {total_budget}

print("=== BUDGET OPTIMIZATION ===")
print(f"Total Budget to Allocate: ${{total_budget:,.0f}}")
print()

# Current allocation
current_spend = {{col: df[col].sum() for col in media_cols if col in df.columns}}
current_total = sum(current_spend.values())

print("Current Allocation:")
for col, spend in current_spend.items():
    pct = spend / current_total * 100 if current_total > 0 else 0
    print(f"  {{col}}: ${{spend:,.0f}} ({{pct:.1f}}%)")

# Simple optimization based on response curves
# In real implementation: use posterior samples and response curves

# Assume diminishing returns - allocate proportional to sqrt of historical effectiveness
effectiveness_scores = {{}}
for col in media_cols:
    if col in df.columns:
        corr = abs(df[col].corr(df['{kpi_column}']))
        effectiveness_scores[col] = np.sqrt(corr)

total_score = sum(effectiveness_scores.values())

optimal_allocation = {{}}
for col, score in effectiveness_scores.items():
    optimal_allocation[col] = total_budget * (score / total_score) if total_score > 0 else total_budget / len(media_cols)

print()
print("Optimal Allocation:")
for col, optimal in optimal_allocation.items():
    current = current_spend.get(col, 0)
    change_pct = ((optimal - current) / current * 100) if current > 0 else 0
    change_indicator = "↑" if optimal > current else "↓" if optimal < current else "="
    print(f"  {{col}}: ${{optimal:,.0f}} ({{change_indicator}} {{abs(change_pct):.1f}}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Current vs Optimal
channels = list(optimal_allocation.keys())
x = np.arange(len(channels))
width = 0.35

current_values = [current_spend.get(c, 0) for c in channels]
optimal_values = [optimal_allocation[c] for c in channels]

axes[0].bar(x - width/2, current_values, width, label='Current', color='steelblue')
axes[0].bar(x + width/2, optimal_values, width, label='Optimal', color='forestgreen')
axes[0].set_xlabel('Channel')
axes[0].set_ylabel('Budget ($)')
axes[0].set_title('Current vs Optimal Budget Allocation')
axes[0].set_xticks(x)
axes[0].set_xticklabels(channels, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart of optimal allocation
axes[1].pie(optimal_values, labels=channels, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Optimal Budget Distribution')

plt.tight_layout()
save_figure('budget_optimization')
plt.close()

print()
print("Optimization charts saved")
'''


SCENARIO_ANALYSIS_CODE = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("{data_path}")
kpi_col = "{kpi_column}"
media_cols = {media_columns}

print("=== WHAT-IF SCENARIO ANALYSIS ===")
print()

baseline_kpi = df[kpi_col].mean()
print(f"Baseline Average {{kpi_col}}: ${{baseline_kpi:,.0f}}")
print()

# Define scenarios
scenarios = [
    {{"name": "10% Budget Increase", "changes": {{c: 1.10 for c in media_cols}}}},
    {{"name": "20% Budget Cut", "changes": {{c: 0.80 for c in media_cols}}}},
    {{"name": "Shift to Digital", "changes": {shift_to_digital}}},
    {{"name": "Double Top Performer", "changes": {double_top}}},
]

# Simple elasticity-based impact estimation
# In real implementation: use posterior predictive samples

results = []
for scenario in scenarios:
    # Estimate impact based on simple elasticity
    total_change = 0
    for col, multiplier in scenario["changes"].items():
        if col in df.columns:
            # Assume 0.1-0.3 elasticity
            elasticity = abs(df[col].corr(df[kpi_col])) * 0.3
            spend_change = (multiplier - 1)
            impact = baseline_kpi * elasticity * spend_change
            total_change += impact
    
    change_pct = total_change / baseline_kpi * 100 if baseline_kpi > 0 else 0
    
    results.append({{
        "scenario": scenario["name"],
        "kpi_change": total_change,
        "change_pct": change_pct
    }})
    
    print(f"{{scenario['name']}}:")
    print(f"  Expected {{kpi_col}} Change: ${{total_change:,.0f}} ({{change_pct:+.1f}}%)")
    print()

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

scenario_names = [r["scenario"] for r in results]
changes = [r["kpi_change"] for r in results]
colors = ['green' if c > 0 else 'red' for c in changes]

bars = ax.barh(scenario_names, changes, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel(f'{{kpi_col}} Change ($)')
ax.set_title('What-If Scenario Analysis')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val, r in zip(bars, changes, results):
    label = f"${{val:,.0f}} ({{r['change_pct']:+.1f}}%)"
    ax.text(val + (max(changes) - min(changes)) * 0.02 if val >= 0 else val - (max(changes) - min(changes)) * 0.02,
            bar.get_y() + bar.get_height()/2, label, va='center', 
            ha='left' if val >= 0 else 'right')

plt.tight_layout()
save_figure('scenario_analysis')
plt.close()

print("Scenario analysis chart saved")
'''


# =============================================================================
# Interpretation Agent
# =============================================================================

INTERPRETATION_SYSTEM_PROMPT = """You are an expert Marketing Mix Model analyst providing business insights.

Your role is to:
1. Translate model results into business-relevant insights
2. Calculate and explain ROI with uncertainty
3. Provide budget optimization recommendations
4. Run what-if scenarios
5. Create executive-ready summaries

Key principles:
- Focus on actionable recommendations
- Quantify expected impact where possible
- Acknowledge uncertainty in estimates
- Consider business constraints (contracts, minimum spend)
- Frame insights in business language, not statistical jargon

Provide clear, concise insights that non-technical stakeholders can act on."""


class InterpretationAgent:
    """
    Agent for MMM interpretation phase.
    
    Responsibilities:
    - Calculate ROI metrics
    - Optimize budget allocation
    - Run scenario analyses
    - Generate recommendations
    """
    
    def __init__(self, llm, code_executor):
        """
        Initialize interpretation agent.
        
        Args:
            llm: LangChain chat model
            code_executor: LocalCodeExecutor instance
        """
        self.llm = llm
        self.executor = code_executor
        self.structured_llm = llm.with_structured_output(InterpretationOutput)
    
    async def interpret(
        self,
        state: MMMWorkflowState,
        context: str = "",
        on_progress = None,
    ) -> dict[str, Any]:
        """
        Execute interpretation phase.
        
        Args:
            state: Current workflow state
            context: RAG context
            on_progress: Progress callback
        
        Returns:
            Dict of state updates
        """
        workflow_id = state.get("workflow_id", "default")
        data_path = state.get("mff_data_path") or (state.get("data_paths", [None])[0])
        target_var = state.get("target_variable", "Revenue")
        media_channels = state.get("media_channels", [])
        model_summary = state.get("model_summary", "")
        
        if not data_path:
            return {
                "error": "No data available for interpretation",
                "current_phase": WorkflowPhase.ERROR,
            }
        
        all_outputs = []
        all_plots = []
        
        # Step 1: ROI Analysis
        if on_progress:
            await on_progress("Interpretation", "Calculating channel ROI...")
        
        roi_code = ROI_ANALYSIS_CODE.format(
            data_path=data_path,
            kpi_column=target_var,
            media_columns=media_channels,
        )
        
        result = await self.executor.execute(roi_code, session_id=workflow_id)
        all_outputs.append(f"## ROI Analysis\n{result.stdout}")
        all_plots.extend([f for f in result.generated_files if f.endswith('.png')])
        
        # Step 2: Budget Optimization
        if on_progress:
            await on_progress("Interpretation", "Optimizing budget allocation...")
        
        opt_code = OPTIMIZATION_CODE.format(
            data_path=data_path,
            kpi_column=target_var,
            media_columns=media_channels,
            total_budget=1000000,  # Example budget
        )
        
        result = await self.executor.execute(opt_code, session_id=workflow_id)
        all_outputs.append(f"## Budget Optimization\n{result.stdout}")
        all_plots.extend([f for f in result.generated_files if f.endswith('.png')])
        
        # Step 3: Scenario Analysis
        if on_progress:
            await on_progress("Interpretation", "Running scenario analysis...")
        
        # Build scenario changes
        shift_to_digital = {c: (1.2 if 'digital' in c.lower() or 'social' in c.lower() or 'search' in c.lower() else 0.9) for c in media_channels}
        double_top = {c: (2.0 if i == 0 else 1.0) for i, c in enumerate(media_channels)}
        
        scenario_code = SCENARIO_ANALYSIS_CODE.format(
            data_path=data_path,
            kpi_column=target_var,
            media_columns=media_channels,
            shift_to_digital=shift_to_digital,
            double_top=double_top,
        )
        
        result = await self.executor.execute(scenario_code, session_id=workflow_id)
        all_outputs.append(f"## Scenario Analysis\n{result.stdout}")
        all_plots.extend([f for f in result.generated_files if f.endswith('.png')])
        
        # Step 4: LLM interpretation
        if on_progress:
            await on_progress("Interpretation", "Generating insights and recommendations...")
        
        combined_output = "\n\n".join(all_outputs)
        
        interpretation_prompt = f"""Analyze these MMM results and provide business insights:

## Model Summary
{model_summary[:1500]}

## Analysis Results
{combined_output[:4000]}

## Context
- Target KPI: {target_var}
- Media Channels: {media_channels}

Provide:
1. Channel-by-channel ROI assessment
2. Optimal budget allocation recommendations
3. Key insights from scenario analysis
4. Prioritized recommendations for stakeholders
5. Executive summary (2-3 paragraphs)"""

        try:
            interpretation = self.structured_llm.invoke([
                SystemMessage(content=INTERPRETATION_SYSTEM_PROMPT),
                HumanMessage(content=interpretation_prompt),
            ])
            
            logger.info(f"Interpretation complete: {len(interpretation.recommendations)} recommendations")
            
            return {
                "roi_estimates": {r.channel: {"roi": r.roi_mean, "spend": r.spend_total} 
                                 for r in interpretation.channel_rois},
                "budget_allocation": {a.channel: a.optimal_spend 
                                     for a in interpretation.optimal_allocation},
                "what_if_scenarios": [s.model_dump() for s in interpretation.what_if_scenarios],
                "key_insights": interpretation.key_insights,
                "recommendations": [r.model_dump() for r in interpretation.recommendations],
                "interpretation_summary": interpretation.executive_summary,
                "interpretation_plots": all_plots,
                "current_phase": WorkflowPhase.COMPLETE,
                "prior_decisions": [Decision(
                    phase=WorkflowPhase.INTERPRETATION,
                    decision_type="final_recommendations",
                    content={
                        "n_recommendations": len(interpretation.recommendations),
                        "n_insights": len(interpretation.key_insights),
                    },
                    rationale=interpretation.executive_summary[:500],
                ).model_dump()],
            }
            
        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            
            # Fallback to raw output
            return {
                "interpretation_summary": combined_output[:3000],
                "interpretation_plots": all_plots,
                "key_insights": ["See analysis output for details"],
                "recommendations": [],
                "current_phase": WorkflowPhase.COMPLETE,
            }
    
    def interpret_sync(
        self,
        state: MMMWorkflowState,
        context: str = "",
    ) -> dict[str, Any]:
        """Synchronous interpretation wrapper."""
        import asyncio
        return asyncio.run(self.interpret(state, context))


# =============================================================================
# LangGraph Node
# =============================================================================

async def interpretation_node(state: MMMWorkflowState, deps: dict) -> dict:
    """
    LangGraph node for interpretation phase.
    
    Args:
        state: Workflow state
        deps: Dependencies (llm, executor, context_manager)
    
    Returns:
        State updates
    """
    llm = deps.get("llm")
    executor = deps.get("executor")
    context_manager = deps.get("context_manager")
    on_progress = deps.get("on_progress")
    
    # Get context
    context = ""
    if context_manager:
        context = context_manager.get_context(
            state.get("workflow_id", "default"),
            "interpretation",
            state.get("user_query", ""),
        )
    
    agent = InterpretationAgent(llm, executor)
    return await agent.interpret(state, context, on_progress)
