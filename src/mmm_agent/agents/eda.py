"""
EDA Agent for MMM Workflow

Handles the second phase of the MMM workflow:
- Loading and validating data
- Data quality assessment
- Exploratory data analysis
- Feature engineering (adstock, saturation)
- Data harmonization to MFF format
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

class DataQualityCheck(BaseModel):
    """A data quality check result."""
    check_name: str
    passed: bool
    severity: str = Field(description="high, medium, low")
    message: str
    affected_variables: list[str] = Field(default_factory=list)
    recommendation: str = ""


class FeatureTransformation(BaseModel):
    """A feature transformation to apply."""
    variable: str
    transformation: str = Field(description="geometric_adstock, hill_saturation, etc.")
    parameters: dict[str, Any]
    rationale: str


class EDAOutput(BaseModel):
    """Complete output from EDA phase."""
    data_summary: dict[str, Any] = Field(description="Summary statistics")
    quality_checks: list[DataQualityCheck] = Field(description="Quality check results")
    recommended_transformations: list[FeatureTransformation] = Field(description="Recommended transformations")
    correlation_insights: list[str] = Field(description="Key correlation findings")
    concerns: list[str] = Field(description="Concerns or issues to address")
    ready_for_modeling: bool = Field(description="Whether data is ready for modeling")
    summary: str = Field(description="Summary of EDA findings")


# =============================================================================
# Code Generation Templates
# =============================================================================

DATA_LOADING_CODE = '''
import pandas as pd
import numpy as np
from pathlib import Path

# Load data files
data_files = {data_files}
dfs = {{}}

for name, path in data_files.items():
    try:
        if path.endswith('.csv'):
            dfs[name] = pd.read_csv(path)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            dfs[name] = pd.read_excel(path)
        elif path.endswith('.parquet'):
            dfs[name] = pd.read_parquet(path)
        print(f"Loaded {{name}}: {{dfs[name].shape}}")
        print(f"  Columns: {{list(dfs[name].columns)}}")
        print()
    except Exception as e:
        print(f"Failed to load {{name}}: {{e}}")

# Combine info
for name, df in dfs.items():
    print(f"\\n=== {{name}} ===")
    print(df.info())
    print("\\nFirst 5 rows:")
    print(df.head())
    print("\\nDescriptive stats:")
    print(df.describe())
'''


DATA_QUALITY_CODE = '''
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("{data_path}")

print("=== DATA QUALITY REPORT ===\\n")

# 1. Missing values
print("1. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({{"count": missing, "percent": missing_pct}})
missing_df = missing_df[missing_df["count"] > 0].sort_values("percent", ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("  No missing values found")
print()

# 2. Data types
print("2. DATA TYPES")
print(df.dtypes)
print()

# 3. Numeric column statistics
print("3. NUMERIC COLUMNS SUMMARY")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"  {{col}}:")
    print(f"    Range: [{{df[col].min():.2f}}, {{df[col].max():.2f}}]")
    print(f"    Mean: {{df[col].mean():.2f}}, Std: {{df[col].std():.2f}}")
    print(f"    Zeros: {{(df[col] == 0).sum()}} ({{(df[col] == 0).mean()*100:.1f}}%)")
    
    # Check for outliers (IQR method)
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        print(f"    Outliers: {{outliers}} ({{outliers/len(df)*100:.1f}}%)")
    print()

# 4. Date column check
print("4. DATE/PERIOD ANALYSIS")
date_cols = [col for col in df.columns if 'date' in col.lower() or 'period' in col.lower()]
for col in date_cols:
    try:
        dates = pd.to_datetime(df[col])
        print(f"  {{col}}:")
        print(f"    Range: {{dates.min()}} to {{dates.max()}}")
        print(f"    Unique periods: {{dates.nunique()}}")
        
        # Check for gaps
        if dates.nunique() > 1:
            date_diff = dates.sort_values().diff().dropna()
            modal_diff = date_diff.mode().iloc[0]
            gaps = (date_diff != modal_diff).sum()
            if gaps > 0:
                print(f"    WARNING: {{gaps}} date gaps detected (expected freq: {{modal_diff}})")
    except:
        print(f"  {{col}}: Could not parse as date")
    print()

# 5. Categorical columns
print("5. CATEGORICAL COLUMNS")
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if col not in date_cols:
        print(f"  {{col}}: {{df[col].nunique()}} unique values")
        if df[col].nunique() <= 10:
            print(f"    Values: {{df[col].unique().tolist()}}")
        print()

print("=== END QUALITY REPORT ===")
'''


EDA_VISUALIZATION_CODE = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("{data_path}")

# Identify columns
kpi_col = "{kpi_column}"
media_cols = {media_columns}
control_cols = {control_columns}
date_col = "{date_column}"

# Parse dates
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

# 1. Time series plot of KPI
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df[date_col], df[kpi_col], 'b-', linewidth=1.5)
ax.set_title(f'{{kpi_col}} Over Time')
ax.set_xlabel('Date')
ax.set_ylabel(kpi_col)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('01_kpi_timeseries')
plt.close()

# 2. Media spend time series
if media_cols:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in media_cols[:5]:  # Limit to 5 channels
        if col in df.columns:
            ax.plot(df[date_col], df[col], label=col, alpha=0.8)
    ax.set_title('Media Spend Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure('02_media_timeseries')
    plt.close()

# 3. Correlation heatmap
numeric_cols = [kpi_col] + [c for c in media_cols + control_cols if c in df.columns]
numeric_cols = [c for c in numeric_cols if c in df.columns and df[c].dtype in ['int64', 'float64']]

if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    save_figure('03_correlation_heatmap')
    plt.close()
    
    # Print correlations with KPI
    print("\\nCorrelations with", kpi_col)
    kpi_corr = corr_matrix[kpi_col].sort_values(ascending=False)
    print(kpi_corr)

# 4. Distribution of KPI
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[kpi_col], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Distribution of {{kpi_col}}')
axes[0].set_xlabel(kpi_col)
axes[0].set_ylabel('Frequency')

# QQ plot
from scipy import stats
stats.probplot(df[kpi_col].dropna(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')
plt.tight_layout()
save_figure('04_kpi_distribution')
plt.close()

# 5. Scatter plots: Media vs KPI
if media_cols:
    n_media = min(len(media_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(media_cols[:n_media]):
        if col in df.columns:
            axes[i].scatter(df[col], df[kpi_col], alpha=0.5, s=20)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(kpi_col)
            axes[i].set_title(f'{{col}} vs {{kpi_col}}')
            
            # Add trend line
            z = np.polyfit(df[col].fillna(0), df[kpi_col].fillna(0), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[col].min(), df[col].max(), 100)
            axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8)
    
    # Hide empty subplots
    for j in range(n_media, 6):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_figure('05_media_vs_kpi_scatter')
    plt.close()

print("\\n=== EDA Visualizations Complete ===")
print("Generated plots saved to working directory")
'''


# =============================================================================
# EDA Agent
# =============================================================================

EDA_SYSTEM_PROMPT = """You are an expert data analyst specializing in Marketing Mix Model (MMM) data preparation.

Your role is to:
1. Assess data quality and identify issues
2. Recommend data transformations (adstock, saturation)
3. Identify correlations and potential confounders
4. Prepare data for Bayesian MMM modeling

Key considerations:
- Media channels typically need adstock transformation (carryover effect)
- Saturation curves capture diminishing returns
- Watch for multicollinearity between media channels
- Control variables should be included to avoid omitted variable bias
- Look for seasonality patterns that need to be controlled

Be specific about issues found and provide actionable recommendations."""


class EDAAgent:
    """
    Agent for MMM exploratory data analysis.
    
    Responsibilities:
    - Load and validate data
    - Generate quality reports
    - Create visualizations
    - Recommend transformations
    """
    
    def __init__(self, llm, code_executor):
        """
        Initialize EDA agent.
        
        Args:
            llm: LangChain chat model
            code_executor: LocalCodeExecutor instance
        """
        self.llm = llm
        self.executor = code_executor
        self.structured_llm = llm.with_structured_output(EDAOutput)
    
    async def analyze(
        self,
        state: MMMWorkflowState,
        context: str = "",
        on_progress = None,
    ) -> dict[str, Any]:
        """
        Execute EDA phase.
        
        Args:
            state: Current workflow state
            context: RAG context
            on_progress: Progress callback
        
        Returns:
            Dict of state updates
        """
        workflow_id = state.get("workflow_id", "default")
        data_paths = state.get("data_paths", [])
        target_var = state.get("target_variable", "Revenue")
        media_channels = state.get("media_channels", [])
        control_vars = state.get("control_variables", [])
        
        if not data_paths:
            return {
                "error": "No data files provided",
                "current_phase": WorkflowPhase.ERROR,
            }
        
        plots = []
        quality_output = ""
        eda_output = ""
        
        # Step 1: Load and profile data
        if on_progress:
            await on_progress("EDA", "Loading and profiling data...")
        
        load_code = DATA_LOADING_CODE.format(
            data_files=str({f"file_{i}": p for i, p in enumerate(data_paths)})
        )
        
        result = await self.executor.execute(
            load_code,
            session_id=workflow_id,
        )
        
        if not result.success:
            logger.error(f"Data loading failed: {result.error}")
            return {
                "error": f"Failed to load data: {result.error}",
                "current_phase": WorkflowPhase.ERROR,
            }
        
        data_summary = result.stdout
        
        # Step 2: Data quality checks
        if on_progress:
            await on_progress("EDA", "Running data quality checks...")
        
        # Use first data file for detailed analysis
        primary_data = data_paths[0]
        
        quality_code = DATA_QUALITY_CODE.format(data_path=primary_data)
        
        result = await self.executor.execute(
            quality_code,
            session_id=workflow_id,
        )
        
        quality_output = result.stdout if result.success else f"Quality check failed: {result.error}"
        
        # Step 3: EDA Visualizations
        if on_progress:
            await on_progress("EDA", "Creating visualizations...")
        
        # Need to infer date column
        viz_code = EDA_VISUALIZATION_CODE.format(
            data_path=primary_data,
            kpi_column=target_var,
            media_columns=media_channels,
            control_columns=control_vars,
            date_column="Period",  # Assume standard name
        )
        
        result = await self.executor.execute(
            viz_code,
            session_id=workflow_id,
        )
        
        if result.success:
            plots = [f for f in result.generated_files if f.endswith('.png')]
            eda_output = result.stdout
        else:
            logger.warning(f"Visualization failed: {result.error}")
        
        # Step 4: LLM analysis of results
        if on_progress:
            await on_progress("EDA", "Analyzing findings...")
        
        analysis_prompt = f"""Analyze the following EDA results for an MMM:

## Data Loading Summary
{data_summary[:2000]}

## Data Quality Report
{quality_output[:2000]}

## EDA Output
{eda_output[:1000]}

## Planning Context
- Target variable: {target_var}
- Media channels: {media_channels}
- Control variables: {control_vars}

## Context
{context[:1000]}

Based on this analysis:
1. Assess data quality and identify issues
2. Recommend appropriate adstock/saturation transformations
3. Note any correlations or concerns
4. Determine if data is ready for modeling"""

        try:
            analysis = self.structured_llm.invoke([
                SystemMessage(content=EDA_SYSTEM_PROMPT),
                HumanMessage(content=analysis_prompt),
            ])
            
            logger.info(f"EDA complete: {len(analysis.quality_checks)} checks, "
                       f"{len(analysis.recommended_transformations)} transformations")
            
            return {
                "data_loaded": True,
                "raw_data_summary": {"text": data_summary[:5000]},
                "data_quality_report": {"text": quality_output},
                "data_quality_issues": [c.model_dump() for c in analysis.quality_checks if not c.passed],
                "correlation_matrix": {},  # Would need to extract from code output
                "feature_transformations": [t.model_dump() for t in analysis.recommended_transformations],
                "mff_data_path": primary_data,  # Would be processed path
                "eda_summary": analysis.summary,
                "eda_plots": plots,
                "current_phase": WorkflowPhase.MODELING if analysis.ready_for_modeling else WorkflowPhase.EDA,
                "prior_decisions": [Decision(
                    phase=WorkflowPhase.EDA,
                    decision_type="data_assessment",
                    content={
                        "quality_issues": len([c for c in analysis.quality_checks if not c.passed]),
                        "transformations": len(analysis.recommended_transformations),
                        "ready": analysis.ready_for_modeling,
                    },
                    rationale=analysis.summary,
                ).model_dump()],
            }
            
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            return {
                "error": str(e),
                "current_phase": WorkflowPhase.ERROR,
            }
    
    def analyze_sync(
        self,
        state: MMMWorkflowState,
        context: str = "",
    ) -> dict[str, Any]:
        """Synchronous analysis wrapper."""
        import asyncio
        return asyncio.run(self.analyze(state, context))


# =============================================================================
# LangGraph Node
# =============================================================================

async def eda_node(state: MMMWorkflowState, deps: dict) -> dict:
    """
    LangGraph node for EDA phase.
    
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
            "eda",
            state.get("user_query", ""),
        )
    
    agent = EDAAgent(llm, executor)
    return await agent.analyze(state, context, on_progress)
