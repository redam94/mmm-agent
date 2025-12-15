"""
Workflow 2: EDA Agent

An agentic workflow that takes the research plan and user data to:
- Load and validate data
- Run exploratory data analysis
- Clean and transform data
- Generate MFF (Master Flat File) format for mmm-framework

Features:
- Code execution for data analysis
- Automated quality checks
- Feature engineering recommendations
- GraphRAG storage of decisions
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from pydantic import BaseModel, Field

from ..config import Settings, get_settings, create_ollama_llm, LLMTask
from ..db import get_neo4j_client, get_graphrag_manager
from ..tools import CodeExecutor, get_code_executor, ExecutionResult
from .state import (
    EDAWorkflowState,
    WorkflowPhase,
    DataQualityIssue,
    FeatureTransformation,
)


# =============================================================================
# Structured Outputs
# =============================================================================

class DataLoadPlan(BaseModel):
    """Plan for loading data files."""
    files_to_load: list[str] = Field(description="Files to load")
    load_code: str = Field(description="Python code to load and merge data")
    expected_shape: str = Field(description="Expected data shape description")


class QualityCheckResult(BaseModel):
    """Result of data quality checks."""
    issues: list[DataQualityIssue] = Field(default_factory=list)
    summary: str = ""
    blocking_issues: bool = False
    recommendations: list[str] = Field(default_factory=list)


class TransformationPlan(BaseModel):
    """Feature transformation plan."""
    transformations: list[FeatureTransformation] = Field(default_factory=list)
    preprocessing_code: str = ""
    adstock_recommendations: dict = Field(default_factory=dict)
    saturation_recommendations: dict = Field(default_factory=dict)


class MFFOutput(BaseModel):
    """MFF generation output."""
    mff_path: str
    row_count: int
    column_count: int
    period_range: str
    variables: list[str]
    dimensions: list[str]


# =============================================================================
# Code Templates
# =============================================================================

DATA_LOAD_TEMPLATE = '''
import pandas as pd
import numpy as np
from pathlib import Path

# Load data files
{load_code}

# Basic info
print("=== DATA SUMMARY ===")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"\\nData types:\\n{{df.dtypes}}")
print(f"\\nFirst rows:\\n{{df.head()}}")

# Save merged data
df.to_csv("merged_data.csv", index=False)
print("\\nSaved merged data to merged_data.csv")
'''

QUALITY_CHECK_TEMPLATE = '''
import pandas as pd
import numpy as np
import json

df = pd.read_csv("merged_data.csv")

issues = []

# Check for missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
for col in df.columns:
    if missing[col] > 0:
        severity = "high" if missing_pct[col] > 20 else "medium" if missing_pct[col] > 5 else "low"
        issues.append({{
            "severity": severity,
            "variable": col,
            "issue_type": "missing_values",
            "description": f"{{missing_pct[col]}}% missing ({{missing[col]}} rows)",
            "recommendation": "Consider imputation or exclusion"
        }})

# Check for outliers (numeric columns)
for col in df.select_dtypes(include=[np.number]).columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3*iqr, q3 + 3*iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if outliers > len(df) * 0.01:
        issues.append({{
            "severity": "medium",
            "variable": col,
            "issue_type": "outliers",
            "description": f"{{outliers}} potential outliers ({{outliers/len(df)*100:.1f}}%)",
            "recommendation": "Review outlier handling strategy"
        }})

# Check for negative values in spend columns
spend_cols = [c for c in df.columns if 'spend' in c.lower() or 'cost' in c.lower()]
for col in spend_cols:
    if (df[col] < 0).any():
        issues.append({{
            "severity": "high",
            "variable": col,
            "issue_type": "negative_values",
            "description": "Contains negative values",
            "recommendation": "Investigate and correct negative spend values"
        }})

# Check date/period column
date_cols = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()]
for col in date_cols:
    try:
        dates = pd.to_datetime(df[col])
        date_range = (dates.max() - dates.min()).days
        if date_range < 365:
            issues.append({{
                "severity": "medium",
                "variable": col,
                "issue_type": "insufficient_history",
                "description": f"Only {{date_range}} days of data (recommend 2+ years)",
                "recommendation": "Consider gathering more historical data"
            }})
    except:
        pass

# Summary statistics
stats = df.describe().to_dict()

print("=== QUALITY CHECK RESULTS ===")
print(f"Total issues found: {{len(issues)}}")
print(f"High severity: {{sum(1 for i in issues if i['severity'] == 'high')}}")
print(f"Medium severity: {{sum(1 for i in issues if i['severity'] == 'medium')}}")
print(f"Low severity: {{sum(1 for i in issues if i['severity'] == 'low')}}")

# Save results
with open("quality_check.json", "w") as f:
    json.dump({{"issues": issues, "stats": stats}}, f, indent=2, default=str)
'''

EDA_ANALYSIS_TEMPLATE = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

df = pd.read_csv("merged_data.csv")

# Parse date column
date_cols = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()]
if date_cols:
    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
    df = df.sort_values(date_cols[0])

print("=== EDA ANALYSIS ===")

# 1. Summary statistics
print("\\n--- Summary Statistics ---")
print(df.describe())

# 2. Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv("correlation_matrix.csv")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title("Variable Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()
    print("\\nSaved correlation_heatmap.png")

# 3. Time series plots for key variables
{time_series_code}

# 4. Distribution analysis
{distribution_code}

# 5. Media spend analysis
spend_cols = [c for c in df.columns if 'spend' in c.lower() or 'cost' in c.lower() or 'media' in c.lower()]
if spend_cols:
    fig, axes = plt.subplots(len(spend_cols), 1, figsize=(12, 4*len(spend_cols)))
    if len(spend_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(spend_cols):
        if date_cols:
            axes[i].plot(df[date_cols[0]], df[col])
        else:
            axes[i].plot(df[col])
        axes[i].set_title(f"{{col}} Over Time")
        axes[i].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig("media_spend_trends.png", dpi=150)
    plt.close()
    print("\\nSaved media_spend_trends.png")

print("\\nEDA analysis complete!")
'''

MFF_GENERATION_TEMPLATE = '''
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("merged_data.csv")

# Configuration
period_col = "{period_column}"
geo_col = {geo_column}
product_col = {product_column}
kpi_col = "{kpi_column}"
media_cols = {media_columns}
control_cols = {control_columns}

print("=== MFF GENERATION ===")
print(f"Period column: {{period_col}}")
print(f"KPI column: {{kpi_col}}")
print(f"Media columns: {{media_cols}}")
print(f"Control columns: {{control_cols}}")

# Parse dates
df[period_col] = pd.to_datetime(df[period_col])

# Ensure weekly aggregation
df['Period'] = df[period_col].dt.to_period('W').dt.start_time

# Build dimension columns
dim_cols = ['Period']
if geo_col and geo_col in df.columns:
    df['Geography'] = df[geo_col]
    dim_cols.append('Geography')
else:
    df['Geography'] = 'National'
    dim_cols.append('Geography')

if product_col and product_col in df.columns:
    df['Product'] = df[product_col]
    dim_cols.append('Product')

# Aggregate to weekly level
agg_dict = {{kpi_col: 'sum'}}
for col in media_cols:
    if col in df.columns:
        agg_dict[col] = 'sum'
for col in control_cols:
    if col in df.columns:
        agg_dict[col] = 'mean'

# Group and aggregate
grouped = df.groupby(dim_cols).agg(agg_dict).reset_index()

# Rename KPI column
grouped = grouped.rename(columns={{kpi_col: 'KPI'}})

# Create MFF format (wide)
mff_df = grouped.copy()
mff_df['Period'] = mff_df['Period'].dt.strftime('%Y-%m-%d')

# Save MFF
mff_path = "mff_data.csv"
mff_df.to_csv(mff_path, index=False)

print(f"\\n=== MFF OUTPUT ===")
print(f"Shape: {{mff_df.shape}}")
print(f"Period range: {{mff_df['Period'].min()}} to {{mff_df['Period'].max()}}")
print(f"Variables: {{list(mff_df.columns)}}")
print(f"\\nFirst rows:\\n{{mff_df.head()}}")
print(f"\\nSaved to {{mff_path}}")

# Save metadata
import json
metadata = {{
    "mff_path": mff_path,
    "row_count": len(mff_df),
    "column_count": len(mff_df.columns),
    "period_range": f"{{mff_df['Period'].min()}} to {{mff_df['Period'].max()}}",
    "variables": list(mff_df.columns),
    "dimensions": dim_cols
}}
with open("mff_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
'''


# =============================================================================
# Workflow Nodes
# =============================================================================

async def initialize_eda(state: EDAWorkflowState) -> dict:
    """Initialize EDA workflow."""
    logger.info("📊 Initializing EDA workflow")
    
    session_id = state.get("session_id") or str(uuid.uuid4())[:8]
    analysis_id = state.get("analysis_id") or f"eda_{session_id}"
    
    # Copy data files to session directory
    executor = get_code_executor()
    session_dir = executor._get_session_dir(session_id)
    
    data_sources = state.get("data_sources", [])
    for source in data_sources:
        source_path = source.get("path")
        if source_path and Path(source_path).exists():
            executor.copy_file_to_session(session_id, source_path)
    
    return {
        "session_id": session_id,
        "analysis_id": analysis_id,
        "current_phase": WorkflowPhase.EDA_LOAD,
        "messages": [f"[{datetime.now().isoformat()}] EDA workflow initialized"],
    }


async def load_data(state: EDAWorkflowState) -> dict:
    """Load and merge data files."""
    logger.info("📂 Loading data files")
    
    session_id = state.get("session_id", "default")
    data_sources = state.get("data_sources", [])
    
    # Generate load code
    settings = get_settings()
    llm = create_ollama_llm(task=LLMTask.CODE, settings=settings)
    
    # Build source description
    sources_desc = "\n".join([
        f"- {s.get('name', 'unknown')}: {s.get('path', 'unknown')} ({s.get('description', '')})"
        for s in data_sources
    ])
    
    prompt = f"""Generate Python code to load and merge these data files:

{sources_desc}

Requirements:
- Load each file (handle CSV, Excel)
- Merge on common columns (dates, geography, etc.)
- Create a single DataFrame called 'df'
- Print summary information

Output only the Python code, no explanations.
"""
    
    try:
        response = llm.invoke([
            SystemMessage(content="Generate clean Python code for data loading. Output only code."),
            HumanMessage(content=prompt)
        ])
        
        load_code = response.content.strip()
        # Clean up code blocks if present
        if "```python" in load_code:
            load_code = load_code.split("```python")[1].split("```")[0]
        elif "```" in load_code:
            load_code = load_code.split("```")[1].split("```")[0]
        
    except Exception as e:
        logger.warning(f"LLM code generation failed: {e}, using default")
        # Default loading for single CSV
        load_code = """
files = list(Path(".").glob("*.csv"))
if files:
    df = pd.read_csv(files[0])
else:
    raise ValueError("No CSV files found")
"""
    
    # Execute load code
    executor = get_code_executor()
    full_code = DATA_LOAD_TEMPLATE.format(load_code=load_code)
    
    result = await executor.execute(full_code, session_id)
    
    if not result.success:
        return {
            "errors": [f"Data loading failed: {result.error}"],
            "current_phase": WorkflowPhase.ERROR,
        }
    
    return {
        "current_phase": WorkflowPhase.EDA_QUALITY,
        "messages": [f"[{datetime.now().isoformat()}] Data loaded: {result.stdout[:200]}"],
    }


async def run_quality_checks(state: EDAWorkflowState) -> dict:
    """Run data quality checks."""
    logger.info("🔍 Running quality checks")
    
    session_id = state.get("session_id", "default")
    executor = get_code_executor()
    
    result = await executor.execute(QUALITY_CHECK_TEMPLATE, session_id)
    
    if not result.success:
        logger.warning(f"Quality check had issues: {result.error}")
    
    # Read quality check results
    quality_data = executor.read_session_file(session_id, "quality_check.json")
    quality_report = {}
    quality_issues = []
    
    if quality_data:
        try:
            quality_report = json.loads(quality_data.decode())
            quality_issues = quality_report.get("issues", [])
        except:
            pass
    
    # Check for blocking issues
    blocking = any(i.get("severity") == "high" for i in quality_issues)
    
    return {
        "data_quality_report": quality_report,
        "quality_issues": quality_issues,
        "current_phase": WorkflowPhase.EDA_TRANSFORM,
        "messages": [f"[{datetime.now().isoformat()}] Quality check complete: {len(quality_issues)} issues found"],
    }


async def run_eda_analysis(state: EDAWorkflowState) -> dict:
    """Run exploratory data analysis."""
    logger.info("📈 Running EDA analysis")
    
    session_id = state.get("session_id", "default")
    research_plan = state.get("research_plan", {})
    
    # Get variable names from plan
    target = research_plan.get("target_variable", "revenue")
    media_channels = research_plan.get("media_channels", [])
    controls = research_plan.get("control_variables", [])
    
    # Generate time series code
    time_series_code = ""
    distribution_code = ""
    
    if target:
        time_series_code = f'''
# KPI time series
if "{target}" in df.columns and date_cols:
    plt.figure(figsize=(12, 4))
    plt.plot(df[date_cols[0]], df["{target}"])
    plt.title("{target} Over Time")
    plt.xlabel("Date")
    plt.ylabel("{target}")
    plt.tight_layout()
    plt.savefig("kpi_timeseries.png", dpi=150)
    plt.close()
    print("Saved kpi_timeseries.png")
'''
        
        distribution_code = f'''
# KPI distribution
if "{target}" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["{target}"], bins=30, edgecolor='black')
    axes[0].set_title("{target} Distribution")
    axes[1].boxplot(df["{target}"])
    axes[1].set_title("{target} Box Plot")
    plt.tight_layout()
    plt.savefig("kpi_distribution.png", dpi=150)
    plt.close()
    print("Saved kpi_distribution.png")
'''
    
    # Execute EDA
    executor = get_code_executor()
    eda_code = EDA_ANALYSIS_TEMPLATE.format(
        time_series_code=time_series_code,
        distribution_code=distribution_code,
    )
    
    result = await executor.execute(eda_code, session_id)
    
    # Collect generated plots
    plots = result.plots if result.plots else []
    
    # Read correlation matrix
    corr_data = executor.read_session_file(session_id, "correlation_matrix.csv")
    correlation_matrix = {}
    if corr_data:
        try:
            import pandas as pd
            import io
            corr_df = pd.read_csv(io.BytesIO(corr_data), index_col=0)
            correlation_matrix = corr_df.to_dict()
        except:
            pass
    
    return {
        "correlation_matrix": correlation_matrix,
        "generated_plots": plots,
        "current_phase": WorkflowPhase.EDA_OUTPUT,
        "messages": [f"[{datetime.now().isoformat()}] EDA analysis complete: {len(plots)} plots generated"],
    }


async def generate_mff(state: EDAWorkflowState) -> dict:
    """Generate MFF format data."""
    logger.info("📋 Generating MFF format")
    
    session_id = state.get("session_id", "default")
    research_plan = state.get("research_plan", {})
    data_sources = state.get("data_sources", [])
    
    # Get configuration
    target = research_plan.get("target_variable", "revenue")
    media_channels = research_plan.get("media_channels", [])
    controls = research_plan.get("control_variables", [])
    
    # Find date column (from first data source or default)
    period_column = "date"
    geo_column = "None"
    product_column = "None"
    
    if data_sources:
        first_source = data_sources[0]
        period_column = first_source.get("period_column", "date")
        geo_column = f'"{first_source.get("geography_column")}"' if first_source.get("geography_column") else "None"
        product_column = f'"{first_source.get("product_column")}"' if first_source.get("product_column") else "None"
    
    # Generate MFF code
    mff_code = MFF_GENERATION_TEMPLATE.format(
        period_column=period_column,
        geo_column=geo_column,
        product_column=product_column,
        kpi_column=target,
        media_columns=json.dumps(media_channels),
        control_columns=json.dumps(controls),
    )
    
    # Execute
    executor = get_code_executor()
    result = await executor.execute(mff_code, session_id)
    
    if not result.success:
        return {
            "errors": [f"MFF generation failed: {result.error}"],
            "current_phase": WorkflowPhase.ERROR,
        }
    
    # Read metadata
    metadata_data = executor.read_session_file(session_id, "mff_metadata.json")
    mff_metadata = {}
    if metadata_data:
        try:
            mff_metadata = json.loads(metadata_data.decode())
        except:
            pass
    
    # Get full path
    session_dir = executor._get_session_dir(session_id)
    mff_path = str(session_dir / "mff_data.csv")
    
    # Store decision in GraphRAG
    graphrag = get_graphrag_manager()
    graphrag.add_decision(
        analysis_id=state.get("analysis_id", ""),
        phase="eda",
        decision_type="mff_generation",
        content=mff_metadata,
        rationale=f"Generated MFF with {len(media_channels)} media channels",
    )
    
    return {
        "mff_data_path": mff_path,
        "cleaned_data_path": str(session_dir / "merged_data.csv"),
        "statistics_summary": mff_metadata,
        "current_phase": WorkflowPhase.EDA_COMPLETE,
        "messages": [f"[{datetime.now().isoformat()}] MFF generated: {mff_path}"],
    }


async def finalize_eda(state: EDAWorkflowState) -> dict:
    """Finalize EDA workflow."""
    logger.info("✅ Finalizing EDA workflow")
    
    # Generate recommendations based on quality issues
    quality_issues = state.get("quality_issues", [])
    
    recommendations = []
    for issue in quality_issues:
        if issue.get("recommendation"):
            recommendations.append(issue["recommendation"])
    
    # Add feature engineering recommendations
    research_plan = state.get("research_plan", {})
    media_channels = research_plan.get("media_channels", [])
    
    for channel in media_channels:
        recommendations.append(f"Apply geometric adstock to {channel} (recommended decay: 0.5-0.8)")
        recommendations.append(f"Apply Hill saturation to {channel}")
    
    return {
        "modeling_recommendations": recommendations,
        "current_phase": WorkflowPhase.EDA_COMPLETE,
        "messages": [f"[{datetime.now().isoformat()}] EDA workflow complete"],
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_load(state: EDAWorkflowState) -> str:
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "quality"


def route_after_quality(state: EDAWorkflowState) -> str:
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "analyze"


def route_after_analyze(state: EDAWorkflowState) -> str:
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "mff"


# =============================================================================
# Workflow Builder
# =============================================================================

class EDAWorkflow:
    """
    EDA Agent Workflow.
    
    Processes data through:
    - Loading and merging
    - Quality checks
    - Exploratory analysis
    - MFF generation
    """
    
    def __init__(
        self,
        settings: Settings | None = None,
        checkpointer: PostgresSaver | None = None,
    ):
        self.settings = settings or get_settings()
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
        self.compiled = None
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph."""
        graph = StateGraph(EDAWorkflowState)
        
        # Add nodes
        graph.add_node("init", initialize_eda)
        graph.add_node("load", load_data)
        graph.add_node("quality", run_quality_checks)
        graph.add_node("analyze", run_eda_analysis)
        graph.add_node("mff", generate_mff)
        graph.add_node("complete", finalize_eda)
        graph.add_node("error", lambda s: {"current_phase": WorkflowPhase.ERROR})
        
        # Add edges
        graph.set_entry_point("init")
        graph.add_edge("init", "load")
        graph.add_conditional_edges("load", route_after_load, {
            "quality": "quality",
            "error": "error",
        })
        graph.add_conditional_edges("quality", route_after_quality, {
            "analyze": "analyze",
            "error": "error",
        })
        graph.add_conditional_edges("analyze", route_after_analyze, {
            "mff": "mff",
            "error": "error",
        })
        graph.add_edge("mff", "complete")
        graph.add_edge("complete", END)
        graph.add_edge("error", END)
        
        return graph
    
    def compile(self) -> "EDAWorkflow":
        """Compile the workflow."""
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        return self
    
    async def run(
        self,
        data_sources: list[dict],
        research_plan: dict | None = None,
        analysis_id: str | None = None,
        config: dict | None = None,
    ) -> EDAWorkflowState:
        """
        Run the EDA workflow.
        
        Args:
            data_sources: List of data source specs
            research_plan: Research plan from Workflow 1
            analysis_id: Optional analysis ID
            config: LangGraph config
        
        Returns:
            Final workflow state with MFF path
        """
        if not self.compiled:
            self.compile()
        
        initial_state: EDAWorkflowState = {
            "data_sources": data_sources,
            "research_plan": research_plan or {},
            "analysis_id": analysis_id,
            "current_phase": WorkflowPhase.EDA_INIT,
            "messages": [],
            "errors": [],
            "quality_issues": [],
            "generated_plots": [],
            "modeling_recommendations": [],
        }
        
        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {
                "thread_id": f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        
        final_state = None
        async for event in self.compiled.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                final_state = state_update
                logger.debug(f"Node {node_name}: {state_update.get('current_phase')}")
        
        return final_state


# =============================================================================
# Factory Function
# =============================================================================

def create_eda_workflow(
    settings: Settings | None = None,
    postgres_url: str | None = None,
) -> EDAWorkflow:
    """Create an EDA workflow instance."""
    settings = settings or get_settings()
    
    checkpointer = None
    if postgres_url or settings.postgres_url:
        try:
            checkpointer = PostgresSaver.from_conn_string(
                postgres_url or settings.postgres_url
            )
        except Exception as e:
            logger.warning(f"Could not create checkpointer: {e}")
    
    workflow = EDAWorkflow(
        settings=settings,
        checkpointer=checkpointer,
    )
    
    return workflow.compile()
