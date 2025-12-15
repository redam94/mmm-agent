"""
Modeling Agent Workflow

Workflow 3: Bayesian MMM fitting and interpretation using PyMC-Marketing.

This workflow:
1. Loads MFF-formatted data from EDA workflow
2. Configures model based on research plan and feature transformations
3. Fits the Bayesian MMM with progress tracking
4. Runs convergence diagnostics
5. Extracts channel contributions and ROI estimates
6. Generates interpretation and visualizations
7. Stores model artifacts with lineage in Neo4j

Uses:
- qwen3:30b for reasoning and interpretation
- qwen3-coder:30b for code generation
- Neo4j for storing model artifacts and lineage
- GraphRAG for retrieving modeling patterns and best practices
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
    ModelingWorkflowState,
    WorkflowPhase,
    ChannelContribution,
    ModelDiagnostics,
)


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts
# =============================================================================

MODELING_SYSTEM_PROMPT = """You are an expert Marketing Mix Modeling (MMM) practitioner specializing in Bayesian inference with PyMC-Marketing.

Your expertise includes:
- Configuring adstock and saturation transformations
- Setting informative priors based on domain knowledge
- Interpreting MCMC diagnostics (R-hat, ESS, divergences)
- Extracting meaningful business insights from posterior distributions
- Channel contribution decomposition and ROI estimation

When configuring models:
- Use geometric adstock for TV, radio, print (longer decay)
- Use shorter decay for digital channels (social, search)
- Apply Hill saturation for diminishing returns
- Set sensible priors based on typical advertising response

When interpreting results:
- Focus on uncertainty quantification (credible intervals)
- Compare channel effectiveness using comparable metrics
- Identify optimization opportunities
- Highlight caveats and limitations
"""

MODEL_CONFIG_PROMPT = """Based on the research plan and data characteristics, generate a complete model configuration.

Consider:
1. Target variable and transformation (log, none)
2. Media channels with appropriate adstock parameters
3. Control variables
4. Prior specifications
5. MCMC sampler settings (draws, tune, chains)

Return a well-structured configuration that balances model complexity with data availability.
"""

INTERPRETATION_PROMPT = """You are interpreting the results of a Bayesian Marketing Mix Model.

Given the model outputs (channel contributions, ROI estimates, diagnostics), provide:
1. Executive summary of key findings
2. Channel-by-channel analysis with business implications
3. Recommendations for budget allocation
4. Caveats and limitations
5. Next steps for optimization

Be specific with numbers and use appropriate uncertainty language (e.g., "95% credible interval").
"""


# =============================================================================
# Structured Output Models
# =============================================================================

class AdstockConfig(BaseModel):
    """Adstock transformation configuration."""
    type: str = Field(description="geometric, weibull_cdf, weibull_pdf")
    alpha: float = Field(default=0.5, description="Decay rate (0-1)")
    l_max: int = Field(default=8, description="Maximum lag periods")
    alpha_prior_mu: float = Field(default=0.5)
    alpha_prior_sigma: float = Field(default=0.2)


class SaturationConfig(BaseModel):
    """Saturation transformation configuration."""
    type: str = Field(description="hill, logistic, michaelis_menten")
    lam_prior_mu: float = Field(default=1.0)
    lam_prior_sigma: float = Field(default=0.5)
    k_prior_mu: float = Field(default=0.5)
    k_prior_sigma: float = Field(default=0.2)


class ChannelConfig(BaseModel):
    """Configuration for a media channel."""
    name: str
    adstock: AdstockConfig
    saturation: SaturationConfig


class ModelConfiguration(BaseModel):
    """Complete model configuration."""
    target_variable: str = Field(description="Target variable name in MFF")
    target_transformation: str = Field(default="log", description="none, log, sqrt")
    
    media_channels: list[ChannelConfig] = Field(default_factory=list)
    control_variables: list[str] = Field(default_factory=list)
    
    date_column: str = Field(default="Period")
    
    # Sampler settings
    draws: int = Field(default=1000)
    tune: int = Field(default=500)
    chains: int = Field(default=4)
    target_accept: float = Field(default=0.9)
    
    # Priors
    intercept_prior_mu: float = Field(default=0.0)
    intercept_prior_sigma: float = Field(default=2.0)
    
    rationale: str = ""


class InterpretationOutput(BaseModel):
    """Model interpretation output."""
    executive_summary: str
    channel_insights: list[str]
    top_performing_channels: list[str]
    underperforming_channels: list[str]
    budget_recommendations: list[str]
    caveats: list[str]
    next_steps: list[str]


# =============================================================================
# Code Templates
# =============================================================================

MODEL_FIT_TEMPLATE = '''
"""
MMM Model Fitting Script
Generated for session: {session_id}
"""

import json
import pickle
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {config_json}

# Load data
print("Loading MFF data...")
data = pd.read_csv("{mff_data_path}")
print(f"Data shape: {{data.shape}}")
print(f"Columns: {{list(data.columns)}}")

# Prepare data
target = CONFIG["target_variable"]
media_channels = [ch["name"] for ch in CONFIG["media_channels"]]
control_vars = CONFIG["control_variables"]
date_col = CONFIG["date_column"]

# Ensure all columns exist
missing_cols = [c for c in [target] + media_channels + control_vars if c not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {{missing_cols}}")

# Sort by date
if date_col in data.columns:
    data = data.sort_values(date_col).reset_index(drop=True)

# Extract arrays
y = data[target].values
X_media = data[media_channels].values
X_control = data[control_vars].values if control_vars else None

n_obs = len(y)
n_media = len(media_channels)
n_control = len(control_vars) if control_vars else 0

print(f"Observations: {{n_obs}}")
print(f"Media channels: {{n_media}}")
print(f"Control variables: {{n_control}}")

# Target transformation
if CONFIG.get("target_transformation") == "log":
    y_model = np.log1p(y)
    print("Applied log transformation to target")
else:
    y_model = y.copy()

# Define adstock function
def geometric_adstock(x, alpha, l_max):
    """Apply geometric adstock transformation."""
    weights = np.array([alpha ** i for i in range(l_max)])
    weights = weights / weights.sum()
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        for j in range(min(i + 1, l_max)):
            result[i] += weights[j] * x[i - j]
    return result

# Define saturation function (Hill)
def hill_saturation(x, k, n):
    """Apply Hill saturation transformation."""
    return x ** n / (k ** n + x ** n)

# Build PyMC model
print("\\nBuilding PyMC model...")
with pm.Model() as mmm_model:
    # Intercept
    intercept = pm.Normal(
        "intercept",
        mu=CONFIG.get("intercept_prior_mu", 0),
        sigma=CONFIG.get("intercept_prior_sigma", 2)
    )
    
    # Media effects
    media_contributions = []
    for i, ch_config in enumerate(CONFIG["media_channels"]):
        ch_name = ch_config["name"]
        x_ch = X_media[:, i]
        
        # Adstock parameters
        adstock_cfg = ch_config.get("adstock", {{}})
        alpha_mu = adstock_cfg.get("alpha_prior_mu", 0.5)
        alpha_sigma = adstock_cfg.get("alpha_prior_sigma", 0.2)
        l_max = adstock_cfg.get("l_max", 8)
        
        alpha = pm.Beta(f"alpha_{{ch_name}}", mu=alpha_mu, sigma=alpha_sigma)
        
        # Saturation parameters
        sat_cfg = ch_config.get("saturation", {{}})
        lam = pm.HalfNormal(f"lam_{{ch_name}}", sigma=sat_cfg.get("lam_prior_sigma", 0.5))
        
        # Apply transformations (simplified for numpy)
        # In practice, use PyMC-Marketing's built-in transforms
        x_adstocked = geometric_adstock(x_ch, alpha_mu, l_max)  # Use prior mean for transform
        x_saturated = x_adstocked / (1 + x_adstocked / x_adstocked.max())  # Simple saturation
        
        # Channel coefficient
        beta = pm.HalfNormal(f"beta_{{ch_name}}", sigma=1)
        
        contribution = beta * x_saturated
        media_contributions.append(contribution)
    
    # Sum media contributions
    total_media = pm.math.sum(media_contributions, axis=0) if media_contributions else 0
    
    # Control effects
    if n_control > 0:
        beta_control = pm.Normal("beta_control", mu=0, sigma=1, shape=n_control)
        control_effect = pm.math.dot(X_control, beta_control)
    else:
        control_effect = 0
    
    # Expected value
    mu = intercept + total_media + control_effect
    
    # Likelihood
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_model)

# Fit model
print("\\nFitting model (this may take several minutes)...")
print(f"Settings: draws={{CONFIG['draws']}}, tune={{CONFIG['tune']}}, chains={{CONFIG['chains']}}")

with mmm_model:
    trace = pm.sample(
        draws=CONFIG["draws"],
        tune=CONFIG["tune"],
        chains=CONFIG["chains"],
        target_accept=CONFIG.get("target_accept", 0.9),
        return_inferencedata=True,
        progressbar=True
    )

print("\\nModel fitting complete!")

# Diagnostics
print("\\nComputing diagnostics...")
summary = az.summary(trace)
print(summary)

# Extract diagnostics
rhat_values = summary["r_hat"].values
ess_bulk_values = summary["ess_bulk"].values
ess_tail_values = summary["ess_tail"].values

diagnostics = {{
    "rhat_max": float(np.nanmax(rhat_values)),
    "rhat_mean": float(np.nanmean(rhat_values)),
    "ess_bulk_min": float(np.nanmin(ess_bulk_values)),
    "ess_tail_min": float(np.nanmin(ess_tail_values)),
    "divergences": int(trace.sample_stats.diverging.sum().values),
    "n_parameters": len(summary)
}}

# Convergence assessment
if diagnostics["rhat_max"] < 1.01 and diagnostics["divergences"] == 0:
    diagnostics["convergence_status"] = "good"
elif diagnostics["rhat_max"] < 1.05 and diagnostics["divergences"] < 10:
    diagnostics["convergence_status"] = "acceptable"
else:
    diagnostics["convergence_status"] = "poor"

print(f"\\nConvergence status: {{diagnostics['convergence_status']}}")
print(f"Max R-hat: {{diagnostics['rhat_max']:.4f}}")
print(f"Divergences: {{diagnostics['divergences']}}")

# Extract channel contributions
print("\\nExtracting channel contributions...")
contributions = []
total_spend = X_media.sum()

for i, ch_config in enumerate(CONFIG["media_channels"]):
    ch_name = ch_config["name"]
    beta_samples = trace.posterior[f"beta_{{ch_name}}"].values.flatten()
    
    contrib_mean = float(np.mean(beta_samples))
    contrib_lower = float(np.percentile(beta_samples, 2.5))
    contrib_upper = float(np.percentile(beta_samples, 97.5))
    
    # Calculate ROI (contribution / spend)
    ch_spend = X_media[:, i].sum()
    if ch_spend > 0:
        roi_mean = contrib_mean / ch_spend * total_spend
        roi_lower = contrib_lower / ch_spend * total_spend
        roi_upper = contrib_upper / ch_spend * total_spend
    else:
        roi_mean = roi_lower = roi_upper = 0.0
    
    contributions.append({{
        "channel": ch_name,
        "contribution_mean": contrib_mean,
        "contribution_lower": contrib_lower,
        "contribution_upper": contrib_upper,
        "roi_mean": roi_mean,
        "roi_lower": roi_lower,
        "roi_upper": roi_upper,
        "spend_total": float(ch_spend)
    }})

# Calculate shares
total_contribution = sum(c["contribution_mean"] for c in contributions)
for c in contributions:
    c["share_of_contribution"] = c["contribution_mean"] / total_contribution if total_contribution > 0 else 0

# Sort by contribution
contributions.sort(key=lambda x: x["contribution_mean"], reverse=True)

print("\\nChannel contributions:")
for c in contributions:
    print(f"  {{c['channel']}}: {{c['contribution_mean']:.4f}} ({{c['share_of_contribution']*100:.1f}}%)")

# Save artifacts
print("\\nSaving artifacts...")

# Save trace
trace_path = "model_trace.nc"
trace.to_netcdf(trace_path)
print(f"Saved trace to {{trace_path}}")

# Save model summary
summary_path = "model_summary.csv"
summary.to_csv(summary_path)
print(f"Saved summary to {{summary_path}}")

# Save results JSON
results = {{
    "diagnostics": diagnostics,
    "channel_contributions": contributions,
    "config_used": CONFIG,
    "data_shape": list(data.shape),
    "fit_timestamp": "{timestamp}"
}}

results_path = "model_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {{results_path}}")

# Generate plots
print("\\nGenerating plots...")

# 1. Channel contribution waterfall
fig, ax = plt.subplots(figsize=(12, 6))
channels = [c["channel"] for c in contributions]
values = [c["contribution_mean"] for c in contributions]
errors = [[c["contribution_mean"] - c["contribution_lower"], c["contribution_upper"] - c["contribution_mean"]] 
          for c in contributions]
errors = np.array(errors).T

ax.barh(channels, values, xerr=errors, capsize=5, color='steelblue', alpha=0.8)
ax.set_xlabel("Contribution (normalized)")
ax.set_title("Channel Contributions with 95% Credible Intervals")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("channel_contributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved channel_contributions.png")

# 2. ROI comparison
fig, ax = plt.subplots(figsize=(12, 6))
rois = [c["roi_mean"] for c in contributions]
roi_errors = [[c["roi_mean"] - c["roi_lower"], c["roi_upper"] - c["roi_mean"]] 
              for c in contributions]
roi_errors = np.array(roi_errors).T

colors = ['green' if r > 1 else 'red' for r in rois]
ax.barh(channels, rois, xerr=roi_errors, capsize=5, color=colors, alpha=0.8)
ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')
ax.set_xlabel("ROI")
ax.set_title("Channel ROI with 95% Credible Intervals")
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("channel_roi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved channel_roi.png")

# 3. Trace plots for key parameters
fig = az.plot_trace(trace, var_names=["intercept", "sigma"])
plt.tight_layout()
plt.savefig("trace_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved trace_plots.png")

# 4. Posterior distributions for betas
beta_vars = [f"beta_{{ch}}" for ch in channels]
fig = az.plot_posterior(trace, var_names=beta_vars)
plt.tight_layout()
plt.savefig("posterior_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved posterior_distributions.png")

print("\\n=== Model fitting complete ===")
print(f"Results saved to model_results.json")
'''


RESPONSE_CURVES_TEMPLATE = '''
"""
Generate response curves for each media channel.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load results
with open("model_results.json", "r") as f:
    results = json.load(f)

# Load data
data = pd.read_csv("{mff_data_path}")
config = results["config_used"]

media_channels = [ch["name"] for ch in config["media_channels"]]

fig, axes = plt.subplots(len(media_channels), 1, figsize=(10, 4*len(media_channels)))
if len(media_channels) == 1:
    axes = [axes]

for i, ch_name in enumerate(media_channels):
    ax = axes[i]
    ch_data = data[ch_name].values
    
    # Generate spend range
    x_range = np.linspace(0, ch_data.max() * 1.5, 100)
    
    # Simple diminishing returns curve (illustrative)
    contrib = results["channel_contributions"][i]
    beta = contrib["contribution_mean"]
    
    # Hill-like response
    k = np.median(ch_data)
    response = beta * x_range / (k + x_range)
    
    ax.plot(x_range, response, 'b-', linewidth=2)
    ax.axvline(ch_data.mean(), color='r', linestyle='--', alpha=0.7, label='Current avg spend')
    ax.fill_between(x_range, response * 0.8, response * 1.2, alpha=0.2)
    
    ax.set_xlabel(f"{ch_name} Spend")
    ax.set_ylabel("Response")
    ax.set_title(f"{ch_name} Response Curve")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("response_curves.png", dpi=150, bbox_inches='tight')
plt.close()

print("Generated response_curves.png")
'''


# =============================================================================
# Workflow Nodes
# =============================================================================

async def initialize_modeling(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
) -> ModelingWorkflowState:
    """Initialize modeling workflow."""
    session_id = state.get("session_id") or str(uuid.uuid4())
    analysis_id = state.get("analysis_id") or str(uuid.uuid4())
    
    logger.info(f"Initializing modeling workflow: session={session_id}")
    
    # Create or update analysis in Neo4j
    await neo4j.create_analysis(
        analysis_id=analysis_id,
        name=f"MMM Modeling {datetime.now().strftime('%Y-%m-%d')}",
        description="Bayesian MMM fitting and interpretation",
    )
    
    await neo4j.update_analysis_status(analysis_id, "modeling_started")
    
    return {
        **state,
        "session_id": session_id,
        "analysis_id": analysis_id,
        "current_phase": WorkflowPhase.MODEL_INIT,
        "messages": [f"Initialized modeling workflow (session: {session_id})"],
    }


async def configure_model(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
    code_executor: CodeExecutor,
) -> ModelingWorkflowState:
    """Configure MMM model based on research plan and data."""
    logger.info("Configuring model parameters")
    
    session_id = state["session_id"]
    mff_data_path = state["mff_data_path"]
    research_plan = state.get("research_plan", {})
    feature_transforms = state.get("feature_transformations", [])
    
    # Retrieve modeling context from GraphRAG
    modeling_context = await graphrag.get_phase_context(
        phase="modeling",
        query="mmm model configuration best practices",
        analysis_id=state.get("analysis_id"),
    )
    
    # Load MFF data to understand structure
    load_code = f'''
import pandas as pd
import json

data = pd.read_csv("{mff_data_path}")
info = {{
    "columns": list(data.columns),
    "shape": list(data.shape),
    "dtypes": {{k: str(v) for k, v in data.dtypes.items()}},
    "numeric_columns": list(data.select_dtypes(include=['number']).columns),
}}
with open("data_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
'''
    
    result = await code_executor.execute(load_code, session_id)
    
    data_info = {}
    if result.success:
        try:
            info_path = code_executor._get_session_dir(session_id) / "data_info.json"
            with open(info_path) as f:
                data_info = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read data info: {e}")
    
    # Build configuration prompt
    config_prompt = f"""
{MODEL_CONFIG_PROMPT}

Research Plan:
{json.dumps(research_plan, indent=2)}

Feature Transformations from EDA:
{json.dumps(feature_transforms, indent=2)}

Data Structure:
{json.dumps(data_info, indent=2)}

Prior Successful Patterns:
{modeling_context}

Generate a ModelConfiguration that:
1. Uses the target variable from the research plan
2. Configures each media channel with appropriate adstock/saturation
3. Includes relevant control variables
4. Sets reasonable MCMC parameters for the data size
"""
    
    # Generate configuration
    llm = create_structured_llm(ModelConfiguration, LLMTask.REASONING)
    
    try:
        config: ModelConfiguration = await llm.ainvoke(config_prompt)
        config_dict = config.model_dump()
        
        logger.info(f"Generated model config with {len(config.media_channels)} channels")
        
    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        # Use fallback configuration
        media_channels = research_plan.get("media_channels", ["tv", "digital", "print"])
        config_dict = {
            "target_variable": research_plan.get("target_variable", "revenue"),
            "target_transformation": "log",
            "media_channels": [
                {
                    "name": ch,
                    "adstock": {"type": "geometric", "alpha": 0.5, "l_max": 8},
                    "saturation": {"type": "hill", "lam_prior_sigma": 0.5},
                }
                for ch in media_channels
            ],
            "control_variables": research_plan.get("control_variables", []),
            "date_column": "Period",
            "draws": 1000,
            "tune": 500,
            "chains": 4,
            "target_accept": 0.9,
            "rationale": "Fallback configuration",
        }
    
    # Store decision
    await graphrag.add_decision(
        analysis_id=state["analysis_id"],
        phase="modeling",
        decision_type="model_configuration",
        decision=f"Configured MMM with {len(config_dict.get('media_channels', []))} channels",
        rationale=config_dict.get("rationale", "Based on research plan and data"),
    )
    
    return {
        **state,
        "current_phase": WorkflowPhase.MODEL_CONFIG,
        "model_config": config_dict,
        "target_variable": config_dict.get("target_variable", "revenue"),
        "media_channels": [ch["name"] for ch in config_dict.get("media_channels", [])],
        "control_variables": config_dict.get("control_variables", []),
        "messages": [
            f"Model configured with {len(config_dict.get('media_channels', []))} channels: "
            f"{', '.join(ch['name'] for ch in config_dict.get('media_channels', []))}"
        ],
    }


async def fit_model(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
    code_executor: CodeExecutor,
) -> ModelingWorkflowState:
    """Fit the Bayesian MMM model."""
    logger.info("Starting model fitting")
    
    session_id = state["session_id"]
    config = state["model_config"]
    mff_data_path = state["mff_data_path"]
    
    # Copy MFF data to session
    await code_executor.copy_file_to_session(mff_data_path, session_id)
    mff_filename = Path(mff_data_path).name
    session_mff_path = f"./{mff_filename}"
    
    # Generate fitting code
    fit_code = MODEL_FIT_TEMPLATE.format(
        session_id=session_id,
        config_json=json.dumps(config, indent=2),
        mff_data_path=session_mff_path,
        timestamp=datetime.now().isoformat(),
    )
    
    # Execute model fitting (long-running)
    start_time = datetime.now()
    result = await code_executor.execute(fit_code, session_id, timeout=1800)  # 30 min timeout
    fit_duration = (datetime.now() - start_time).total_seconds()
    
    if not result.success:
        logger.error(f"Model fitting failed: {result.error}")
        return {
            **state,
            "current_phase": WorkflowPhase.ERROR,
            "errors": [f"Model fitting failed: {result.error}"],
        }
    
    # Load results
    session_dir = code_executor._get_session_dir(session_id)
    results_path = session_dir / "model_results.json"
    
    try:
        with open(results_path) as f:
            model_results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return {
            **state,
            "current_phase": WorkflowPhase.ERROR,
            "errors": [f"Failed to load model results: {e}"],
        }
    
    # Extract paths
    model_artifact_path = str(session_dir / "model_trace.nc")
    
    logger.info(f"Model fitting complete in {fit_duration:.1f}s")
    
    return {
        **state,
        "current_phase": WorkflowPhase.MODEL_FIT,
        "model_artifact_path": model_artifact_path,
        "inference_data_path": model_artifact_path,
        "fit_duration": fit_duration,
        "convergence_diagnostics": model_results.get("diagnostics", {}),
        "channel_contributions": model_results.get("channel_contributions", []),
        "generated_plots": [
            str(session_dir / "channel_contributions.png"),
            str(session_dir / "channel_roi.png"),
            str(session_dir / "trace_plots.png"),
            str(session_dir / "posterior_distributions.png"),
        ],
        "messages": [f"Model fitted in {fit_duration:.1f} seconds"],
    }


async def run_diagnostics(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
) -> ModelingWorkflowState:
    """Run and evaluate convergence diagnostics."""
    logger.info("Running model diagnostics")
    
    diagnostics = state.get("convergence_diagnostics", {})
    
    # Evaluate convergence
    recommendations = []
    status = diagnostics.get("convergence_status", "unknown")
    
    if diagnostics.get("rhat_max", 999) > 1.1:
        recommendations.append("High R-hat values detected. Consider increasing tune iterations.")
    
    if diagnostics.get("ess_bulk_min", 0) < 100:
        recommendations.append("Low effective sample size. Consider increasing draws or chains.")
    
    if diagnostics.get("divergences", 0) > 0:
        recommendations.append(f"{diagnostics['divergences']} divergences detected. Consider reparameterization.")
    
    if status == "good":
        recommendations.append("Convergence looks good. Proceed with interpretation.")
    elif status == "acceptable":
        recommendations.append("Convergence is acceptable but could be improved with more samples.")
    else:
        recommendations.append("Poor convergence. Results should be interpreted with caution.")
    
    # Store diagnostic decision
    await graphrag.add_decision(
        analysis_id=state["analysis_id"],
        phase="modeling",
        decision_type="diagnostics",
        decision=f"Convergence status: {status}",
        rationale="; ".join(recommendations),
    )
    
    return {
        **state,
        "current_phase": WorkflowPhase.MODEL_DIAGNOSE,
        "convergence_status": status,
        "messages": [f"Diagnostics complete: {status}. {len(recommendations)} recommendations."],
    }


async def interpret_results(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
    graphrag: GraphRAGManager,
    code_executor: CodeExecutor,
) -> ModelingWorkflowState:
    """Interpret model results and generate insights."""
    logger.info("Interpreting model results")
    
    session_id = state["session_id"]
    contributions = state.get("channel_contributions", [])
    diagnostics = state.get("convergence_diagnostics", {})
    config = state.get("model_config", {})
    
    # Generate response curves
    response_code = RESPONSE_CURVES_TEMPLATE.format(
        mff_data_path=f"./{Path(state['mff_data_path']).name}"
    )
    await code_executor.execute(response_code, session_id)
    
    # Build interpretation prompt
    interpret_prompt = f"""
{INTERPRETATION_PROMPT}

Model Configuration:
- Target: {config.get('target_variable', 'revenue')}
- Channels: {[ch.get('name') for ch in config.get('media_channels', [])]}
- Controls: {config.get('control_variables', [])}

Convergence Diagnostics:
- Status: {diagnostics.get('convergence_status', 'unknown')}
- Max R-hat: {diagnostics.get('rhat_max', 'N/A')}
- Divergences: {diagnostics.get('divergences', 'N/A')}

Channel Contributions:
{json.dumps(contributions, indent=2)}

Please provide a comprehensive interpretation.
"""
    
    # Generate interpretation
    llm = create_structured_llm(InterpretationOutput, LLMTask.REASONING)
    
    try:
        interpretation: InterpretationOutput = await llm.ainvoke(interpret_prompt)
        
        interpretation_summary = f"""
## Executive Summary
{interpretation.executive_summary}

## Channel Insights
{chr(10).join('- ' + insight for insight in interpretation.channel_insights)}

## Budget Recommendations
{chr(10).join('- ' + rec for rec in interpretation.budget_recommendations)}

## Caveats
{chr(10).join('- ' + cav for cav in interpretation.caveats)}

## Next Steps
{chr(10).join('- ' + step for step in interpretation.next_steps)}
"""
        
        recommendations = interpretation.budget_recommendations + interpretation.next_steps
        
    except Exception as e:
        logger.error(f"Interpretation generation failed: {e}")
        interpretation_summary = "Interpretation could not be generated automatically."
        recommendations = []
    
    # Calculate ROI summary
    roi_estimates = {}
    for ch in contributions:
        roi_estimates[ch["channel"]] = {
            "mean": ch.get("roi_mean", 0),
            "lower": ch.get("roi_lower", 0),
            "upper": ch.get("roi_upper", 0),
        }
    
    # Calculate total media contribution
    total_media = sum(ch.get("contribution_mean", 0) for ch in contributions)
    
    # Store model artifact in Neo4j
    await neo4j.create_model_artifact(
        artifact_id=str(uuid.uuid4()),
        analysis_id=state["analysis_id"],
        artifact_type="bayesian_mmm",
        artifact_path=state["model_artifact_path"],
        metrics={
            "convergence_status": diagnostics.get("convergence_status", "unknown"),
            "rhat_max": diagnostics.get("rhat_max", 0),
            "divergences": diagnostics.get("divergences", 0),
            "total_media_contribution": total_media,
        },
    )
    
    # Store interpretation
    await graphrag.add_decision(
        analysis_id=state["analysis_id"],
        phase="modeling",
        decision_type="interpretation",
        decision="Model interpretation complete",
        rationale=interpretation_summary[:500],  # Truncate for storage
    )
    
    # Update generated plots
    session_dir = code_executor._get_session_dir(session_id)
    plots = state.get("generated_plots", [])
    response_curves_path = str(session_dir / "response_curves.png")
    if os.path.exists(response_curves_path):
        plots.append(response_curves_path)
    
    return {
        **state,
        "current_phase": WorkflowPhase.MODEL_INTERPRET,
        "interpretation_summary": interpretation_summary,
        "recommendations": recommendations,
        "roi_estimates": roi_estimates,
        "total_media_contribution": total_media,
        "generated_plots": plots,
        "messages": ["Model interpretation complete"],
    }


async def finalize_modeling(
    state: ModelingWorkflowState,
    neo4j: Neo4jClient,
) -> ModelingWorkflowState:
    """Finalize modeling workflow."""
    logger.info("Finalizing modeling workflow")
    
    # Update analysis status
    await neo4j.update_analysis_status(
        state["analysis_id"],
        "modeling_complete"
    )
    
    return {
        **state,
        "current_phase": WorkflowPhase.MODEL_COMPLETE,
        "messages": ["Modeling workflow complete"],
    }


# =============================================================================
# Workflow Graph Builder
# =============================================================================

def route_after_init(state: ModelingWorkflowState) -> str:
    """Route after initialization."""
    if state.get("errors"):
        return "error"
    return "configure"


def route_after_config(state: ModelingWorkflowState) -> str:
    """Route after configuration."""
    if state.get("errors"):
        return "error"
    return "fit"


def route_after_fit(state: ModelingWorkflowState) -> str:
    """Route after fitting."""
    if state.get("errors"):
        return "error"
    return "diagnose"


def route_after_diagnose(state: ModelingWorkflowState) -> str:
    """Route after diagnostics."""
    if state.get("errors"):
        return "error"
    
    # Check if convergence is acceptable
    status = state.get("convergence_status", "unknown")
    if status == "poor":
        # Could implement retry logic here
        pass
    
    return "interpret"


def route_after_interpret(state: ModelingWorkflowState) -> str:
    """Route after interpretation."""
    return "complete"


class ModelingWorkflow:
    """Modeling Agent workflow for Bayesian MMM fitting."""
    
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
        workflow = StateGraph(ModelingWorkflowState)
        
        # Add nodes
        workflow.add_node(
            "init",
            lambda s: asyncio.get_event_loop().run_until_complete(
                initialize_modeling(s, self.neo4j, self.graphrag)
            )
        )
        workflow.add_node(
            "configure",
            lambda s: asyncio.get_event_loop().run_until_complete(
                configure_model(s, self.neo4j, self.graphrag, self.code_executor)
            )
        )
        workflow.add_node(
            "fit",
            lambda s: asyncio.get_event_loop().run_until_complete(
                fit_model(s, self.neo4j, self.code_executor)
            )
        )
        workflow.add_node(
            "diagnose",
            lambda s: asyncio.get_event_loop().run_until_complete(
                run_diagnostics(s, self.neo4j, self.graphrag)
            )
        )
        workflow.add_node(
            "interpret",
            lambda s: asyncio.get_event_loop().run_until_complete(
                interpret_results(s, self.neo4j, self.graphrag, self.code_executor)
            )
        )
        workflow.add_node(
            "complete",
            lambda s: asyncio.get_event_loop().run_until_complete(
                finalize_modeling(s, self.neo4j)
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
        workflow.add_conditional_edges("configure", route_after_config)
        workflow.add_conditional_edges("fit", route_after_fit)
        workflow.add_conditional_edges("diagnose", route_after_diagnose)
        workflow.add_conditional_edges("interpret", route_after_interpret)
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
        mff_data_path: str,
        research_plan: dict | None = None,
        feature_transformations: list[dict] | None = None,
        analysis_id: str | None = None,
    ) -> ModelingWorkflowState:
        """Run the modeling workflow."""
        initial_state: ModelingWorkflowState = {
            "mff_data_path": mff_data_path,
            "research_plan": research_plan or {},
            "feature_transformations": feature_transformations or [],
            "analysis_id": analysis_id,
            "messages": [],
            "errors": [],
        }
        
        app = self.compile()
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        final_state = await app.ainvoke(initial_state, config)
        
        return final_state


async def create_modeling_workflow(
    neo4j_uri: str = settings.NEO4J_URI,
    neo4j_user: str = settings.NEO4J_USER,
    neo4j_password: str = settings.NEO4J_PASSWORD,
    postgres_url: str | None = settings.ASYNC_DATABASE_URL,
) -> ModelingWorkflow:
    """Factory function to create modeling workflow with dependencies."""
    neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    graphrag = GraphRAGManager(neo4j)
    code_executor = CodeExecutor()
    
    checkpointer = None
    if postgres_url:
        checkpointer = AsyncPostgresSaver.from_conn_string(postgres_url)
        await checkpointer.setup()
    
    return ModelingWorkflow(
        neo4j=neo4j,
        graphrag=graphrag,
        code_executor=code_executor,
        checkpointer=checkpointer,
    )
