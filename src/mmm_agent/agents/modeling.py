"""
Modeling Agent for MMM Workflow

Handles the third phase of the MMM workflow:
- Building Bayesian MMM specification
- Fitting model (async with progress)
- Convergence diagnostics
- Channel contribution estimation
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

class ModelSpecification(BaseModel):
    """MMM model specification."""
    adstock_config: dict[str, dict] = Field(description="Adstock config per channel")
    saturation_config: dict[str, dict] = Field(description="Saturation config per channel")
    control_config: list[str] = Field(description="Control variables to include")
    seasonality_config: dict = Field(description="Seasonality specification")
    trend_config: dict = Field(description="Trend specification")
    hierarchical: bool = Field(description="Whether to use hierarchical model")
    rationale: str = Field(description="Rationale for specification")


class DiagnosticsOutput(BaseModel):
    """Model diagnostics summary."""
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int
    convergence_status: str = Field(description="good, acceptable, poor")
    recommendations: list[str] = Field(description="Recommendations for improvement")


class ChannelResult(BaseModel):
    """Results for a single channel."""
    channel: str
    contribution_mean: float
    contribution_lower: float
    contribution_upper: float
    share_of_contribution: float
    adstock_alpha: float
    saturation_lambda: float


class ModelingOutput(BaseModel):
    """Complete output from modeling phase."""
    model_spec: ModelSpecification
    diagnostics: DiagnosticsOutput
    channel_results: list[ChannelResult]
    total_media_contribution: float
    model_r2: float
    summary: str


# =============================================================================
# Model Fitting Code Templates
# =============================================================================

MODEL_FIT_CODE = '''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("{data_path}")

# Configuration
kpi_col = "{kpi_column}"
date_col = "{date_column}"
media_cols = {media_columns}
control_cols = {control_columns}

print("=== MODEL CONFIGURATION ===")
print(f"KPI: {{kpi_col}}")
print(f"Media channels: {{media_cols}}")
print(f"Controls: {{control_cols}}")
print(f"Data shape: {{df.shape}}")
print()

# Parse dates and sort
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

# Prepare data matrices
y = df[kpi_col].values
X_media = df[media_cols].fillna(0).values
X_controls = df[[c for c in control_cols if c in df.columns]].fillna(0).values if control_cols else None

print(f"y shape: {{y.shape}}")
print(f"X_media shape: {{X_media.shape}}")
if X_controls is not None:
    print(f"X_controls shape: {{X_controls.shape}}")

# Standardize
y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

X_media_scaled = X_media / (X_media.mean(axis=0) + 1e-8)

print()
print("=== FITTING BAYESIAN MMM ===")
print("This may take several minutes...")

try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    
    # Build model
    with pm.Model() as mmm:
        # Coordinates
        n_obs = len(y)
        n_channels = len(media_cols)
        
        # Intercept
        intercept = pm.Normal("intercept", mu=0, sigma=0.5)
        
        # Adstock parameters (geometric decay)
        adstock_alpha = pm.Beta("adstock_alpha", alpha=2, beta=2, shape=n_channels)
        
        # Saturation parameters
        saturation_lambda = pm.Gamma("saturation_lambda", alpha=3, beta=1, shape=n_channels)
        
        # Channel coefficients (positive effects)
        beta_media = pm.HalfNormal("beta_media", sigma=0.5, shape=n_channels)
        
        # Apply transformations and compute contributions
        contributions = []
        for i in range(n_channels):
            # Simple geometric adstock
            x_i = X_media_scaled[:, i]
            
            # Cumulative adstock effect (simplified)
            x_adstocked = x_i  # Would apply proper adstock in full implementation
            
            # Logistic saturation
            x_saturated = 1 - pt.exp(-saturation_lambda[i] * x_adstocked)
            
            # Channel contribution
            contrib_i = beta_media[i] * x_saturated
            contributions.append(contrib_i)
        
        # Stack contributions
        media_contribution = pt.stack(contributions, axis=1).sum(axis=1)
        
        # Control effects
        if X_controls is not None and X_controls.shape[1] > 0:
            beta_controls = pm.Normal("beta_controls", mu=0, sigma=0.5, 
                                     shape=X_controls.shape[1])
            control_contribution = pt.dot(X_controls, beta_controls)
        else:
            control_contribution = 0
        
        # Seasonality (Fourier terms)
        t = np.arange(n_obs)
        n_fourier = 2
        fourier_features = []
        for k in range(1, n_fourier + 1):
            fourier_features.append(np.sin(2 * np.pi * k * t / 52))
            fourier_features.append(np.cos(2 * np.pi * k * t / 52))
        X_fourier = np.column_stack(fourier_features)
        
        beta_seasonality = pm.Normal("beta_seasonality", mu=0, sigma=0.3, 
                                    shape=X_fourier.shape[1])
        seasonality = pt.dot(X_fourier, beta_seasonality)
        
        # Combine
        mu = intercept + media_contribution + control_contribution + seasonality
        
        # Noise
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        
        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)
        
        # Store contributions for analysis
        pm.Deterministic("channel_contributions", pt.stack(contributions, axis=1))
        pm.Deterministic("media_total", media_contribution)
    
    print("Model built successfully")
    print()
    
    # Sample
    with mmm:
        trace = pm.sample(
            draws={n_draws},
            tune={n_tune},
            chains={n_chains},
            target_accept={target_accept},
            random_seed=42,
            progressbar=True,
        )
    
    print()
    print("=== DIAGNOSTICS ===")
    
    # Compute diagnostics
    rhat = az.rhat(trace)
    ess = az.ess(trace)
    
    rhat_max = float(rhat.max().to_array().max())
    ess_bulk_min = float(ess.to_array().min())
    
    print(f"R-hat max: {{rhat_max:.4f}}")
    print(f"ESS min: {{ess_bulk_min:.0f}}")
    print(f"Divergences: {{trace.sample_stats.diverging.sum().values}}")
    
    if rhat_max < 1.05 and ess_bulk_min > 400:
        print("Convergence: GOOD")
    elif rhat_max < 1.1:
        print("Convergence: ACCEPTABLE")
    else:
        print("Convergence: POOR - consider simplifying model")
    
    print()
    print("=== CHANNEL CONTRIBUTIONS ===")
    
    # Get posterior means for contributions
    channel_contribs = trace.posterior["channel_contributions"].mean(dim=["chain", "draw"]).values
    total_contrib = channel_contribs.sum()
    
    for i, channel in enumerate(media_cols):
        contrib = channel_contribs[:, i].sum()
        share = contrib / total_contrib * 100 if total_contrib > 0 else 0
        print(f"{{channel}}: {{contrib:.4f}} ({{share:.1f}}% of media)")
    
    print()
    print("=== PARAMETER ESTIMATES ===")
    print(az.summary(trace, var_names=["intercept", "adstock_alpha", "saturation_lambda", 
                                       "beta_media", "sigma"]))
    
    # Save trace
    trace.to_netcdf("model_trace.nc")
    print()
    print("Model trace saved to model_trace.nc")
    
    # Predictions
    with mmm:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    
    y_pred = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
    y_pred_unscaled = y_pred * y_std + y_mean
    
    # Model fit
    ss_res = ((y - y_pred_unscaled) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    mape = np.abs((y - y_pred_unscaled) / y).mean() * 100
    
    print()
    print(f"Model R²: {{r2:.4f}}")
    print(f"MAPE: {{mape:.2f}}%")
    
except ImportError as e:
    print(f"PyMC not available: {{e}}")
    print("Please install: pip install pymc arviz")
except Exception as e:
    print(f"Model fitting failed: {{e}}")
    import traceback
    traceback.print_exc()
'''


SIMPLE_MODEL_CODE = '''
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("{data_path}")

kpi_col = "{kpi_column}"
date_col = "{date_column}"
media_cols = {media_columns}
control_cols = {control_columns}

print("=== SIMPLE MMM (Ridge Regression) ===")
print("Running frequentist baseline model...")
print()

# Prepare data
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

y = df[kpi_col].values
X_media = df[media_cols].fillna(0).values

# Add simple adstock (exponential decay)
def geometric_adstock(x, alpha=0.5, l_max=8):
    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(t + 1, l_max)):
            result[t] += x[t - lag] * (alpha ** lag)
    return result

X_adstocked = np.column_stack([
    geometric_adstock(X_media[:, i]) 
    for i in range(X_media.shape[1])
])

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_adstocked)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Fit Ridge regression
model = Ridge(alpha=1.0, positive=True)
model.fit(X_scaled, y_scaled)

# Predictions
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Metrics
ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
mape = np.abs((y - y_pred) / y).mean() * 100

print(f"Model R²: {{r2:.4f}}")
print(f"MAPE: {{mape:.2f}}%")
print()

# Channel contributions
print("=== CHANNEL CONTRIBUTIONS ===")
contributions = X_scaled * model.coef_
total_contrib = contributions.sum()

for i, channel in enumerate(media_cols):
    contrib = contributions[:, i].sum()
    share = contrib / total_contrib * 100 if total_contrib > 0 else 0
    coef = model.coef_[i]
    print(f"{{channel}}: coef={{coef:.4f}}, share={{share:.1f}}%")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df[date_col], y, 'b-', label='Actual', alpha=0.8)
ax.plot(df[date_col], y_pred, 'r--', label='Predicted', alpha=0.8)
ax.set_title(f'Actual vs Predicted {{kpi_col}} (R²={{r2:.3f}})')
ax.set_xlabel('Date')
ax.set_ylabel(kpi_col)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('model_fit')
plt.close()

print()
print("=== MODEL COMPLETE ===")
print("Plots saved to working directory")
'''


# =============================================================================
# Modeling Agent
# =============================================================================

MODELING_SYSTEM_PROMPT = """You are an expert in Bayesian Marketing Mix Modeling using PyMC.

Your role is to:
1. Specify appropriate model structure based on EDA findings
2. Configure adstock and saturation transformations
3. Interpret convergence diagnostics
4. Explain channel contributions

Key modeling principles:
- Start simple, add complexity based on diagnostics
- Use informative priors when possible
- Check for multicollinearity between channels
- Ensure adequate ESS and R-hat for reliable inference
- Consider model comparison (WAIC, LOO) for specification

Provide clear explanations of results and flag any concerns."""


class ModelingAgent:
    """
    Agent for MMM modeling phase.
    
    Responsibilities:
    - Build model specification
    - Execute model fitting
    - Analyze convergence
    - Extract contributions
    """
    
    def __init__(self, llm, code_executor, use_bayesian: bool = True):
        """
        Initialize modeling agent.
        
        Args:
            llm: LangChain chat model
            code_executor: LocalCodeExecutor instance
            use_bayesian: Whether to use Bayesian (True) or frequentist (False)
        """
        self.llm = llm
        self.executor = code_executor
        self.use_bayesian = use_bayesian
    
    async def model(
        self,
        state: MMMWorkflowState,
        context: str = "",
        on_progress = None,
    ) -> dict[str, Any]:
        """
        Execute modeling phase.
        
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
        control_vars = state.get("control_variables", [])
        
        if not data_path:
            return {
                "error": "No data available for modeling",
                "current_phase": WorkflowPhase.ERROR,
            }
        
        if on_progress:
            await on_progress("Modeling", "Building model specification...")
        
        # Choose model code template
        if self.use_bayesian:
            model_code = MODEL_FIT_CODE.format(
                data_path=data_path,
                kpi_column=target_var,
                date_column="Period",
                media_columns=media_channels,
                control_columns=control_vars,
                n_draws=500,  # Small for POC
                n_tune=300,
                n_chains=2,
                target_accept=0.9,
            )
        else:
            model_code = SIMPLE_MODEL_CODE.format(
                data_path=data_path,
                kpi_column=target_var,
                date_column="Period",
                media_columns=media_channels,
                control_columns=control_vars,
            )
        
        if on_progress:
            await on_progress("Modeling", "Fitting model (this may take a few minutes)...")
        
        # Execute model fitting
        result = await self.executor.execute(
            model_code,
            session_id=workflow_id,
            validate=True,
        )
        
        model_output = result.stdout
        plots = [f for f in result.generated_files if f.endswith('.png')]
        model_artifacts = [f for f in result.generated_files if f.endswith('.nc')]
        
        if not result.success:
            logger.error(f"Model fitting failed: {result.error}")
            
            # Try simpler model
            if self.use_bayesian and on_progress:
                await on_progress("Modeling", "Bayesian fit failed, trying frequentist fallback...")
                
                simple_code = SIMPLE_MODEL_CODE.format(
                    data_path=data_path,
                    kpi_column=target_var,
                    date_column="Period",
                    media_columns=media_channels,
                    control_columns=control_vars,
                )
                
                result = await self.executor.execute(simple_code, session_id=workflow_id)
                model_output = result.stdout
                plots = [f for f in result.generated_files if f.endswith('.png')]
        
        # Analyze results with LLM
        if on_progress:
            await on_progress("Modeling", "Analyzing model results...")
        
        analysis_prompt = f"""Analyze these MMM modeling results:

## Model Output
{model_output[:4000]}

## Configuration
- Target: {target_var}
- Media channels: {media_channels}
- Controls: {control_vars}

Based on this output:
1. Summarize convergence diagnostics (if Bayesian)
2. Interpret channel contributions
3. Assess model fit quality
4. Provide recommendations"""

        try:
            analysis = self.llm.invoke([
                SystemMessage(content=MODELING_SYSTEM_PROMPT),
                HumanMessage(content=analysis_prompt),
            ])
            
            summary = analysis.content
        except Exception as e:
            summary = f"Model output:\n{model_output[:2000]}"
        
        logger.info(f"Modeling complete: {len(plots)} plots, {len(model_artifacts)} artifacts")
        
        return {
            "model_config": {
                "bayesian": self.use_bayesian,
                "media_channels": media_channels,
                "controls": control_vars,
            },
            "model_artifact_path": model_artifacts[0] if model_artifacts else "",
            "inference_data_path": model_artifacts[0] if model_artifacts else "",
            "convergence_diagnostics": {},  # Would parse from output
            "channel_contributions": [],  # Would parse from output
            "model_summary": summary,
            "model_plots": plots,
            "model_status": "completed" if result.success else "fallback",
            "current_phase": WorkflowPhase.INTERPRETATION,
            "prior_decisions": [Decision(
                phase=WorkflowPhase.MODELING,
                decision_type="model_fit",
                content={
                    "bayesian": self.use_bayesian,
                    "success": result.success,
                    "artifacts": model_artifacts,
                },
                rationale=summary[:500],
            ).model_dump()],
        }
    
    def model_sync(
        self,
        state: MMMWorkflowState,
        context: str = "",
    ) -> dict[str, Any]:
        """Synchronous modeling wrapper."""
        import asyncio
        return asyncio.run(self.model(state, context))


# =============================================================================
# LangGraph Node
# =============================================================================

async def modeling_node(state: MMMWorkflowState, deps: dict) -> dict:
    """
    LangGraph node for modeling phase.
    
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
    use_bayesian = deps.get("use_bayesian", False)  # Default to simple for POC
    
    # Get context
    context = ""
    if context_manager:
        context = context_manager.get_context(
            state.get("workflow_id", "default"),
            "modeling",
            state.get("user_query", ""),
        )
    
    agent = ModelingAgent(llm, executor, use_bayesian=use_bayesian)
    return await agent.model(state, context, on_progress)
