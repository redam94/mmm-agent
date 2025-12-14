# Agentic MMM Framework: Architecture Blueprint

Building an AI-powered Marketing Mix Model system requires orchestrating LLM agents, graph databases, async computation, and Bayesian inference into a cohesive architecture. **The optimal stack combines PyMC-Marketing for modeling, LangGraph for agent orchestration, Neo4j for knowledge persistence, and Ray for distributed model fitting**—enabling a four-phase workflow that progressively builds causal understanding from data to actionable insights.

This architecture separates interactive agent workflows from computationally intensive model fitting, uses knowledge graphs for cross-phase context sharing, and implements incremental Bayesian modeling that starts simple and adds complexity based on diagnostics. The result is a system where analysts can iteratively design, explore, model, and interpret marketing effectiveness through natural conversation.

## Multi-LLM integration with provider-agnostic abstraction

LangChain's `init_chat_model()` provides the cleanest abstraction for supporting multiple LLM providers at runtime without code changes. This approach allows seamless switching between Ollama (local development), Claude (complex reasoning), GPT-4 (structured outputs), and Gemini (cost optimization) based on task requirements.

```python
from langchain.chat_models import init_chat_model

# Create configurable model with runtime switching
configurable_model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    configurable_fields="any",
    config_prefix="llm",
    temperature=0
)

# Switch providers at runtime based on task complexity
result = configurable_model.invoke(
    "Analyze model residuals...",
    config={"configurable": {
        "llm_model": "gpt-4o",
        "llm_model_provider": "openai"
    }}
)
```

**Provider routing strategy**: Route simple data queries to faster/cheaper models (GPT-4o-mini, local Ollama), reserve Claude for complex causal reasoning, and use GPT-4o for structured outputs requiring strict JSON adherence. Implement caching at the LangChain level to avoid redundant API calls across phases.

## Four-phase workflow architecture with LangGraph

The core workflow progresses through **Planning → EDA → Modeling → Interpretation** phases, each managed by specialized agents that inherit context from prior phases. LangGraph's `StateGraph` provides typed state management that persists decisions across phases while enabling checkpointing for session resumability.

```python
from typing import TypedDict, Annotated, Optional, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

class MMMWorkflowState(TypedDict):
    messages: Annotated[list, add]
    current_phase: str
    
    # Planning outputs → inherited by downstream phases
    research_questions: List[str]
    target_variable: str
    media_channels: List[str]
    control_variables: List[str]
    causal_hypotheses: List[dict]
    
    # EDA outputs
    data_quality_report: dict
    feature_transformations: List[dict]  # adstock/saturation configs
    correlation_insights: List[str]
    
    # Modeling outputs
    model_config: dict
    inference_data_path: str  # ArviZ NetCDF artifact
    convergence_diagnostics: dict
    channel_contributions: dict
    
    # Interpretation outputs
    roi_estimates: dict
    optimization_recommendations: List[dict]
    confidence_intervals: dict
    
    # Shared context
    knowledge_graph_context: str
    prior_decisions: Annotated[List[dict], add]

# Phase nodes with context inheritance
def planning_node(state: MMMWorkflowState):
    # Query KG for similar past analyses
    kg_context = query_successful_mmm_patterns(state["research_questions"])
    
    plan = planning_agent.invoke({
        "user_requirements": state["messages"][-1],
        "prior_successful_patterns": kg_context
    })
    
    return {
        "media_channels": plan.channels,
        "control_variables": plan.controls,
        "causal_hypotheses": plan.dag_structure,
        "current_phase": "eda",
        "prior_decisions": [{"phase": "planning", "decision": plan}]
    }

def eda_node(state: MMMWorkflowState):
    # Inherit planning context, query KG for feature patterns
    feature_patterns = query_feature_engineering_patterns(state["media_channels"])
    
    eda_results = eda_agent.invoke({
        "channels": state["media_channels"],
        "controls": state["control_variables"],
        "prior_transformations": feature_patterns
    })
    
    return {
        "feature_transformations": eda_results.transformations,
        "data_quality_report": eda_results.quality_issues,
        "current_phase": "modeling"
    }

workflow = StateGraph(MMMWorkflowState)
workflow.add_node("planning", planning_node)
workflow.add_node("eda", eda_node)
workflow.add_node("modeling", modeling_node)
workflow.add_node("interpretation", interpretation_node)

workflow.add_edge(START, "planning")
workflow.add_edge("planning", "eda")
workflow.add_edge("eda", "modeling")
workflow.add_edge("modeling", "interpretation")
workflow.add_edge("interpretation", END)

# PostgreSQL checkpointing for production
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph = workflow.compile(checkpointer=checkpointer)
```

**Supervisor pattern selection**: Use a flat supervisor for the 4-phase workflow (centralized control, clear phase boundaries). Reserve hierarchical agent architectures for complex sub-tasks within phases, like having the Modeling phase supervisor coordinate separate agents for prior selection, MCMC sampling, and diagnostics.

## Knowledge graph schema for MMM context sharing

Neo4j emerges as the **recommended graph database** for this use case—mature LangChain integration, robust Cypher queries, and proven scalability for enterprise knowledge graphs. FalkorDB is a viable alternative when ultra-low latency GraphRAG queries are paramount.

```cypher
// Core MMM Domain Schema
// Variables with causal metadata
CREATE (v:Variable {
    id: 'tv_spend',
    name: 'TV Media Spend',
    type: 'continuous',
    domain: 'media',
    unit: 'USD',
    typical_lag: [0, 1, 2, 3],
    saturation_type: 'hill'
})

// Causal relationships with confidence scores
CREATE (tv:Variable {id: 'tv_spend'})-[:CAUSES {
    mechanism: 'brand_awareness_lift',
    lag_periods: [0, 1, 2, 3, 4],
    confidence: 0.82,
    evidence_source: 'geo_lift_test_q3_2024',
    adjustment_set: ['seasonality', 'competitor_spend']
}]->(sales:Variable {id: 'revenue'})

// Feature engineering decisions with lineage
CREATE (fe:FeatureDecision {
    id: 'fe_tv_adstock_v2',
    timestamp: datetime(),
    transformation: 'geometric_adstock',
    parameters: {decay_rate: 0.7, l_max: 8},
    rationale: 'Posterior analysis showed peak carryover at week 3'
})
CREATE (tv)-[:DERIVED_INTO]->(tv_adstock:Feature)
CREATE (fe)-[:PRODUCED]->(tv_adstock)

// Model artifacts with full lineage
CREATE (m:Model {
    id: 'mmm_v2.3_2024Q4',
    framework: 'pymc_marketing',
    artifact_path: 's3://models/mmm_v2.3.nc',
    metrics: {mape: 0.12, r2: 0.87, rhat_max: 1.01},
    training_period: ['2022-01-01', '2024-10-01']
})
CREATE (m)-[:TRAINED_ON]->(features:Dataset)
CREATE (m)-[:USES_VARIABLE]->(tv_adstock)
```

### GraphRAG retrieval pattern

Combine vector similarity search with graph traversal for context-aware retrieval that doesn't overwhelm agent context windows:

```python
from langchain_neo4j import Neo4jVector, GraphCypherQAChain

class MMMContextRetriever:
    def __init__(self, graph, vector_store):
        self.graph = graph
        self.vectors = vector_store
        self.context_budget = {
            "graph_relationships": 0.4,
            "vector_chunks": 0.3,
            "phase_history": 0.2,
            "system": 0.1
        }
    
    def get_phase_context(self, phase: str, state: dict, max_tokens: int = 4000):
        if phase == "planning":
            # Similar successful analyses
            similar = self.vectors.similarity_search(
                state["user_query"], k=3, 
                filter={"node_type": "Analysis"}
            )
            patterns = self.graph.query("""
                MATCH (a:Analysis)-[:PRODUCED]->(m:Model)
                WHERE m.metrics.mape < 0.15
                MATCH (a)-[:MADE_DECISION]->(d:Decision {phase: 'planning'})
                RETURN a.query, d.rationale, m.metrics
                ORDER BY m.metrics.mape LIMIT 5
            """)
            return self._compress_context(similar, patterns, max_tokens * 0.7)
        
        elif phase == "modeling":
            # Successful model configs for these variable types
            channels = state.get("media_channels", [])
            configs = self.graph.query("""
                MATCH (m:Model)-[:USES_VARIABLE]->(v:Variable)
                WHERE v.id IN $channels AND m.metrics.rhat_max < 1.05
                RETURN m.framework, m.hyperparameters, m.metrics
                ORDER BY m.metrics.r2 DESC LIMIT 3
            """, {"channels": channels})
            
            # Causal structure priors
            causal_paths = self.graph.query("""
                MATCH path = (v1:Variable)-[:CAUSES*1..2]->(v2:Variable)
                WHERE v1.id IN $channels AND v2.id = 'revenue'
                RETURN path, [r IN relationships(path) | r.confidence]
            """, {"channels": channels})
            
            return {"model_patterns": configs, "causal_structure": causal_paths}
```

## Async architecture separating agents from model fitting

**Ray is the recommended choice for Bayesian model fitting jobs**—native GPU scheduling, built-in progress reporting, and seamless scaling from laptop to cluster. Use **ARQ for async agent I/O operations** and **Redis pub/sub for real-time progress streaming**.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                          │
├─────────────────────────────────────────────────────────────────┤
│  WebSocket /ws/agent/{session}   │  POST /jobs/fit-mmm          │
│  (streaming agent responses)     │  (queue to Ray)              │
└─────────────────────────────────────────────────────────────────┘
              │                                   │
              ▼                                   ▼
    ┌──────────────────┐              ┌─────────────────────┐
    │ LangGraph Agent  │              │ Redis               │
    │ (sync, in-proc)  │              │ • Job queue         │
    └──────────────────┘              │ • Pub/sub progress  │
                                      │ • Result cache      │
                                      └─────────────────────┘
                                                │
                                                ▼
                                      ┌─────────────────────┐
                                      │ Ray Workers         │
                                      │ • PyMC sampling     │
                                      │ • GPU acceleration  │
                                      │ • Progress callbacks│
                                      └─────────────────────┘
```

### WebSocket streaming for agent chat + SSE for model progress

```python
from fastapi import FastAPI, WebSocket
from sse_starlette.sse import EventSourceResponse
import redis.asyncio as redis

app = FastAPI()

# WebSocket for bidirectional agent chat
@app.websocket("/ws/agent/{session_id}")
async def agent_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    state_manager = AgentStateManager(redis_client, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["action"] == "chat":
                # Get current state including any completed model results
                state = await state_manager.get_state()
                context = f"Latest model: {state.get('last_model_summary', 'None')}"
                
                # Stream agent response
                async for token in agent.stream(data["message"], context=context):
                    await websocket.send_json({"type": "token", "content": token})
                await websocket.send_json({"type": "complete"})
                
            elif data["action"] == "fit_model":
                # Submit to Ray, link to session
                job_id = await submit_ray_job(data["config"])
                await state_manager.link_job(job_id)
                await websocket.send_json({"type": "job_started", "job_id": job_id})
                
    except WebSocketDisconnect:
        pass

# SSE for model fitting progress (auto-reconnect, unidirectional)
@app.get("/jobs/{job_id}/progress")
async def stream_progress(job_id: str):
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"job:{job_id}:progress")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                yield {"event": "progress", "data": json.dumps(data)}
                
                if data.get("status") in ["completed", "failed"]:
                    yield {"event": "complete", "data": json.dumps(data)}
                    break
    
    return EventSourceResponse(event_generator())
```

### MCMC progress streaming with Ray

```python
import ray
from ray import train

@ray.remote(num_gpus=1)
def fit_mmm_model(config: dict, job_id: str, redis_url: str):
    import pymc as pm
    import arviz as az
    import redis
    
    r = redis.from_url(redis_url)
    
    # Custom callback for progress streaming
    class StreamingCallback:
        def __init__(self, job_id, redis_client, update_freq=100):
            self.job_id = job_id
            self.r = redis_client
            self.freq = update_freq
        
        def __call__(self, trace, draw):
            if draw.draw_idx % self.freq == 0:
                progress = {
                    "iteration": draw.draw_idx,
                    "chain": draw.chain,
                    "tuning": draw.tuning,
                    "divergences": int(trace.sample_stats.diverging.sum()),
                    "mean_tree_depth": float(trace.sample_stats.tree_depth.mean())
                }
                self.r.publish(f"job:{self.job_id}:progress", json.dumps(progress))
    
    with pm.Model() as model:
        # Build model from config...
        callback = StreamingCallback(job_id, r)
        trace = pm.sample(
            draws=config["draws"],
            tune=config["tune"],
            callback=callback,
            nuts_sampler="numpyro"  # GPU-accelerated
        )
    
    # Save artifact
    idata = az.from_pymc(trace)
    artifact_path = f"s3://models/{job_id}.nc"
    idata.to_netcdf(artifact_path)
    
    r.publish(f"job:{job_id}:progress", json.dumps({
        "status": "completed",
        "artifact_path": artifact_path,
        "diagnostics": {
            "rhat_max": float(az.rhat(idata).max()),
            "ess_min": float(az.ess(idata).min())
        }
    }))
    
    return artifact_path
```

## Incremental Bayesian model building with PyMC-Marketing

**PyMC-Marketing is the recommended framework**—it outperforms Google's Meridian with **40% lower error in channel contribution estimates** while using 3-5x less memory. Start simple and add complexity based on diagnostics.

### Progressive model complexity workflow

```
Model 0: Baseline linear regression
    ↓ Check R², residual patterns
Model 1: Add geometric adstock (single decay parameter)
    ↓ Check posterior decay estimates make business sense
Model 2: Add saturation curves (Hill functions per channel)
    ↓ Check convergence: R-hat < 1.01, ESS > 400
Model 3: Add controls (seasonality Fourier terms, holidays, trend)
    ↓ Posterior predictive checks
Model 4: Add hierarchical structure (if multi-geo)
    ↓ Compare WAIC/LOO between models
Model 5: Time-varying parameters (if residuals show temporal patterns)
```

### Using posteriors as priors for complex models

```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

# Stage 1: Simple model to learn adstock decay
simple_config = {
    "adstock_alpha": Prior("Beta", alpha=1, beta=3),  # Uninformative
    "saturation_lam": Prior("Gamma", alpha=3, beta=1)
}

simple_mmm = MMM(
    model_config=simple_config,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    channel_columns=["tv", "digital", "social"],
    control_columns=["trend"]  # Start minimal
)
simple_mmm.fit(X, y, draws=1000, tune=500)

# Extract posterior summaries
posterior = simple_mmm.fit_result.posterior
adstock_mean = float(posterior["adstock_alpha"].mean())
adstock_std = float(posterior["adstock_alpha"].std())

# Stage 2: Informed priors for complex model
informed_config = {
    "adstock_alpha": Prior("Normal", 
                          mu=adstock_mean, 
                          sigma=adstock_std * 2),  # 2x std for flexibility
    "saturation_lam": Prior("Normal",
                           mu=float(posterior["saturation_lam"].mean()),
                           sigma=float(posterior["saturation_lam"].std()) * 2)
}

complex_mmm = MMM(
    model_config=informed_config,
    adstock=GeometricAdstock(l_max=12),  # Longer memory
    saturation=LogisticSaturation(),
    channel_columns=["tv", "digital", "social"],
    control_columns=["trend", "seasonality", "holiday_effect"],
    yearly_seasonality=4  # Fourier terms
)
complex_mmm.fit(X, y, draws=2000, tune=1000)
```

### Convergence-driven complexity decisions

```python
import arviz as az

def assess_model_and_recommend(idata, residuals):
    """Analyze diagnostics and recommend next complexity step"""
    diagnostics = {
        "rhat_max": float(az.rhat(idata).max()),
        "ess_bulk_min": float(az.ess(idata, method="bulk").min()),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "residual_autocorr": compute_durbin_watson(residuals)
    }
    
    recommendations = []
    
    if diagnostics["rhat_max"] > 1.05:
        recommendations.append("SIMPLIFY: High R-hat suggests poor mixing. "
                             "Consider removing parameters or tightening priors.")
    
    if diagnostics["divergences"] > 0:
        recommendations.append("REPARAMETERIZE: Divergences indicate geometry issues. "
                             "Try non-centered parameterization or increase target_accept.")
    
    if diagnostics["ess_bulk_min"] < 400:
        recommendations.append("EXTEND: Low ESS. Run longer chains or simplify model.")
    
    if abs(diagnostics["residual_autocorr"] - 2.0) > 0.5:
        recommendations.append("ADD_TVP: Residual autocorrelation suggests time-varying "
                             "intercept needed (Gaussian Process or random walk).")
    
    if not recommendations:
        recommendations.append("READY: Model converged well. Consider adding complexity "
                             "if substantive questions remain.")
    
    return diagnostics, recommendations
```

## Causal inference integration with DoWhy

Combine MMM with causal inference methods to validate assumptions and quantify sensitivity to unmeasured confounding. **DoWhy provides the principled framework for encoding causal assumptions as DAGs** and testing robustness.

```python
from dowhy import CausalModel
import dowhy.gcm as gcm

# Define causal graph encoding marketing domain knowledge
model = CausalModel(
    data=df,
    treatment="tv_spend",
    outcome="revenue",
    graph="""digraph {
        tv_spend -> brand_awareness -> revenue;
        tv_spend -> revenue;
        seasonality -> tv_spend;
        seasonality -> revenue;
        competitor_spend -> tv_spend;
        competitor_spend -> revenue;
    }"""
)

# Identify estimand using backdoor criterion
identified_estimand = model.identify_effect()
# Output: Adjustment set = {seasonality, competitor_spend}

# Estimate effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

# Sensitivity analysis for unmeasured confounding
refutation = model.refute_estimate(
    identified_estimand, estimate,
    method_name="add_unobserved_common_cause",
    confounders_effect_on_treatment="binary_flip",
    confounders_effect_on_outcome="linear",
    effect_strength_on_treatment=0.1,
    effect_strength_on_outcome=0.1
)
# Reports: Effect nullified if confounder explains >X% of variance
```

### Counterfactual scenarios with Structural Causal Models

```python
# Fit invertible structural causal model for counterfactuals
scm = gcm.InvertibleStructuralCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(scm, observed_data)
gcm.fit(scm, observed_data)

# What-if: 30% TV budget reallocation to digital
counterfactual_sales = gcm.counterfactual_samples(
    scm,
    {
        'tv_spend': lambda x: x * 0.7,
        'digital_spend': lambda x: x * 1.3 * (original_tv / original_digital)
    },
    observed_data=current_period_data
)

incremental_impact = counterfactual_sales["revenue"].mean() - observed_revenue
```

## Sandboxed code execution with E2B

**E2B (e2b.dev) is the recommended sandbox** for agent code execution—Firecracker microVMs with ~150ms startup, persistent sessions for multi-step analysis, and clean LangChain integration. Use Modal for GPU-intensive Bayesian sampling.

```python
from e2b_code_interpreter import Sandbox

# Create custom template with MMM packages pre-installed
class MMMSandbox:
    def __init__(self, session_id: str):
        self.sandbox = Sandbox.create(template="mmm-analytics")
        self.session_id = session_id
    
    async def execute_eda(self, code: str) -> dict:
        # Validate code before execution
        is_valid, msg = self.validate_code(code)
        if not is_valid:
            return {"error": msg}
        
        execution = self.sandbox.run_code(
            code,
            timeout=300,  # 5 min timeout
            on_stdout=lambda x: self.stream_output(x, "stdout"),
            on_stderr=lambda x: self.stream_output(x, "stderr")
        )
        
        return {
            "results": execution.results,
            "stdout": execution.logs.stdout,
            "error": str(execution.error) if execution.error else None
        }
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """AST-based validation blocking dangerous operations"""
        import ast
        BLOCKED = {'os', 'subprocess', 'sys', 'socket', 'shutil'}
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, 'module', None) or node.names[0].name
                    if module.split('.')[0] in BLOCKED:
                        return False, f"Blocked module: {module}"
            return True, "Valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

# LangGraph tool integration
from langchain_core.tools import tool

@tool
def execute_analysis_code(code: str, session_id: str) -> dict:
    """Execute Python code for data analysis in isolated sandbox."""
    sandbox = get_or_create_sandbox(session_id)
    return sandbox.execute_eda(code)
```

## Model artifact management with ArviZ and MLflow

```python
from pymc_marketing.mlflow import log_inference_data, autolog
import mlflow
import arviz as az

mlflow.set_experiment("mmm_production")

# Auto-log all PyMC-Marketing runs
autolog()

with mlflow.start_run(run_name=f"mmm_v{version}"):
    # Log configuration
    mlflow.log_params({
        "channels": channel_columns,
        "adstock_l_max": 8,
        "saturation_type": "hill",
        "data_hash": hashlib.md5(X.tobytes()).hexdigest()[:8]
    })
    
    # Fit model
    mmm.fit(X, y, draws=2000, tune=1000, nuts_sampler="numpyro")
    
    # Log InferenceData artifact
    log_inference_data(mmm.fit_result, "inference_data.nc")
    
    # Log diagnostics
    mlflow.log_metrics({
        "rhat_max": float(az.rhat(mmm.fit_result).max()),
        "ess_min": float(az.ess(mmm.fit_result).min()),
        "mape": compute_mape(y, mmm.predict(X).mean())
    })
    
    # Save full model for what-if scenarios
    mmm.save("mmm_model.nc")
    mlflow.log_artifact("mmm_model.nc")

# On-demand loading for interpretation phase
loaded_mmm = MMM.load("s3://models/mmm_v2.3.nc")
optimal_budget = loaded_mmm.optimize_budget(
    budget=1_000_000,
    constraints={"min_channel_spend": 50_000}
)
```

## Implementation sequence and technology stack

| Component | Primary Choice | Alternative | Rationale |
|-----------|---------------|-------------|-----------|
| **LLM Orchestration** | LangGraph | — | Native state management, checkpointing, tool integration |
| **LLM Abstraction** | `init_chat_model()` | `configurable_alternatives()` | Runtime provider switching |
| **Graph Database** | Neo4j | FalkorDB | Mature ecosystem, LangChain integration |
| **Message Queue** | Redis + Ray | Celery | Ray native ML support, Redis pub/sub |
| **Bayesian MMM** | PyMC-Marketing | NumPyro | Best accuracy/performance ratio |
| **Causal Inference** | DoWhy | CausalML | Principled DAG-based framework |
| **Code Sandbox** | E2B | Modal (GPU) | Fast startup, persistent sessions |
| **Artifact Storage** | MLflow + S3 | DVC | Native PyMC integration |
| **Progress Streaming** | SSE (model) + WS (chat) | — | SSE auto-reconnects, WS bidirectional |

### Recommended implementation phases

**Phase 1 (Weeks 1-3): Foundation**
- Set up LangGraph workflow with 4-phase state schema
- Implement PostgreSQL checkpointing
- Deploy Neo4j with base MMM schema
- Create multi-LLM provider abstraction

**Phase 2 (Weeks 4-6): Core Modeling**
- Integrate PyMC-Marketing with incremental complexity patterns
- Implement Ray workers for MCMC sampling
- Build MCMC progress streaming via Redis pub/sub
- Create E2B sandbox for EDA code execution

**Phase 3 (Weeks 7-9): Context & Retrieval**
- Implement GraphRAG retrieval for phase context
- Build knowledge graph persistence for decisions/artifacts
- Create context budget management for LLM windows

**Phase 4 (Weeks 10-12): Causal & Interpretation**
- Integrate DoWhy for causal validation
- Build counterfactual scenario engine
- Implement budget optimization tools
- Create interpretation agents with structured outputs

This architecture enables analysts to progress from "What drives our sales?" through causal hypothesis formation, rigorous Bayesian estimation, and actionable budget recommendations—all through natural conversation with persistent, auditable context across sessions.