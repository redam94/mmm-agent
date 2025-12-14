"""
MMM Workflow Orchestration - LangGraph StateGraph for multi-phase MMM analysis.

This module defines the main workflow that orchestrates the four phases:
Planning → EDA → Modeling → Interpretation

The workflow uses conditional edges for error handling and supports
progress callbacks for real-time updates.
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import MMMWorkflowState, WorkflowPhase
from .config import Settings, get_settings, create_chat_model, get_llm_config
from .agents import (
    PlanningAgent, planning_node,
    EDAAgent, eda_node,
    ModelingAgent, modeling_node,
    InterpretationAgent, interpretation_node,
)
from .tools.code_executor import LocalCodeExecutor
from .tools.rag_context import ContextManager
from .tools.web_search import ResearchTool


logger = logging.getLogger(__name__)


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_planning(state: MMMWorkflowState) -> Literal["eda", "error"]:
    """Route after planning phase based on state."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    if state.get("target_variable") and state.get("media_channels"):
        return "eda"
    logger.warning("Planning incomplete - missing target or channels")
    return "error"


def route_after_eda(state: MMMWorkflowState) -> Literal["modeling", "error"]:
    """Route after EDA phase based on data quality."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    # Check if we have sufficient data quality
    quality_report = state.get("data_quality_report", {})
    if quality_report.get("blocking_issues"):
        logger.warning(f"EDA found blocking issues: {quality_report['blocking_issues']}")
        return "error"
    return "modeling"


def route_after_modeling(state: MMMWorkflowState) -> Literal["interpretation", "error"]:
    """Route after modeling phase based on convergence."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    # Check convergence diagnostics
    diagnostics = state.get("convergence_diagnostics", {})
    if diagnostics.get("convergence_status") == "failed":
        logger.warning("Model failed to converge")
        # For POC, continue anyway with warning
    return "interpretation"


def route_after_interpretation(state: MMMWorkflowState) -> Literal["complete", "error"]:
    """Route after interpretation - always complete unless error."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "complete"


# ============================================================================
# Terminal Nodes
# ============================================================================

def complete_node(state: MMMWorkflowState) -> Dict[str, Any]:
    """Terminal node for successful completion."""
    logger.info("Workflow completed successfully")
    return {
        "current_phase": WorkflowPhase.COMPLETE,
        "messages": [f"[{datetime.now().isoformat()}] Workflow completed successfully"],
    }


def error_node(state: MMMWorkflowState) -> Dict[str, Any]:
    """Terminal node for error handling."""
    errors = state.get("errors", [])
    logger.error(f"Workflow ended with errors: {errors}")
    return {
        "current_phase": WorkflowPhase.ERROR,
        "messages": [f"[{datetime.now().isoformat()}] Workflow ended with errors"],
    }


# ============================================================================
# Workflow Builder
# ============================================================================

class MMMWorkflow:
    """
    Main MMM Workflow orchestrator.
    
    Manages the LangGraph StateGraph and provides methods for:
    - Building and compiling the workflow
    - Running with progress callbacks
    - Accessing intermediate results
    
    Example:
        workflow = MMMWorkflow()
        result = await workflow.run(
            data_sources=[{"path": "sales.csv", "type": "sales"}],
            business_context="Q4 marketing effectiveness analysis",
            progress_callback=lambda phase, msg: print(f"{phase}: {msg}")
        )
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_bayesian: bool = False,  # False for POC (Ridge regression)
        enable_web_search: bool = True,
    ):
        """
        Initialize the MMM workflow.
        
        Args:
            settings: Configuration settings (uses defaults if None)
            use_bayesian: Whether to use full Bayesian MMM (False = Ridge regression)
            enable_web_search: Whether to enable web search for research
        """
        self.settings = settings or get_settings()
        self.use_bayesian = use_bayesian
        self.enable_web_search = enable_web_search
        
        # Initialize shared resources
        self.code_executor = LocalCodeExecutor()
        self.context_manager = ContextManager()
        self.research_tool = ResearchTool() if enable_web_search else None
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled = None
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        # Create the graph with our state schema
        graph = StateGraph(MMMWorkflowState)
        
        # Add nodes for each phase
        graph.add_node("planning", self._create_planning_node())
        graph.add_node("eda", self._create_eda_node())
        graph.add_node("modeling", self._create_modeling_node())
        graph.add_node("interpretation", self._create_interpretation_node())
        graph.add_node("complete", complete_node)
        graph.add_node("error", error_node)
        
        # Set entry point
        graph.set_entry_point("planning")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "planning",
            route_after_planning,
            {"eda": "eda", "error": "error"}
        )
        graph.add_conditional_edges(
            "eda", 
            route_after_eda,
            {"modeling": "modeling", "error": "error"}
        )
        graph.add_conditional_edges(
            "modeling",
            route_after_modeling,
            {"interpretation": "interpretation", "error": "error"}
        )
        graph.add_conditional_edges(
            "interpretation",
            route_after_interpretation,
            {"complete": "complete", "error": "error"}
        )
        
        # Terminal nodes go to END
        graph.add_edge("complete", END)
        graph.add_edge("error", END)
        
        return graph
    
    def _create_planning_node(self) -> Callable:
        """Create the planning node function with injected dependencies."""
        llm_config = get_llm_config(self.settings, task_type="reasoning")
        llm = create_chat_model(llm_config)
        
        async def node(state: MMMWorkflowState) -> Dict[str, Any]:
            # Get context from context_manager
            context = ""
            if self.context_manager:
                context = self.context_manager.get_context(
                    state.get("workflow_id", "default"),
                    "planning",
                    state.get("user_query", ""),
                )
            
            agent = PlanningAgent(llm=llm, research_tool=self.research_tool)
            return await agent.plan(state, context)
        
        return node

    def _create_eda_node(self) -> Callable:
        """Create the EDA node function with injected dependencies."""
        llm_config = get_llm_config(self.settings, task_type="code")
        llm = create_chat_model(llm_config)
        
        async def node(state: MMMWorkflowState) -> Dict[str, Any]:
            context = ""
            if self.context_manager:
                context = self.context_manager.get_context(
                    state.get("workflow_id", "default"),
                    "eda",
                    "",
                )
            
            agent = EDAAgent(llm=llm, code_executor=self.code_executor)
            return await agent.analyze(state, context)
        
        return node

    def _create_modeling_node(self) -> Callable:
        """Create the modeling node function with injected dependencies."""
        llm_config = get_llm_config(self.settings, task_type="code")
        llm = create_chat_model(llm_config)
        
        async def node(state: MMMWorkflowState) -> Dict[str, Any]:
            context = ""
            if self.context_manager:
                context = self.context_manager.get_context(
                    state.get("workflow_id", "default"),
                    "modeling",
                    "",
                )
            
            agent = ModelingAgent(
                llm=llm,
                code_executor=self.code_executor,
                use_bayesian=self.use_bayesian,
            )
            return await agent.model(state, context)
        
        return node

    def _create_interpretation_node(self) -> Callable:
        """Create the interpretation node function with injected dependencies."""
        llm_config = get_llm_config(self.settings, task_type="reasoning")
        llm = create_chat_model(llm_config)
        
        async def node(state: MMMWorkflowState) -> Dict[str, Any]:
            context = ""
            if self.context_manager:
                context = self.context_manager.get_context(
                    state.get("workflow_id", "default"),
                    "interpretation",
                    "",
                )
            
            agent = InterpretationAgent(llm=llm, code_executor=self.code_executor)
            return await agent.interpret(state, context)
        
        return node
    
    def compile(self, checkpointer: Optional[MemorySaver] = None) -> "MMMWorkflow":
        """
        Compile the workflow graph.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            
        Returns:
            Self for method chaining
        """
        self.compiled = self.graph.compile(checkpointer=checkpointer)
        return self
    
    async def run(
        self,
        data_sources: list[Dict[str, Any]],
        business_context: str,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MMMWorkflowState:
        """
        Run the complete MMM workflow.
        
        Args:
            data_sources: List of data source specifications with paths and metadata
            business_context: Description of the business problem and objectives
            progress_callback: Optional callback(phase, message) for progress updates
            config: Optional LangGraph config (thread_id, etc.)
            
        Returns:
            Final workflow state with all results
        """
        if not self.compiled:
            self.compile()
        
        # Initialize state
        initial_state: MMMWorkflowState = {
            "messages": [f"[{datetime.now().isoformat()}] Starting MMM workflow"],
            "current_phase": WorkflowPhase.PLANNING,
            "data_sources": data_sources,
            "business_context": business_context,
            "errors": [],
            # Planning outputs
            "research_questions": [],
            "target_variable": None,
            "media_channels": [],
            "control_variables": [],
            "causal_hypotheses": [],
            # EDA outputs
            "data_quality_report": {},
            "feature_transformations": [],
            "correlation_matrix": None,
            "mff_data_path": None,
            # Modeling outputs
            "model_artifact_path": None,
            "convergence_diagnostics": {},
            "channel_contributions": {},
            # Interpretation outputs
            "roi_estimates": {},
            "budget_allocation": {},
            "what_if_scenarios": [],
            "recommendations": [],
            # Context
            "rag_context": {},
            "web_search_results": [],
            "prior_decisions": [],
        }
        
        # Set up config with thread_id for checkpointing
        run_config = config or {}
        if "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": f"mmm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}
        
        # Run with streaming for progress updates
        final_state = None
        
        async for event in self.compiled.astream(initial_state, config=run_config):
            # Extract node name and state updates
            for node_name, state_update in event.items():
                if progress_callback:
                    phase = state_update.get("current_phase", WorkflowPhase.PLANNING)
                    messages = state_update.get("messages", [])
                    if messages:
                        progress_callback(phase.value if hasattr(phase, 'value') else str(phase), messages[-1])
                
                # Keep track of final state
                final_state = state_update
        
        # Clean up code executor
        self.code_executor.cleanup_session()
        
        return final_state
    
    def get_visualization(self) -> str:
        """Get a Mermaid diagram of the workflow."""
        if not self.compiled:
            self.compile()
        return self.compiled.get_graph().draw_mermaid()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_workflow(
    provider: str = "ollama",
    model: Optional[str] = None,
    use_bayesian: bool = False,
    enable_web_search: bool = True,
) -> MMMWorkflow:
    """
    Create an MMM workflow with specified configuration.
    
    Args:
        provider: LLM provider ("ollama", "openai", "anthropic", "gemini")
        model: Optional model name override
        use_bayesian: Whether to use Bayesian MMM (False = Ridge regression)
        enable_web_search: Whether to enable web search
        
    Returns:
        Configured MMMWorkflow instance
    """
    from .config import LLMProvider
    
    settings = Settings(
        provider=LLMProvider(provider),
        default_model=model or Settings().default_model,
    )
    
    return MMMWorkflow(
        settings=settings,
        use_bayesian=use_bayesian,
        enable_web_search=enable_web_search,
    )


async def run_mmm_analysis(
    data_sources: list[Dict[str, Any]],
    business_context: str,
    provider: str = "ollama",
    use_bayesian: bool = False,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> MMMWorkflowState:
    """
    Run a complete MMM analysis with default settings.
    
    Args:
        data_sources: List of data source specifications
        business_context: Business problem description
        provider: LLM provider to use
        use_bayesian: Whether to use Bayesian MMM
        progress_callback: Optional progress callback
        
    Returns:
        Final workflow state with all results
        
    Example:
        result = await run_mmm_analysis(
            data_sources=[
                {"path": "sales_data.csv", "type": "sales"},
                {"path": "media_spend.csv", "type": "media"},
            ],
            business_context="Analyze Q4 2024 marketing effectiveness for Product X",
            provider="anthropic",
        )
        print(result["recommendations"])
    """
    workflow = create_workflow(
        provider=provider,
        use_bayesian=use_bayesian,
        enable_web_search=True,
    )
    
    return await workflow.run(
        data_sources=data_sources,
        business_context=business_context,
        progress_callback=progress_callback,
    )


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for running MMM workflow."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run MMM Analysis Workflow")
    parser.add_argument(
        "--data", "-d",
        required=True,
        nargs="+",
        help="Data source files (CSV, Excel, or Parquet)"
    )
    parser.add_argument(
        "--context", "-c",
        required=True,
        help="Business context description"
    )
    parser.add_argument(
        "--provider", "-p",
        default="ollama",
        choices=["ollama", "openai", "anthropic", "gemini"],
        help="LLM provider"
    )
    parser.add_argument(
        "--bayesian", "-b",
        action="store_true",
        help="Use Bayesian MMM (requires PyMC)"
    )
    parser.add_argument(
        "--output", "-o",
        default="mmm_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Build data sources from file paths
    data_sources = []
    for path in args.data:
        # Infer type from filename
        source_type = "unknown"
        lower_path = path.lower()
        if "sales" in lower_path or "revenue" in lower_path:
            source_type = "sales"
        elif "media" in lower_path or "spend" in lower_path or "marketing" in lower_path:
            source_type = "media"
        elif "weather" in lower_path or "temp" in lower_path:
            source_type = "external"
        
        data_sources.append({"path": path, "type": source_type})
    
    # Progress callback
    def progress(phase: str, message: str):
        print(f"[{phase.upper()}] {message}")
    
    # Run the workflow
    print(f"\n{'='*60}")
    print("MMM AGENT WORKFLOW")
    print(f"{'='*60}")
    print(f"Data Sources: {args.data}")
    print(f"Provider: {args.provider}")
    print(f"Bayesian: {args.bayesian}")
    print(f"{'='*60}\n")
    
    result = asyncio.run(run_mmm_analysis(
        data_sources=data_sources,
        business_context=args.context,
        provider=args.provider,
        use_bayesian=args.bayesian,
        progress_callback=progress,
    ))
    
    # Save results
    # Convert state to JSON-serializable format
    output = {
        "status": result.get("current_phase", "unknown"),
        "recommendations": result.get("recommendations", []),
        "roi_estimates": result.get("roi_estimates", {}),
        "budget_allocation": result.get("budget_allocation", {}),
        "channel_contributions": result.get("channel_contributions", {}),
        "data_quality_report": result.get("data_quality_report", {}),
        "errors": result.get("errors", []),
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    
    if result.get("recommendations"):
        print("\nTop Recommendations:")
        for i, rec in enumerate(result["recommendations"][:3], 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
