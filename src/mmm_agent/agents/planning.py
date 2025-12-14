"""
Planning Agent for MMM Workflow

Handles the first phase of the MMM workflow:
- Understanding user requirements
- Identifying KPI and target variables
- Suggesting media channels and controls
- Formulating causal hypotheses
- Defining data requirements
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from ..state import MMMWorkflowState, WorkflowPhase, Decision


# =============================================================================
# Structured Outputs
# =============================================================================

class CausalHypothesis(BaseModel):
    """A causal hypothesis for the MMM."""
    treatment: str = Field(description="Treatment variable (e.g., TV spend)")
    outcome: str = Field(description="Outcome variable (e.g., Revenue)")
    mechanism: str = Field(description="Hypothesized mechanism")
    expected_lag: list[int] = Field(description="Expected lag periods in weeks")
    confounders: list[str] = Field(description="Potential confounders")
    confidence: float = Field(description="Confidence in hypothesis (0-1)", ge=0, le=1)


class DataRequirements(BaseModel):
    """Data requirements for the MMM."""
    minimum_periods: int = Field(description="Minimum number of time periods needed")
    required_dimensions: list[str] = Field(description="Required dimensions (e.g., Geography)")
    required_variables: list[str] = Field(description="Variables that must be present")
    nice_to_have_variables: list[str] = Field(description="Variables that would improve model")


class PlanningOutput(BaseModel):
    """Complete output from planning phase."""
    research_questions: list[str] = Field(description="Key research questions to answer")
    target_variable: str = Field(description="Main KPI/target variable")
    media_channels: list[str] = Field(description="Media channels to model")
    control_variables: list[str] = Field(description="Control variables to include")
    causal_hypotheses: list[CausalHypothesis] = Field(description="Causal hypotheses")
    data_requirements: DataRequirements = Field(description="Data requirements")
    summary: str = Field(description="Summary of planning decisions")


# =============================================================================
# Planning Agent
# =============================================================================

PLANNING_SYSTEM_PROMPT = """You are an expert Marketing Mix Model (MMM) analyst helping to plan an MMM analysis.

Your role is to:
1. Understand the user's business question and objectives
2. Identify the appropriate KPI/target variable
3. Suggest relevant media channels to analyze
4. Recommend control variables to avoid confounding
5. Formulate causal hypotheses about marketing effects
6. Define data requirements

Key MMM Principles:
- Media effects typically have carryover (adstock) and diminishing returns (saturation)
- Common confounders: seasonality, economic conditions, competitor activity, pricing
- Need sufficient data variation to estimate effects (2+ years weekly data ideal)
- Geographic or product variation helps identification

When making recommendations, explain your reasoning and flag any concerns."""


PLANNING_USER_PROMPT = """# User Query
{user_query}

# Available Data Files
{data_paths}

# Context
{context}

Please analyze this request and provide a comprehensive MMM planning output including:
1. Research questions to answer
2. Target variable (KPI)
3. Media channels to analyze
4. Control variables to include
5. Causal hypotheses with mechanisms
6. Data requirements

Be specific and practical in your recommendations."""


class PlanningAgent:
    """
    Agent for MMM planning phase.
    
    Responsibilities:
    - Parse user requirements
    - Research industry context
    - Formulate analysis plan
    - Define data needs
    """
    
    def __init__(self, llm, research_tool=None):
        """
        Initialize planning agent.
        
        Args:
            llm: LangChain chat model
            research_tool: Optional web research tool
        """
        self.llm = llm
        self.research_tool = research_tool
        self.structured_llm = llm.with_structured_output(PlanningOutput)
    
    async def plan(
        self,
        state: MMMWorkflowState,
        context: str = "",
        on_progress = None,
    ) -> dict[str, Any]:
        """
        Execute planning phase.
        
        Args:
            state: Current workflow state
            context: RAG/search context
            on_progress: Progress callback
        
        Returns:
            Dict of state updates
        """
        user_query = state.get("user_query", "")
        data_paths = state.get("data_paths", [])
        
        if on_progress:
            await on_progress("Planning", "Analyzing requirements...")
        
        # Optionally do web research
        web_results = []
        if self.research_tool and user_query:
            if on_progress:
                await on_progress("Planning", "Researching industry context...")
            
            try:
                web_results = await self.research_tool.research(
                    user_query,
                    phase="planning",
                    context=context,
                )
                
                # Add to context
                if web_results:
                    web_context = "\n".join([
                        f"- {r['title']}: {r['snippet'][:200]}"
                        for r in web_results[:5]
                    ])
                    context = f"{context}\n\n## Web Research\n{web_context}"
            except Exception as e:
                logger.warning(f"Web research failed: {e}")
        
        if on_progress:
            await on_progress("Planning", "Generating analysis plan...")
        
        # Generate plan
        prompt = PLANNING_USER_PROMPT.format(
            user_query=user_query,
            data_paths="\n".join(f"- {p}" for p in data_paths) if data_paths else "No data files provided",
            context=context or "No additional context",
        )
        
        try:
            result = self.structured_llm.invoke([
                SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            
            logger.info(f"Planning complete: {len(result.media_channels)} channels, "
                       f"{len(result.control_variables)} controls")
            
            # Build state update
            return {
                "research_questions": result.research_questions,
                "target_variable": result.target_variable,
                "media_channels": result.media_channels,
                "control_variables": result.control_variables,
                "causal_hypotheses": [h.model_dump() for h in result.causal_hypotheses],
                "data_requirements": result.data_requirements.model_dump(),
                "planning_summary": result.summary,
                "web_search_results": web_results,
                "current_phase": WorkflowPhase.EDA,
                "prior_decisions": [Decision(
                    phase=WorkflowPhase.PLANNING,
                    decision_type="analysis_plan",
                    content={
                        "target": result.target_variable,
                        "channels": result.media_channels,
                        "controls": result.control_variables,
                    },
                    rationale=result.summary,
                ).model_dump()],
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                "error": str(e),
                "current_phase": WorkflowPhase.ERROR,
            }
    
    def plan_sync(
        self,
        state: MMMWorkflowState,
        context: str = "",
    ) -> dict[str, Any]:
        """Synchronous planning wrapper."""
        import asyncio
        return asyncio.run(self.plan(state, context))


# =============================================================================
# LangGraph Node
# =============================================================================

async def planning_node(state: MMMWorkflowState, deps: dict) -> dict:
    """
    LangGraph node for planning phase.
    
    Args:
        state: Workflow state
        deps: Dependencies (llm, research_tool, context_manager)
    
    Returns:
        State updates
    """
    llm = deps.get("llm")
    research_tool = deps.get("research_tool")
    context_manager = deps.get("context_manager")
    on_progress = deps.get("on_progress")
    
    # Get context
    context = ""
    if context_manager:
        context = context_manager.get_context(
            state.get("workflow_id", "default"),
            "planning",
            state.get("user_query", ""),
        )
    
    agent = PlanningAgent(llm, research_tool)
    return await agent.plan(state, context, on_progress)
