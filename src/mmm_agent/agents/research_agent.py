"""
Workflow 1: Research Agent

An agentic workflow that helps users create a plan for:
- Data processing requirements
- Data collection for MMM
- Variable selection
- Causal hypothesis formation

Features:
- Web search for domain knowledge
- GraphRAG for storing and retrieving patterns
- User feedback integration
- Structured planning output
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from pydantic import BaseModel, Field

from ..config import Settings, get_settings, create_ollama_llm, LLMTask
from ..db import get_neo4j_client, get_graphrag_manager, Analysis
from ..tools import SearchAgent, research_topic, format_search_results
from .state import (
    ResearchWorkflowState,
    WorkflowPhase,
    ResearchPlan,
    CausalHypothesis,
    DataRequirements,
)


# =============================================================================
# Structured Outputs
# =============================================================================

class ResearchQuestions(BaseModel):
    """Generated research questions."""
    questions: list[str] = Field(description="Research questions to explore")
    reasoning: str = Field(description="Why these questions matter")


class VariableSelection(BaseModel):
    """Variable selection output."""
    target_variable: str = Field(description="Main KPI/target variable")
    media_channels: list[str] = Field(description="Media channels to model")
    control_variables: list[str] = Field(description="Control variables to include")
    rationale: str = Field(description="Reasoning for selections")


class PlanningOutput(BaseModel):
    """Complete planning phase output."""
    research_questions: list[str] = Field(default_factory=list)
    target_variable: str = "revenue"
    media_channels: list[str] = Field(default_factory=list)
    control_variables: list[str] = Field(default_factory=list)
    causal_hypotheses: list[CausalHypothesis] = Field(default_factory=list)
    data_requirements: DataRequirements = Field(default_factory=DataRequirements)
    summary: str = ""


# =============================================================================
# System Prompts
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are an expert Marketing Mix Model (MMM) research analyst.

Your role is to:
1. Understand the user's business question and objectives
2. Research best practices and patterns for similar MMM analyses
3. Generate insightful research questions
4. Identify relevant data requirements

Use the provided context from previous successful analyses and web research.
Be thorough but practical in your recommendations.
"""

PLANNING_SYSTEM_PROMPT = """You are an expert MMM planning specialist.

Based on the research conducted, your role is to:
1. Identify the appropriate KPI/target variable
2. Recommend media channels to analyze
3. Suggest control variables to avoid confounding
4. Formulate testable causal hypotheses
5. Define specific data requirements

Provide clear rationale for each recommendation.
Consider practical constraints and data availability.
"""


# =============================================================================
# Workflow Nodes
# =============================================================================

async def initialize_research(state: ResearchWorkflowState) -> dict:
    """Initialize the research workflow."""
    logger.info("🔬 Initializing research workflow")
    
    # Generate IDs
    session_id = state.get("session_id") or str(uuid.uuid4())[:8]
    analysis_id = state.get("analysis_id") or f"analysis_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create analysis in Neo4j
    neo4j = get_neo4j_client()
    analysis = Analysis(
        id=analysis_id,
        query=state.get("user_query", ""),
        user_id=state.get("user_id"),
        status="research",
    )
    neo4j.create_analysis(analysis)
    
    return {
        "session_id": session_id,
        "analysis_id": analysis_id,
        "current_phase": WorkflowPhase.RESEARCH_SEARCH,
        "messages": [f"[{datetime.now().isoformat()}] Research workflow initialized"],
    }


async def conduct_web_research(state: ResearchWorkflowState) -> dict:
    """Conduct web research on the user's topic."""
    logger.info("🔍 Conducting web research")
    
    user_query = state.get("user_query", "")
    business_context = state.get("business_context", "")
    
    # Get LLM for research
    settings = get_settings()
    llm = create_ollama_llm(task=LLMTask.REASONING, settings=settings)
    
    # Create search agent
    agent = SearchAgent(
        llm=llm,
        max_iterations=2,
        results_per_query=4,
        enable_crawl=True,
        max_crawl_urls=3,
    )
    
    # Conduct research
    search_context = await agent.search(
        query=f"{user_query} marketing mix model best practices",
        context=business_context,
    )
    
    # Store results in GraphRAG
    graphrag = get_graphrag_manager()
    if search_context.results and search_context.synthesis:
        graphrag.add_research_result(
            analysis_id=state.get("analysis_id", ""),
            query=user_query,
            summary=search_context.synthesis.summary,
            sources=[{"title": r.title, "url": r.url, "snippet": r.content[:200]} 
                    for r in search_context.results[:5]],
            insights=search_context.synthesis.key_insights,
        )
    
    # Get prior patterns from graph
    prior_patterns = graphrag.neo4j.get_successful_mmm_patterns(
        query=user_query,
        max_mape=0.15,
        limit=3,
    )
    
    return {
        "web_search_results": [r.model_dump() for r in search_context.results],
        "research_summary": search_context.synthesis.summary if search_context.synthesis else "",
        "domain_insights": search_context.synthesis.key_insights if search_context.synthesis else [],
        "prior_patterns": prior_patterns,
        "current_phase": WorkflowPhase.RESEARCH_PLAN,
        "messages": [f"[{datetime.now().isoformat()}] Web research completed: {len(search_context.results)} results"],
    }


async def generate_research_plan(state: ResearchWorkflowState) -> dict:
    """Generate a comprehensive research plan."""
    logger.info("📋 Generating research plan")
    
    user_query = state.get("user_query", "")
    business_context = state.get("business_context", "")
    research_summary = state.get("research_summary", "")
    domain_insights = state.get("domain_insights", [])
    prior_patterns = state.get("prior_patterns", [])
    
    # Get LLM
    settings = get_settings()
    llm = create_ollama_llm(task=LLMTask.PLANNING, settings=settings)
    
    # Build context
    context_parts = []
    if research_summary:
        context_parts.append(f"Research Summary:\n{research_summary}")
    if domain_insights:
        context_parts.append(f"Key Insights:\n" + "\n".join(f"- {i}" for i in domain_insights))
    if prior_patterns:
        context_parts.append(f"Successful Prior Patterns:\n{prior_patterns[:2]}")
    
    context = "\n\n".join(context_parts)
    
    # Generate structured plan
    structured_llm = llm.with_structured_output(PlanningOutput)
    
    prompt = f"""Create a comprehensive MMM analysis plan based on:

User's Question: {user_query}

Business Context: {business_context}

Research Findings:
{context}

Provide:
1. Key research questions to answer
2. Target variable (KPI) recommendation
3. Media channels to include
4. Control variables needed
5. Causal hypotheses with mechanisms
6. Specific data requirements

Be practical and actionable. Consider typical data availability constraints.
"""
    
    try:
        plan = structured_llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        # Convert to dict for state
        plan_dict = plan.model_dump()
        
        # Store decisions in GraphRAG
        graphrag = get_graphrag_manager()
        graphrag.add_decision(
            analysis_id=state.get("analysis_id", ""),
            phase="planning",
            decision_type="variable_selection",
            content={
                "target": plan.target_variable,
                "media": plan.media_channels,
                "controls": plan.control_variables,
            },
            rationale=plan.summary,
        )
        
        return {
            "research_plan": plan_dict,
            "target_variable": plan.target_variable,
            "media_channels": plan.media_channels,
            "control_variables": plan.control_variables,
            "causal_hypotheses": [h.model_dump() for h in plan.causal_hypotheses],
            "data_requirements": plan.data_requirements.model_dump(),
            "current_phase": WorkflowPhase.HUMAN_INPUT,
            "messages": [f"[{datetime.now().isoformat()}] Research plan generated"],
        }
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {
            "errors": [f"Planning failed: {str(e)}"],
            "current_phase": WorkflowPhase.ERROR,
        }


async def collect_user_feedback(state: ResearchWorkflowState) -> dict:
    """
    Collect user feedback on the plan.
    
    This is a human-in-the-loop node that pauses for user input.
    In production, this would be handled by the API layer.
    """
    logger.info("👤 Awaiting user feedback")
    
    # This node is designed to pause the workflow
    # The actual feedback collection happens through the API
    
    return {
        "current_phase": WorkflowPhase.HUMAN_INPUT,
        "messages": [f"[{datetime.now().isoformat()}] Awaiting user feedback on plan"],
    }


async def process_feedback(state: ResearchWorkflowState) -> dict:
    """Process user feedback and update plan if needed."""
    logger.info("🔄 Processing user feedback")
    
    feedback = state.get("user_feedback", "")
    plan_approved = state.get("plan_approved", False)
    
    if plan_approved or not feedback:
        # Plan approved or no feedback - proceed
        return {
            "current_phase": WorkflowPhase.RESEARCH_COMPLETE,
            "messages": [f"[{datetime.now().isoformat()}] Plan approved"],
        }
    
    # User provided feedback - refine the plan
    settings = get_settings()
    llm = create_ollama_llm(task=LLMTask.PLANNING, settings=settings)
    
    current_plan = state.get("research_plan", {})
    
    structured_llm = llm.with_structured_output(PlanningOutput)
    
    prompt = f"""Refine the MMM analysis plan based on user feedback.

Current Plan:
{current_plan}

User Feedback:
{feedback}

Update the plan to address the feedback while maintaining best practices.
"""
    
    try:
        refined_plan = structured_llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        plan_dict = refined_plan.model_dump()
        
        # Store refined decision
        graphrag = get_graphrag_manager()
        graphrag.add_decision(
            analysis_id=state.get("analysis_id", ""),
            phase="planning",
            decision_type="plan_refinement",
            content=plan_dict,
            rationale=f"Refined based on feedback: {feedback[:200]}",
        )
        
        # Store user feedback
        graphrag.add_user_feedback(
            analysis_id=state.get("analysis_id", ""),
            feedback=feedback,
            context=f"Planning phase refinement",
        )
        
        return {
            "research_plan": plan_dict,
            "target_variable": refined_plan.target_variable,
            "media_channels": refined_plan.media_channels,
            "control_variables": refined_plan.control_variables,
            "causal_hypotheses": [h.model_dump() for h in refined_plan.causal_hypotheses],
            "data_requirements": refined_plan.data_requirements.model_dump(),
            "current_phase": WorkflowPhase.RESEARCH_COMPLETE,
            "messages": [f"[{datetime.now().isoformat()}] Plan refined based on feedback"],
        }
        
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        return {
            "errors": [f"Refinement failed: {str(e)}"],
            "current_phase": WorkflowPhase.RESEARCH_COMPLETE,  # Proceed with original
        }


async def finalize_research(state: ResearchWorkflowState) -> dict:
    """Finalize the research workflow."""
    logger.info("✅ Finalizing research workflow")
    
    # Update analysis status in Neo4j
    neo4j = get_neo4j_client()
    neo4j.update_analysis_status(
        analysis_id=state.get("analysis_id", ""),
        status="research_complete",
    )
    
    return {
        "current_phase": WorkflowPhase.RESEARCH_COMPLETE,
        "messages": [f"[{datetime.now().isoformat()}] Research workflow complete"],
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_init(state: ResearchWorkflowState) -> str:
    """Route after initialization."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "search"


def route_after_search(state: ResearchWorkflowState) -> str:
    """Route after web research."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "plan"


def route_after_plan(state: ResearchWorkflowState) -> str:
    """Route after plan generation."""
    if state.get("current_phase") == WorkflowPhase.ERROR:
        return "error"
    return "feedback"


def route_after_feedback(state: ResearchWorkflowState) -> Literal["process", "complete"]:
    """Route after feedback collection."""
    # Check if we have feedback to process
    if state.get("user_feedback"):
        return "process"
    if state.get("plan_approved"):
        return "complete"
    # Default: wait for feedback (this node will be re-entered)
    return "complete"


# =============================================================================
# Workflow Builder
# =============================================================================

class ResearchWorkflow:
    """
    Research Agent Workflow.
    
    Helps users create MMM plans through:
    - Web research for best practices
    - Knowledge graph retrieval for patterns
    - Structured planning
    - User feedback integration
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
        graph = StateGraph(ResearchWorkflowState)
        
        # Add nodes
        graph.add_node("init", initialize_research)
        graph.add_node("search", conduct_web_research)
        graph.add_node("plan", generate_research_plan)
        graph.add_node("feedback", collect_user_feedback)
        graph.add_node("process", process_feedback)
        graph.add_node("complete", finalize_research)
        graph.add_node("error", lambda s: {"current_phase": WorkflowPhase.ERROR})
        
        # Add edges
        graph.set_entry_point("init")
        graph.add_conditional_edges("init", route_after_init, {
            "search": "search",
            "error": "error",
        })
        graph.add_conditional_edges("search", route_after_search, {
            "plan": "plan",
            "error": "error",
        })
        graph.add_conditional_edges("plan", route_after_plan, {
            "feedback": "feedback",
            "error": "error",
        })
        graph.add_conditional_edges("feedback", route_after_feedback, {
            "process": "process",
            "complete": "complete",
        })
        graph.add_edge("process", "complete")
        graph.add_edge("complete", END)
        graph.add_edge("error", END)
        
        return graph
    
    def compile(self) -> "ResearchWorkflow":
        """Compile the workflow."""
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        return self
    
    async def run(
        self,
        user_query: str,
        business_context: str = "",
        user_id: str | None = None,
        config: dict | None = None,
    ) -> ResearchWorkflowState:
        """
        Run the research workflow.
        
        Args:
            user_query: User's business question
            business_context: Additional context
            user_id: Optional user identifier
            config: LangGraph config
        
        Returns:
            Final workflow state with research plan
        """
        if not self.compiled:
            self.compile()
        
        initial_state: ResearchWorkflowState = {
            "user_query": user_query,
            "business_context": business_context,
            "user_id": user_id,
            "current_phase": WorkflowPhase.RESEARCH_INIT,
            "messages": [],
            "errors": [],
            "web_search_results": [],
            "domain_insights": [],
            "prior_patterns": [],
            "plan_approved": False,
        }
        
        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {
                "thread_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        
        final_state = None
        async for event in self.compiled.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                final_state = state_update
                logger.debug(f"Node {node_name}: {state_update.get('current_phase')}")
        
        return final_state
    
    async def submit_feedback(
        self,
        thread_id: str,
        feedback: str,
        approved: bool = False,
    ) -> ResearchWorkflowState:
        """
        Submit user feedback to continue workflow.
        
        Args:
            thread_id: Workflow thread ID
            feedback: User's feedback
            approved: Whether plan is approved
        
        Returns:
            Updated workflow state
        """
        if not self.compiled:
            self.compile()
        
        # Update state with feedback
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state and update
        current_state = self.compiled.get_state(config)
        
        update = {
            "user_feedback": feedback,
            "plan_approved": approved,
        }
        
        # Continue execution
        final_state = None
        async for event in self.compiled.astream(update, config=config):
            for node_name, state_update in event.items():
                final_state = state_update
        
        return final_state


# =============================================================================
# Factory Function
# =============================================================================

def create_research_workflow(
    settings: Settings | None = None,
    postgres_url: str | None = None,
) -> ResearchWorkflow:
    """
    Create a research workflow instance.
    
    Args:
        settings: Optional settings override
        postgres_url: PostgreSQL URL for checkpointing
    
    Returns:
        Compiled ResearchWorkflow
    """
    settings = settings or get_settings()
    
    checkpointer = None
    if postgres_url or settings.postgres_url:
        try:
            checkpointer = PostgresSaver.from_conn_string(
                postgres_url or settings.postgres_url
            )
        except Exception as e:
            logger.warning(f"Could not create checkpointer: {e}")
    
    workflow = ResearchWorkflow(
        settings=settings,
        checkpointer=checkpointer,
    )
    
    return workflow.compile()
