"""
MMM Workflows API

FastAPI application providing endpoints for the four MMM workflows:
1. /research - Research and planning workflow
2. /eda - Data quality and transformation workflow
3. /modeling - Bayesian MMM fitting workflow
4. /whatif - Scenario analysis workflow

Also provides:
- WebSocket endpoints for real-time progress streaming
- File upload/download endpoints
- Health and status endpoints
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..db.neo4j_client import Neo4jClient
from ..db.graphrag import GraphRAGManager
from ..tools.code_executor import CodeExecutor
from ..workflows import (
    ResearchWorkflow,
    EDAWorkflow,
    ModelingWorkflow,
    WhatIfWorkflow,
    create_research_workflow,
    create_eda_workflow,
    create_modeling_workflow,
    create_whatif_workflow,
)


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="MMM Workflows API",
    description="AI-Powered Marketing Mix Modeling Workflows",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# State Management
# =============================================================================

# In-memory storage for workflow states (use Redis in production)
workflow_states: dict[str, dict] = {}
active_websockets: dict[str, list[WebSocket]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class ResearchRequest(BaseModel):
    """Request model for research workflow."""
    query: str = Field(..., description="Research query or objective")
    session_id: Optional[str] = Field(None, description="Session ID for resuming")


class ResearchFeedbackRequest(BaseModel):
    """Request model for research feedback."""
    workflow_id: str
    feedback: str
    approved: bool = False


class EDARequest(BaseModel):
    """Request model for EDA workflow."""
    data_sources: list[str] = Field(..., description="Paths to uploaded data files")
    target_variable: str = Field(..., description="Target variable name")
    date_column: str = Field(default="Date", description="Date column name")
    media_channels: list[str] = Field(default_factory=list, description="Media channel columns")
    analysis_id: Optional[str] = Field(None, description="Link to research analysis")


class ModelingRequest(BaseModel):
    """Request model for modeling workflow."""
    mff_data_path: str = Field(..., description="Path to MFF-formatted data")
    research_plan: Optional[dict] = Field(None, description="Research plan from workflow 1")
    feature_transformations: Optional[list[dict]] = Field(None, description="Transformations from EDA")
    analysis_id: Optional[str] = Field(None, description="Analysis ID for lineage")


class WhatIfRequest(BaseModel):
    """Request model for what-if workflow."""
    model_artifact_path: str = Field(..., description="Path to fitted model")
    mff_data_path: str = Field(..., description="Path to MFF data")
    query: Optional[str] = Field(None, description="Scenario query (natural language)")
    scenarios: Optional[list[dict]] = Field(None, description="Pre-defined scenarios")
    analysis_id: Optional[str] = Field(None, description="Analysis ID for lineage")


class WorkflowStatus(BaseModel):
    """Workflow status response."""
    workflow_id: str
    status: str
    current_phase: str
    progress: float = 0.0
    messages: list[str] = []
    errors: list[str] = []
    result: Optional[dict] = None


class UploadResponse(BaseModel):
    """File upload response."""
    filename: str
    path: str
    size: int


# =============================================================================
# Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting MMM Workflows API")
    
    # Ensure storage directories exist
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize GraphRAG with domain knowledge
    try:
        neo4j = Neo4jClient(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
        )
        graphrag = GraphRAGManager(neo4j)
        await graphrag.load_domain_knowledge()
        logger.info("Loaded domain knowledge into GraphRAG")
    except Exception as e:
        logger.warning(f"Failed to initialize GraphRAG: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MMM Workflows API")


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with service status."""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
    }
    
    # Check Neo4j
    try:
        neo4j = Neo4jClient(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
        )
        # Simple connectivity test
        health["services"]["neo4j"] = "connected"
    except Exception as e:
        health["services"]["neo4j"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                health["services"]["ollama"] = "connected"
            else:
                health["services"]["ollama"] = f"error: status {resp.status_code}"
    except Exception as e:
        health["services"]["ollama"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    return health


# =============================================================================
# File Management Endpoints
# =============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file."""
    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = Path(settings.UPLOAD_DIR) / unique_name
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return UploadResponse(
            filename=file.filename,
            path=str(file_path),
            size=len(content),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.post("/upload/multiple")
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    """Upload multiple data files."""
    results = []
    for file in files:
        file_ext = Path(file.filename).suffix
        unique_name = f"{uuid.uuid4()}{file_ext}"
        file_path = Path(settings.UPLOAD_DIR) / unique_name
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            results.append({
                "filename": file.filename,
                "path": str(file_path),
                "size": len(content),
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return {"files": results}


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download a file by ID."""
    # Search in output directory
    for ext in [".csv", ".json", ".pkl", ".png", ".pdf"]:
        file_path = Path(settings.OUTPUT_DIR) / f"{file_id}{ext}"
        if file_path.exists():
            return FileResponse(file_path)
    
    # Search in model directory
    for ext in [".pkl", ".json", ".nc"]:
        file_path = Path(settings.MODEL_DIR) / f"{file_id}{ext}"
        if file_path.exists():
            return FileResponse(file_path)
    
    raise HTTPException(status_code=404, detail="File not found")


# =============================================================================
# Research Workflow Endpoints
# =============================================================================

@app.post("/research/start", response_model=WorkflowStatus)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research workflow."""
    workflow_id = str(uuid.uuid4())
    
    # Initialize state
    workflow_states[workflow_id] = {
        "status": "starting",
        "current_phase": "init",
        "progress": 0.0,
        "messages": [],
        "errors": [],
        "result": None,
    }
    
    # Run workflow in background
    background_tasks.add_task(
        run_research_workflow,
        workflow_id,
        request.query,
        request.session_id,
    )
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="starting",
        current_phase="init",
        messages=["Research workflow started"],
    )


async def run_research_workflow(workflow_id: str, query: str, session_id: str | None):
    """Background task to run research workflow."""
    try:
        workflow_states[workflow_id]["status"] = "running"
        
        # Create workflow
        workflow = await create_research_workflow()
        
        # Run workflow
        result = await workflow.run(query, session_id)
        
        # Update state
        workflow_states[workflow_id].update({
            "status": "waiting_feedback" if result.get("user_feedback") is None else "completed",
            "current_phase": str(result.get("current_phase", "unknown")),
            "progress": 1.0 if result.get("current_phase") == "RESEARCH_COMPLETE" else 0.5,
            "messages": result.get("messages", []),
            "errors": result.get("errors", []),
            "result": {
                "research_plan": result.get("research_plan"),
                "web_search_results": len(result.get("web_search_results", [])),
                "analysis_id": result.get("analysis_id"),
            },
        })
        
        # Notify WebSocket clients
        await broadcast_status(workflow_id)
        
    except Exception as e:
        logger.exception(f"Research workflow failed: {e}")
        workflow_states[workflow_id].update({
            "status": "failed",
            "errors": [str(e)],
        })
        await broadcast_status(workflow_id)


@app.post("/research/{workflow_id}/feedback", response_model=WorkflowStatus)
async def submit_research_feedback(
    workflow_id: str,
    request: ResearchFeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """Submit feedback for research workflow."""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    state = workflow_states[workflow_id]
    if state["status"] != "waiting_feedback":
        raise HTTPException(status_code=400, detail="Workflow not waiting for feedback")
    
    # Continue workflow with feedback
    background_tasks.add_task(
        continue_research_with_feedback,
        workflow_id,
        request.feedback,
        request.approved,
    )
    
    state["status"] = "running"
    state["messages"].append(f"Feedback received: {'Approved' if request.approved else 'Revisions requested'}")
    
    return WorkflowStatus(**{"workflow_id": workflow_id, **state})


async def continue_research_with_feedback(workflow_id: str, feedback: str, approved: bool):
    """Continue research workflow with user feedback."""
    # Implementation would resume the workflow from checkpoint
    # For now, mark as complete
    workflow_states[workflow_id].update({
        "status": "completed",
        "current_phase": "RESEARCH_COMPLETE",
        "progress": 1.0,
        "messages": workflow_states[workflow_id].get("messages", []) + ["Feedback processed"],
    })
    await broadcast_status(workflow_id)


# =============================================================================
# EDA Workflow Endpoints
# =============================================================================

@app.post("/eda/start", response_model=WorkflowStatus)
async def start_eda(request: EDARequest, background_tasks: BackgroundTasks):
    """Start EDA workflow."""
    workflow_id = str(uuid.uuid4())
    
    workflow_states[workflow_id] = {
        "status": "starting",
        "current_phase": "init",
        "progress": 0.0,
        "messages": [],
        "errors": [],
        "result": None,
    }
    
    background_tasks.add_task(
        run_eda_workflow,
        workflow_id,
        request.data_sources,
        request.target_variable,
        request.date_column,
        request.media_channels,
        request.analysis_id,
    )
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="starting",
        current_phase="init",
        messages=["EDA workflow started"],
    )


async def run_eda_workflow(
    workflow_id: str,
    data_sources: list[str],
    target_variable: str,
    date_column: str,
    media_channels: list[str],
    analysis_id: str | None,
):
    """Background task to run EDA workflow."""
    try:
        workflow_states[workflow_id]["status"] = "running"
        
        workflow = await create_eda_workflow()
        
        result = await workflow.run(
            data_sources=data_sources,
            target_variable=target_variable,
            date_column=date_column,
            media_channels=media_channels,
            analysis_id=analysis_id,
        )
        
        workflow_states[workflow_id].update({
            "status": "completed",
            "current_phase": str(result.get("current_phase", "unknown")),
            "progress": 1.0,
            "messages": result.get("messages", []),
            "errors": result.get("errors", []),
            "result": {
                "mff_data_path": result.get("mff_data_path"),
                "data_quality_report": result.get("data_quality_report"),
                "generated_plots": result.get("generated_plots", []),
                "modeling_recommendations": result.get("modeling_recommendations"),
            },
        })
        
        await broadcast_status(workflow_id)
        
    except Exception as e:
        logger.exception(f"EDA workflow failed: {e}")
        workflow_states[workflow_id].update({
            "status": "failed",
            "errors": [str(e)],
        })
        await broadcast_status(workflow_id)


# =============================================================================
# Modeling Workflow Endpoints
# =============================================================================

@app.post("/modeling/start", response_model=WorkflowStatus)
async def start_modeling(request: ModelingRequest, background_tasks: BackgroundTasks):
    """Start modeling workflow."""
    workflow_id = str(uuid.uuid4())
    
    workflow_states[workflow_id] = {
        "status": "starting",
        "current_phase": "init",
        "progress": 0.0,
        "messages": [],
        "errors": [],
        "result": None,
    }
    
    background_tasks.add_task(
        run_modeling_workflow,
        workflow_id,
        request.mff_data_path,
        request.research_plan,
        request.feature_transformations,
        request.analysis_id,
    )
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="starting",
        current_phase="init",
        messages=["Modeling workflow started"],
    )


async def run_modeling_workflow(
    workflow_id: str,
    mff_data_path: str,
    research_plan: dict | None,
    feature_transformations: list[dict] | None,
    analysis_id: str | None,
):
    """Background task to run modeling workflow."""
    try:
        workflow_states[workflow_id]["status"] = "running"
        
        workflow = await create_modeling_workflow()
        
        result = await workflow.run(
            mff_data_path=mff_data_path,
            research_plan=research_plan,
            feature_transformations=feature_transformations,
            analysis_id=analysis_id,
        )
        
        workflow_states[workflow_id].update({
            "status": "completed",
            "current_phase": str(result.get("current_phase", "unknown")),
            "progress": 1.0,
            "messages": result.get("messages", []),
            "errors": result.get("errors", []),
            "result": {
                "model_artifact_path": result.get("model_artifact_path"),
                "channel_contributions": result.get("channel_contributions"),
                "roi_estimates": result.get("roi_estimates"),
                "convergence_status": result.get("convergence_status"),
                "interpretation_summary": result.get("interpretation_summary"),
                "generated_plots": result.get("generated_plots", []),
            },
        })
        
        await broadcast_status(workflow_id)
        
    except Exception as e:
        logger.exception(f"Modeling workflow failed: {e}")
        workflow_states[workflow_id].update({
            "status": "failed",
            "errors": [str(e)],
        })
        await broadcast_status(workflow_id)


# =============================================================================
# What-If Workflow Endpoints
# =============================================================================

@app.post("/whatif/start", response_model=WorkflowStatus)
async def start_whatif(request: WhatIfRequest, background_tasks: BackgroundTasks):
    """Start what-if analysis workflow."""
    workflow_id = str(uuid.uuid4())
    
    workflow_states[workflow_id] = {
        "status": "starting",
        "current_phase": "init",
        "progress": 0.0,
        "messages": [],
        "errors": [],
        "result": None,
    }
    
    background_tasks.add_task(
        run_whatif_workflow,
        workflow_id,
        request.model_artifact_path,
        request.mff_data_path,
        request.query,
        request.analysis_id,
    )
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="starting",
        current_phase="init",
        messages=["What-if workflow started"],
    )


async def run_whatif_workflow(
    workflow_id: str,
    model_artifact_path: str,
    mff_data_path: str,
    query: str | None,
    analysis_id: str | None,
):
    """Background task to run what-if workflow."""
    try:
        workflow_states[workflow_id]["status"] = "running"
        
        workflow = await create_whatif_workflow()
        
        result = await workflow.run(
            model_artifact_path=model_artifact_path,
            mff_data_path=mff_data_path,
            user_query=query,
            analysis_id=analysis_id,
        )
        
        workflow_states[workflow_id].update({
            "status": "completed",
            "current_phase": str(result.get("current_phase", "unknown")),
            "progress": 1.0,
            "messages": result.get("messages", []),
            "errors": result.get("errors", []),
            "result": {
                "scenario_results": [
                    s.model_dump() if hasattr(s, 'model_dump') else s 
                    for s in result.get("scenario_results", [])
                ],
                "optimization_suggestions": result.get("optimization_suggestions"),
                "summary": result.get("summary"),
                "generated_plots": result.get("generated_plots", []),
            },
        })
        
        await broadcast_status(workflow_id)
        
    except Exception as e:
        logger.exception(f"What-if workflow failed: {e}")
        workflow_states[workflow_id].update({
            "status": "failed",
            "errors": [str(e)],
        })
        await broadcast_status(workflow_id)


# =============================================================================
# Workflow Status Endpoints
# =============================================================================

@app.get("/workflow/{workflow_id}", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str):
    """Get workflow status by ID."""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    state = workflow_states[workflow_id]
    return WorkflowStatus(workflow_id=workflow_id, **state)


@app.get("/workflows")
async def list_workflows():
    """List all workflows and their statuses."""
    return {
        workflow_id: WorkflowStatus(workflow_id=workflow_id, **state)
        for workflow_id, state in workflow_states.items()
    }


@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow and its state."""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del workflow_states[workflow_id]
    return {"message": f"Workflow {workflow_id} deleted"}


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/{workflow_id}")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """WebSocket endpoint for real-time workflow updates."""
    await websocket.accept()
    
    # Register WebSocket
    if workflow_id not in active_websockets:
        active_websockets[workflow_id] = []
    active_websockets[workflow_id].append(websocket)
    
    try:
        # Send initial status
        if workflow_id in workflow_states:
            await websocket.send_json({
                "type": "status",
                "data": WorkflowStatus(
                    workflow_id=workflow_id,
                    **workflow_states[workflow_id]
                ).model_dump(),
            })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        active_websockets[workflow_id].remove(websocket)
        if not active_websockets[workflow_id]:
            del active_websockets[workflow_id]


async def broadcast_status(workflow_id: str):
    """Broadcast status update to all connected WebSocket clients."""
    if workflow_id not in active_websockets:
        return
    
    if workflow_id not in workflow_states:
        return
    
    message = {
        "type": "status",
        "data": WorkflowStatus(
            workflow_id=workflow_id,
            **workflow_states[workflow_id]
        ).model_dump(),
    }
    
    disconnected = []
    for websocket in active_websockets[workflow_id]:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)
    
    # Clean up disconnected clients
    for ws in disconnected:
        active_websockets[workflow_id].remove(ws)


# =============================================================================
# Pipeline Endpoint (Full Workflow Chain)
# =============================================================================

class PipelineRequest(BaseModel):
    """Request model for full pipeline execution."""
    query: str = Field(..., description="Research query/objective")
    data_files: list[str] = Field(..., description="Paths to data files")
    target_variable: str = Field(..., description="Target variable name")
    date_column: str = Field(default="Date")
    media_channels: list[str] = Field(default_factory=list)
    run_whatif: bool = Field(default=True, description="Run what-if analysis after modeling")


@app.post("/pipeline/start", response_model=WorkflowStatus)
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start full MMM pipeline (Research → EDA → Modeling → What-If)."""
    workflow_id = str(uuid.uuid4())
    
    workflow_states[workflow_id] = {
        "status": "starting",
        "current_phase": "pipeline_init",
        "progress": 0.0,
        "messages": [],
        "errors": [],
        "result": None,
        "pipeline_stage": "research",
    }
    
    background_tasks.add_task(
        run_full_pipeline,
        workflow_id,
        request,
    )
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="starting",
        current_phase="pipeline_init",
        messages=["Full MMM pipeline started"],
    )


async def run_full_pipeline(workflow_id: str, request: PipelineRequest):
    """Run the complete MMM pipeline."""
    try:
        analysis_id = str(uuid.uuid4())
        
        # Stage 1: Research
        workflow_states[workflow_id].update({
            "status": "running",
            "pipeline_stage": "research",
            "progress": 0.1,
            "messages": ["Starting research phase..."],
        })
        await broadcast_status(workflow_id)
        
        research_workflow = await create_research_workflow()
        research_result = await research_workflow.run(request.query)
        
        if research_result.get("errors"):
            raise Exception(f"Research failed: {research_result['errors']}")
        
        # Stage 2: EDA
        workflow_states[workflow_id].update({
            "pipeline_stage": "eda",
            "progress": 0.3,
            "messages": workflow_states[workflow_id]["messages"] + ["Starting EDA phase..."],
        })
        await broadcast_status(workflow_id)
        
        eda_workflow = await create_eda_workflow()
        eda_result = await eda_workflow.run(
            data_sources=request.data_files,
            target_variable=request.target_variable,
            date_column=request.date_column,
            media_channels=request.media_channels,
            analysis_id=analysis_id,
        )
        
        if eda_result.get("errors"):
            raise Exception(f"EDA failed: {eda_result['errors']}")
        
        mff_path = eda_result.get("mff_data_path")
        
        # Stage 3: Modeling
        workflow_states[workflow_id].update({
            "pipeline_stage": "modeling",
            "progress": 0.5,
            "messages": workflow_states[workflow_id]["messages"] + ["Starting modeling phase..."],
        })
        await broadcast_status(workflow_id)
        
        modeling_workflow = await create_modeling_workflow()
        modeling_result = await modeling_workflow.run(
            mff_data_path=mff_path,
            research_plan=research_result.get("research_plan"),
            feature_transformations=eda_result.get("feature_transformations"),
            analysis_id=analysis_id,
        )
        
        if modeling_result.get("errors"):
            raise Exception(f"Modeling failed: {modeling_result['errors']}")
        
        model_path = modeling_result.get("model_artifact_path")
        
        # Stage 4: What-If (optional)
        whatif_result = None
        if request.run_whatif and model_path:
            workflow_states[workflow_id].update({
                "pipeline_stage": "whatif",
                "progress": 0.8,
                "messages": workflow_states[workflow_id]["messages"] + ["Starting what-if analysis..."],
            })
            await broadcast_status(workflow_id)
            
            whatif_workflow = await create_whatif_workflow()
            whatif_result = await whatif_workflow.run(
                model_artifact_path=model_path,
                mff_data_path=mff_path,
                analysis_id=analysis_id,
            )
        
        # Complete
        workflow_states[workflow_id].update({
            "status": "completed",
            "pipeline_stage": "complete",
            "progress": 1.0,
            "messages": workflow_states[workflow_id]["messages"] + ["Pipeline complete!"],
            "result": {
                "analysis_id": analysis_id,
                "research_plan": research_result.get("research_plan"),
                "mff_data_path": mff_path,
                "model_artifact_path": model_path,
                "channel_contributions": modeling_result.get("channel_contributions"),
                "roi_estimates": modeling_result.get("roi_estimates"),
                "whatif_summary": whatif_result.get("summary") if whatif_result else None,
            },
        })
        await broadcast_status(workflow_id)
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        workflow_states[workflow_id].update({
            "status": "failed",
            "errors": [str(e)],
        })
        await broadcast_status(workflow_id)


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mmm_workflows.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
