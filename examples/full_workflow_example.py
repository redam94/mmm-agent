#!/usr/bin/env python
"""
MMM Workflows - Example Usage

This example demonstrates how to use all four MMM workflows:
1. Research Agent - Web search and planning
2. EDA Agent - Data cleaning and transformation
3. Modeling Agent - Bayesian MMM fitting
4. What-If Agent - Scenario analysis

Prerequisites:
- Neo4j running at localhost:7687
- PostgreSQL running at localhost:5432
- Ollama running at localhost:11434 with qwen3:30b and qwen3-coder:30b

Run: python examples/full_workflow_example.py
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_research_workflow():
    """Run the research workflow."""
    from mmm_agent.workflows import create_research_workflow
    
    logger.info("=" * 60)
    logger.info("STEP 1: Research Workflow")
    logger.info("=" * 60)
    
    # Create workflow
    workflow = await create_research_workflow()
    
    # Define research query
    query = """
    We want to build a Marketing Mix Model for a CPG brand.
    The brand sells beverages across multiple retail channels.
    We have 2 years of weekly data including:
    - Sales by region
    - Media spend (TV, Digital, Social, Search)
    - Promotions
    - Pricing
    - Distribution
    
    What should we consider for the model?
    """
    
    # Run workflow
    result = await workflow.run(query)
    
    # Print results
    logger.info(f"Research complete!")
    logger.info(f"Analysis ID: {result.get('analysis_id')}")
    
    if result.get("research_plan"):
        plan = result["research_plan"]
        logger.info(f"Target Variable: {plan.get('target_variable')}")
        logger.info(f"Media Channels: {plan.get('media_channels')}")
        logger.info(f"Control Variables: {plan.get('control_variables')}")
    
    return result


async def run_eda_workflow(analysis_id: str = None):
    """Run the EDA workflow with sample data."""
    from mmm_workflows.workflows import create_eda_workflow
    
    logger.info("=" * 60)
    logger.info("STEP 2: EDA Workflow")
    logger.info("=" * 60)
    
    # Create sample data
    sample_data_path = create_sample_data()
    
    # Create workflow
    workflow = await create_eda_workflow()
    
    # Run workflow
    result = await workflow.run(
        data_sources=[sample_data_path],
        target_variable="Sales",
        date_column="Date",
        media_channels=["TV_Spend", "Digital_Spend", "Social_Spend", "Search_Spend"],
        analysis_id=analysis_id,
    )
    
    # Print results
    logger.info(f"EDA complete!")
    logger.info(f"MFF Data Path: {result.get('mff_data_path')}")
    logger.info(f"Generated Plots: {len(result.get('generated_plots', []))}")
    
    if result.get("data_quality_report"):
        report = result["data_quality_report"]
        logger.info(f"Quality Issues: {len(report.get('issues', []))}")
    
    return result


async def run_modeling_workflow(mff_data_path: str, research_plan: dict = None, analysis_id: str = None):
    """Run the modeling workflow."""
    from mmm_workflows.workflows import create_modeling_workflow
    
    logger.info("=" * 60)
    logger.info("STEP 3: Modeling Workflow")
    logger.info("=" * 60)
    
    # Create workflow
    workflow = await create_modeling_workflow()
    
    # Run workflow
    result = await workflow.run(
        mff_data_path=mff_data_path,
        research_plan=research_plan,
        analysis_id=analysis_id,
    )
    
    # Print results
    logger.info(f"Modeling complete!")
    logger.info(f"Model Artifact: {result.get('model_artifact_path')}")
    logger.info(f"Convergence: {result.get('convergence_status')}")
    
    if result.get("channel_contributions"):
        logger.info("Channel Contributions:")
        for contrib in result["channel_contributions"]:
            logger.info(f"  {contrib.get('channel')}: {contrib.get('contribution_mean', 0):.2f}")
    
    return result


async def run_whatif_workflow(model_path: str, mff_data_path: str, analysis_id: str = None):
    """Run the what-if analysis workflow."""
    from mmm_workflows.workflows import create_whatif_workflow
    
    logger.info("=" * 60)
    logger.info("STEP 4: What-If Workflow")
    logger.info("=" * 60)
    
    # Create workflow
    workflow = await create_whatif_workflow()
    
    # Run workflow with scenario query
    result = await workflow.run(
        model_artifact_path=model_path,
        mff_data_path=mff_data_path,
        user_query="What if we increase TV spend by 20% and decrease Digital by 10%?",
        analysis_id=analysis_id,
    )
    
    # Print results
    logger.info(f"What-If analysis complete!")
    
    if result.get("scenario_results"):
        logger.info("Scenario Results:")
        for scenario in result["scenario_results"]:
            name = scenario.scenario_name if hasattr(scenario, 'scenario_name') else scenario.get('scenario_name')
            impact = scenario.incremental_impact if hasattr(scenario, 'incremental_impact') else scenario.get('incremental_impact', 0)
            logger.info(f"  {name}: {impact:+.2f}")
    
    if result.get("optimization_suggestions"):
        logger.info("Optimization Suggestions:")
        for rec in result["optimization_suggestions"][:3]:
            logger.info(f"  {rec.get('channel')}: {rec.get('rationale', '')[:50]}...")
    
    return result


def create_sample_data() -> str:
    """Create sample MMM data for testing."""
    import numpy as np
    import pandas as pd
    
    logger.info("Creating sample data...")
    
    # Generate 104 weeks of data (2 years)
    np.random.seed(42)
    n_weeks = 104
    
    dates = pd.date_range(start="2022-01-01", periods=n_weeks, freq="W")
    
    # Generate media spend
    tv_spend = np.random.uniform(50000, 200000, n_weeks)
    digital_spend = np.random.uniform(20000, 100000, n_weeks)
    social_spend = np.random.uniform(10000, 50000, n_weeks)
    search_spend = np.random.uniform(15000, 75000, n_weeks)
    
    # Generate controls
    price = np.random.uniform(2.5, 3.5, n_weeks)
    distribution = np.random.uniform(0.7, 0.95, n_weeks)
    promo = np.random.choice([0, 1], n_weeks, p=[0.7, 0.3])
    
    # Generate seasonality
    week_of_year = np.array([d.isocalendar()[1] for d in dates])
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * week_of_year / 52)
    
    # Generate sales with effects
    base_sales = 100000
    tv_effect = 0.5 * np.sqrt(tv_spend)
    digital_effect = 0.3 * np.log1p(digital_spend)
    social_effect = 0.2 * np.sqrt(social_spend)
    search_effect = 0.4 * np.log1p(search_spend)
    price_effect = -20000 * (price - 3)
    dist_effect = 50000 * distribution
    promo_effect = 10000 * promo
    
    sales = (
        base_sales 
        + tv_effect 
        + digital_effect 
        + social_effect 
        + search_effect 
        + price_effect 
        + dist_effect 
        + promo_effect
    ) * seasonality + np.random.normal(0, 5000, n_weeks)
    
    sales = np.maximum(sales, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales.round(0).astype(int),
        "TV_Spend": tv_spend.round(0).astype(int),
        "Digital_Spend": digital_spend.round(0).astype(int),
        "Social_Spend": social_spend.round(0).astype(int),
        "Search_Spend": search_spend.round(0).astype(int),
        "Price": price.round(2),
        "Distribution": distribution.round(3),
        "Promotion": promo,
    })
    
    # Save to file
    data_dir = Path("./data/uploads")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / "sample_mmm_data.csv"
    df.to_csv(file_path, index=False)
    
    logger.info(f"Sample data saved to: {file_path}")
    logger.info(f"Data shape: {df.shape}")
    
    return str(file_path)


async def run_full_pipeline():
    """Run the complete MMM pipeline."""
    logger.info("=" * 60)
    logger.info("MMM Workflows - Full Pipeline Demo")
    logger.info("=" * 60)
    
    # Step 1: Research
    research_result = await run_research_workflow()
    analysis_id = research_result.get("analysis_id")
    research_plan = research_result.get("research_plan")
    
    # Step 2: EDA
    eda_result = await run_eda_workflow(analysis_id)
    mff_data_path = eda_result.get("mff_data_path")
    
    if not mff_data_path:
        logger.error("EDA did not produce MFF data. Check for errors.")
        return
    
    # Step 3: Modeling
    modeling_result = await run_modeling_workflow(
        mff_data_path=mff_data_path,
        research_plan=research_plan,
        analysis_id=analysis_id,
    )
    model_path = modeling_result.get("model_artifact_path")
    
    if not model_path:
        logger.error("Modeling did not produce artifact. Check for errors.")
        return
    
    # Step 4: What-If
    whatif_result = await run_whatif_workflow(
        model_path=model_path,
        mff_data_path=mff_data_path,
        analysis_id=analysis_id,
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Analysis ID: {analysis_id}")
    logger.info(f"MFF Data: {mff_data_path}")
    logger.info(f"Model Artifact: {model_path}")
    
    return {
        "analysis_id": analysis_id,
        "research_plan": research_plan,
        "mff_data_path": mff_data_path,
        "model_path": model_path,
        "channel_contributions": modeling_result.get("channel_contributions"),
        "scenario_results": whatif_result.get("scenario_results"),
    }


async def run_api_demo():
    """Demonstrate API usage."""
    import httpx
    
    logger.info("=" * 60)
    logger.info("API Demo")
    logger.info("=" * 60)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Health check
        resp = await client.get(f"{base_url}/health")
        logger.info(f"Health: {resp.json()}")
        
        # Upload sample data
        sample_path = create_sample_data()
        with open(sample_path, "rb") as f:
            resp = await client.post(
                f"{base_url}/upload",
                files={"file": ("sample_data.csv", f, "text/csv")}
            )
        upload_result = resp.json()
        logger.info(f"Uploaded: {upload_result['path']}")
        
        # Start research workflow
        resp = await client.post(
            f"{base_url}/research/start",
            json={"query": "Build MMM for beverage brand with 2 years of weekly data"}
        )
        workflow = resp.json()
        logger.info(f"Research workflow started: {workflow['workflow_id']}")
        
        # Poll for completion
        while True:
            resp = await client.get(f"{base_url}/workflow/{workflow['workflow_id']}")
            status = resp.json()
            logger.info(f"Status: {status['status']} - {status['current_phase']}")
            
            if status["status"] in ["completed", "failed", "waiting_feedback"]:
                break
            
            await asyncio.sleep(5)
        
        logger.info(f"Final status: {status}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run API demo
        asyncio.run(run_api_demo())
    elif len(sys.argv) > 1 and sys.argv[1] == "--step":
        # Run specific step
        step = sys.argv[2] if len(sys.argv) > 2 else "research"
        
        if step == "research":
            asyncio.run(run_research_workflow())
        elif step == "eda":
            asyncio.run(run_eda_workflow())
        elif step == "sample":
            create_sample_data()
    else:
        # Run full pipeline
        asyncio.run(run_full_pipeline())
