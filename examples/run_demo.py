#!/usr/bin/env python3
"""
MMM Agent POC - Example Usage Script

This script demonstrates the complete MMM workflow for stakeholder demos.
It generates sample data and runs the full analysis pipeline.

Usage:
    python run_demo.py                    # Run with sample data
    python run_demo.py --data path/to/data.csv --context "Your business context"
    python run_demo.py --provider anthropic  # Use Claude
    python run_demo.py --bayesian           # Use full Bayesian MMM
"""

import asyncio
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmm_agent.workflow import MMMWorkflow, run_mmm_analysis, create_workflow
from mmm_agent.data.sample_generator import generate_mmm_sample_data
from mmm_agent.state import WorkflowPhase


# ============================================================================
# Demo Configuration
# ============================================================================

DEMO_BUSINESS_CONTEXT = """
We are a consumer electronics company analyzing Q4 2024 marketing effectiveness.

Key business questions:
1. Which marketing channels are driving the most revenue?
2. Are we seeing diminishing returns on any channels?
3. How should we reallocate budget for Q1 2025?
4. What is the ROI of each marketing channel?

Constraints:
- Total marketing budget is $130K per week
- TV has minimum commitment of $20K/week
- We want to maintain presence across all channels
- Q1 typically sees 15% lower baseline sales than Q4
"""


# ============================================================================
# Progress Display
# ============================================================================

class ProgressTracker:
    """Track and display workflow progress."""
    
    def __init__(self):
        self.phase_symbols = {
            "planning": "📋",
            "eda": "🔍",
            "modeling": "⚙️",
            "interpretation": "💡",
            "complete": "✅",
            "error": "❌",
        }
        self.current_phase = None
        
    def update(self, phase: str, message: str):
        """Display progress update."""
        symbol = self.phase_symbols.get(phase.lower(), "⏳")
        
        # Print phase header on phase change
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"\n{'='*60}")
            print(f"{symbol} PHASE: {phase.upper()}")
            print(f"{'='*60}")
        
        # Print message (truncate if too long)
        if len(message) > 200:
            print(f"  {message[:200]}...")
        else:
            print(f"  {message}")


# ============================================================================
# Demo Runner
# ============================================================================

async def run_demo(
    data_path: str = None,
    business_context: str = None,
    provider: str = "ollama",
    use_bayesian: bool = False,
    output_dir: str = None,
):
    """
    Run the MMM demo workflow.
    
    Args:
        data_path: Path to data file (None = generate sample data)
        business_context: Business context description
        provider: LLM provider to use
        use_bayesian: Whether to use Bayesian MMM
        output_dir: Directory for outputs
    """
    progress = ProgressTracker()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       🎯 MMM AGENT POC - MARKETING MIX MODELING              ║
    ║                                                              ║
    ║   An AI-powered workflow for marketing effectiveness        ║
    ║   analysis using Bayesian methods and multi-source          ║
    ║   data harmonization.                                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"./mmm_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate or load data
    if data_path is None:
        print("📊 Generating sample MMM data...")
        data_dir = os.path.join(output_dir, "data")
        generate_mmm_sample_data(output_dir=data_dir)
        
        # Use combined file
        data_sources = [
            {
                "path": os.path.join(data_dir, "mmm_data_combined.csv"),
                "type": "combined",
                "description": "Combined sales, media spend, and external factors",
            }
        ]
        
        # Or use separate files for multi-source demo
        # data_sources = [
        #     {"path": os.path.join(data_dir, "sales_data.csv"), "type": "sales"},
        #     {"path": os.path.join(data_dir, "media_spend.csv"), "type": "media"},
        #     {"path": os.path.join(data_dir, "external_factors.csv"), "type": "external"},
        # ]
    else:
        data_sources = [{"path": data_path, "type": "unknown"}]
    
    # Use provided context or demo context
    context = business_context or DEMO_BUSINESS_CONTEXT
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   LLM Provider: {provider}")
    print(f"   Bayesian MMM: {'Yes' if use_bayesian else 'No (Ridge regression)'}")
    print(f"   Output Dir: {output_dir}")
    print(f"\n📂 Data Sources:")
    for ds in data_sources:
        print(f"   - {ds['path']} ({ds['type']})")
    
    print(f"\n📝 Business Context:")
    for line in context.strip().split("\n")[:5]:
        print(f"   {line.strip()}")
    if context.count("\n") > 5:
        print("   ...")
    
    input("\n Press Enter to start the workflow...")
    
    # Create and run workflow
    try:
        workflow = create_workflow(
            provider=provider,
            use_bayesian=use_bayesian,
            enable_web_search=True,
        )
        
        result = await workflow.run(
            data_sources=data_sources,
            business_context=context,
            progress_callback=progress.update,
        )
        
        # Save results
        results_file = os.path.join(output_dir, "results.json")
        save_results(result, results_file)
        
        # Display summary
        display_summary(result)
        
        print(f"\n📁 Full results saved to: {results_file}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(state: dict, filepath: str):
    """Save workflow results to JSON."""
    # Extract serializable results
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": str(state.get("current_phase", "unknown")),
        "planning": {
            "target_variable": state.get("target_variable"),
            "media_channels": state.get("media_channels", []),
            "control_variables": state.get("control_variables", []),
            "research_questions": state.get("research_questions", []),
        },
        "eda": {
            "data_quality_report": state.get("data_quality_report", {}),
            "feature_transformations": state.get("feature_transformations", []),
        },
        "modeling": {
            "convergence_diagnostics": state.get("convergence_diagnostics", {}),
            "channel_contributions": state.get("channel_contributions", {}),
            "model_artifact_path": state.get("model_artifact_path"),
        },
        "interpretation": {
            "roi_estimates": state.get("roi_estimates", {}),
            "budget_allocation": state.get("budget_allocation", {}),
            "what_if_scenarios": state.get("what_if_scenarios", []),
            "recommendations": state.get("recommendations", []),
        },
        "errors": state.get("errors", []),
    }
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def display_summary(state: dict):
    """Display a summary of the workflow results."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     📊 RESULTS SUMMARY                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Status
    phase = state.get("current_phase")
    if phase == WorkflowPhase.COMPLETE:
        print("   ✅ Workflow completed successfully!\n")
    else:
        print(f"   ⚠️ Workflow ended in phase: {phase}\n")
    
    # ROI Estimates
    roi = state.get("roi_estimates", {})
    if roi:
        print("   📈 ROI Estimates:")
        for channel, value in roi.items():
            if isinstance(value, (int, float)):
                print(f"      {channel}: {value:.2f}")
            else:
                print(f"      {channel}: {value}")
        print()
    
    # Channel Contributions
    contributions = state.get("channel_contributions", {})
    if contributions:
        print("   📊 Channel Contributions:")
        for channel, value in contributions.items():
            if isinstance(value, (int, float)):
                print(f"      {channel}: {value:.1%}")
            else:
                print(f"      {channel}: {value}")
        print()
    
    # Budget Allocation
    allocation = state.get("budget_allocation", {})
    if allocation:
        print("   💰 Recommended Budget Allocation:")
        for channel, value in allocation.items():
            if isinstance(value, (int, float)):
                print(f"      {channel}: ${value:,.0f}")
            else:
                print(f"      {channel}: {value}")
        print()
    
    # Top Recommendations
    recommendations = state.get("recommendations", [])
    if recommendations:
        print("   💡 Key Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            # Truncate long recommendations
            if len(str(rec)) > 100:
                rec = str(rec)[:100] + "..."
            print(f"      {i}. {rec}")
        print()
    
    # Errors
    errors = state.get("errors", [])
    if errors:
        print("   ⚠️ Warnings/Errors:")
        for err in errors[:3]:
            print(f"      - {err}")
        print()


# ============================================================================
# Interactive Demo Mode
# ============================================================================

async def interactive_demo():
    """Run an interactive demo with user prompts."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             🎮 INTERACTIVE MMM DEMO                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Get LLM provider
    print("Available LLM providers:")
    print("  1. ollama (local, free)")
    print("  2. openai (GPT-4)")
    print("  3. anthropic (Claude)")
    print("  4. gemini (Google)")
    
    choice = input("\nSelect provider [1-4, default=1]: ").strip() or "1"
    providers = {"1": "ollama", "2": "openai", "3": "anthropic", "4": "gemini"}
    provider = providers.get(choice, "ollama")
    
    # Bayesian option
    use_bayesian = input("Use full Bayesian MMM? (requires PyMC) [y/N]: ").lower() == "y"
    
    # Custom data?
    data_path = input("\nPath to data file [Enter for sample data]: ").strip() or None
    
    # Custom context?
    print("\nBusiness context [Enter for demo context, or paste your own]:")
    context = input().strip() or None
    
    # Run
    await run_demo(
        data_path=data_path,
        business_context=context,
        provider=provider,
        use_bayesian=use_bayesian,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MMM Agent POC Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                          # Run with sample data
  python run_demo.py --interactive            # Interactive mode
  python run_demo.py --provider anthropic     # Use Claude
  python run_demo.py --data my_data.csv       # Use custom data
  python run_demo.py --bayesian               # Full Bayesian MMM
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        help="Path to data file (CSV/Excel/Parquet)"
    )
    parser.add_argument(
        "--context", "-c",
        help="Business context description"
    )
    parser.add_argument(
        "--provider", "-p",
        default="ollama",
        choices=["ollama", "openai", "anthropic", "gemini"],
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--bayesian", "-b",
        action="store_true",
        help="Use full Bayesian MMM (requires PyMC)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_demo())
    else:
        asyncio.run(run_demo(
            data_path=args.data,
            business_context=args.context,
            provider=args.provider,
            use_bayesian=args.bayesian,
            output_dir=args.output,
        ))


if __name__ == "__main__":
    main()
