# MMM Agent POC

**Agentic Marketing Mix Modeling Framework** - An AI-powered workflow for marketing effectiveness analysis using LangGraph, multi-LLM support, and Bayesian methods.

## Overview

This proof-of-concept demonstrates an agentic approach to Marketing Mix Modeling (MMM) that combines:

- **Multi-phase workflow**: Planning → EDA → Modeling → Interpretation
- **LLM-powered analysis**: Automatic hypothesis generation, data quality assessment, and insight extraction
- **Provider flexibility**: Works with Ollama (local), OpenAI, Anthropic, or Google Gemini
- **Bayesian & classical methods**: Full PyMC Bayesian MMM or Ridge regression fallback
- **Local code execution**: Safe subprocess-based code execution with AST validation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MMM Agent Workflow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │ Planning │───▶│   EDA    │───▶│ Modeling │───▶│Interpret││
│  │  Agent   │    │  Agent   │    │  Agent   │    │ Agent  ││
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬───┘│
│       │               │               │               │     │
│       ▼               ▼               ▼               ▼     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Shared State (LangGraph)                   ││
│  │  • Research questions  • Data quality report            ││
│  │  • Causal hypotheses   • Model diagnostics              ││
│  │  • ROI estimates       • Budget recommendations         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Tools Layer                          ││
│  │  • Local Code Executor   • RAG Context Manager          ││
│  │  • Data Harmonizer       • Web Search (DuckDuckGo)      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mmm-agent.git
cd mmm-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with pip
pip install -e .
```

### With Bayesian Support

```bash
pip install -e ".[bayesian]"
```

### Full Installation (all features)

```bash
pip install -e ".[all]"
```

### LLM Provider Setup

#### Ollama (Local, Free)
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1
export OLLAMA_MODEL=llama3.1
```

#### OpenAI
```bash
export OPENAI_API_KEY=your-key-here
```

#### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY=your-key-here
```

#### Google Gemini
```bash
export GOOGLE_API_KEY=your-key-here
```

## Quick Start

### Run Demo with Sample Data

```bash
cd examples
python run_demo.py
```

### Interactive Mode

```bash
python run_demo.py --interactive
```

### With Your Own Data

```bash
python run_demo.py \
    --data /path/to/your/data.csv \
    --context "Your business context description" \
    --provider anthropic
```

### Programmatic Usage

```python
import asyncio
from mmm_agent.workflow import create_workflow

async def main():
    workflow = create_workflow(use_bayesian=False)
    
    result = await workflow.run(
        data_path="sales_data.csv",
        business_context="""
        E-commerce company analyzing Q4 marketing effectiveness.
        Key channels: TV, Digital, Social, Search.
        Primary KPI: Revenue.
        """,
        kpi_column="revenue",
        media_columns=["tv_spend", "digital_spend", "social_spend", "search_spend"],
        date_column="date",
    )
    
    print(f"ROI Estimates: {result['roi_estimates']}")
    print(f"Recommendations: {result['recommendations']}")

asyncio.run(main())
```

## Workflow Phases

### 1. Planning Phase
- Analyzes business context and data structure
- Generates research questions
- Forms causal hypotheses (treatment → outcome with mechanisms)
- Identifies potential confounders
- Optional web research for domain context

### 2. EDA Phase
- Loads and profiles data
- Checks data quality (missing values, outliers, date gaps)
- Generates visualizations (time series, correlations, distributions)
- Recommends feature transformations

### 3. Modeling Phase
- Builds model specification (adstock, saturation, controls)
- Fits Bayesian MMM (PyMC) or Ridge regression fallback
- Calculates convergence diagnostics
- Estimates channel contributions

### 4. Interpretation Phase
- Calculates ROI by channel
- Optimizes budget allocation
- Runs what-if scenarios
- Generates actionable recommendations

## Configuration

### Environment Variables

```bash
# LLM Provider (ollama, openai, anthropic, gemini)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-1.5-pro

# Workflow
CODE_TIMEOUT=120
MAX_RETRIES=3
```

### Data Format

The agent accepts CSV, Excel, or Parquet files with:

| Column Type | Description | Example |
|------------|-------------|---------|
| Date | Time dimension | `date`, `week`, `period` |
| KPI | Target variable | `revenue`, `sales`, `conversions` |
| Media | Marketing spend | `tv_spend`, `digital_spend` |
| Control | External factors | `price`, `promo`, `weather` |
| Geography | Optional dimension | `region`, `market` |
| Product | Optional dimension | `brand`, `sku` |

## Project Structure

```
mmm_agent_poc/
├── src/mmm_agent/
│   ├── __init__.py
│   ├── config.py          # LLM provider configuration
│   ├── state.py           # LangGraph state schema
│   ├── workflow.py        # Main workflow orchestration
│   ├── agents/
│   │   ├── planning.py    # Planning phase agent
│   │   ├── eda.py         # EDA phase agent
│   │   ├── modeling.py    # Modeling phase agent
│   │   └── interpretation.py  # Interpretation agent
│   ├── tools/
│   │   ├── code_executor.py   # Local code execution
│   │   ├── data_harmonizer.py # Multi-source data alignment
│   │   ├── rag_context.py     # RAG context manager
│   │   └── web_search.py      # Web search integration
│   └── data/
│       └── sample_generator.py # Sample data generation
├── examples/
│   └── run_demo.py        # Demo script
├── tests/
│   └── test_workflow.py   # Test suite
├── pyproject.toml         # Project configuration
└── README.md
```

## Key Design Decisions

### Local Code Executor
- Uses subprocess isolation instead of E2B for POC simplicity
- AST validation blocks dangerous operations (subprocess, socket, shutil)
- Automatic matplotlib Agg backend injection for headless execution

### Multi-LLM Support
- Uses LangChain's `init_chat_model()` for runtime provider switching
- Task-based routing (reasoning → Claude, code → GPT-4o, fast → Ollama)
- Graceful fallback when providers unavailable

### Ridge Regression Fallback
- Full Bayesian MMM requires PyMC which can be complex to install
- Ridge regression provides channel contributions without uncertainty
- `use_bayesian=False` (default) for reliable demos

## Extending the Framework

### Adding a New Phase

```python
from mmm_agent.state import MMMWorkflowState

async def my_custom_node(state: MMMWorkflowState) -> dict:
    # Access previous phase results
    roi = state.get("roi_estimates", {})
    
    # Do custom analysis
    results = analyze(roi)
    
    # Return state updates
    return {
        "custom_results": results,
        "messages": ["Custom analysis complete"],
    }
```

### Custom Tools

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Description of what this tool does."""
    return perform_custom_operation(query)
```

## Production Roadmap

This POC demonstrates the core concepts. For production:

1. **Replace RAG**: Use ChromaDB or Neo4j for vector/graph storage
2. **Distributed Fitting**: Integrate Ray for parallel model fitting
3. **Full Bayesian**: Use PyMC-Marketing for hierarchical models
4. **API Layer**: Add FastAPI endpoints for async execution
5. **Monitoring**: Add observability with LangSmith or similar

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
