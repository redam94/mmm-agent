# MMM Workflows

AI-Powered Marketing Mix Modeling with Agentic Workflows

## Overview

MMM Workflows is a comprehensive system for Marketing Mix Modeling (MMM) using four specialized AI agents:

1. **Research Agent** - Web search and planning for data collection strategy
2. **EDA Agent** - Data quality assessment and transformation to MFF format
3. **Modeling Agent** - Bayesian MMM fitting using PyMC-Marketing
4. **What-If Agent** - Scenario analysis and budget optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MMM Workflows API                            │
│                       (FastAPI + WebSocket)                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐        ┌─────────────────┐        ┌───────────────┐
│   Research    │   →    │      EDA        │   →    │   Modeling    │
│    Agent      │        │     Agent       │        │    Agent      │
└───────────────┘        └─────────────────┘        └───────────────┘
        │                          │                          │
        │                          │                          ▼
        │                          │                 ┌───────────────┐
        │                          │                 │   What-If     │
        │                          │                 │    Agent      │
        │                          │                 └───────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │                    GraphRAG Layer                        │
        │              (Neo4j + ChromaDB + PostgreSQL)             │
        └─────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Ollama (qwen3:30b, qwen3-coder:30b) | Reasoning and code generation |
| Orchestration | LangGraph | Workflow state management |
| Graph DB | Neo4j | Knowledge graph, lineage tracking |
| Vector DB | ChromaDB | Semantic search |
| Checkpointing | PostgreSQL | Workflow persistence |
| Pub/Sub | Redis | Real-time progress streaming |
| Modeling | PyMC-Marketing | Bayesian MMM |
| API | FastAPI | REST + WebSocket endpoints |

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- NVIDIA GPU (optional, for faster LLM inference)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-org/mmm-workflows.git
cd mmm-workflows
```

2. **Start infrastructure services**
```bash
docker-compose up -d neo4j postgres redis chromadb ollama
```

3. **Pull Ollama models**
```bash
docker exec -it mmm-ollama ollama pull qwen3:30b
docker exec -it mmm-ollama ollama pull qwen3-coder:30b
```

4. **Install Python dependencies**
```bash
pip install -e .
```

5. **Initialize databases**
```bash
python scripts/init_db.py
```

6. **Start the API**
```bash
uvicorn mmm_workflows.api.main:app --reload
```

### Full Docker Deployment

```bash
docker-compose up -d
```

Access the API at http://localhost:8000/docs

## Usage

### Python API

```python
import asyncio
from mmm_workflows.workflows import (
    create_research_workflow,
    create_eda_workflow,
    create_modeling_workflow,
    create_whatif_workflow,
)

async def run_mmm_pipeline():
    # Step 1: Research
    research = await create_research_workflow()
    research_result = await research.run(
        "Build MMM for CPG brand with 2 years of weekly data"
    )
    
    # Step 2: EDA
    eda = await create_eda_workflow()
    eda_result = await eda.run(
        data_sources=["data/sales.csv"],
        target_variable="Sales",
        media_channels=["TV_Spend", "Digital_Spend"],
    )
    
    # Step 3: Modeling
    modeling = await create_modeling_workflow()
    model_result = await modeling.run(
        mff_data_path=eda_result["mff_data_path"],
        research_plan=research_result["research_plan"],
    )
    
    # Step 4: What-If Analysis
    whatif = await create_whatif_workflow()
    whatif_result = await whatif.run(
        model_artifact_path=model_result["model_artifact_path"],
        mff_data_path=eda_result["mff_data_path"],
        user_query="What if we increase TV spend by 20%?",
    )
    
    return whatif_result

asyncio.run(run_mmm_pipeline())
```

### REST API

```bash
# Start research workflow
curl -X POST http://localhost:8000/research/start \
  -H "Content-Type: application/json" \
  -d '{"query": "Build MMM for beverage brand"}'

# Check status
curl http://localhost:8000/workflow/{workflow_id}

# Upload data
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sales.csv"

# Start EDA
curl -X POST http://localhost:8000/eda/start \
  -H "Content-Type: application/json" \
  -d '{
    "data_sources": ["/path/to/uploaded/file.csv"],
    "target_variable": "Sales",
    "media_channels": ["TV_Spend", "Digital_Spend"]
  }'
```

### WebSocket (Real-time Updates)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{workflow_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status:', data.data.status);
  console.log('Progress:', data.data.progress);
};
```

## Workflows

### 1. Research Agent

**Purpose**: Conducts web research and creates a structured research plan.

**Inputs**:
- User query describing the MMM objective

**Outputs**:
- Research plan with target variable, channels, controls
- Causal hypotheses
- Data requirements

**Phases**:
1. Initialize analysis
2. Conduct web search (DuckDuckGo + crawl4ai)
3. Generate research plan
4. Collect user feedback (optional)
5. Finalize

### 2. EDA Agent

**Purpose**: Cleans data and transforms to Marketing Measurement Framework (MFF) format.

**Inputs**:
- Data file paths
- Target variable name
- Media channel columns

**Outputs**:
- MFF-formatted data
- Quality report
- EDA visualizations
- Modeling recommendations

**Phases**:
1. Load and merge data
2. Run quality checks
3. Generate EDA analysis
4. Transform to MFF format
5. Finalize with recommendations

### 3. Modeling Agent

**Purpose**: Fits Bayesian MMM and extracts insights.

**Inputs**:
- MFF data path
- Research plan (optional)
- Feature transformations (optional)

**Outputs**:
- Fitted model artifact
- Channel contributions
- ROI estimates
- Convergence diagnostics
- Interpretation summary

**Phases**:
1. Initialize and configure
2. Generate model configuration
3. Fit Bayesian MMM
4. Run diagnostics
5. Interpret results

### 4. What-If Agent

**Purpose**: Scenario analysis and budget optimization.

**Inputs**:
- Model artifact path
- MFF data path
- Scenario query (natural language)

**Outputs**:
- Scenario comparison results
- Optimization recommendations
- Sensitivity analysis
- Summary with actionable insights

**Phases**:
1. Load model artifacts
2. Parse scenario queries
3. Run scenario analysis
4. Generate optimization suggestions
5. Create summary

## Configuration

Environment variables:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REASONING_MODEL=qwen3:30b
OLLAMA_CODING_MODEL=qwen3-coder:30b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/mmm
ASYNC_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mmm

# Redis
REDIS_URL=redis://localhost:6379

# Storage
UPLOAD_DIR=/data/uploads
MODEL_DIR=/data/models
OUTPUT_DIR=/data/outputs
```

## Project Structure

```
mmm-workflows/
├── src/
│   └── mmm_workflows/
│       ├── __init__.py
│       ├── config.py              # Settings and LLM configuration
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py            # FastAPI application
│       ├── db/
│       │   ├── __init__.py
│       │   ├── neo4j_client.py    # Neo4j graph operations
│       │   └── graphrag.py        # Combined graph + vector retrieval
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── web_search.py      # DuckDuckGo + crawl4ai
│       │   └── code_executor.py   # Sandboxed Python execution
│       └── workflows/
│           ├── __init__.py
│           ├── state.py           # LangGraph state definitions
│           ├── research_agent.py  # Workflow 1
│           ├── eda_agent.py       # Workflow 2
│           ├── modeling_agent.py  # Workflow 3
│           └── whatif_agent.py    # Workflow 4
├── scripts/
│   └── init_db.py                 # Database initialization
├── examples/
│   └── full_workflow_example.py   # Usage examples
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## GraphRAG Knowledge Persistence

The system uses a hybrid GraphRAG approach:

- **Neo4j**: Stores entities (Variables, Analyses, Decisions, Models) and relationships
- **ChromaDB**: Stores document embeddings for semantic search
- **PostgreSQL**: Stores LangGraph checkpoints for workflow resumability

### Knowledge Flow

1. **Research Agent** stores:
   - Web search results with sources
   - Research synthesis
   - Planning decisions

2. **EDA Agent** stores:
   - Data quality decisions
   - Feature transformation rationale
   - MFF generation metadata

3. **Modeling Agent** stores:
   - Model configuration rationale
   - Convergence decisions
   - Interpretation insights

4. **What-If Agent** retrieves:
   - Prior successful patterns
   - Similar scenarios
   - Optimization best practices

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## References

- [PyMC-Marketing](https://www.pymc-marketing.io/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Neo4j](https://neo4j.com/)
- [Ollama](https://ollama.ai/)
