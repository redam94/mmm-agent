"""
MMM Agent Tools

Tools for the MMM agent workflow:
- Code execution (local subprocess)
- Data harmonization (MFF processing)
- RAG context retrieval
- Web search
"""

from .code_executor import (
    LocalCodeExecutor,
    ExecutionResult,
    validate_code,
    get_executor,
    create_code_execution_tool,
)

from .data_harmonizer import (
    DataHarmonizer,
    DataSourceSpec,
    VariableMapping,
    DimensionMapping,
    AlignmentReport,
    AlignmentIssue,
    auto_detect_source,
    create_data_harmonization_tool,
)

from .rag_context import (
    SimpleRAG,
    Document,
    SearchResult,
    WorkflowHistory,
    ContextManager,
    get_context_manager,
)

from .web_search import (
    WebSearcher,
    SearchResult as WebSearchResult,
    ResearchTool,
    get_web_searcher,
    create_web_search_tool,
    create_research_tool,
)

__all__ = [
    # Code Executor
    "LocalCodeExecutor",
    "ExecutionResult",
    "validate_code",
    "get_executor",
    "create_code_execution_tool",
    # Data Harmonizer
    "DataHarmonizer",
    "DataSourceSpec",
    "VariableMapping",
    "DimensionMapping",
    "AlignmentReport",
    "AlignmentIssue",
    "auto_detect_source",
    "create_data_harmonization_tool",
    # RAG
    "SimpleRAG",
    "Document",
    "SearchResult",
    "WorkflowHistory",
    "ContextManager",
    "get_context_manager",
    # Web Search
    "WebSearcher",
    "WebSearchResult",
    "ResearchTool",
    "get_web_searcher",
    "create_web_search_tool",
    "create_research_tool",
]
