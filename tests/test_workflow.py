"""
Tests for MMM Agent POC.

Run with: pytest tests/ -v
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmm_agent.state import MMMWorkflowState, WorkflowPhase
from mmm_agent.config import Settings, LLMProvider, get_llm_config
from mmm_agent.tools.code_executor import LocalCodeExecutor, validate_code
from mmm_agent.tools.data_harmonizer import DataHarmonizer, auto_detect_source
from mmm_agent.tools.rag_context import SimpleRAG, ContextManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample MMM data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    
    data = pd.DataFrame({
        "date": dates,
        "revenue": 100000 + np.random.normal(0, 5000, 52).cumsum(),
        "tv_spend": np.random.uniform(10000, 30000, 52),
        "digital_spend": np.random.uniform(5000, 20000, 52),
        "social_spend": np.random.uniform(2000, 10000, 52),
        "price": np.random.uniform(9.99, 14.99, 52),
        "promo_flag": np.random.choice([0, 1], 52),
    })
    
    return data


@pytest.fixture
def sample_data_file(sample_data, tmp_path):
    """Save sample data to a temporary CSV file."""
    filepath = tmp_path / "test_data.csv"
    sample_data.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def initial_state(sample_data_file):
    """Create initial workflow state for testing."""
    return {
        "current_phase": WorkflowPhase.PLANNING,
        "messages": [],
        "errors": [],
        "data_sources": [sample_data_file],
        "business_context": "Test business context",
        "kpi_column": "revenue",
        "media_columns": ["tv_spend", "digital_spend", "social_spend"],
        "date_column": "date",
        "geography_column": None,
        "product_column": None,
        "control_columns": ["price", "promo_flag"],
        "do_web_research": False,
        "research_questions": [],
        "target_variable": "",
        "media_channels": [],
        "control_variables": [],
        "causal_hypotheses": [],
        "data_quality_report": {},
        "feature_transformations": [],
        "correlation_matrix": {},
        "mff_data_path": "",
        "model_artifact_path": "",
        "convergence_diagnostics": {},
        "channel_contributions": {},
        "roi_estimates": {},
        "budget_allocation": {},
        "what_if_scenarios": [],
        "recommendations": [],
        "rag_context": [],
        "web_search_results": [],
        "prior_decisions": [],
        "generated_artifacts": [],
    }


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration module."""
    
    def test_settings_defaults(self):
        """Test default settings."""
        settings = Settings()
        assert settings.llm_provider == LLMProvider.OLLAMA
        assert settings.code_timeout == 120
        
    def test_get_llm_config(self):
        """Test LLM config generation."""
        settings = Settings()
        
        # Test different task types
        reasoning_config = get_llm_config("reasoning", settings)
        assert reasoning_config is not None
        assert "model" in reasoning_config
        
        fast_config = get_llm_config("fast", settings)
        assert fast_config is not None


# =============================================================================
# Code Executor Tests
# =============================================================================

class TestCodeExecutor:
    """Tests for local code executor."""
    
    def test_validate_code_safe(self):
        """Test validation of safe code."""
        safe_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3]})
print(df.sum())
"""
        is_valid, error = validate_code(safe_code)
        assert is_valid, f"Safe code should be valid: {error}"
    
    def test_validate_code_dangerous_import(self):
        """Test validation blocks dangerous imports."""
        dangerous_code = """
import subprocess
subprocess.run(['ls', '-la'])
"""
        is_valid, error = validate_code(dangerous_code)
        assert not is_valid
        assert "subprocess" in error.lower()
    
    def test_validate_code_dangerous_call(self):
        """Test validation blocks dangerous function calls."""
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        is_valid, error = validate_code(dangerous_code)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_executor_simple_code(self):
        """Test execution of simple code."""
        executor = LocalCodeExecutor(timeout_seconds=30)
        
        result = await executor.execute("""
import pandas as pd
print("Hello from executor")
print(1 + 1)
""")
        
        assert result.success
        assert "Hello from executor" in result.stdout
        assert "2" in result.stdout
        
        executor.cleanup_session()
    
    @pytest.mark.asyncio
    async def test_executor_with_data(self):
        """Test execution with input data."""
        executor = LocalCodeExecutor(timeout_seconds=30)
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        result = await executor.execute(
            code="""
print(f"Rows: {len(input_data)}")
print(f"Sum of a: {input_data['a'].sum()}")
""",
            input_data=df,
        )
        
        assert result.success
        assert "Rows: 3" in result.stdout
        assert "Sum of a: 6" in result.stdout
        
        executor.cleanup_session()
    
    @pytest.mark.asyncio
    async def test_executor_timeout(self):
        """Test execution timeout."""
        executor = LocalCodeExecutor(timeout_seconds=2)
        
        result = await executor.execute("""
import time
time.sleep(10)
""")
        
        assert not result.success
        assert "timeout" in result.error.lower() or result.stderr
        
        executor.cleanup_session()


# =============================================================================
# Data Harmonizer Tests
# =============================================================================

class TestDataHarmonizer:
    """Tests for data harmonization."""
    
    def test_auto_detect_source(self, sample_data_file):
        """Test automatic source detection."""
        spec = auto_detect_source(sample_data_file)
        
        assert spec is not None
        assert spec.period_column == "date"
        assert len(spec.variable_mappings) > 0
    
    def test_harmonizer_basic(self, sample_data_file):
        """Test basic harmonization."""
        harmonizer = DataHarmonizer(target_frequency="W")
        spec = auto_detect_source(sample_data_file)
        
        mff_data, report = harmonizer.harmonize([spec])
        
        assert report.success
        assert not mff_data.empty
        assert "Period" in mff_data.columns or "period" in mff_data.columns.str.lower()


# =============================================================================
# RAG Context Tests
# =============================================================================

class TestRAGContext:
    """Tests for RAG context management."""
    
    def test_simple_rag_search(self):
        """Test simple RAG search."""
        rag = SimpleRAG()
        
        results = rag.search("adstock transformation")
        assert len(results) > 0
        assert any("adstock" in r.lower() for r in results)
    
    def test_simple_rag_phase_context(self):
        """Test phase-specific context retrieval."""
        rag = SimpleRAG()
        
        planning_context = rag.get_phase_context(WorkflowPhase.PLANNING)
        assert len(planning_context) > 0
        
        modeling_context = rag.get_phase_context(WorkflowPhase.MODELING)
        assert len(modeling_context) > 0
    
    def test_context_manager(self):
        """Test context manager."""
        manager = ContextManager()
        
        # Add a decision
        manager.add_decision(
            workflow_id="test",
            phase="planning",
            decision="Selected revenue as KPI",
            reasoning="Primary business metric",
        )
        
        # Get context
        context = manager.get_context(
            workflow_id="test",
            phase=WorkflowPhase.EDA,
            query="data quality",
        )
        
        assert "domain_knowledge" in context
        assert "history" in context


# =============================================================================
# State Tests
# =============================================================================

class TestState:
    """Tests for workflow state."""
    
    def test_workflow_phase_enum(self):
        """Test workflow phase enum."""
        assert WorkflowPhase.PLANNING.value == "planning"
        assert WorkflowPhase.COMPLETE.value == "complete"
    
    def test_initial_state_structure(self, initial_state):
        """Test initial state has required fields."""
        required_fields = [
            "current_phase",
            "data_sources",
            "business_context",
            "kpi_column",
            "media_columns",
        ]
        
        for field in required_fields:
            assert field in initial_state


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests (require mocked LLM)."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test workflow can be created."""
        from mmm_agent.workflow import MMMWorkflowBuilder
        
        # This will fail without LLM, but tests the import structure
        with pytest.raises(Exception):
            # Expected to fail without proper LLM setup
            builder = MMMWorkflowBuilder()
    
    def test_sample_data_generator(self, tmp_path):
        """Test sample data generation."""
        from mmm_agent.data.sample_generator import generate_mmm_sample_data
        
        output_dir = str(tmp_path / "data")
        generate_mmm_sample_data(output_dir=output_dir)
        
        # Check files were created
        assert os.path.exists(os.path.join(output_dir, "mmm_data_combined.csv"))
        
        # Load and verify
        df = pd.read_csv(os.path.join(output_dir, "mmm_data_combined.csv"))
        assert len(df) > 0
        assert "date" in df.columns
        assert "revenue" in df.columns


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
