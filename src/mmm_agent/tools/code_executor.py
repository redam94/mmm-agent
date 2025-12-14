"""
Local Code Executor for MMM Agent

Executes Python code in a sandboxed subprocess environment.
Adapted from the variable-importance repository.

Features:
- Executes Python code in subprocess
- Captures stdout, stderr
- Tracks generated files (plots, data, models)
- AST-based code validation for safety
- Timeout handling
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger


# =============================================================================
# Code Validation
# =============================================================================

# Modules blocked for security
BLOCKED_MODULES = {
    "subprocess", "socket", "shutil", "ftplib", "telnetlib",
    "smtplib", "http.server", "socketserver", "multiprocessing"
}

# Allowed modules for MMM analysis
ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels",
    "sklearn", "pymc", "arviz", "xarray", "json", "csv", "pickle",
    "datetime", "pathlib", "os.path", "warnings", "typing", "functools",
    "itertools", "collections", "math", "re", "io",
}


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate Python code using AST analysis.
    
    Returns:
        (is_valid, message)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module in BLOCKED_MODULES:
                    return False, f"Blocked module: {module}"
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                if module in BLOCKED_MODULES:
                    return False, f"Blocked module: {module}"
        
        # Block exec/eval with dynamic strings
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile"):
                    # Check if argument is a literal string (safer)
                    if node.args and not isinstance(node.args[0], ast.Constant):
                        return False, f"Dynamic {node.func.id}() not allowed"
    
    return True, "Valid"


# =============================================================================
# Matplotlib Setup Code
# =============================================================================

MATPLOTLIB_SETUP = '''
# === Auto-injected setup ===
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

# Working directory for outputs
import os
WORKING_DIR = "{working_dir}"
os.chdir(WORKING_DIR)

def save_figure(name: str, fig=None):
    """Save current figure with standardized naming."""
    if fig is None:
        fig = plt.gcf()
    path = os.path.join(WORKING_DIR, f"{{name}}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {{name}}.png")
    return path

# === End setup ===

'''


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    error: str | None
    execution_time_seconds: float
    generated_files: list[str] = field(default_factory=list)
    working_dir: str = ""
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "generated_files": self.generated_files,
            "working_dir": self.working_dir,
        }


# =============================================================================
# Code Executor
# =============================================================================

class LocalCodeExecutor:
    """
    Execute Python code in a sandboxed subprocess.
    
    Features:
    - Subprocess isolation
    - Output capture
    - File tracking
    - Timeout handling
    - Input data passing via pickle
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        max_output_lines: int = 1000,
        working_dir: Path | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_lines = max_output_lines
        self.base_working_dir = working_dir or Path(tempfile.gettempdir()) / "mmm_agent"
        self.base_working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalCodeExecutor initialized (timeout={timeout_seconds}s)")
    
    def _get_working_dir(self, session_id: str) -> Path:
        """Get or create working directory for a session."""
        work_dir = self.base_working_dir / session_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def _prepare_code(
        self,
        code: str,
        working_dir: Path,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        """Prepare code with setup and optional input data loading."""
        prepared = MATPLOTLIB_SETUP.format(working_dir=str(working_dir))
        
        # Add input data loading if provided
        if input_data:
            import pickle
            data_file = working_dir / "_input_data.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(input_data, f)
            
            prepared += f'''
# === Load input data ===
import pickle
with open("{data_file}", 'rb') as f:
    __INPUT__ = pickle.load(f)
# Unpack to global scope
for __k, __v in __INPUT__.items():
    globals()[__k] = __v
print(f"Loaded input data: {{list(__INPUT__.keys())}}")
# === End input data ===

'''
        
        prepared += code
        return prepared
    
    def _truncate_output(self, text: str) -> str:
        """Truncate output if too long."""
        lines = text.split('\n')
        if len(lines) > self.max_output_lines:
            kept = lines[:self.max_output_lines]
            return '\n'.join(kept) + f"\n\n[... truncated {len(lines) - self.max_output_lines} lines ...]"
        return text
    
    async def execute(
        self,
        code: str,
        session_id: str = "default",
        input_data: dict[str, Any] | None = None,
        validate: bool = True,
        on_progress: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """
        Execute Python code asynchronously.
        
        Args:
            code: Python code to execute
            session_id: Session identifier for working directory
            input_data: Optional dict to pass to code via pickle
            validate: Whether to validate code before execution
            on_progress: Optional callback for progress updates
        
        Returns:
            ExecutionResult with outputs and artifacts
        """
        start_time = datetime.now()
        working_dir = self._get_working_dir(session_id)
        
        # Validate code
        if validate:
            is_valid, msg = validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=msg,
                    error=msg,
                    execution_time_seconds=0,
                    working_dir=str(working_dir),
                )
        
        if on_progress:
            on_progress("Preparing code execution...")
        
        # Track files before
        files_before = set(working_dir.rglob("*"))
        
        # Prepare code
        prepared_code = self._prepare_code(code, working_dir, input_data)
        
        # Save to temp file
        code_file = working_dir / f"_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        code_file.write_text(prepared_code)
        
        if on_progress:
            on_progress(f"Executing in {working_dir}...")
        
        try:
            # Run in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir),
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
                
                stdout = self._truncate_output(stdout_bytes.decode('utf-8', errors='replace'))
                stderr = self._truncate_output(stderr_bytes.decode('utf-8', errors='replace'))
                success = process.returncode == 0
                error = None if success else f"Exit code: {process.returncode}"
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout = ""
                stderr = ""
                success = False
                error = f"Timeout after {self.timeout_seconds}s"
            
            # Track new files
            files_after = set(working_dir.rglob("*"))
            generated_files = [
                str(f.relative_to(working_dir))
                for f in (files_after - files_before)
                if f.is_file() and not f.name.startswith('_')
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                error=error,
                execution_time_seconds=execution_time,
                generated_files=generated_files,
                working_dir=str(working_dir),
            )
            
            if on_progress:
                status = "✅ Success" if success else f"❌ Failed: {error}"
                on_progress(f"{status} ({execution_time:.2f}s)")
            
            logger.info(f"Execution {'succeeded' if success else 'failed'}: {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                error=f"Exception: {e}",
                execution_time_seconds=execution_time,
                working_dir=str(working_dir),
            )
    
    def execute_sync(
        self,
        code: str,
        session_id: str = "default",
        input_data: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Synchronous wrapper for execute()."""
        return asyncio.run(self.execute(code, session_id, input_data))
    
    def get_session_files(self, session_id: str) -> list[Path]:
        """Get all files in a session's working directory."""
        working_dir = self._get_working_dir(session_id)
        return [f for f in working_dir.rglob("*") if f.is_file() and not f.name.startswith('_')]
    
    def read_session_file(self, session_id: str, filename: str) -> bytes | None:
        """Read a file from a session's working directory."""
        working_dir = self._get_working_dir(session_id)
        file_path = working_dir / filename
        if file_path.exists():
            return file_path.read_bytes()
        return None
    
    def cleanup_session(self, session_id: str):
        """Clean up a session's working directory."""
        import shutil
        working_dir = self._get_working_dir(session_id)
        if working_dir.exists():
            shutil.rmtree(working_dir)
            logger.info(f"Cleaned up session: {session_id}")


# =============================================================================
# LangChain Tool Integration
# =============================================================================

def create_code_execution_tool(executor: LocalCodeExecutor):
    """Create a LangChain tool for code execution."""
    from langchain_core.tools import tool
    
    @tool
    async def execute_python_code(
        code: str,
        session_id: str = "default",
    ) -> dict:
        """
        Execute Python code for data analysis.
        
        Use this tool to:
        - Load and analyze CSV/Excel data
        - Create visualizations
        - Perform statistical analysis
        - Process data for MMM
        
        The code runs in an isolated environment with pandas, numpy,
        matplotlib, seaborn, scipy, statsmodels, and sklearn available.
        
        Args:
            code: Python code to execute
            session_id: Session ID for file persistence
        
        Returns:
            Dict with success status, output, and generated files
        """
        result = await executor.execute(code, session_id)
        return result.to_dict()
    
    return execute_python_code


# =============================================================================
# Factory
# =============================================================================

_executor_instance: LocalCodeExecutor | None = None


def get_executor(
    timeout_seconds: int = 300,
    working_dir: Path | None = None,
) -> LocalCodeExecutor:
    """Get or create the global code executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = LocalCodeExecutor(
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
        )
    return _executor_instance
