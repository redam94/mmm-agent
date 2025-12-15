"""
Code Executor for MMM Workflows

Executes Python code in a sandboxed subprocess environment.
Based on the variable_importance repository patterns.

Features:
- Subprocess-based isolation
- AST validation for security
- File tracking for generated outputs
- Timeout handling
- Session-based working directories
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import Settings, get_settings


# =============================================================================
# Code Validation
# =============================================================================

# Blocked modules for security
BLOCKED_MODULES = {
    "subprocess", "socket", "shutil", "ftplib", "telnetlib",
    "smtplib", "http.server", "socketserver", "multiprocessing",
    "os.system", "pty", "fcntl", "grp", "pwd", "crypt",
}

# Allowed modules for MMM analysis
ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels",
    "sklearn", "pymc", "arviz", "xarray", "json", "csv", "pickle",
    "datetime", "pathlib", "os.path", "warnings", "typing", "functools",
    "itertools", "collections", "math", "re", "io", "hashlib",
    "pymc_marketing", "numpyro", "jax", "cloudpickle", "tqdm",
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
        
        # Block dangerous built-in calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile", "open"):
                    # Allow open for reading files
                    if node.func.id == "open":
                        continue
                    return False, f"Blocked function: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                # Block os.system, os.popen, etc.
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "os":
                        if node.func.attr in ("system", "popen", "spawn", "fork"):
                            return False, f"Blocked: os.{node.func.attr}"
    
    return True, "Code validated successfully"


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error: str | None = None
    execution_time: float = 0.0
    generated_files: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": str(self.return_value) if self.return_value else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "generated_files": self.generated_files,
            "plots": self.plots,
        }


# =============================================================================
# Code Executor
# =============================================================================

class CodeExecutor:
    """
    Execute Python code in isolated subprocess.
    
    Features:
    - Session-based working directories
    - File tracking
    - Timeout handling
    - Output capture
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        working_dir: Path | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.timeout = timeout_seconds
        self.base_dir = working_dir or self.settings.code_working_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._sessions: dict[str, Path] = {}
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get or create session working directory."""
        if session_id not in self._sessions:
            session_dir = self.base_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self._sessions[session_id] = session_dir
        return self._sessions[session_id]
    
    def _generate_wrapper_code(
        self,
        code: str,
        session_dir: Path,
        input_data: dict | None = None,
    ) -> str:
        """Generate wrapper code with setup and teardown."""
        wrapper = f'''
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set working directory
os.chdir("{session_dir}")

# Import common libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Input data (if provided)
_INPUT_DATA = {json.dumps(input_data) if input_data else "{}"}

# Track generated files
_GENERATED_FILES = []
_ORIGINAL_SAVEFIG = plt.savefig

def _tracked_savefig(*args, **kwargs):
    """Track saved figures."""
    if args:
        _GENERATED_FILES.append(str(args[0]))
    return _ORIGINAL_SAVEFIG(*args, **kwargs)

plt.savefig = _tracked_savefig

# Helper functions
def load_input_data():
    """Load input data passed to this execution."""
    return _INPUT_DATA

def save_output(data, filename="output.json"):
    """Save output data as JSON."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _GENERATED_FILES.append(filename)
    return filename

def list_files(pattern="*"):
    """List files in working directory."""
    from pathlib import Path
    return list(Path(".").glob(pattern))

# ============ USER CODE ============
try:
{_indent_code(code)}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    raise

# ============ END USER CODE ============

# Report generated files
print(f"\\n__GENERATED_FILES__: {{json.dumps(_GENERATED_FILES)}}")
'''
        return wrapper


def _indent_code(code: str, spaces: int = 4) -> str:
    """Indent code block."""
    indent = " " * spaces
    return "\n".join(indent + line for line in code.split("\n"))


class CodeExecutor:
    """
    Execute Python code in isolated subprocess.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        working_dir: Path | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.timeout = timeout_seconds
        self.base_dir = working_dir or self.settings.code_working_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Path] = {}
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get or create session working directory."""
        if session_id not in self._sessions:
            session_dir = self.base_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self._sessions[session_id] = session_dir
        return self._sessions[session_id]
    
    def _generate_wrapper_code(
        self,
        code: str,
        session_dir: Path,
        input_data: dict | None = None,
    ) -> str:
        """Generate wrapper code with setup and teardown."""
        indent = "    "
        indented_code = "\n".join(indent + line for line in code.split("\n"))
        
        wrapper = f'''
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.chdir("{session_dir}")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

_INPUT_DATA = {json.dumps(input_data) if input_data else "{}"}
_GENERATED_FILES = []
_ORIGINAL_SAVEFIG = plt.savefig

def _tracked_savefig(*args, **kwargs):
    if args:
        _GENERATED_FILES.append(str(args[0]))
    return _ORIGINAL_SAVEFIG(*args, **kwargs)

plt.savefig = _tracked_savefig

def load_input_data():
    return _INPUT_DATA

def save_output(data, filename="output.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    _GENERATED_FILES.append(filename)
    return filename

def list_files(pattern="*"):
    from pathlib import Path
    return list(Path(".").glob(pattern))

try:
{indented_code}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    raise

print(f"\\n__GENERATED_FILES__: {{json.dumps(_GENERATED_FILES)}}")
'''
        return wrapper
    
    async def execute(
        self,
        code: str,
        session_id: str = "default",
        input_data: dict | None = None,
        validate: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            session_id: Session ID for file persistence
            input_data: Optional input data dict
            validate: Whether to validate code first
        
        Returns:
            ExecutionResult with outputs
        """
        start_time = datetime.now()
        
        # Validate code
        if validate:
            is_valid, msg = validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    error=msg,
                )
        
        # Get session directory
        session_dir = self._get_session_dir(session_id)
        
        # Generate wrapper code
        wrapper_code = self._generate_wrapper_code(code, session_dir, input_data)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir=session_dir,
        ) as f:
            f.write(wrapper_code)
            script_path = f.name
        
        try:
            # Execute in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(session_dir),
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {self.timeout}s",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Parse generated files
            generated_files = []
            plots = []
            
            if "__GENERATED_FILES__:" in stdout_str:
                parts = stdout_str.split("__GENERATED_FILES__:")
                stdout_str = parts[0].strip()
                try:
                    files_json = parts[1].strip()
                    generated_files = json.loads(files_json)
                    
                    # Separate plots
                    for f in generated_files:
                        if f.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                            plots.append(str(session_dir / f))
                        else:
                            generated_files.append(str(session_dir / f))
                except:
                    pass
            
            # Check for actual files in directory
            for f in session_dir.glob("*"):
                if f.is_file() and not f.name.startswith("_"):
                    if f.suffix in [".png", ".jpg", ".svg", ".pdf"]:
                        if str(f) not in plots:
                            plots.append(str(f))
                    elif f.suffix in [".csv", ".json", ".parquet"]:
                        if str(f) not in generated_files:
                            generated_files.append(str(f))
            
            success = process.returncode == 0
            
            return ExecutionResult(
                success=success,
                stdout=stdout_str,
                stderr=stderr_str,
                error=stderr_str if not success else None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                generated_files=generated_files,
                plots=plots,
            )
        
        finally:
            # Cleanup temp script
            try:
                Path(script_path).unlink()
            except:
                pass
    
    def execute_sync(
        self,
        code: str,
        session_id: str = "default",
        input_data: dict | None = None,
    ) -> ExecutionResult:
        """Synchronous code execution."""
        return asyncio.run(self.execute(code, session_id, input_data))
    
    def get_session_files(self, session_id: str) -> list[Path]:
        """Get all files in session directory."""
        session_dir = self._get_session_dir(session_id)
        return [f for f in session_dir.rglob("*") if f.is_file()]
    
    def read_session_file(self, session_id: str, filename: str) -> bytes | None:
        """Read a file from session directory."""
        session_dir = self._get_session_dir(session_id)
        file_path = session_dir / filename
        if file_path.exists():
            return file_path.read_bytes()
        return None
    
    def cleanup_session(self, session_id: str):
        """Clean up session directory."""
        import shutil
        
        if session_id in self._sessions:
            session_dir = self._sessions[session_id]
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
            del self._sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    def copy_file_to_session(
        self,
        session_id: str,
        source_path: str | Path,
        dest_name: str | None = None,
    ) -> Path:
        """Copy a file into session directory."""
        import shutil
        
        session_dir = self._get_session_dir(session_id)
        source = Path(source_path)
        dest_name = dest_name or source.name
        dest = session_dir / dest_name
        
        shutil.copy2(source, dest)
        logger.debug(f"Copied {source} to {dest}")
        return dest


# =============================================================================
# LangChain Tool
# =============================================================================

def create_code_execution_tool(executor: CodeExecutor):
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
        - Create visualizations (plots are saved to files)
        - Perform statistical analysis
        - Process data for MMM
        
        Available libraries: pandas, numpy, matplotlib, seaborn,
        scipy, statsmodels, sklearn, pymc, arviz
        
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
# Singleton Factory
# =============================================================================

_executor: CodeExecutor | None = None


def get_code_executor(settings: Settings | None = None) -> CodeExecutor:
    """Get or create singleton code executor."""
    global _executor
    if _executor is None:
        settings = settings or get_settings()
        _executor = CodeExecutor(
            timeout_seconds=settings.code_execution_timeout,
            working_dir=settings.code_working_dir,
            settings=settings,
        )
    return _executor
