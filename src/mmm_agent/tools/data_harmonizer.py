"""
Data Harmonization for MMM

Handles multiple data sources and aligns them on:
- Period (date/time dimension)
- Geography (market/region dimension)
- Product (brand/sku dimension)
- Variable name standardization

Outputs data in MFF (Master Flat File) format ready for BayesianMMM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


# =============================================================================
# Data Source Configuration
# =============================================================================

class DimensionMapping(BaseModel):
    """Mapping for a dimension column."""
    source_column: str
    target_name: str  # Standard name: Period, Geography, Product
    value_mapping: dict[str, str] = Field(default_factory=dict)
    date_format: str | None = None  # For period columns


class VariableMapping(BaseModel):
    """Mapping for value columns to MFF variables."""
    source_columns: list[str]  # Columns in source data
    variable_role: Literal["kpi", "media", "control", "auxiliary"]
    variable_prefix: str = ""  # Prefix to add to variable names
    value_scale: float = 1.0  # Scaling factor
    aggregation: Literal["sum", "mean", "last", "first"] = "sum"


class DataSourceSpec(BaseModel):
    """Specification for a data source."""
    name: str
    path: str
    source_type: Literal["csv", "excel", "parquet"] = "csv"
    
    # Dimension mappings
    period_column: str
    period_format: str = "%Y-%m-%d"
    geography_column: str | None = None
    product_column: str | None = None
    
    # Optional dimension value mappings
    geography_mapping: dict[str, str] = Field(default_factory=dict)
    product_mapping: dict[str, str] = Field(default_factory=dict)
    
    # Variable mappings
    variables: list[VariableMapping] = Field(default_factory=list)
    
    # Default handling
    fill_missing: float | None = None


# =============================================================================
# Alignment Report
# =============================================================================

@dataclass
class AlignmentIssue:
    """An alignment issue found during data harmonization."""
    severity: str  # "error", "warning", "info"
    source: str
    dimension: str
    issue_type: str
    message: str
    affected_rows: int = 0
    recommendation: str = ""


@dataclass
class AlignmentReport:
    """Report from data harmonization process."""
    success: bool
    sources_processed: int
    total_rows: int
    aligned_rows: int
    
    # Dimension coverage
    periods: list[str]
    geographies: list[str]
    products: list[str]
    variables: list[str]
    
    # Issues
    issues: list[AlignmentIssue] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    data_hash: str = ""
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "sources_processed": self.sources_processed,
            "total_rows": self.total_rows,
            "aligned_rows": self.aligned_rows,
            "periods": len(self.periods),
            "geographies": len(self.geographies),
            "products": len(self.products),
            "variables": len(self.variables),
            "issues": [
                {
                    "severity": i.severity,
                    "source": i.source,
                    "dimension": i.dimension,
                    "type": i.issue_type,
                    "message": i.message,
                }
                for i in self.issues
            ],
            "created_at": self.created_at.isoformat(),
            "data_hash": self.data_hash,
        }


# =============================================================================
# Data Harmonizer
# =============================================================================

class DataHarmonizer:
    """
    Harmonize multiple data sources into MFF format.
    
    MFF Format:
    - Period (date column)
    - Geography (optional)
    - Product (optional)
    - Variable (variable name)
    - Value (numeric value)
    
    Or wide format:
    - Period, Geography, Product + one column per variable
    """
    
    def __init__(
        self,
        target_frequency: Literal["D", "W", "M"] = "W",
        target_date_format: str = "%Y-%m-%d",
        output_format: Literal["long", "wide"] = "wide",
    ):
        self.target_frequency = target_frequency
        self.target_date_format = target_date_format
        self.output_format = output_format
        self.issues: list[AlignmentIssue] = []
    
    def _load_source(self, spec: DataSourceSpec) -> pd.DataFrame:
        """Load a data source."""
        path = Path(spec.path)
        
        if spec.source_type == "csv":
            df = pd.read_csv(path)
        elif spec.source_type == "excel":
            df = pd.read_excel(path)
        elif spec.source_type == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unknown source type: {spec.source_type}")
        
        logger.info(f"Loaded {spec.name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _parse_period(
        self,
        df: pd.DataFrame,
        spec: DataSourceSpec,
    ) -> pd.DataFrame:
        """Parse and standardize period column."""
        try:
            df["Period"] = pd.to_datetime(df[spec.period_column], format=spec.period_format)
        except Exception as e:
            # Try automatic parsing
            df["Period"] = pd.to_datetime(df[spec.period_column], infer_datetime_format=True)
            self.issues.append(AlignmentIssue(
                severity="warning",
                source=spec.name,
                dimension="Period",
                issue_type="date_format",
                message=f"Used inferred date format instead of {spec.period_format}",
            ))
        
        # Align to target frequency
        if self.target_frequency == "W":
            # Align to week start (Monday)
            df["Period"] = df["Period"].dt.to_period("W-MON").dt.start_time
        elif self.target_frequency == "M":
            df["Period"] = df["Period"].dt.to_period("M").dt.start_time
        # Daily needs no alignment
        
        return df
    
    def _map_dimension(
        self,
        df: pd.DataFrame,
        source_col: str | None,
        target_name: str,
        mapping: dict[str, str],
        source_name: str,
    ) -> pd.DataFrame:
        """Map a dimension column to standard values."""
        if source_col is None:
            df[target_name] = "Total"
            return df
        
        if source_col not in df.columns:
            self.issues.append(AlignmentIssue(
                severity="error",
                source=source_name,
                dimension=target_name,
                issue_type="missing_column",
                message=f"Column '{source_col}' not found",
            ))
            df[target_name] = "Unknown"
            return df
        
        # Apply mapping
        if mapping:
            df[target_name] = df[source_col].map(mapping).fillna(df[source_col])
        else:
            df[target_name] = df[source_col].astype(str)
        
        return df
    
    def _extract_variables(
        self,
        df: pd.DataFrame,
        spec: DataSourceSpec,
    ) -> pd.DataFrame:
        """Extract and melt variable columns to long format."""
        # Get dimension columns
        dim_cols = ["Period"]
        if "Geography" in df.columns:
            dim_cols.append("Geography")
        if "Product" in df.columns:
            dim_cols.append("Product")
        
        # Process each variable mapping
        long_dfs = []
        
        for var_map in spec.variables:
            # Get columns that exist
            existing_cols = [c for c in var_map.source_columns if c in df.columns]
            
            if not existing_cols:
                self.issues.append(AlignmentIssue(
                    severity="warning",
                    source=spec.name,
                    dimension="Variable",
                    issue_type="missing_columns",
                    message=f"None of {var_map.source_columns} found",
                ))
                continue
            
            # Melt to long format
            melted = df[dim_cols + existing_cols].melt(
                id_vars=dim_cols,
                value_vars=existing_cols,
                var_name="Variable",
                value_name="Value",
            )
            
            # Apply prefix and scaling
            if var_map.variable_prefix:
                melted["Variable"] = var_map.variable_prefix + "_" + melted["Variable"]
            
            melted["Value"] = melted["Value"] * var_map.value_scale
            
            # Add role
            melted["Role"] = var_map.variable_role
            
            long_dfs.append(melted)
        
        if not long_dfs:
            return pd.DataFrame()
        
        return pd.concat(long_dfs, ignore_index=True)
    
    def process_source(self, spec: DataSourceSpec) -> pd.DataFrame:
        """Process a single data source."""
        df = self._load_source(spec)
        
        # Parse period
        df = self._parse_period(df, spec)
        
        # Map dimensions
        df = self._map_dimension(
            df, spec.geography_column, "Geography",
            spec.geography_mapping, spec.name
        )
        df = self._map_dimension(
            df, spec.product_column, "Product",
            spec.product_mapping, spec.name
        )
        
        # Extract variables
        long_df = self._extract_variables(df, spec)
        
        # Add source tracking
        long_df["_Source"] = spec.name
        
        return long_df
    
    def harmonize(
        self,
        sources: list[DataSourceSpec],
        output_path: str | None = None,
    ) -> tuple[pd.DataFrame, AlignmentReport]:
        """
        Harmonize multiple data sources into unified MFF format.
        
        Args:
            sources: List of data source specifications
            output_path: Optional path to save output
        
        Returns:
            (harmonized_df, alignment_report)
        """
        self.issues = []
        
        # Process each source
        long_dfs = []
        total_rows = 0
        
        for spec in sources:
            try:
                source_df = self.process_source(spec)
                total_rows += len(source_df)
                long_dfs.append(source_df)
                logger.info(f"Processed {spec.name}: {len(source_df)} variable-observations")
            except Exception as e:
                self.issues.append(AlignmentIssue(
                    severity="error",
                    source=spec.name,
                    dimension="",
                    issue_type="processing_error",
                    message=str(e),
                ))
        
        if not long_dfs:
            report = AlignmentReport(
                success=False,
                sources_processed=0,
                total_rows=0,
                aligned_rows=0,
                periods=[],
                geographies=[],
                products=[],
                variables=[],
                issues=self.issues,
            )
            return pd.DataFrame(), report
        
        # Combine all sources
        combined = pd.concat(long_dfs, ignore_index=True)
        
        # Get dimension columns
        dim_cols = ["Period"]
        if "Geography" in combined.columns and combined["Geography"].nunique() > 1:
            dim_cols.append("Geography")
        if "Product" in combined.columns and combined["Product"].nunique() > 1:
            dim_cols.append("Product")
        
        # Aggregate duplicates
        grouped = combined.groupby(dim_cols + ["Variable"], as_index=False).agg({
            "Value": "sum",
            "Role": "first",
            "_Source": lambda x: ",".join(sorted(set(x))),
        })
        
        # Check for alignment issues
        self._check_alignment(grouped, dim_cols)
        
        # Convert to wide format if requested
        if self.output_format == "wide":
            output_df = self._to_wide_format(grouped, dim_cols)
        else:
            output_df = grouped
        
        # Create report
        report = AlignmentReport(
            success=len([i for i in self.issues if i.severity == "error"]) == 0,
            sources_processed=len(sources),
            total_rows=total_rows,
            aligned_rows=len(output_df),
            periods=sorted(combined["Period"].dt.strftime(self.target_date_format).unique().tolist()),
            geographies=sorted(combined["Geography"].unique().tolist()) if "Geography" in combined.columns else ["Total"],
            products=sorted(combined["Product"].unique().tolist()) if "Product" in combined.columns else ["Total"],
            variables=sorted(combined["Variable"].unique().tolist()),
            issues=self.issues,
            data_hash=hashlib.md5(output_df.to_json().encode()).hexdigest()[:8],
        )
        
        # Save if requested
        if output_path:
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved harmonized data to {output_path}")
        
        return output_df, report
    
    def _check_alignment(self, df: pd.DataFrame, dim_cols: list[str]):
        """Check for alignment issues in combined data."""
        # Check for gaps in period
        periods = df["Period"].unique()
        if len(periods) > 1:
            period_range = pd.date_range(
                periods.min(), periods.max(),
                freq={"D": "D", "W": "W-MON", "M": "MS"}[self.target_frequency]
            )
            missing_periods = set(period_range) - set(periods)
            if missing_periods:
                self.issues.append(AlignmentIssue(
                    severity="warning",
                    source="combined",
                    dimension="Period",
                    issue_type="gaps",
                    message=f"{len(missing_periods)} missing periods detected",
                    recommendation="Fill or interpolate missing periods",
                ))
        
        # Check variable coverage by source
        for var in df["Variable"].unique():
            var_sources = df[df["Variable"] == var]["_Source"].unique()
            if len(var_sources) > 1:
                logger.info(f"Variable '{var}' from multiple sources: {var_sources}")
    
    def _to_wide_format(self, df: pd.DataFrame, dim_cols: list[str]) -> pd.DataFrame:
        """Convert long format to wide format."""
        # Drop role and source columns for pivot
        pivot_df = df.drop(columns=["Role", "_Source"], errors="ignore")
        
        # Pivot to wide
        wide = pivot_df.pivot_table(
            index=dim_cols,
            columns="Variable",
            values="Value",
            aggfunc="sum",
        ).reset_index()
        
        # Flatten column names
        wide.columns.name = None
        
        return wide


# =============================================================================
# LangChain Tool Integration
# =============================================================================

def create_data_harmonization_tool(working_dir: Path):
    """Create a LangChain tool for data harmonization."""
    from langchain_core.tools import tool
    
    @tool
    def harmonize_data_sources(
        sources_json: str,
        output_filename: str = "harmonized_data.csv",
    ) -> dict:
        """
        Harmonize multiple data sources into MFF format for MMM.
        
        This tool aligns data on Period, Geography, and Product dimensions,
        standardizes variable names, and outputs a unified dataset.
        
        Args:
            sources_json: JSON string with list of source specifications.
                Each source should have:
                - name: Source identifier
                - path: File path
                - period_column: Name of date column
                - period_format: Date format (e.g., "%Y-%m-%d")
                - geography_column: Optional geo column
                - product_column: Optional product column
                - variables: List of {source_columns, variable_role, variable_prefix}
            output_filename: Name for output file
        
        Returns:
            Dict with success status, output path, and alignment report
        """
        try:
            sources_data = json.loads(sources_json)
            sources = [DataSourceSpec(**s) for s in sources_data]
        except Exception as e:
            return {"success": False, "error": f"Invalid sources JSON: {e}"}
        
        harmonizer = DataHarmonizer(
            target_frequency="W",
            output_format="wide",
        )
        
        output_path = working_dir / output_filename
        df, report = harmonizer.harmonize(sources, str(output_path))
        
        return {
            "success": report.success,
            "output_path": str(output_path),
            "rows": len(df),
            "variables": report.variables,
            "report": report.to_dict(),
        }
    
    return harmonize_data_sources


# =============================================================================
# Convenience Functions
# =============================================================================

def auto_detect_source(path: str) -> DataSourceSpec:
    """
    Auto-detect data source configuration from file.
    
    Attempts to identify:
    - Date columns
    - Dimension columns (geo, product)
    - Numeric value columns
    """
    df = pd.read_csv(path)
    
    # Find date column
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(10))
            date_cols.append(col)
        except:
            pass
    
    period_col = date_cols[0] if date_cols else df.columns[0]
    
    # Find likely dimension columns (low cardinality strings)
    dim_candidates = []
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() < 50:
            if col.lower() in ["geo", "geography", "region", "market", "country", "state"]:
                dim_candidates.append(("geography", col))
            elif col.lower() in ["product", "brand", "sku", "category"]:
                dim_candidates.append(("product", col))
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Build spec
    spec = DataSourceSpec(
        name=Path(path).stem,
        path=path,
        period_column=period_col,
        geography_column=next((c for t, c in dim_candidates if t == "geography"), None),
        product_column=next((c for t, c in dim_candidates if t == "product"), None),
        variables=[
            VariableMapping(
                source_columns=numeric_cols,
                variable_role="media",  # Default, should be overridden
            )
        ],
    )
    
    return spec
