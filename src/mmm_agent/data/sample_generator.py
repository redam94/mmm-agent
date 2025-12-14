"""
Sample Data Generator for MMM Agent POC Demo.

Generates realistic marketing mix data for demonstrating the workflow:
- Daily/Weekly sales data with seasonality
- Media spend across multiple channels
- External factors (weather, holidays, etc.)

The data follows realistic patterns including:
- Adstock effects (carryover from media spend)
- Saturation (diminishing returns at high spend)
- Seasonality and trend
- Confounders and noise
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os


def generate_adstock(x: np.ndarray, decay: float = 0.7) -> np.ndarray:
    """Apply geometric adstock transformation."""
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = x[i] + decay * result[i-1]
    return result


def generate_saturation(x: np.ndarray, alpha: float = 0.5, lam: float = 1.0) -> np.ndarray:
    """Apply Hill saturation transformation."""
    return alpha * (1 - np.exp(-lam * x / x.mean()))


def generate_mmm_sample_data(
    n_periods: int = 156,  # 3 years of weekly data
    frequency: str = "W",
    start_date: str = "2022-01-01",
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate sample MMM data with realistic patterns.
    
    Args:
        n_periods: Number of time periods
        frequency: Time frequency ("D" for daily, "W" for weekly, "M" for monthly)
        start_date: Start date for the data
        seed: Random seed for reproducibility
        output_dir: Directory to save CSV files (None = don't save)
        
    Returns:
        Tuple of (sales_df, media_df, external_df)
    """
    np.random.seed(seed)
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_periods, freq=frequency)
    
    # =========================================================================
    # Generate Media Spend Data
    # =========================================================================
    
    # Base spend levels (different scales per channel)
    tv_base = 50000 + 20000 * np.random.randn(n_periods).cumsum() / 10
    tv_base = np.clip(tv_base, 20000, 100000)
    
    digital_base = 30000 + 15000 * np.random.randn(n_periods).cumsum() / 10
    digital_base = np.clip(digital_base, 10000, 80000)
    
    social_base = 15000 + 10000 * np.random.randn(n_periods).cumsum() / 10
    social_base = np.clip(social_base, 5000, 50000)
    
    search_base = 25000 + 12000 * np.random.randn(n_periods).cumsum() / 10
    search_base = np.clip(search_base, 8000, 60000)
    
    print_base = 10000 + 5000 * np.random.randn(n_periods).cumsum() / 10
    print_base = np.clip(print_base, 2000, 30000)
    
    # Add weekly patterns (higher spend in certain periods)
    week_of_year = dates.isocalendar().week.values
    holiday_boost = np.where((week_of_year >= 46) & (week_of_year <= 52), 1.3, 1.0)  # Q4 boost
    summer_dip = np.where((week_of_year >= 26) & (week_of_year <= 35), 0.85, 1.0)  # Summer dip
    
    # Apply seasonal patterns
    tv_spend = tv_base * holiday_boost * summer_dip + np.random.randn(n_periods) * 5000
    digital_spend = digital_base * holiday_boost + np.random.randn(n_periods) * 3000
    social_spend = social_base * holiday_boost + np.random.randn(n_periods) * 2000
    search_spend = search_base * holiday_boost + np.random.randn(n_periods) * 2500
    print_spend = print_base * summer_dip + np.random.randn(n_periods) * 1000
    
    # Ensure non-negative
    tv_spend = np.clip(tv_spend, 0, None)
    digital_spend = np.clip(digital_spend, 0, None)
    social_spend = np.clip(social_spend, 0, None)
    search_spend = np.clip(search_spend, 0, None)
    print_spend = np.clip(print_spend, 0, None)
    
    media_df = pd.DataFrame({
        "date": dates,
        "tv_spend": tv_spend.round(2),
        "digital_spend": digital_spend.round(2),
        "social_spend": social_spend.round(2),
        "search_spend": search_spend.round(2),
        "print_spend": print_spend.round(2),
    })
    
    # =========================================================================
    # Generate External Factors
    # =========================================================================
    
    # Temperature (affects behavior - higher in summer)
    day_of_year = dates.dayofyear.values
    temp_seasonal = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    temperature = temp_seasonal + np.random.randn(n_periods) * 3
    
    # Competitor activity (random but autocorrelated)
    competitor_spend = 40000 + np.random.randn(n_periods).cumsum() * 3000
    competitor_spend = np.clip(competitor_spend, 20000, 80000)
    
    # Economic indicator (GDP proxy, slowly varying)
    gdp_index = 100 + np.random.randn(n_periods).cumsum() * 0.5
    gdp_index = np.clip(gdp_index, 95, 110)
    
    # Holiday indicator
    is_holiday = np.where((week_of_year >= 51) | (week_of_year <= 1), 1, 0)  # Christmas/NY
    is_holiday = is_holiday | np.where((week_of_year >= 12) & (week_of_year <= 13), 1, 0)  # Easter
    is_holiday = is_holiday | np.where(week_of_year == 47, 1, 0)  # Black Friday
    
    external_df = pd.DataFrame({
        "date": dates,
        "temperature": temperature.round(1),
        "competitor_spend": competitor_spend.round(2),
        "gdp_index": gdp_index.round(2),
        "is_holiday": is_holiday.astype(int),
    })
    
    # =========================================================================
    # Generate Sales Data (with true effects)
    # =========================================================================
    
    # True effect parameters (what we want the model to recover)
    TRUE_EFFECTS = {
        "tv": {"beta": 0.15, "adstock": 0.7, "saturation_alpha": 0.8},
        "digital": {"beta": 0.25, "adstock": 0.3, "saturation_alpha": 0.6},
        "social": {"beta": 0.12, "adstock": 0.4, "saturation_alpha": 0.5},
        "search": {"beta": 0.30, "adstock": 0.2, "saturation_alpha": 0.7},
        "print": {"beta": 0.05, "adstock": 0.6, "saturation_alpha": 0.4},
    }
    
    # Base sales (intercept)
    base_sales = 500000
    
    # Trend (slight growth over time)
    trend = np.linspace(0, 50000, n_periods)
    
    # Seasonality (weekly pattern within year)
    seasonality = 30000 * np.sin(2 * np.pi * day_of_year / 365) + \
                  15000 * np.sin(4 * np.pi * day_of_year / 365)
    
    # Media effects with adstock and saturation
    tv_effect = generate_saturation(
        generate_adstock(tv_spend, TRUE_EFFECTS["tv"]["adstock"]),
        TRUE_EFFECTS["tv"]["saturation_alpha"]
    ) * TRUE_EFFECTS["tv"]["beta"] * 1000
    
    digital_effect = generate_saturation(
        generate_adstock(digital_spend, TRUE_EFFECTS["digital"]["adstock"]),
        TRUE_EFFECTS["digital"]["saturation_alpha"]
    ) * TRUE_EFFECTS["digital"]["beta"] * 1000
    
    social_effect = generate_saturation(
        generate_adstock(social_spend, TRUE_EFFECTS["social"]["adstock"]),
        TRUE_EFFECTS["social"]["saturation_alpha"]
    ) * TRUE_EFFECTS["social"]["beta"] * 1000
    
    search_effect = generate_saturation(
        generate_adstock(search_spend, TRUE_EFFECTS["search"]["adstock"]),
        TRUE_EFFECTS["search"]["saturation_alpha"]
    ) * TRUE_EFFECTS["search"]["beta"] * 1000
    
    print_effect = generate_saturation(
        generate_adstock(print_spend, TRUE_EFFECTS["print"]["adstock"]),
        TRUE_EFFECTS["print"]["saturation_alpha"]
    ) * TRUE_EFFECTS["print"]["beta"] * 1000
    
    # External effects
    temp_effect = -500 * (temperature - 20)  # Optimal at 20C
    competitor_effect = -0.3 * competitor_spend  # Negative effect
    gdp_effect = 2000 * (gdp_index - 100)  # Positive effect
    holiday_effect = 50000 * is_holiday  # Big holiday boost
    
    # Combine all effects
    sales = (
        base_sales +
        trend +
        seasonality +
        tv_effect +
        digital_effect +
        social_effect +
        search_effect +
        print_effect +
        temp_effect +
        competitor_effect +
        gdp_effect +
        holiday_effect +
        np.random.randn(n_periods) * 20000  # Random noise
    )
    
    # Ensure non-negative
    sales = np.clip(sales, 100000, None)
    
    # Units sold (derived from sales with some price variation)
    avg_price = 25 + np.random.randn(n_periods) * 2
    units = (sales / avg_price).astype(int)
    
    sales_df = pd.DataFrame({
        "date": dates,
        "revenue": sales.round(2),
        "units_sold": units,
        "avg_price": avg_price.round(2),
    })
    
    # =========================================================================
    # Save to CSV if output directory specified
    # =========================================================================
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        sales_df.to_csv(os.path.join(output_dir, "sales_data.csv"), index=False)
        media_df.to_csv(os.path.join(output_dir, "media_spend.csv"), index=False)
        external_df.to_csv(os.path.join(output_dir, "external_factors.csv"), index=False)
        
        # Also create a combined wide-format file
        combined = sales_df.merge(media_df, on="date").merge(external_df, on="date")
        combined.to_csv(os.path.join(output_dir, "mmm_data_combined.csv"), index=False)
        
        # Create ground truth file for validation
        ground_truth = pd.DataFrame({
            "channel": ["tv", "digital", "social", "search", "print"],
            "true_beta": [TRUE_EFFECTS[c]["beta"] for c in ["tv", "digital", "social", "search", "print"]],
            "true_adstock": [TRUE_EFFECTS[c]["adstock"] for c in ["tv", "digital", "social", "search", "print"]],
            "true_saturation": [TRUE_EFFECTS[c]["saturation_alpha"] for c in ["tv", "digital", "social", "search", "print"]],
        })
        ground_truth.to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)
        
        print(f"Generated sample data in {output_dir}:")
        print(f"  - sales_data.csv: {len(sales_df)} rows")
        print(f"  - media_spend.csv: {len(media_df)} rows")
        print(f"  - external_factors.csv: {len(external_df)} rows")
        print(f"  - mmm_data_combined.csv: Combined wide format")
        print(f"  - ground_truth.csv: True effect parameters")
    
    return sales_df, media_df, external_df


def generate_multi_geography_data(
    geographies: list = ["US", "UK", "DE"],
    n_periods: int = 104,  # 2 years weekly
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate multi-geography MMM data in MFF format.
    
    Args:
        geographies: List of geography codes
        n_periods: Number of time periods per geography
        output_dir: Directory to save CSV
        
    Returns:
        DataFrame in MFF (long) format
    """
    all_data = []
    
    for i, geo in enumerate(geographies):
        # Generate base data with different seed per geography
        sales_df, media_df, external_df = generate_mmm_sample_data(
            n_periods=n_periods,
            seed=42 + i * 100,
            output_dir=None,
        )
        
        # Scale by geography (US is largest)
        geo_scales = {"US": 1.0, "UK": 0.6, "DE": 0.5}
        scale = geo_scales.get(geo, 0.4)
        
        # Combine and melt to long format
        combined = sales_df.merge(media_df, on="date").merge(external_df, on="date")
        
        # Scale numeric columns
        for col in combined.columns:
            if col != "date" and combined[col].dtype in ["float64", "int64"]:
                combined[col] = combined[col] * scale
        
        # Melt to long format
        id_vars = ["date"]
        value_vars = [c for c in combined.columns if c != "date"]
        
        melted = combined.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="variable",
            value_name="value",
        )
        
        melted["geography"] = geo
        melted["product"] = "ALL"  # Single product for simplicity
        melted["period"] = melted["date"].dt.strftime("%Y-%m-%d")
        
        all_data.append(melted[["period", "geography", "product", "variable", "value"]])
    
    # Combine all geographies
    mff_df = pd.concat(all_data, ignore_index=True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mff_df.to_csv(os.path.join(output_dir, "mmm_data_mff.csv"), index=False)
        print(f"Generated MFF format data: {len(mff_df)} rows")
    
    return mff_df


if __name__ == "__main__":
    # Generate sample data for demo
    output_dir = "./sample_data"
    
    print("Generating single-geography sample data...")
    sales, media, external = generate_mmm_sample_data(output_dir=output_dir)
    
    print("\nGenerating multi-geography MFF data...")
    mff = generate_multi_geography_data(output_dir=output_dir)
    
    print("\nSample data generation complete!")
    print(f"\nSales data preview:\n{sales.head()}")
    print(f"\nMedia data preview:\n{media.head()}")
    print(f"\nMFF data preview:\n{mff.head(10)}")
