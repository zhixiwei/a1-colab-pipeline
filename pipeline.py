"""
Orchestrator: load cleaned Orbis parquets → compute outcome variables → save CSVs.

Usage on Colab:
    from pipeline import run_all
    results = run_all()
"""

import os
import polars as pl
from config import (
    MASTER_DIR, OUTPUT_DIR, FIRM_COL, YEAR_COL,
    COUNTRY_COL, INDUSTRY_COL, SALES_COL,
)
from outcomes.markup import compute_firm_markup, compute_markup_decomposition
from outcomes.op_covariance import compute_op_covariance
from outcomes.dispersion import compute_dispersion
from outcomes.concentration import compute_concentration


def load_data() -> pl.LazyFrame:
    """Load all parquets from MASTER_DIR, derive country and SIC2."""
    print(f"Loading parquets from {MASTER_DIR}/ ...")
    df = pl.scan_parquet(os.path.join(MASTER_DIR, "*.parquet"))

    # Derive country from bvd_id[:2]
    df = df.with_columns(
        pl.col(FIRM_COL).str.slice(0, 2).alias(COUNTRY_COL)
    )

    # Truncate SIC to 2-digit (string)
    df = df.with_columns(
        pl.col(INDUSTRY_COL).cast(pl.Utf8).str.slice(0, 2).alias(INDUSTRY_COL)
    )

    n = df.select(pl.count()).collect().item()
    print(f"  Loaded {n:,} firm-year observations")
    return df


def run_all(save: bool = True) -> dict:
    """
    Run all outcome modules and save results.

    Returns dict of outcome name → DataFrame for inspection.
    """
    df = load_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # 1. Markup decomposition
    print("\n[1/4] Computing markup decomposition ...")
    df_markup = compute_firm_markup(df)
    markup_decomp = compute_markup_decomposition(df_markup)
    results["markup_decomp"] = markup_decomp
    print(f"  → {markup_decomp.height} industry-pairs")

    if save:
        # Save firm-level markup panel as parquet (large)
        firm_markup = df_markup.collect()
        firm_path = os.path.join(OUTPUT_DIR, "firm_panel_markup.parquet")
        firm_markup.write_parquet(firm_path)
        print(f"  Saved firm panel: {firm_path}")

        # Save industry decomposition as CSV
        csv_path = os.path.join(OUTPUT_DIR, "industry_markup_decomp.csv")
        markup_decomp.write_csv(csv_path)
        print(f"  Saved decomposition: {csv_path}")

    # 2. OP covariance
    print("\n[2/4] Computing OP covariance ...")
    opcov = compute_op_covariance(df)
    results["op_covariance"] = opcov
    print(f"  → {opcov.height} industry-pairs")

    if save:
        csv_path = os.path.join(OUTPUT_DIR, "industry_op_covariance.csv")
        opcov.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    # 3. Dispersion
    print("\n[3/4] Computing MRPK/MRPL/markup dispersion ...")
    disp = compute_dispersion(df)
    results["dispersion"] = disp
    print(f"  → {disp.height} industry-pairs")

    if save:
        csv_path = os.path.join(OUTPUT_DIR, "industry_dispersion.csv")
        disp.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    # 4. Concentration & turbulence
    print("\n[4/4] Computing CR4, n_firms, turbulence ...")
    conc = compute_concentration(df)
    results["concentration"] = conc
    print(f"  → {conc.height} industry-pairs")

    if save:
        csv_path = os.path.join(OUTPUT_DIR, "industry_concentration.csv")
        conc.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}/")
    return results


if __name__ == "__main__":
    run_all()
