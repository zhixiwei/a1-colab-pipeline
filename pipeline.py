"""
Orchestrator: load cleaned Orbis parquets → compute outcome variables → save CSVs.

Usage on Colab:
    from pipeline import run_all
    results = run_all()              # full run
    results = run_all(test=True)     # smoke test with one parquet
"""

import os
import glob
import polars as pl
from config import (
    MASTER_DIR, OUTPUT_DIR, FIRM_COL, YEAR_COL,
    COUNTRY_COL, INDUSTRY_COL, SALES_COL,
)
from outcomes.markup import compute_firm_markup, compute_markup_decomposition, compute_markup_decomposition_block2
from outcomes.op_covariance import compute_op_covariance, compute_op_covariance_block2
from outcomes.dispersion import compute_dispersion, compute_dispersion_block2
from outcomes.concentration import compute_concentration, compute_concentration_block2


def load_data(test: bool = False) -> pl.LazyFrame:
    """Load parquets from MASTER_DIR, derive country and SIC2.

    If test=True, loads only the first parquet file (smoke test).
    """
    pattern = os.path.join(MASTER_DIR, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {MASTER_DIR}/")

    if test:
        files = files[:1]
        print(f"SMOKE TEST: loading only {os.path.basename(files[0])}")

    print(f"Loading {len(files)} parquet file(s) from {MASTER_DIR}/ ...")
    df = pl.scan_parquet(files)

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


def run_all(save: bool = True, test: bool = False) -> dict:
    """
    Run all outcome modules and save results.

    Args:
        save: Write output CSVs/parquets to OUTPUT_DIR.
        test: If True, load only one parquet file (smoke test).

    Returns dict of outcome name → DataFrame for inspection.
    """
    df = load_data(test=test)

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


def run_block2_mediators(save: bool = True, test: bool = False) -> dict:
    """
    Compute Block 2 industry mediators: long-differences from 2010 base.

    Same outcome modules, but differenced from PRESHOCK_BASE (2010)
    to BLOCK2_HORIZONS (2013, 2014).

    Saves to {OUTPUT_DIR}/block2/ subfolder.
    """
    df = load_data(test=test)

    block2_dir = os.path.join(OUTPUT_DIR, "block2")
    os.makedirs(block2_dir, exist_ok=True)
    results = {}

    # 1. Markup decomposition (needs firm-level markup first)
    print("\n[Block2 1/4] Computing markup decomposition (2010 base) ...")
    df_markup = compute_firm_markup(df)
    markup_b2 = compute_markup_decomposition_block2(df_markup)
    results["markup_decomp"] = markup_b2
    print(f"  → {markup_b2.height} industry-pairs")

    if save:
        csv_path = os.path.join(block2_dir, "industry_markup_decomp_2010base.csv")
        markup_b2.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    # 2. OP covariance
    print("\n[Block2 2/4] Computing OP covariance (2010 base) ...")
    opcov_b2 = compute_op_covariance_block2(df)
    results["op_covariance"] = opcov_b2
    print(f"  → {opcov_b2.height} industry-pairs")

    if save:
        csv_path = os.path.join(block2_dir, "industry_op_covariance_2010base.csv")
        opcov_b2.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    # 3. Dispersion
    print("\n[Block2 3/4] Computing dispersion (2010 base) ...")
    disp_b2 = compute_dispersion_block2(df)
    results["dispersion"] = disp_b2
    print(f"  → {disp_b2.height} industry-pairs")

    if save:
        csv_path = os.path.join(block2_dir, "industry_dispersion_2010base.csv")
        disp_b2.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    # 4. Concentration & turbulence
    print("\n[Block2 4/4] Computing CR4, n_firms, turbulence (2010 base) ...")
    conc_b2 = compute_concentration_block2(df)
    results["concentration"] = conc_b2
    print(f"  → {conc_b2.height} industry-pairs")

    if save:
        csv_path = os.path.join(block2_dir, "industry_concentration_2010base.csv")
        conc_b2.write_csv(csv_path)
        print(f"  Saved: {csv_path}")

    print(f"\nBlock 2 mediators done. Saved to {block2_dir}/")
    return results


if __name__ == "__main__":
    run_all()
