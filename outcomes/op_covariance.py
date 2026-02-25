"""
Olley-Pakes covariance decomposition of allocative efficiency.

Labor productivity: φ_it = log(sales_it / employees_it)

OP covariance per industry-year:
  OPcov_jt = Σ_i (s_it − s̄_t)(φ_it − φ̄_t)

Positive OPcov = more productive firms have larger market shares.
Long-difference: LD_OPcov_2012_20YY = OPcov_{j,20YY} − OPcov_{j,2012}
"""

import polars as pl
from config import (
    FIRM_COL, YEAR_COL, COUNTRY_COL, INDUSTRY_COL,
    SALES_COL, EMPLOYEES_COL, TREATMENT_YEAR, HORIZONS,
)


def _compute_opcov_by_year(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute OP covariance per industry-year."""
    return (
        df
        .filter(
            pl.col(EMPLOYEES_COL).is_not_null()
            & (pl.col(EMPLOYEES_COL) > 0)
            & pl.col(SALES_COL).is_not_null()
            & (pl.col(SALES_COL) > 0)
        )
        .with_columns([
            (pl.col(SALES_COL) / pl.col(SALES_COL).sum().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("share"),
            (pl.col(SALES_COL) / pl.col(EMPLOYEES_COL)).log().alias("labor_prod"),
        ])
        # Demeaned share and productivity within industry-year
        .with_columns([
            (pl.col("share") - pl.col("share").mean().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("d_share"),
            (pl.col("labor_prod") - pl.col("labor_prod").mean().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("d_prod"),
        ])
        .with_columns(
            (pl.col("d_share") * pl.col("d_prod")).alias("cov_term")
        )
        .group_by([COUNTRY_COL, INDUSTRY_COL, YEAR_COL])
        .agg(
            pl.col("cov_term").sum().alias("opcov")
        )
    )


def compute_op_covariance(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute long-difference OP covariance.

    Returns industry-level DataFrame with columns:
      fic_code, borrower_sic,
      LD_OPcov_2012_2013, LD_OPcov_2012_2014, LD_OPcov_2012_2015
    """
    opcov = _compute_opcov_by_year(df).collect()

    base = opcov.filter(pl.col(YEAR_COL) == TREATMENT_YEAR).select([
        COUNTRY_COL, INDUSTRY_COL,
        pl.col("opcov").alias("opcov_base"),
    ])

    results = []
    for h in HORIZONS:
        horizon = opcov.filter(pl.col(YEAR_COL) == h)
        ld = (
            base.join(horizon, on=[COUNTRY_COL, INDUSTRY_COL], how="inner")
            .with_columns(
                (pl.col("opcov") - pl.col("opcov_base")).alias(f"LD_OPcov_2012_{h}")
            )
            .select([COUNTRY_COL, INDUSTRY_COL, f"LD_OPcov_2012_{h}"])
        )
        results.append(ld)

    if not results:
        return pl.DataFrame()

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[COUNTRY_COL, INDUSTRY_COL], how="left")

    out = out.rename({
        COUNTRY_COL: "fic_code",
        INDUSTRY_COL: "borrower_sic",
    })
    return out
