"""
Within-industry dispersion of revenue products of inputs (MRPK, MRPL)
and markup dispersion.

  sigma_MRPL_jt = std(log(sales_it / employees_it))  within j,t
  sigma_MRPK_jt = std(log(sales_it / assets_it))     within j,t
  sigma_markup_jt = std(log(sales_it / cogs_it))      within j,t

Higher dispersion → more misallocation (Hsieh & Klenow 2009).
Long-difference: LD_sigma_*_2012_20YY
"""

import polars as pl
from config import (
    YEAR_COL, COUNTRY_COL, INDUSTRY_COL,
    SALES_COL, EMPLOYEES_COL, ASSETS_COL, COGS_COL,
    TREATMENT_YEAR, HORIZONS,
)


def _compute_dispersion_by_year(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute within-industry std of log ratios per industry-year."""
    return (
        df
        .filter(
            pl.col(SALES_COL).is_not_null() & (pl.col(SALES_COL) > 0)
        )
        .with_columns([
            pl.when(
                pl.col(EMPLOYEES_COL).is_not_null() & (pl.col(EMPLOYEES_COL) > 0)
            ).then(
                (pl.col(SALES_COL) / pl.col(EMPLOYEES_COL)).log()
            ).alias("log_mrpl"),

            pl.when(
                pl.col(ASSETS_COL).is_not_null() & (pl.col(ASSETS_COL) > 0)
            ).then(
                (pl.col(SALES_COL) / pl.col(ASSETS_COL)).log()
            ).alias("log_mrpk"),

            pl.when(
                pl.col(COGS_COL).is_not_null() & (pl.col(COGS_COL) > 0)
            ).then(
                (pl.col(SALES_COL) / pl.col(COGS_COL)).log()
            ).alias("log_markup"),
        ])
        .group_by([COUNTRY_COL, INDUSTRY_COL, YEAR_COL])
        .agg([
            pl.col("log_mrpl").std().alias("sigma_mrpl"),
            pl.col("log_mrpk").std().alias("sigma_mrpk"),
            pl.col("log_markup").std().alias("sigma_markup"),
        ])
    )


def compute_dispersion(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute long-difference dispersion measures.

    Returns industry-level DataFrame with columns:
      fic_code, borrower_sic,
      LD_sigma_MRPL_2012_2013, ..., LD_sigma_MRPK_2012_2013, ...,
      LD_sigma_markup_2012_2013, ...
    """
    disp = _compute_dispersion_by_year(df).collect()

    base = (
        disp.filter(pl.col(YEAR_COL) == TREATMENT_YEAR)
        .select([
            COUNTRY_COL, INDUSTRY_COL,
            pl.col("sigma_mrpl").alias("base_mrpl"),
            pl.col("sigma_mrpk").alias("base_mrpk"),
            pl.col("sigma_markup").alias("base_markup"),
        ])
    )

    results = []
    for h in HORIZONS:
        horizon = disp.filter(pl.col(YEAR_COL) == h)
        ld = (
            base.join(horizon, on=[COUNTRY_COL, INDUSTRY_COL], how="inner")
            .with_columns([
                (pl.col("sigma_mrpl") - pl.col("base_mrpl")).alias(f"LD_sigma_MRPL_2012_{h}"),
                (pl.col("sigma_mrpk") - pl.col("base_mrpk")).alias(f"LD_sigma_MRPK_2012_{h}"),
                (pl.col("sigma_markup") - pl.col("base_markup")).alias(f"LD_sigma_markup_2012_{h}"),
            ])
            .select([
                COUNTRY_COL, INDUSTRY_COL,
                f"LD_sigma_MRPL_2012_{h}",
                f"LD_sigma_MRPK_2012_{h}",
                f"LD_sigma_markup_2012_{h}",
            ])
        )
        results.append(ld)

    if not results:
        return pl.DataFrame()

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[COUNTRY_COL, INDUSTRY_COL], how="outer")

    out = out.rename({
        COUNTRY_COL: "fic_code",
        INDUSTRY_COL: "borrower_sic",
    })
    return out
