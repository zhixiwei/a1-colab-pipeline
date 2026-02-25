"""
Industry concentration and turbulence measures.

  CR4_jt      = sum of top-4 firm sales shares within industry j at time t
  n_firms_jt  = count of active firms (positive sales)
  turbulence_jt = Σ_i |s_it − s_{i,t-1}|  (total absolute reallocation)

Long-differences relative to 2012.
"""

import polars as pl
from config import (
    FIRM_COL, YEAR_COL, COUNTRY_COL, INDUSTRY_COL,
    SALES_COL, TREATMENT_YEAR, HORIZONS,
)


def _compute_concentration_by_year(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute CR4, n_firms per industry-year."""
    base = (
        df
        .filter(pl.col(SALES_COL).is_not_null() & (pl.col(SALES_COL) > 0))
        .with_columns(
            (pl.col(SALES_COL) / pl.col(SALES_COL).sum().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("share")
        )
    )

    # CR4: rank shares within industry-year, sum top 4
    cr4 = (
        base
        .with_columns(
            pl.col("share")
            .rank(method="ordinal", descending=True)
            .over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL])
            .alias("rank")
        )
        .filter(pl.col("rank") <= 4)
        .group_by([COUNTRY_COL, INDUSTRY_COL, YEAR_COL])
        .agg(pl.col("share").sum().alias("cr4"))
    )

    # n_firms
    n_firms = (
        base
        .group_by([COUNTRY_COL, INDUSTRY_COL, YEAR_COL])
        .agg(pl.col(FIRM_COL).n_unique().alias("n_firms"))
    )

    return cr4.join(n_firms, on=[COUNTRY_COL, INDUSTRY_COL, YEAR_COL], how="left")


def _compute_turbulence(df: pl.LazyFrame) -> pl.DataFrame:
    """Compute turbulence = sum of |Δshare| between consecutive years."""
    collected = (
        df
        .filter(pl.col(SALES_COL).is_not_null() & (pl.col(SALES_COL) > 0))
        .with_columns(
            (pl.col(SALES_COL) / pl.col(SALES_COL).sum().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("share")
        )
        .collect()
    )

    results = []
    for h in HORIZONS:
        base = collected.filter(pl.col(YEAR_COL) == TREATMENT_YEAR)
        horizon = collected.filter(pl.col(YEAR_COL) == h)

        merged = base.join(
            horizon,
            on=[FIRM_COL, COUNTRY_COL, INDUSTRY_COL],
            suffix="_h",
        )

        turb = (
            merged
            .with_columns(
                (pl.col("share_h") - pl.col("share")).abs().alias("abs_d_share")
            )
            .group_by([COUNTRY_COL, INDUSTRY_COL])
            .agg(pl.col("abs_d_share").sum().alias(f"LD_turbulence_2012_{h}"))
        )
        results.append(turb)

    if not results:
        return pl.DataFrame()

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[COUNTRY_COL, INDUSTRY_COL], how="left")
    return out


def compute_concentration(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute long-difference CR4, n_firms, and turbulence.

    Returns industry-level DataFrame with columns:
      fic_code, borrower_sic,
      LD_CR4_2012_2013, ..., LD_n_firms_2012_2013, ...,
      LD_turbulence_2012_2013, ...
    """
    conc = _compute_concentration_by_year(df).collect()

    base = (
        conc.filter(pl.col(YEAR_COL) == TREATMENT_YEAR)
        .select([
            COUNTRY_COL, INDUSTRY_COL,
            pl.col("cr4").alias("base_cr4"),
            pl.col("n_firms").alias("base_n_firms"),
        ])
    )

    results = []
    for h in HORIZONS:
        horizon = conc.filter(pl.col(YEAR_COL) == h)
        ld = (
            base.join(horizon, on=[COUNTRY_COL, INDUSTRY_COL], how="inner")
            .with_columns([
                (pl.col("cr4") - pl.col("base_cr4")).alias(f"LD_CR4_2012_{h}"),
                (pl.col("n_firms") - pl.col("base_n_firms")).alias(f"LD_n_firms_2012_{h}"),
            ])
            .select([
                COUNTRY_COL, INDUSTRY_COL,
                f"LD_CR4_2012_{h}",
                f"LD_n_firms_2012_{h}",
            ])
        )
        results.append(ld)

    if not results:
        return pl.DataFrame()

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[COUNTRY_COL, INDUSTRY_COL], how="left")

    # Add turbulence
    turb = _compute_turbulence(df)
    if turb.height > 0:
        out = out.join(turb, on=[COUNTRY_COL, INDUSTRY_COL], how="left")

    out = out.rename({
        COUNTRY_COL: "fic_code",
        INDUSTRY_COL: "borrower_sic",
    })
    return out
