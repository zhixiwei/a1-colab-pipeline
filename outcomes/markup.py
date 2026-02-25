"""
Markup proxy and De Loecker (2018) within/between/cross decomposition.

Firm-level markup: μ_it = sales_it / cogs_it
Industry = country × SIC2

De Loecker decomposition of aggregate markup change:
  Δμ̄ = Within + Between + Cross
  Within  = Σ_i s̄_i (μ_it − μ_{i,t-1})
  Between = Σ_i μ̄_i (s_it − s_{i,t-1})
  Cross   = Σ_i (μ_it − μ_{i,t-1})(s_it − s_{i,t-1})
where s_it = firm sales share within its industry-year.
"""

import polars as pl
from config import (
    FIRM_COL, YEAR_COL, COUNTRY_COL, INDUSTRY_COL,
    SALES_COL, COGS_COL, TREATMENT_YEAR, HORIZONS,
)


def compute_firm_markup(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add firm-level markup and within-industry sales share."""
    return (
        df
        .filter(
            pl.col(COGS_COL).is_not_null()
            & (pl.col(COGS_COL) > 0)
            & pl.col(SALES_COL).is_not_null()
            & (pl.col(SALES_COL) > 0)
        )
        .with_columns(
            (pl.col(SALES_COL) / pl.col(COGS_COL)).alias("markup")
        )
        # Outlier trim: keep markup in [0.5, 10]
        .filter((pl.col("markup") >= 0.5) & (pl.col("markup") <= 10))
        # Within-industry sales share
        .with_columns(
            (pl.col(SALES_COL) / pl.col(SALES_COL).sum().over([COUNTRY_COL, INDUSTRY_COL, YEAR_COL]))
            .alias("share")
        )
    )


def _decompose_pair(df_t0: pl.DataFrame, df_t1: pl.DataFrame) -> dict:
    """
    Decompose markup change between two cross-sections.
    Returns dict with within, between, cross components per industry.
    """
    # Inner join: firms present in both periods
    merged = df_t0.join(
        df_t1,
        on=[FIRM_COL, COUNTRY_COL, INDUSTRY_COL],
        suffix="_1",
    )

    if merged.height == 0:
        return None

    # Average share and average markup across the two periods
    merged = merged.with_columns([
        ((pl.col("share") + pl.col("share_1")) / 2).alias("s_bar"),
        ((pl.col("markup") + pl.col("markup_1")) / 2).alias("mu_bar"),
        (pl.col("markup_1") - pl.col("markup")).alias("d_mu"),
        (pl.col("share_1") - pl.col("share")).alias("d_s"),
    ])

    # Aggregate by industry (country × SIC2)
    decomp = (
        merged
        .group_by([COUNTRY_COL, INDUSTRY_COL])
        .agg([
            (pl.col("s_bar") * pl.col("d_mu")).sum().alias("within"),
            (pl.col("mu_bar") * pl.col("d_s")).sum().alias("between"),
            (pl.col("d_mu") * pl.col("d_s")).sum().alias("cross"),
        ])
    )
    return decomp


def compute_markup_decomposition(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute long-difference De Loecker decomposition.

    Returns industry-level DataFrame with columns:
      fic_code, borrower_sic,
      LD_Within_2012_2013, LD_Within_2012_2014, LD_Within_2012_2015,
      LD_Between_2012_2013, ..., LD_Cross_2012_2013, ...
    """
    collected = df.collect()

    # Base year cross-section
    base = collected.filter(pl.col(YEAR_COL) == TREATMENT_YEAR)

    results = []
    for h in HORIZONS:
        horizon_df = collected.filter(pl.col(YEAR_COL) == h)
        decomp = _decompose_pair(base, horizon_df)
        if decomp is None:
            continue
        decomp = decomp.rename({
            "within":  f"LD_Within_2012_{h}",
            "between": f"LD_Between_2012_{h}",
            "cross":   f"LD_Cross_2012_{h}",
        })
        results.append(decomp)

    if not results:
        return pl.DataFrame()

    # Join all horizons
    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[COUNTRY_COL, INDUSTRY_COL], how="outer")

    # Rename to match IV pipeline convention
    out = out.rename({
        COUNTRY_COL: "fic_code",
        INDUSTRY_COL: "borrower_sic",
    })
    return out
