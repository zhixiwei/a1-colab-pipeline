# a1-colab-pipeline

Constructs new outcome variables (markup decomposition, OP covariance, MRPK/MRPL dispersion, concentration) from cleaned Orbis firm-level data on Google Drive.

## Setup

1. Open `run_on_colab.ipynb` on Google Colab
2. Mount Google Drive (cell 1)
3. Clone this repo (cell 2 — update the GitHub URL first)
4. Install deps and run (cells 3–4)

## Output

CSVs saved to `Drive/Project_Credit Supply and Market Share Reallocation/zhixi/new_outcomes/`:

| File | Contents |
|------|----------|
| `industry_markup_decomp.csv` | De Loecker within/between/cross decomposition |
| `industry_op_covariance.csv` | Olley-Pakes allocative efficiency |
| `industry_dispersion.csv` | sigma(MRPL), sigma(MRPK), sigma(markup) |
| `industry_concentration.csv` | CR4, n_firms, turbulence |
| `firm_panel_markup.parquet` | Firm-level markup and shares (for diagnostics) |

All industry-level files have columns: `fic_code`, `borrower_sic`, `LD_*_2012_2013`, `LD_*_2012_2014`, `LD_*_2012_2015`.

## Local workflow

After Colab run, download CSVs to `Dropbox/A1_project/produced data/new_outcomes/`, then:
1. Run `get data/09_NewOutcome_Supervision_Prep.ipynb` to merge with supervision index + IV instruments
2. Run `Gropp paper data/First Stage/IV_Supervision_Markup.R` for IV estimation

## Dependencies

- `polars` (lazy scan for large parquets)
- `pyarrow` (parquet I/O)
