"""Central configuration for Colab outcome-variable pipeline."""

# Google Drive paths (mounted at /content/drive on Colab)
DRIVE_ROOT = "/content/drive/MyDrive/Project_Credit Supply and Market Share Reallocation/zhixi"
MASTER_DIR = f"{DRIVE_ROOT}/master_with_sic_links"   # cleaned firm-year parquets
OUTPUT_DIR = f"{DRIVE_ROOT}/new_outcomes"              # where CSVs land

# Time structure
YEARS = range(2009, 2016)        # 2009–2015
PRESHOCK_YEARS = [2009, 2010]    # baseline for pre-shock averages
TREATMENT_YEAR = 2012            # EBA treatment year
HORIZONS = [2013, 2014, 2015]    # post-treatment windows

# Column names in the cleaned Orbis parquets
INDUSTRY_COL = "sic"             # SIC-2 level
COUNTRY_COL = "_country_"        # derived from bvd_id[:2]
FIRM_COL = "bvd_id"
YEAR_COL = "year"
SALES_COL = "sales"
COGS_COL = "costs_of_goods_sold"
EMPLOYEES_COL = "number_of_employees"
ASSETS_COL = "total_assets"
STAFF_COL = "costs_of_employees"
