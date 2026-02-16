# Data

This project uses publicly available data from **Kenneth French's Data Library**.

## Required Datasets

Download the following two datasets (monthly frequency) from:  
👉 https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

| Dataset | Expected Filename |
|---|---|
| Fama/French 5 Factors (2x3) | `F-F_Research_Data_5_Factors_2x3.CSV` |
| 49 Industry Portfolios | `49_Industry_Portfolios.CSV` |

Place both files in this `data/` directory before running the notebook.

## Notes

- Files are in CSV format with a header section that should be skipped (`skiprows=3`)
- Data is in **monthly frequency**, expressed as **percentages** (the code converts to decimals)
- The notebook expects monthly returns from **July 1963 onward** — both files cover this range
- The High-Tech sector return is constructed by averaging four industries: `LabEq`, `Chips`, `Softw`, `Telcm`

## Data is NOT included in this repository

Per Kenneth French's data library terms, the raw data files are not redistributed here. Please download them directly from the source link above.
