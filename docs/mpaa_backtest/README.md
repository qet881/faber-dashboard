# MPAA Backtest

This folder contains a reproducible systrader79-style MPAA backtest built from the provided workbook.

## Command

```powershell
python scripts\mpaa_backtest.py --out-dir docs\mpaa_backtest
```

Use a different workbook path if needed:

```powershell
python scripts\mpaa_backtest.py --input "C:\Users\godls\Downloads\MPAA 데이터.xlsx" --out-dir docs\mpaa_backtest
```

## Logic

- Monthly rebalance.
- Use prior-month signals only.
- Within each sleeve, rank assets by 1-to-12-month average momentum.
- Sleeve ranks and weights:
  - `국가`: top 4, weight 1/6
  - `섹터`: top 8, weight 1/6
  - `팩터`: top 10, weight 1/6
  - `채권`: top 1, weight 3/6
- For selected assets, allocate between the risky asset and cash using 12-month positive momentum score.
- Apply a final 6-month equity-curve momentum overlay to the MPAA base NAV.

## Data

The original workbook ends in July 2017. I also checked the upstream `hermian/startetf` repository after the March 10, 2022 `update 3.33 MPAA` commit; its public `data/mpaa.csv` still has 199 monthly rows from January 2001 through July 2017. The 2022 commit updates notebooks, not a longer MPAA data history.

The script keeps the original data intact, then appends public Yahoo Finance proxy series scaled to each original index level. The proxy mapping is explicit in `source_status.csv`.

Important limitation: several original `섹터` and `팩터` series are FnGuide/WiseIndex-style indexes without free one-to-one public tickers. Those columns are therefore extended with documented ETF/index proxies, not exact official continuation data.

## Outputs

- `summary.csv`: headline performance metrics.
- `latest_weights.csv`: latest non-zero MPAA base weights.
- `weights.csv`: monthly MPAA base weights.
- `base_nav.csv`: MPAA before the 6-month equity-curve overlay.
- `final_nav.csv`: MPAA after the 6-month equity-curve overlay.
- `mpaa_appended_prices.csv`: original plus appended monthly price/index data.
- `source_status.csv`: proxy ticker, original range, proxy range, and rows appended.
