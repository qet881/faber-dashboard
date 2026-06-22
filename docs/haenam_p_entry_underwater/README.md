# HaenamP Entry Timing and Underwater Test

## Setup

- Run date: 2026-06-17
- Engine: exact HaenamP strategy data and simulator from `app.py`
- Entry dates: monthly rebalance dates from 2000-01-03 to 2026-06-17
- Entry method: normalize the already-simulated HaenamP NAV to 100 at each entry date.
- High-entry definitions:
  - `strategy_ath_month_ends`: the already-running HaenamP NAV was at an all-time high on that month-end.
  - `stock_basket_ath_month_ends`: the 50/50 KOSPI200/Nasdaq100 basket was at an all-time high on that month-end.

## Entry Timing Summary

| Group | Count | Median recovery | P90 recovery | Max recovery | Worst 1Y return | Median 1Y return | Worst 3Y return | Worst full MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All month-ends | 305 | 3d | 22d | 1y 3m | -5.15% | 8.18% | 8.70% | -10.02% |
| HaenamP ATH entries | 136 | 3d | 1m | 1y 3m | -5.15% | 7.84% | 8.70% | -10.02% |
| Stock-basket ATH entries | 70 | 3d | 1m | 1y 3m | -5.15% | 8.42% | 10.22% | -10.02% |

## Underwater Summary

- Underwater ratio: 88.6%
- Current drawdown: -0.07%
- Ulcer Index: 2.43
- Recovered underwater periods: 345
- Median recovered period: 6d
- Average recovered period: 26d
- P90 recovered period: 1m
- Longest period: 2012-10-04 -> 2014-06-20 (1y 8m, recovered=True)
- Deepest period: 2008-10-08 -> 2008-11-03 -> 2008-11-25 (-10.02%, recovered=True)

## Worst 1Y Entry Starts

| group | start | 1Y return | 1Y MDD | recover |
| --- | --- | --- | --- | --- |
| strategy_ath_month_ends | 2007-12-28 | 15.46% | -10.02% | 5d |
| all_month_ends | 2007-11-30 | 16.34% | -10.02% | 3d |
| all_month_ends | 2007-12-28 | 15.46% | -10.02% | 5d |
| strategy_ath_month_ends | 2008-03-31 | 14.01% | -10.02% | 7d |
| all_month_ends | 2008-03-31 | 14.01% | -10.02% | 7d |
| all_month_ends | 2008-07-31 | 15.53% | -10.02% | 1d |
| all_month_ends | 2008-09-30 | 11.94% | -10.02% | 1d |
| strategy_ath_month_ends | 2008-09-30 | 11.94% | -10.02% | 1d |
| strategy_ath_month_ends | 2008-05-30 | 9.88% | -10.02% | 7d |
| all_month_ends | 2008-06-30 | 11.48% | -10.02% | 1d |

## Interpretation

- Entering at a HaenamP all-time high was not catastrophic in this sample. The worst full-period drawdown after those entries still stayed near the historical HaenamP MDD range.
- Stock-market all-time-high entries were also manageable because HaenamP may still hold bonds, gold, or cash depending on momentum.
- The uncomfortable part is not the depth alone; it is time underwater. The strategy spends a meaningful share of days below its own prior high, so investors should expect long flat or recovery stretches even when MDD is modest.
- This supports using HaenamP as a main candidate, but it does not support performance chasing after a strong month. The rule is strongest when followed mechanically at scheduled rebalance dates.
