# 공포분할매수 백테스트 초안

- 기간: 2011-02-28 ~ 2026-05-31
- 월말 종가로 신호를 계산하고 다음 달 수익률에 적용
- 기본 구조: 한국 20%, 미국 20%, 한국 추가 25%, 미국 추가 25%, 잔여 방어 10% 포함
- 한국 공포 신호는 코스피200 기준, 미국 집행은 TIGER 미국나스닥100 기준
- 방어자산 cash는 현금 0% 수익률, basket은 GLD/TLT/현금 1:1:1 월간 리밸런싱 프록시

## CAGR 상위

| name | kr_exec | rule | defense | final_nav | cagr | mdd | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline 50KR/50US StockOnly | baseline | baseline | baseline | 2801.79 | 24.48% | -31.18% | 1.14 |
| SemiSplit/ma_reclaim/basket | KR_Semi_50_50 | ma_reclaim | basket | 892.05 | 15.19% | -18.93% | 1.15 |
| Semi50/or/basket | KR_Semi_50_50 | or | basket | 743.87 | 13.83% | -10.02% | 1.33 |
| Semi50/annual5_live/basket | KR_Semi_50_50 | annual5_live | basket | 726.37 | 13.65% | -10.02% | 1.32 |
| Semi50/ma_down/basket | KR_Semi_50_50 | ma_down | basket | 721.88 | 13.61% | -10.02% | 1.31 |
| Semi50/ma_reclaim/basket | KR_Semi_50_50 | ma_reclaim | basket | 700.68 | 13.38% | -15.50% | 1.26 |
| Baseline 20KR/20US/60DefenseBasket | baseline | baseline | baseline | 665.84 | 13.12% | -17.62% | 1.31 |
| Semi50/drawdown/basket | KR_Semi_50_50 | drawdown | basket | 668.73 | 13.04% | -9.68% | 1.29 |
| Attack/or/basket | KR_Attack_25_37_37 | or | basket | 657.77 | 12.92% | -9.47% | 1.31 |
| Semi50/and/basket | KR_Semi_50_50 | and | basket | 648.83 | 12.81% | -9.68% | 1.28 |
| Attack/annual5_live/basket | KR_Attack_25_37_37 | annual5_live | basket | 642.23 | 12.74% | -9.47% | 1.30 |
| Attack/ma_down/basket | KR_Attack_25_37_37 | ma_down | basket | 638.28 | 12.69% | -9.47% | 1.30 |

## MDD 방어 상위

| name | kr_exec | rule | defense | final_nav | cagr | mdd | sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KOSPI200/drawdown/cash | KOSPI200 | drawdown | cash | 227.69 | 5.54% | -4.31% | 1.09 |
| KOSPI200/and/cash | KOSPI200 | and | cash | 217.79 | 5.24% | -4.31% | 1.04 |
| Attack/ma_down/cash | KR_Attack_25_37_37 | ma_down | cash | 358.07 | 8.72% | -4.77% | 1.20 |
| Attack/or/cash | KR_Attack_25_37_37 | or | cash | 374.10 | 9.04% | -4.77% | 1.22 |
| Attack/drawdown/cash | KR_Attack_25_37_37 | drawdown | cash | 316.04 | 7.84% | -4.77% | 1.20 |
| Attack/and/cash | KR_Attack_25_37_37 | and | cash | 302.34 | 7.52% | -4.77% | 1.16 |
| Semi50/drawdown/cash | KR_Semi_50_50 | drawdown | cash | 351.86 | 8.60% | -4.92% | 1.20 |
| Semi50/and/cash | KR_Semi_50_50 | and | cash | 336.63 | 8.28% | -4.92% | 1.17 |
| SemiSplit/drawdown/cash | KR_Semi_50_50 | drawdown | cash | 331.10 | 8.17% | -4.92% | 1.15 |
| SemiSplit/annual5_live/cash | KR_Semi_50_50 | annual5_live | cash | 321.91 | 7.97% | -4.92% | 1.13 |
| SemiSplit/ma_down/cash | KR_Semi_50_50 | ma_down | cash | 314.24 | 7.80% | -4.92% | 1.12 |
| SemiSplit/and/cash | KR_Semi_50_50 | and | cash | 301.58 | 7.51% | -4.92% | 1.09 |

## 한국 매수 대상 평균

| kr_exec | final_nav | cagr | mdd | sharpe |
| --- | --- | --- | --- | --- |
| baseline | 1733.81 | 18.80% | -24.40% | 1.23 |
| KR_Semi_50_50 | 518.73 | 10.92% | -8.85% | 1.20 |
| KR_Attack_25_37_37 | 480.33 | 10.40% | -7.92% | 1.23 |
| KOSPI200 | 331.55 | 7.75% | -9.04% | 1.10 |

## 공포 기준 평균

| rule | final_nav | cagr | mdd | sharpe |
| --- | --- | --- | --- | --- |
| baseline | 1733.81 | 18.80% | -24.40% | 1.23 |
| or | 483.54 | 10.39% | -7.73% | 1.23 |
| ma_reclaim | 499.91 | 10.38% | -12.55% | 1.08 |
| annual5_live | 467.36 | 10.15% | -8.17% | 1.21 |
| ma_down | 461.49 | 10.04% | -7.97% | 1.20 |
| drawdown | 441.06 | 9.69% | -7.73% | 1.20 |
| and | 420.64 | 9.34% | -7.84% | 1.16 |

## 해석 메모

- 이 결과는 룰 후보를 줄이기 위한 초안이다. 실제 해남A 앱의 채권/금 프록시 및 세금/비용과는 다를 수 있다.
- `annual5_live`는 미국 신호만 진행 중 연봉 5년선 비슷한 기준을 쓰고, 한국은 월봉선 기준을 쓴다.
- 다음 단계에서는 상위 후보를 기존 `app.py`의 해남A 데이터 로더와 연결해 동일 프록시로 재검증하는 것이 좋다.
