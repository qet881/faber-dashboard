# Macro Cycle Evidence Pack

- Generated: 2026-06-19 14:04:17
- Requested as-of date: 2026-06-19
- Role: evidence package for GPT-5.5/Codex judgment, not a deterministic scorecard.

## Judgment Instruction

Use the evidence below to judge the US macro cycle directly. Do not score mechanically. Classify the economy as Recovery, Growth, Slowdown, or Recession. If evidence is transitional, name the transition path and explain why one side dominates. Mention conflicting evidence.

Portfolio output is allowed, but it must be framed as a macro satellite/overlay guide. It must not override Haenam P's trend-following/rebalancing rules unless the user explicitly asks for a separate strategy.

## Leading Indicators

| Asset | Ticker | Status | Latest date | Latest | 3M | 6M | 12M | 5Y | >200D MA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | ^GSPC | ok | 2026-06-18 | 7500.58 | 13.22% | 10.71% | 25.41% | 80.02% | True |
| NASDAQ Composite | ^IXIC | ok | 2026-06-18 | 26517.93 | 19.71% | 15.26% | 35.67% | 89.00% | True |
| Dow Jones Industrial Average | ^DJI | ok | 2026-06-18 | 51564.70 | 11.55% | 7.53% | 22.27% | 54.90% | True |
| Russell 2000 | ^RUT | ok | 2026-06-18 | 2979.77 | 20.22% | 18.82% | 41.02% | 33.16% | True |

## Leading Macro

| Indicator | FRED | Status | Latest date | Latest | 3M delta | 6M delta | 12M delta | 6M dir | Reading note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISM Manufacturing PMI | TradingEconomics:business-confidence | ok | May 2026 | 54.00 | n/a | n/a | n/a | rising | Trading Economics latest/previous snapshot fallback because the public FRED PMI series was unavailable. Source: https://tradingeconomics.com/united-states/business-confidence |
| University of Michigan Consumer Sentiment | UMCSENT | ok | 2026-04-01 | 49.80 | -6.60 | -3.80 | -2.40 | falling | Consumer expectations and sentiment, treated as leading pressure. |
| 10Y minus 2Y Treasury spread | T10Y2Y | ok | 2026-06-18 | 0.27 | -0.23 | -0.39 | -0.17 | falling | Curve steepening after inversion can mark late slowdown/recession transition. |

## Coincident Indicators

| Indicator | FRED | Status | Latest date | Latest | 3M delta | 6M delta | 12M delta | 6M dir | Reading note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Industrial Production Index | INDPRO | ok | 2026-05-01 | 102.65 | 0.70 | 1.61 | 1.68 | rising | Use 12-month moving average and YoY direction, not noisy MoM alone. |
| Advance Retail Sales | RSAFS | ok | 2026-05-01 | 763705.00 | 22427.00 | 28987.00 | 49137.00 | rising | Use YoY/12-month trend as annual retail-sales proxy. |
| Capacity Utilization | TCU | ok | 2026-05-01 | 76.17 | 0.32 | 0.78 | 0.28 | rising | Coincident production-cycle pressure. |

## Lagging Indicators

| Indicator | FRED | Status | Latest date | Latest | 3M delta | 6M delta | 12M delta | 6M dir | Reading note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real GDP | GDPC1 | ok | 2026-01-01 | 24152.66 | 96.91 | 125.82 | 604.45 | rising | Quarterly GDP, included because annual GDP is too slow for current-cycle reads. |
| Real GDP annual growth | A191RL1A225NBEA | ok | 2025-01-01 | 2.10 | -0.70 | -0.70 | -0.70 | falling | Annual GDP growth, slow but useful as a final confirmation. |
| Unemployment Rate | UNRATE | ok | 2026-05-01 | 4.30 | -0.10 | -0.20 | 0.00 | falling | Low and flat can be late-cycle; rising from lows is slowdown/recession evidence. |
| Average Hourly Earnings | CES0500000003 | ok | 2026-05-01 | 37.53 | 0.26 | 0.53 | 1.25 | rising | Lagging wage pressure. |
| Average Weekly Hours, Manufacturing | AWHMAN | ok | 2026-05-01 | 41.60 | 0.00 | 0.30 | 0.60 | rising | Labor-cycle deterioration often appears before layoffs fully show. |

## Market Rotation

| Rotation | Ticker | Status | Latest date | 3M | 6M | 12M | 6M dir | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM vs EM | VEA/EEM | ok | 2026-06-18 | -8.37% | -11.44% | -13.37% | falling | DM stronger supports recovery/slowdown/recession; EM stronger supports growth. |
| US vs Non-US | VTI/VEA | ok | 2026-06-18 | 0.50% | -5.71% | -5.14% | falling | US stronger supports recovery/slowdown/recession; Non-US stronger supports growth. |
| NASDAQ vs S&P 500 | QQQ/SPY | ok | 2026-06-18 | 9.83% | 9.52% | 10.99% | rising | NASDAQ stronger supports recovery/growth; weakness supports slowdown. |
| Dow vs S&P 500 | DIA/SPY | ok | 2026-06-18 | -1.32% | -2.68% | -2.07% | falling | Dow stronger supports slowdown defensiveness. |
| Growth vs Value | IWF/IWD | ok | 2026-06-18 | -0.49% | -9.44% | -6.39% | falling | Growth stronger supports recovery; value/defensive strength supports slowdown. |
| Mega-cap vs Small-cap | OEF/IWM | ok | 2026-06-18 | -5.15% | -8.24% | -10.78% | falling | Mega-cap strength supports recession/defensive concentration. |
| Dollar index | DX-Y.NYB | ok | 2026-06-19 | 1.81% | 2.47% | 2.14% | unknown | USD strength supports slowdown/recession; weakness supports growth. |

## Sector Rotation

| Sector | Ticker | Status | Latest date | 3M | 6M | 12M | 6M dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technology vs S&P 500 | XLK/SPY | ok | 2026-06-18 | 22.41% | 21.86% | 25.74% | rising |
| Communication Services vs S&P 500 | XLC/SPY | ok | 2026-06-18 | -14.89% | -15.03% | -15.51% | falling |
| Consumer Discretionary vs S&P 500 | XLY/SPY | ok | 2026-06-18 | -6.45% | -13.62% | -11.45% | falling |
| Financials vs S&P 500 | XLF/SPY | ok | 2026-06-18 | -3.12% | -10.99% | -14.54% | falling |
| Industrials vs S&P 500 | XLI/SPY | ok | 2026-06-18 | -3.23% | 5.84% | 1.72% | rising |
| Health Care vs S&P 500 | XLV/SPY | ok | 2026-06-18 | -10.17% | -12.06% | -9.96% | falling |
| Real Estate vs S&P 500 | XLRE/SPY | ok | 2026-06-18 | -7.42% | -0.93% | -14.42% | falling |
| Energy vs S&P 500 | XLE/SPY | ok | 2026-06-18 | -18.39% | 11.13% | -0.58% | rising |
| Materials vs S&P 500 | XLB/SPY | ok | 2026-06-18 | -5.42% | 4.05% | -4.20% | rising |
| Consumer Staples vs S&P 500 | XLP/SPY | ok | 2026-06-18 | -10.69% | -4.20% | -16.03% | falling |
| Utilities vs S&P 500 | XLU/SPY | ok | 2026-06-18 | -15.02% | -5.51% | -9.83% | falling |

## Required Final Output Shape

**[경기 국면 최종 진단]**
* 현재 국면: [회복 / 성장 / 둔화 / 침체]
* 진단 신뢰도: [높음 / 보통 / 낮음] (이유 요약)
* 과도기 여부: [아님 / 회복->성장 / 성장->둔화 / 둔화->침체 / 침체->회복]
* 국면 위치: [초입 / 초중반 / 중반 / 후반 / 말기] (대략 개월 범위와 근거)

**[경제 지표 분석 근거]**
* 선행 지표 추세: ...
* 동행 지표 추세: ...
* 후행 지표 추세: ...

**[시장 로테이션 교차 검증]**
* 현재 시장 강세 자산/지수/업종과 경기국면 진단의 일치 여부

**[포트폴리오 대응 지침]**
* 해남P와의 관계: [독립/위성/오버레이/적용 안 함]
* 위험자산 포지션 행동: [선호 / 수익실현 / 분할 축소 / 방어 완료]
* 100% 기준 매크로 포트폴리오: [주식/채권/현금/금/대체 또는 기타 합산 100%]
* 주식 내부 100% 배분: [지역/스타일/섹터 합산 100%]
* 추천 탑픽 섹터/스타일: ...
* 금지: 개인 맞춤 확정 매수·매도 지시처럼 쓰지 말 것. 국면 기반 자산배분 가이드로 제한.
