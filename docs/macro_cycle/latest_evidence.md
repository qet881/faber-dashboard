# Macro Cycle Evidence Pack

- Generated: 2026-06-23 13:57:36
- Requested as-of date: 2026-06-23
- Role: evidence package for GPT-5.5/Codex judgment, not a deterministic scorecard.

## Judgment Instruction

Use the evidence below to judge the US macro cycle directly. Do not score mechanically. Classify the economy as Recovery, Growth, Slowdown, or Recession. If evidence is transitional, name the transition path and explain why one side dominates. Mention conflicting evidence.

Portfolio output is allowed, but it must be framed as a macro satellite/overlay guide. It must not override Haenam P's trend-following/rebalancing rules unless the user explicitly asks for a separate strategy.

## Leading Indicators

| Asset | Ticker | Status | Latest date | Latest | 3M | 6M | 12M | 5Y | >200D MA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | ^GSPC | ok | 2026-06-22 | 7472.79 | 14.85% | 8.64% | 25.22% | 75.98% | True |
| NASDAQ Composite | ^IXIC | ok | 2026-06-22 | 26166.60 | 20.88% | 11.69% | 34.55% | 83.58% | True |
| Dow Jones Industrial Average | ^DJI | ok | 2026-06-22 | 51712.71 | 13.46% | 6.93% | 22.52% | 52.34% | True |
| Russell 2000 | ^RUT | ok | 2026-06-22 | 3004.40 | 23.21% | 17.42% | 42.44% | 30.86% | True |

## Leading Macro

| Indicator | FRED | Status | Latest date | Latest | 3M delta | 6M delta | 12M delta | 6M dir | Reading note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISM Manufacturing PMI | TradingEconomics:business-confidence | ok | May 2026 | 54.00 | n/a | n/a | n/a | rising | Trading Economics latest/previous snapshot fallback because the public FRED PMI series was unavailable. Source: https://tradingeconomics.com/united-states/business-confidence Accumulated locally (2 months) in pmi_history.csv; trend grows as more runs are stored. |
| University of Michigan Consumer Sentiment | UMCSENT | ok | 2026-04-01 | 49.80 | -6.60 | -3.80 | -2.40 | falling | Consumer expectations and sentiment, treated as leading pressure. |
| 10Y minus 2Y Treasury spread | T10Y2Y | ok | 2026-06-22 | 0.27 | -0.24 | -0.46 | -0.21 | falling | Curve steepening after inversion can mark late slowdown/recession transition. |
| Manufacturers' New Orders: Total Manufacturing | AMTMNO | ok | 2026-04-01 | 662728.00 | 43376.00 | 57327.00 | 69191.00 | rising | Live FRED proxy for the demand side of the ISM PMI. Use 6M/12M direction; falling new orders lead a manufacturing slowdown. |
| Empire State Mfg General Business Conditions | GACDISA066MSFRBNY | ok | 2026-06-01 | 5.70 | 5.90 | 9.40 | 20.60 | rising | Live regional Fed diffusion index (oscillates around 0). A trend-able PMI proxy: below 0 and falling supports slowdown. |
| Philadelphia Fed Mfg General Activity | GACDFSA066MSFRBPHI | ok | 2026-06-01 | 10.30 | -7.80 | 19.10 | 11.70 | rising | Live regional Fed diffusion index (oscillates around 0). Cross-check with Empire State and PMI direction. |

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
| DM vs EM | VEA/EEM | ok | 2026-06-22 | -8.82% | -11.47% | -13.58% | falling | DM stronger supports recovery/slowdown/recession; EM stronger supports growth. |
| US vs Non-US | VTI/VEA | ok | 2026-06-22 | -1.54% | -6.60% | -6.09% | falling | US stronger supports recovery/slowdown/recession; Non-US stronger supports growth. |
| NASDAQ vs S&P 500 | QQQ/SPY | ok | 2026-06-22 | 10.44% | 9.32% | 11.26% | rising | NASDAQ stronger supports recovery/growth; weakness supports slowdown. |
| Dow vs S&P 500 | DIA/SPY | ok | 2026-06-22 | -1.07% | -1.39% | -1.72% | falling | Dow stronger supports slowdown defensiveness. |
| Growth vs Value | IWF/IWD | ok | 2026-06-22 | -1.36% | -11.66% | -7.21% | falling | Growth stronger supports recovery; value/defensive strength supports slowdown. |
| Mega-cap vs Small-cap | OEF/IWM | ok | 2026-06-22 | -6.52% | -9.40% | -12.13% | falling | Mega-cap strength supports recession/defensive concentration. |
| Dollar index | DX-Y.NYB | ok | 2026-06-23 | 2.12% | 3.17% | 2.67% | unknown | USD strength supports slowdown/recession; weakness supports growth. |

## Sector Rotation

| Sector | Ticker | Status | Latest date | 3M | 6M | 12M | 6M dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technology vs S&P 500 | XLK/SPY | ok | 2026-06-22 | 23.73% | 21.43% | 27.02% | rising |
| Communication Services vs S&P 500 | XLC/SPY | ok | 2026-06-22 | -16.77% | -15.69% | -16.72% | falling |
| Consumer Discretionary vs S&P 500 | XLY/SPY | ok | 2026-06-22 | -6.92% | -13.68% | -12.99% | falling |
| Financials vs S&P 500 | XLF/SPY | ok | 2026-06-22 | -4.09% | -10.40% | -14.20% | falling |
| Industrials vs S&P 500 | XLI/SPY | ok | 2026-06-22 | -1.76% | 6.49% | 2.44% | rising |
| Health Care vs S&P 500 | XLV/SPY | ok | 2026-06-22 | -9.50% | -10.82% | -8.65% | falling |
| Real Estate vs S&P 500 | XLRE/SPY | ok | 2026-06-22 | -4.29% | 1.79% | -13.26% | rising |
| Energy vs S&P 500 | XLE/SPY | ok | 2026-06-22 | -19.70% | 13.44% | -0.26% | rising |
| Materials vs S&P 500 | XLB/SPY | ok | 2026-06-22 | -3.73% | 4.28% | -3.46% | rising |
| Consumer Staples vs S&P 500 | XLP/SPY | ok | 2026-06-22 | -11.04% | -2.22% | -17.12% | falling |
| Utilities vs S&P 500 | XLU/SPY | ok | 2026-06-22 | -11.79% | -2.35% | -9.51% | falling |

## Investor Sentiment (Fear / Greed)

Contrarian layer: the quantitative shadow of the psychology cycle. Fear/capitulation extremes are contrarian-bullish; complacency/greed extremes are late-cycle warnings. Cross-validation only, not the primary cycle judge.

| Signal | Ticker | Status | Latest date | Latest | 3M | 1Y %ile | Dir | Reading note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VIX (equity fear gauge) | ^VIX | ok | 2026-06-22 | 17.28 | -35.47% | 55.16% | falling | High and spiking = fear/capitulation (contrarian buy zone); low and flat = complacency/euphoria near tops. |
| High-Yield OAS credit spread | n/a | ok | 2026-06-19 | 2.66 | -0.61 | n/a | falling | Widening = credit stress/capitulation; tight and falling = risk appetite/late-cycle complacency. |
| S&P 500 drawdown from trailing high | ^GSPC | ok | 2026-06-22 | -1.80% | n/a | n/a | unknown | Deep drawdown = capitulation/contrarian-buy territory; near 0% = at highs, late-cycle complacency. |

## Asset Peak Order

Money rotates, so assets tend to top in sequence (equities first, consumer/luxury next, real estate last). Each row shows how far the asset sits below its own trailing high and how long ago that high was. Use the sequence to gauge how late the cycle is.

| Asset | Ticker | Status | Own peak date | From peak | Months since peak | At high | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US Equity (S&P 500) | ^GSPC | ok | 2026-06-02 | -1.80% | 0.7 | True | Leads the cycle; usually tops first. |
| Consumer Discretionary (XLY) | XLY | ok | 2026-01-12 | -7.32% | 5.3 | False | Consumer/luxury spending proxy; tends to top after equities broadly. |
| Real Estate (XLRE) | XLRE | ok | 2026-06-12 | -2.10% | 0.3 | True | Rate-sensitive; listed real estate proxy. |
| US Home Prices (Case-Shiller) | CSUSHPINSA | ok | 2025-06-01 | -0.50% | 9.0 | False | Physical real estate; slowest to turn, usually tops last. Reported with a ~2 month lag. |

## Qualitative Top-Signals (manual input)

No reliable data feed exists for these (the video's bookstore and human indicators). Leave blank if unknown. The agent must treat any answer as **soft confirmation only**, never as a primary cycle judge.

| Signal | Prompt | Current reading |
| --- | --- | --- |
| 서점 지표 | 베스트셀러가 가치투자/원칙서(침체·바닥 분위기) vs 차트/파동/일목균형표 등 기술적 매매서(과열·고점 분위기) 중 어느 쪽인가? | (수동 입력 — 비워두면 미사용) |
| 인간 지표 | 평소 주식과 무관하던 대중·지인이 갑자기 추격매수에 뛰어드는 강세장 끝물 신호가 보이는가? | (수동 입력 — 비워두면 미사용) |
| 거래 열기 | 주변에서 레버리지·빚투·신규계좌 개설 분위기가 과열인가, 아니면 관심을 끊고 무기력(항복/우울)한가? | (수동 입력 — 비워두면 미사용) |

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

**[투자자 심리 · 고점 신호]**
* 공포/탐욕(VIX·신용스프레드·낙폭): [항복/공포 / 중립 / 탐욕/과열] -> 역발상 해석
* 자산 고점 순서(주식->소비재->부동산): 각 자산의 고점대비 위치로 본 사이클 성숙도
* 정성 신호(서점/인간 지표): 수동 입력값이 있으면 보조 확인용으로만 언급, 없으면 '입력 없음'

**[포트폴리오 대응 지침]**
* 해남P와의 관계: [독립/위성/오버레이/적용 안 함]
* 위험자산 포지션 행동: [선호 / 수익실현 / 분할 축소 / 방어 완료]
* 100% 기준 매크로 포트폴리오: [주식/채권/현금/금/대체 또는 기타 합산 100%]
* 주식 내부 100% 배분: [지역/스타일/섹터 합산 100%]
* 추천 탑픽 섹터/스타일: ...
* 금지: 개인 맞춤 확정 매수·매도 지시처럼 쓰지 말 것. 국면 기반 자산배분 가이드로 제한.
