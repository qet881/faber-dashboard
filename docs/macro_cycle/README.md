# Macro Cycle Agent

This folder supports a GPT-5.5/Codex-driven macro-cycle judgment workflow.

The goal is not to make the Streamlit app mechanically score the economy. The
goal is to build a clean evidence pack, then let the agent reason like a macro
analyst against fixed guardrails.

## Why This Is Separate From Haenam P

Haenam P is a rules-based trend/rebalancing strategy. The macro-cycle agent is
a discretionary macro diagnosis layer.

Use it as:

- A cycle regime report.
- A satellite portfolio guide.
- A macro risk overlay for discussion.

Do not use it as:

- A silent replacement for Haenam P.
- An automatic buy/sell engine.
- A reason to mutate Haenam P rules without a separate backtest.

## Workflow

1. Generate the latest evidence pack.

```powershell
python scripts\macro_cycle_evidence.py
```

2. Ask GPT-5.5/Codex to judge the cycle using:

- `docs/macro_cycle/latest_evidence.md`
- `docs/macro_cycle/JUDGMENT_PROMPT.md`

3. Save the final report as a dated markdown file under this folder.

Suggested path:

```text
docs/macro_cycle/report-YYYY-MM-DD.md
```

## Evidence Layers

The evidence pack covers four indicator families plus two contrarian/qualitative layers:

- Leading / coincident / lagging macro + leading price indices (the core 4-stage cycle engine).
- ISM PMI headline is proprietary, so only latest/previous is scraped; the script
  accumulates each reading into `pmi_history.csv` so a real 3M/6M/12M trend rebuilds
  over repeated runs. Live FRED proxies (new orders, Empire State, Philadelphia Fed)
  give an immediate manufacturing-direction read.
- Market and sector rotation (cross-validation only).
- Investor sentiment (fear/greed): VIX, high-yield credit spread, S&P drawdown from
  its trailing high. Read contrarianly — fear extremes near bottoms, complacency near tops.
- Asset peak order: equities, consumer/luxury, and real estate (XLRE + Case-Shiller)
  measured against their own trailing highs to gauge cycle maturity.
- Qualitative top-signals (bookstore index, human index): manual input only, no data
  feed. Soft confirmation, never the primary judge.

## Portfolio Boundary

Portfolio guidance is allowed, but only as a macro-cycle allocation guide.

Recommended v0 scope:

- Cycle phase.
- Phase location such as early, mid, late, or terminal.
- Approximate phase age as a range when the transition anchor is visible.
- Confidence.
- Evidence and conflicting evidence.
- Risk-asset posture.
- Sector/style tilt.
- 100% macro model allocation.
- 100% equity-sleeve allocation.
- Explicit relationship to Haenam P.

Avoid v0 scope:

- Personalized all-in/all-out instructions.
- Position sizing tied to a specific account.
- Automatic override of existing strategy weights.

## Future App Integration

After the workflow is stable, the Streamlit app can show:

- Latest macro-cycle report.
- Evidence tables.
- Rotation dashboard.
- Optional macro overlay note next to existing strategy outputs.

The app should display the judgment; the agent should make the judgment.

## Phase Age

Cycle age should be estimated from transition anchors, not a hard four-year
timer. A four-year cycle is a useful prior, but real cycles compress or extend.

Preferred language:

- "성장 초중반, 대략 3-6개월차"
- "둔화 후반, 대략 9-15개월차"
- "앵커가 불명확해 개월 추정 신뢰도 낮음"

Avoid language:

- "정확히 7개월차"
- "4년 주기상 반드시 다음 국면"
- "이 날짜부터 확정"
