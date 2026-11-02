# Archived top-ten BUY replay

[Results and data](../analysis/results/archived_replay/) · [Replay source](../analysis/replay_archived_signals.py) · [Sensitivity table](../analysis/results/archived_replay/sensitivity.csv)

This test evaluates the earlier archived top-ten BUY workflow, **separately from the declared 30-minute ML Supertrend portfolio**. It consumes rankings recorded in emailed signals. It does not reconstruct past model predictions or demonstrate that adaptive Supertrend beats fixed Supertrend.

## Data and timing

The original archive contains 1,268 messages across 255 selected dates, 7 September 2025–14 October 2026. The first message for each date supplies ten ranked BUY candidates. The normalized ranking file contains only date, rank, symbol, score and signal; no email addresses, message bodies, receipt headers or account identifiers are published. The private receipt audit checks the original archive separately.

Replay period: **8 September 2025–30 October 2026**, 290 sessions. Signals become eligible at the first session open strictly after their signal date. Multiple signal dates mapping to one execution session use the latest eligible list. The last list carries forward through 30 October. There are no fresh archived signals after 14 October.

Prices: Alpaca SIP daily bars, split-adjusted. No raw market-price cache is distributed. The public normalized QQQ series and simulated equity/fill records are supporting outputs. Supply your own licensed bars to rerun the engine; provenance.json records input hashes. Dividends and cash interest are omitted. These are historical simulated executions, not broker-account returns.

## Execution rules

- Initial cash $100,000; $10,000 all-in cash debit per new name; up to ten holdings; no borrowing. Fractional shares reproduce the historical simulation.
- Rank order determines entries; only nonnegative-score BUY candidates qualify. No duplicate entry while held.
- Sell if the name is absent from the eligible list or the previous close is at/below 95% of its execution entry price or at/above 115%. Execute at the following open, with adverse slippage and fees. A name still on the list can be repurchased after an exit.
- Mark remaining holdings at daily closes. Ten remain open at the end of the baseline; no final liquidation is invented. The baseline applies 5 bps fees plus 5 bps slippage on each side.

## Completed sensitivity tests

| Case | Total return | Max drawdown | Sessions |
| --- | ---: | ---: | ---: |
| Zero modeled cost | 64.45% | -19.47% | 290 |
| 10 bps per side | 51.54% | -21.10% | 290 |
| 20 bps per side | 39.11% | -22.81% | 290 |
| 10 bps + one extra session for ranking eligibility | 43.49% | -21.91% | 289 |

All settings were held fixed across these tests; these are sensitivity cases, not a search for the best parameter set. The delayed case begins 9 September, one session later, and delays ranking-driven changes only. Stop and target exits still use the prior close and next open. Costs can change future holdings through available cash and entry-relative stop thresholds, so the return difference is not simply a subtraction of aggregate fees.

Higher costs reduce baseline total return by 12.43 percentage points; delaying rankings reduces it by 8.05 points. Profitability is materially execution-sensitive. No 15-year validation or guarantee of freedom from bias is claimed.

The standalone replay reproduced all **1,254 baseline fills** (632 BUY, 622 SELL) and all **290 daily equity marks** against the previously saved simulation. Dates, symbols, actions, shares, execution prices, fees, cash debits/credits and cash balances match within floating-point tolerance. This establishes reproduction of the saved historical workflow; it does not validate broker execution or reconstruct the original ranking model.

## Reproduce

```sh
python -m pip install -r requirements.txt
python analysis/replay_archived_signals.py \
  analysis/results/archived_replay/dated_buy_rankings.csv \
  /path/to/provider_daily_bars.csv \
  --output-dir outputs/archived_replay
python analysis/audit_published_results.py
```

Daily bar columns: `date,symbol,open,close`; include QQQ to define the trading-session calendar and cover every held symbol/date. Missing prices fail the run. Retain the recorded date range and corporate-action treatment. Use the provenance input hashes to distinguish an exact data match from a replay on a refreshed provider history.

CAGR uses elapsed calendar days (365.2425/year). Drawdown includes the initial $100,000 capital in the running peak. Daily Sharpe uses all marked session returns, including the first session relative to initial capital, 252 sessions/year, sample standard deviation and zero risk-free rate. Monthly returns compound; the first month is partial. Neither Sharpe nor CAGR is a forecast.

## Trade expectancy

Expected return per trade (trade expectancy) is the mean net gain or loss as a percentage of the capital committed to a trade. A positive value means average gains exceeded average losses and modeled costs in the observed sample. It describes an average across trades; it is neither a win probability nor a guaranteed outcome for the next trade.

**Net Trade Expectancy Equation:** `e = p W - (1-p) D - c`

- `p`: fraction of completed trades with positive gross P&L, before modeled costs.
- `W`: mean gross return among those winners, in percentage points of entry cash debit.
- `D`: positive magnitude of mean gross return among the remaining trades, in the same units.
- `c`: mean round-trip fees plus adverse slippage, in the same units.
- `e`: average return after both fees and slippage. A 1% result means $100 per $10,000 committed, on average.

The archived replay has **622 closed trades**. Using their recorded fills gives `p = 47.7492%`, `W = 8.06654%`, `D = 5.42310%`, and `c = 0.200818%`. These produce **e = +0.817278%**, or **$81.73 net per $10,000 position on average**. Ten open positions are excluded. Return terms use the actual all-in entry debit as denominator; gross P&L removes both entry and exit execution frictions, and costs are then subtracted once. Small discrepancies from the rounded PDF inputs come from rounding.

The baseline net winning-trade fraction is 46.62%; it differs from `p` because costs turn some small gross gains into net losses. A win rate below 50% can coexist with positive expectancy when winners are sufficiently larger than losers. Average trade expectancy is distinct from cumulative account return: positions overlap, capital is reused, and open holdings still affect account equity.

[Per-trade calculations](../analysis/results/archived_replay/closed_trade_expectancy.csv) · [Full-precision summary](../analysis/results/archived_replay/expectancy_summary.json)

These are historical descriptive statistics. Dependence between concurrent trades and changing market conditions limits their use as forecasts.
