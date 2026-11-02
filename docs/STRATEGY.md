# From data to a decision

The hypothesis is that a decline which makes a fresh low and recently oversold momentum becomes more attractive only after a confirmed upward change in its volatility-adjusted trend.

The implementation uses a close-weighted price and smoothed true range to construct trailing boundaries. A larger multiplier creates wider boundaries; a longer smoothing length reacts more slowly to volatility. The learning loop changes these settings as completed hypothetical trades accumulate.

| Stage | Observation | Consequence |
| --- | --- | --- |
| Structural setup | Fresh low stored during a downtrend | Marks a possible reversal setup; no order yet |
| Confirmation | Close crosses the previous opposing trend boundary upward | Trend changes upward |
| Momentum | RSI(14) below 30 on the current or preceding 49 candles | Prior selling pressure satisfies the declared filter |
| Spacing | At least 10 candles since the last BUY or SELL | Avoids repeated clustered signals |
| Account capacity | Liquidity, exposure, sector, shared-theme and loss limits all pass | Size a long order after bar confirmation |
| Exit | Confirmed SELL or earlier protective stop | Close an existing long; do not open a short |

Fresh-low detection is more specific than a rolling-minimum touch: the source compares the rolling-low value with its earlier value and requires close below that earlier low. Baseline structural sensitivity is 30; its comparison offset derives from sensitivity / 10. Stored flag updates also depend on signal spacing. Preserve the code's order of operations in any reproduction.

The declared baseline uses 30-minute regular-session bars, Reversal mode, hlcc4 price basis, initial ATR length 24 and multiplier 1.4, RSI length 14, momentum memory 50 bars and spacing 10 bars. Volume-surge and major-level filtering are off. Adaptive learning is on; tick pressure is off. Plot bands can remain locked at baseline while the signal engine uses adaptive bands. Use signal outputs for decisions.

Portfolio limits are at most ten concurrent positions, 10% of marked equity per name and 100% maximum total stock exposure, subject to the separate stop-risk and concentration budgets. Settings live in `config/declaration.json`. The CSV runner does not enforce them or execute trades. Long-only does not mean the learner lacks a short simulation; the learner and the account are separate parts of the system.

When several entries compete for capacity, process earlier confirmed signals first and use alphabetical ticker order for a timestamp tie. Apply the remaining portfolio, sector, cluster and planned-loss budgets after each accepted order. This belongs to the declared account overlay; the single-symbol CSV runner does not implement a portfolio order queue.
