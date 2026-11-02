# Validation and evidence

The included unit tests check the learning machinery's direction, batch readiness and stride handling using scalar learning observations. They contain no generated OHLCV dataset and make no investment-performance claim.

No private broker-account records or raw downloaded market-data cache is included. A separate archived top-ten ranking replay and cost/delay sensitivity results are documented in [BACKTEST.md](BACKTEST.md). These do not test the declared 30-minute strategy or adaptive-vs-fixed performance. Causality must still be checked on complete, provider-sourced market histories; these learning-unit tests are not a lookahead audit.

## To test whether adaptation contributes

1. Freeze the baseline before looking at the test returns. Use identical initial settings, universe, candle history, execution and sizing for adaptive and fixed versions.
2. Use point-in-time universe membership, corporate-action-consistent OHLCV, timezone-aware timestamps and a fixed warm-up. Keep raw input provenance and checksums.
3. Replay chronologically. Allow only matured probe labels to update state. If any parameters are fitted between blocks, purge training labels crossing the boundary and account for open positions consistently.
4. Fill after confirmation and include brokerage, spread, slippage, missing fills and stops. The built-in research example does not do this for you.
5. Compare adaptive and fixed Supertrend, SPY, and a 40% SPY / 60% cash portfolio on identical dates. Report costs, exposure, drawdown, daily return convention and risk-free assumption.
6. Repeat nearby initial lengths and multipliers, higher execution costs and an extra candle's fill delay. Save every candidate, including failures.
7. Test trending, ranging, high-volatility and abrupt gap periods. Report sparse/no-signal coverage rather than replacing missing intraday bars with daily data.

Python/Pine parity is unverified. Their initialization, indicator calculations, micro-batch details and intrabar execution can differ. A matching name and configuration are not evidence of identical results. Python tests do not verify Pine's historical/realtime behaviour.

The strategy declaration's separate risk experiments test conditional stock-book losses. They do not validate the learning advantage or simulate every trading decision. An attractive risk scenario or paper-account return is not evidence that adaptive settings beat fixed settings.
