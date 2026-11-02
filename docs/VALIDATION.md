# Validation and evidence

The included unit tests check the learning machinery's direction, batch readiness and stride handling using scalar learning observations. They contain no generated OHLCV dataset and make no investment-performance claim.

No private broker-account records or raw downloaded market-data cache is included. The archived top-ten ranking replay and cost/delay sensitivity in [BACKTEST.md](BACKTEST.md) follow rules separate from the declaration. A direct adaptive-vs-fixed 30-minute portfolio replay is documented in [DIRECT_MODEL_BACKTEST.md](DIRECT_MODEL_BACKTEST.md); it covers 70 sessions and a selected 17-name sample. The replay is sequential and charges costs, but is not a point-in-time universe study or long-run out-of-sample validation. The scalar learning-unit tests are not a complete lookahead or Python/Pine parity audit.

## To test whether adaptation contributes

1. Freeze the baseline before looking at the test returns. Use identical initial settings, universe, candle history, execution and sizing for adaptive and fixed versions.
2. Use point-in-time universe membership, corporate-action-consistent OHLCV, timezone-aware timestamps and a fixed warm-up. Keep raw input provenance and checksums.
3. Replay chronologically. Allow only matured probe labels to update state. If any parameters are fitted between blocks, purge training labels crossing the boundary and account for open positions consistently.
4. Fill after confirmation and include brokerage, spread, slippage, missing fills and stops. The direct replay applies these conventions to the available 70-session sample; the code and results are documented in [DIRECT_MODEL_BACKTEST.md](DIRECT_MODEL_BACKTEST.md).
5. Extend the current adaptive/fixed comparison across a longer, point-in-time universe and include SPY and a 75% SPY / 25% cash benchmark on identical dates. Report costs, exposure, drawdown, daily return convention and risk-free assumption.
6. Repeat nearby initial lengths and multipliers, higher execution costs and an extra candle's fill delay. Save every candidate, including failures.
7. Test trending, ranging, high-volatility and abrupt gap periods. Report sparse/no-signal coverage rather than replacing missing intraday bars with daily data.

Python/Pine parity is unverified. Their initialization, indicator calculations, micro-batch details and intrabar execution can differ. A matching name and configuration are not evidence of identical results. Python tests do not verify Pine's historical/realtime behaviour. The direct portfolio replay does not establish Pine parity.

The strategy declaration's separate risk experiments test conditional stock-book losses. The 70-session adaptive/fixed replay tests portfolio decisions on a selected sample, but does not establish a durable learning advantage or simulate every historical regime. An attractive risk scenario or paper-account return is not evidence that adaptive settings beat fixed settings.
