# Timing, state and execution

- Use completed 30-minute regular-session candles. CSV timestamps identify completed candle closes with explicit offsets.
- Confirm a signal at candle close; the earliest causal historical market-order fill is the next bar open, plus costs. The runner emits signals only.
- Learning probes are internal hypothetical trades. Their simulated entries are candle opens and their results mature at the close of the fifth candle, including the entry candle.
- Freeze model configuration before evaluation, but permit its declared online update rules to evolve state chronologically. Do not retune thresholds using the test interval.
- A Python `run()` resets the model. Re-running a shorter history is not equivalent to continuing a longer run. The snapshot omits full probe, rolling-window and bar-history state; it is not a complete restart checkpoint.
- Use a fixed warm-up start. Reconstruct a complete history for each offline evaluation; do not claim partial snapshot resume reproduces uninterrupted execution.
- The source's risk guard and probe execution ledger are not a broker account risk manager. The declaration's stops, cash, sector, concentration and drawdown limits require a separate portfolio/execution layer.
- The Pine strategy wrapper has basic entry/close handling, sizing properties and zero default cost assumptions. Configure realistic properties and disable tick pressure. It does not enforce all declared portfolio controls.
- A stop is an intended exit trigger. Overnight gaps, halts and poor liquidity can produce larger losses than the planned stop distance.

No live-account credentials, order-routing adapters or services are part of this folder.
