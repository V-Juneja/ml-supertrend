# Separate bands-only overlay

Use `tradingview/ml_supertrend_bands_only.pine`. This is a standalone indicator with the model's adaptive bands. It creates no strategy orders or BUY/SELL markers. The `ml_supertrend_visual.pine` filename is also provided as an alias of this bands-only version.

## Add it independently

1. Open Pine Editor, create a new indicator and replace the entire default template with the complete file. Include everything from `//@version=5` to the final `plot(...)` line. Pasting only the declaration or calculation section cannot create the bands.
2. Choose Add to chart. The chart legend reads **ML Supertrend - Bands Only**.
3. Toggle that indicator's eye icon to show or hide the bands. Hide your separate strategy if you also want to remove its order markers from a screenshot. The indicator and strategy can be toggled independently.
4. The overlay defaults to the ML trailing bands and a faint corridor. Fixed baseline and raw-envelope comparisons are optional in Bands overlay settings. There are no learning tables or BUY/SELL labels on this overlay.
5. Use regular-session 30-minute standard candles. Keep tick pressure off for the declared baseline.

## Verification

The complete bands-only script was compiled and run in TradingView on a DELL 30-minute chart in this session. The bands were visible with no compiler/runtime error for the overlay, and its eye toggle was tested off and on. This verifies that chart configuration; it is not proof of Pine/Python parity or all possible input combinations.

The file includes a top-level plot and visible band plots. Its current code satisfies the indicator output requirement. Keep the whole file when copying into Pine Editor.

The core keeps hypothetical probe calculations for learning, but displays none of those trades. A forming candle can move before close; the default carry setting holds the last confirmed model-band values on that candle. The model uses fixed-baseline and raw-envelope calculations separately from the learning proposals.
