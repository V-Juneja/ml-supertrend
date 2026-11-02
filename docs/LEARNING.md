# Machine learning mechanism

The model trains its indicator live through online, performance-conditioned updates as completed market candles arrive. Its training observations are simulated probe outcomes. They are generated from the same price stream that produces signals, but they are independent of actual portfolio positions. A profitable probe is evidence about a short price interval, not proof that an account trade would have filled profitably.

## Probe labels

For direction `s`, entry `P`, fifth-candle close `C`, and entry smoothed true range `A_entry`:

```text
price_gain = s × (C - P)
return_fraction = price_gain / P
volatility_normalized_gain g = price_gain / A_entry
simulated_dollar_gain = return_fraction × configured_probe_notional
```

`s = +1` for a long and `-1` for a short. The implementation counts the entry candle as candle one. A probe opened on candle k matures at the close of candle k+4. Its eventual result must not be used on earlier candles. The worst adverse move is measured from the low of the holding window for a long and its high for a short, then divided by entry volatility.

Example: buying at 100 and observing a fifth close of 102 with entry volatility 2 gives `g = +1`; the matching short has `g = -1`. Dollar probe values are illustrative learning quantities, not account returns.

## Three feedback routes

**Rolling proposals.** Separate long/short windows retain recent outcomes (baseline 30). A win rate below 38% and negative average outcome produces a proposal to widen bands and lengthen smoothing. A win rate above 62% and positive average outcome proposes the opposite. Downside measures, streaks and other coded conditions can modify the net proposal. The rolling proposal step includes `learning_rate / sqrt(sample_count)`; do not describe all update sizes as increasing with sample count.

**Small batches.** Each completed probe contributes a confidence score and an adverse-excursion measure. Its effective batch weight is:

```text
sample_weight = confidence / (1 + adverse_excursion_in_ATR_units)
weighted_mean_gain = sum(sample_weight × g) / sum(sample_weight)
```

Confidence here is a bounded internal weight, not a calibrated likelihood of profit. A larger adverse excursion lowers the sample's influence at the same confidence. Batch quality also uses weighted profit factor and a downside-return ratio. The master dial sets batch size, stride, blend weight and change caps. This is a designed update rule, not gradient descent on an estimated forecasting loss.

**Context memory.** An 8 × 8 grid stores performance-conditioned parameter suggestions across trend/range score and relative volatility. The trend/range score combines Hurst-based persistence, inverse entropy and ADX-based trend strength; each describes past prices. Cell updates use exponential smoothing. Nearby cells are blended with Gaussian-distance weights. The grid's confidence contribution rises with its count; its recommendation weight is capped at 0.65. A decaying short-term trace buffer can consolidate representative outcomes into this memory.

## Blend and update

Global proposals and grid proposals are blended by the grid weight. Available small-batch recommendations are blended separately. Long/short batch contributions can receive unequal internal weights according to the completed batch outcomes. These are weights on parameter-change proposals; they do not allocate account capital between long and short books.

A simplified final update is:

```text
theta_next = theta_now + alpha × (bounded_rounded_target - theta_now)
```

`theta` is an adaptive setting, `alpha` the fraction of the move applied, and the target the proposal after rounding and bounds. For 1.40, target 1.50 and alpha 0.20, the updated value is 1.42. The full source also includes deadbands, per-update caps, update scheduling and periodic baseline reversion. Signal ATR length remains between 5 and 100; multiplier between 0.5 and 5.0.

The core also carries probe stop/target/breakout-related settings. These internal settings do not implement the declaration's actual protective stop or portfolio loss controls. The declared Reversal strategy and external risk overlay determine actual account behaviour.

## What is not established

The code implements adaptation; it does not prove a learning advantage. Context cells can be sparse, overlapping probe outcomes are dependent, and regime scores can lag a change in conditions. Compare against an otherwise identical fixed-parameter baseline on untouched chronological data, including realistic costs and all strategy decisions.
