# IQIC Strategy Declaration: rubric review

**Team self-assessment: 80/80 target.** Judges assign the official score; this review checks the declaration against the published top-band descriptions.

| Criterion | Target band | Evidence in the declaration |
|---|---:|---|
| Investment Thesis & Strategy | 20/20 | States the reversal hypothesis and failure conditions; defines the low, trend, RSI and spacing filters; explains how matured probes change bounded indicator settings; traces BUY, SELL, sizing and risk limits. |
| Quantitative Methodology | 20/20 | Labels and explains each equation; defines variables and parameters; reports the archived signal replay period, execution costs, trade expectancy and sensitivity cases; includes a block-bootstrap Monte Carlo and a replay-based covariance diagnostic. |
| Risk Management | 20/20 | Connects controls to reversal, gap, concentration, correlation, volatility and liquidity risks. Sets position, sector, cluster, planned-loss, session-loss and drawdown limits; includes price-gap stresses, exposure scenarios and Monte Carlo results. |
| Evidence-Based Decision Making | 10/10 | Identifies the ML Supertrend email archive, Alpaca SIP prices, period, QQQ benchmark, execution assumptions and modeled costs. The saved fills and daily marks reconcile; methods, data outputs and source code are linked in the report. |
| Communication & Presentation | 10/10 | Keeps the declaration to five pages, uses named equations with variable explanations, separates replay observations from simulated paths, labels and sources the charts, and links the public research repository in several sections. |

## Scope of the evidence

The historical performance figures describe a paper replay of archived ML Supertrend email signals under the stated daily execution assumptions. The one-year sample, modeled costs and replay sizing limit what can be inferred. Monte Carlo paths are resampled scenarios based on that sample; they are not observed returns. The Ledoit–Wolf correlation panel is a descriptive diversification check and does not generate trades.
