# IQIC Strategy Declaration: rubric review

**Strict team self-assessment: 73/80.** This is our pre-submission estimate against the published bands, not the official judge score.

| Criterion | Target band | Evidence in the declaration |
|---|---:|---|
| Investment Thesis & Strategy | 18/20 | Clear reversal hypothesis and failure condition; signal filters, adaptive settings, entry, exit and sizing are defined. The economic case for a short-term rebound after strong selling remains a hypothesis rather than an independently established effect. |
| Quantitative Methodology | 18/20 | Equations, variables, probe learning, archived replay, costs, sensitivity, trade statistics and 50,000-path block bootstrap are documented. The replay is daily email-list execution, so it does not independently measure the declared 30-minute signal engine's account performance. |
| Risk Management | 19/20 | Controls address stop gaps, name and sector concentration, liquidity, aggregate exposure, session losses and drawdown, with deterministic and simulated stress cases. Several limits are design choices and have not been validated in live competition fills. |
| Evidence-Based Decision Making | 9/10 | Uses archived ML Supertrend signals, Alpaca SIP price data, QQQ comparison, source code, fill reconciliation and cost/delay sensitivity. The evidence is limited to one replay window and its specific execution assumptions. |
| Communication & Presentation | 9/10 | Five-page structure, labeled equations, defined variables, interpreted charts and repeated repository links improve clarity. The machine-learning section remains information-dense for readers new to the method. |

## Scope of the evidence

The historical performance figures describe a paper replay of archived ML Supertrend email signals under the stated daily execution assumptions. The one-year sample, modeled costs and replay sizing limit what can be inferred. Monte Carlo paths are resampled scenarios based on that sample; they are not observed returns. The official result remains with IQIC judges.
