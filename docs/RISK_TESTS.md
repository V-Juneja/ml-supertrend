# Declared-portfolio risk testing

The risk script uses declared exposures and explicit assumptions, not paper-account positions or returns.

```sh
python -m pip install -r requirements-plot.txt
python analysis/run_risk.py --output-dir analysis/results/declared_risk
python analysis/check_model.py /path/to/completed_30m_bars.csv
```

`run_risk.py` creates 96 terminal cases: horizons 15/126 sessions × stock exposures 20/40/60/75% × annual stock volatility 25/45/65% × pairwise log-shock correlation 0/.35/.65/.90. Each case uses 100,000 draws; seed 20261101. Each simulated book uses six equal return streams, frozen shares and a fixed uninvested balance. This generic sensitivity setup is not an implementable portfolio under the strategy's five-position limit. Gaussian prices have zero expected stock drift; cash interest and costs are zero.

Four daily models use 40,000 paths each: Gaussian, variance-standardized t(5) log shocks, and Gaussian with a common 10% or 20% day-one stock gap. Each path is compared at 40%, 60% and 75% exposure. Shared daily chi-square scaling gives heavy-tailed innovations. The t(5) stress has no finite expected stock-price drift and is not an expected-return forecast.

Liquidation occurs at the first daily close at least 6% below the running marked-equity peak; proceeds remain uninvested. This does not model intraday fills, fresh signals, liquidity shortages or halts. Stop-overshoot arithmetic tests realized loss at 1, 1.5, 2 and 3 times the planned stop loss. A 6% liquidation trigger cannot guarantee a 6% maximum loss.

At 45% volatility and .65 correlation, 15-session 95% expected shortfall is 3.54%, 7.09%, 10.63% and 13.29% at 20%, 40%, 60% and 75% exposure. The modeled 6% drawdown probabilities and daily-close liquidation expected shortfalls for the 75% book are recorded in `analysis/results/declared_risk/daily_risk.csv`. These are conditional model results, not observed or forecast strategy returns.

`analysis/results/declared_risk` includes the completed model outputs and dashboard. Parameter/prefix check outputs use recorded provider-sourced DELL bars and are labelled implementation diagnostics. The raw market-data cache is not redistributed. Matching signal counts do not establish a learning advantage; a cost-aware portfolio replay remains necessary.

## Historically calibrated model comparison

```sh
python -m pip install -r requirements-risk.txt
python analysis/calibrated_risk.py /path/to/provider_bar_directory
```

The calibration run uses six historical asset return series as an illustrative cross-sector risk basket. These are not a declaration of actual holdings. At 40% gross exposure, each series has 6.67% account weight; at 75%, each has 12.5%. The recorded provider cache supplies 59 aligned close-to-close daily log returns from 2 July through 25 September 2026. Returns use the last observed regular-session bar per local trading date, with no missing-price filling. Input checksums identify every source file.

Ledoit-Wolf covariance shrinkage estimates cross-asset risk, blending the noisy sample covariance toward a scaled identity. Fitted shrinkage intensity is 11.00%. The model comparison runs 50,000 fifteen-session paths per family and scales the same asset paths to 40%, 60% and 75% exposure. Seeds are 20261102-20261106.

| Risk model | ES at 40% exposure | ES at 75% exposure |
| --- | ---: | ---: |
| Calibrated Gaussian | 3.50% | 6.56% |
| Variance-standardized Student-t(5) | 3.56% | 6.68% |
| Five-session circular block bootstrap | 2.92% | 5.48% |
| Fitted volatility × 1.5; correlation .90 | 9.36% | 17.55% |
| Common 20% day-one price gap | 10.81% | 20.27% |

The block bootstrap resamples whole cross-asset return vectors in five-session blocks, retaining contemporaneous co-movement and short sequences. It is demeaned at the log-return level with a Gaussian variance correction; this does not imply zero expected arithmetic return. Gaussian scenarios impose zero expected stock-price drift. Student-t log shocks have finite variance but no finite positive exponential moment; they are tail-stress experiments, not expected-price-return forecasts.

Liquidation can lock in losses before recovery. At 75% exposure, expected shortfall after the common 20% gap is 20.27% without liquidation and 16.44% with the daily-close trigger. In the calibrated Gaussian case at 75%, 7.18% of paths reach the 6% drawdown trigger. A single protective rule need not improve every risk statistic. The 59-return sample is short, the six assets were selected as an illustrative cross-sector basket, and Monte Carlo standard errors measure simulation noise only, not parameter or regime uncertainty.

Calibration outputs in `analysis/results/calibrated` include the covariance/correlation estimates and scenario summaries. The raw price cache and observed daily-return table are not redistributed. This is a risk-model comparison, not a model of signal-driven portfolio returns or an out-of-sample performance test.

The current declaration permits at most ten simultaneous positions and 10% per name, within the 100% gross cap; ten positions at 10% each use the full allowance. The saved six-asset risk basket is a reference exposure experiment rather than a portfolio satisfying the five-position constraint. Historical archived-ranking replay rules and results retain their original ten-holding allocation.
