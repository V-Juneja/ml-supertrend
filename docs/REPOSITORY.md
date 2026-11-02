# Repository layout

```text
ml-supertrend/
├── README.md
├── reports/Strategy_Declaration.pdf
├── src/ml_supertrend.py
├── tradingview/                  # Pine strategy and bands-only overlay
├── config/declaration.json
├── run_signals.py
├── analysis/
│   ├── replay_archived_signals.py
│   ├── audit_published_results.py
│   ├── check_model.py
│   ├── calibrated_risk.py
│   ├── run_risk.py
│   └── results/
│       ├── archived_replay/      # Dated rankings, simulated fills and equity, cost/delay tests
│       ├── dell_checks/          # Signal sensitivity and parameter diagnostics
│       └── calibrated/          # Monte Carlo and covariance outputs
├── docs/                        # Strategy, learning, formulas, execution and backtest methods
│   └── images/                  # Equity, drawdown, monthly returns and DELL charts
├── examples/
└── tests/
```

Start with [the declaration](../reports/Strategy_Declaration.pdf), [backtest methodology](BACKTEST.md), or [how the learner works](LEARNING.md). The ranking replay and ML Supertrend signal engine have separate evidence scopes.
