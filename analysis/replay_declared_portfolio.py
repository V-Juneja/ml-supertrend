"""Replay the declared long-only 30-minute portfolio on supplied SIP bars.

Bars must be one SYMBOL_30Min.csv per name with columns time,open,high,low,
close,volume and timezone-aware UTC timestamps. The engine uses only regular
session bars, emits signals after completed candles, and fills at the next
available bar open. The script does not download or fabricate market data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ml_supertrend import (  # noqa: E402
    MLSupertrend,
    SupertrendConfig,
    atr_series,
)

START = pd.Timestamp("2026-07-02", tz="America/New_York")
END = pd.Timestamp("2026-10-09 23:59:59", tz="America/New_York")
INITIAL = 100_000.0
FRICTION = 5e-4  # 5 bp on each side, applied adversely to fills.
SECTOR = {
    "AMD": "Technology", "CRM": "Technology", "CRWD": "Technology",
    "DELL": "Technology", "FTNT": "Technology", "INTC": "Technology",
    "MRVL": "Technology", "MU": "Technology", "NET": "Technology",
    "PANW": "Technology", "SNOW": "Technology",
    "BNS": "Financials", "HOOD": "Financials", "ING": "Financials",
    "MET": "Financials", "MPC": "Energy", "VLO": "Energy",
}
TECH_CLUSTER = set(SECTOR) - {"BNS", "HOOD", "ING", "MET", "MPC", "VLO"}


def load_symbol(path: Path, adaptive: bool) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close", "volume"}
    if required - set(d.columns):
        raise ValueError(f"{path.name} missing columns: {sorted(required - set(d.columns))}")
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="raise")
    if d.time.isna().any() or d.time.duplicated().any() or not d.time.is_monotonic_increasing:
        raise ValueError(f"{path.name} timestamps must be unique and increasing")
    numeric = d[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (d[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path.name} contains invalid OHLCV")
    if (d.high < d[["open", "close", "low"]].max(axis=1)).any() or (d.low > d[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{path.name} contains inconsistent candle ranges")
    local = d.time.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    d = d.loc[(minute >= 570) & (minute < 960)].copy()
    d["session"] = d.time.dt.tz_convert("America/New_York").dt.date
    d = d.sort_values("time").set_index("time")
    if len(d) < 100:
        raise ValueError(f"Insufficient regular-session warm-up: {path.name}")
    cfg = SupertrendConfig(
        signal_mode="Reversal", enable_tick_pressure=False,
        enable_adaptive=adaptive, atr_period=24, multiplier=1.4,
    )
    d["atr24"] = atr_series(d.high.to_numpy(float), d.low.to_numpy(float),
                             d.close.to_numpy(float), 24)
    result = MLSupertrend(cfg).run(d)
    result["session"] = d.session
    # Use only the previous 20 fully completed sessions for the liquidity rule.
    daily_dv = (d.close * d.volume).groupby(d.session).sum()
    adv = daily_dv.rolling(20, min_periods=20).mean().shift(1)
    result["adv20"] = result.session.map(adv)
    return result


def mark_equity(cash: float, positions: dict, px: dict) -> float:
    return cash + sum(pos["qty"] * px.get(sym, pos["last_close"])
                      for sym, pos in positions.items())


def simulate(data: dict[str, pd.DataFrame], friction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    if any(set(d.index) != set(common) for d in data.values()):
        raise ValueError("Symbols do not share identical regular-session timestamps; no bars are silently discarded")
    common = [t for t in common
              if START <= t.tz_convert("America/New_York") <= END]
    if len(common) < 100:
        raise ValueError(f"Only {len(common)} common in-period bars")
    cash = INITIAL
    positions: dict[str, dict] = {}
    closed: list[dict] = []
    daily_marks: list[dict] = []
    peak = INITIAL
    day_start = INITIAL
    day_paused = False
    session_pause_triggered = False
    active_session = None
    halt_pending = False
    halted = False
    halt_triggered = False
    prev_time_by_symbol = {sym: None for sym in data}
    prev_signal: dict[str, dict] = {}
    last_close = {sym: float(d.close.iloc[0]) for sym, d in data.items()}

    for t in common:
        local = t.tz_convert("America/New_York")
        session = local.date()
        if session != active_session:
            active_session = session
            day_start = mark_equity(cash, positions, last_close)
            day_paused = False
            session_pause_triggered = False

        rows = {sym: d.loc[t] for sym, d in data.items()}
        opens = {sym: float(r.open) for sym, r in rows.items()}
        closes = {sym: float(r.close) for sym, r in rows.items()}

        def close_position(sym: str, raw_px: float, reason: str) -> None:
            nonlocal cash
            pos = positions.pop(sym)
            fill = raw_px * (1 - friction)
            cash += pos["qty"] * fill
            pnl = pos["qty"] * (fill - pos["entry"])
            closed.append({"symbol": sym, "entry_time": pos["time"],
                           "exit_time": t, "shares": pos["qty"],
                           "entry_fill": pos["entry"], "exit_fill": fill,
                           "net_pnl": pnl, "net_return_pct": pnl / (pos["qty"] * pos["entry"]) * 100,
                           "reason": reason})

        # Close SELLs known at the preceding candle close; the next open is the fill.
        for sym in sorted(positions):
            sig = prev_signal.get(sym)
            if sig and sig["sell"]:
                close_position(sym, opens[sym], "SELL signal")

        # A drawdown halt liquidates at the first available next-bar open.
        if halt_pending:
            for sym in sorted(list(positions)):
                close_position(sym, opens[sym], "6% drawdown halt")
            halted = True
            halt_pending = False

        # Existing protective stops: a gap through the stop fills at the open;
        # otherwise the stop price is used when the candle low crosses it.
        for sym in sorted(list(positions)):
            pos = positions[sym]
            if opens[sym] <= pos["stop"]:
                close_position(sym, opens[sym], "protective stop")
            elif float(rows[sym].low) <= pos["stop"]:
                close_position(sym, pos["stop"], "protective stop")

        # Earlier completed signal first; alphabetical ticker order breaks ties.
        if not halted and not day_paused:
            open_marks = {sym: opens[sym] for sym in positions}
            equity_open = mark_equity(cash, positions, open_marks)
            for sym in sorted(data):
                sig = prev_signal.get(sym)
                if not sig or not sig["buy"] or sym in positions:
                    continue
                if sym not in SECTOR or sig["adv20"] < 20_000_000 or opens[sym] < 5:
                    continue
                if len(positions) >= 5:
                    continue
                px = opens[sym] * (1 + friction)
                stop_dist = 2 * sig["atr24"]
                if not np.isfinite(stop_dist) or stop_dist <= 0:
                    continue
                # Re-mark all current positions at this bar's opens before sizing.
                eq = mark_equity(cash, positions, open_marks)
                gross = sum(p["qty"] * open_marks[s] for s, p in positions.items())
                sector_value = sum(p["qty"] * open_marks[s] for s, p in positions.items()
                                   if SECTOR[s] == SECTOR[sym])
                tech_value = sum(p["qty"] * open_marks[s] for s, p in positions.items()
                                 if s in TECH_CLUSTER)
                risk_open = sum(p["qty"] * max(p["entry"] - p["stop"], 0)
                                for p in positions.values())
                value_room = min(.15 * eq, .75 * eq - gross,
                                 .20 * eq - sector_value)
                if sym in TECH_CLUSTER:
                    value_room = min(value_room, .25 * eq - tech_value)
                risk_room = min(.005 * eq, .02 * eq - risk_open)
                qty = int(max(0, min(value_room / px, risk_room / stop_dist)))
                qty = min(qty, int(cash // px))
                if qty <= 0:
                    continue
                cash -= qty * px
                positions[sym] = {"qty": qty, "entry": px, "stop": px - stop_dist,
                                  "time": t, "last_close": closes[sym],
                                  "sector": SECTOR[sym]}
                open_marks[sym] = opens[sym]
                equity_open = mark_equity(cash, positions, open_marks)

        # A new position is exposed to the remaining range of its fill bar.
        for sym in sorted(list(positions)):
            pos = positions[sym]
            if float(rows[sym].low) <= pos["stop"]:
                close_position(sym, pos["stop"], "protective stop")

        for sym, pos in positions.items():
            pos["last_close"] = closes[sym]
            last_close[sym] = closes[sym]
        equity = mark_equity(cash, positions, closes)
        peak = max(peak, equity)
        if equity <= day_start * .98:
            day_paused = True
            session_pause_triggered = True
        if equity <= peak * .94 and not halted:
            halt_pending = True
            halt_triggered = True
        daily_marks.append({"time": t, "session": session, "equity": equity,
                            "cash": cash, "positions": len(positions),
                            "gross_exposure": (sum(p["qty"] * closes[s] for s, p in positions.items()) / equity
                                               if equity else 0),
                            "session_pause_triggered": session_pause_triggered,
                            "halt_triggered": halt_triggered})

        # The completed candle's signal is eligible only at its next bar open.
        prev_signal = {sym: {"buy": bool(r.buy_signal), "sell": bool(r.sell_signal),
                             "atr24": float(r.atr24), "adv20": float(r.adv20) if pd.notna(r.adv20) else 0.0}
                       for sym, r in rows.items()}

    daily = pd.DataFrame(daily_marks).drop_duplicates("session", keep="last")
    daily["return"] = daily.equity.pct_change().fillna(daily.equity.iloc[0] / INITIAL - 1)
    trades = pd.DataFrame(closed)
    return daily, trades


def summarize(daily: pd.DataFrame, trades: pd.DataFrame, model: str, cost_bps: float) -> dict:
    eq = daily.equity.to_numpy(float)
    ret = daily["return"].to_numpy(float)
    dd = eq / np.maximum.accumulate(np.r_[INITIAL, eq])[1:] - 1
    gain = trades.loc[trades.net_pnl > 0, "net_pnl"].sum() if len(trades) else 0
    loss = -trades.loc[trades.net_pnl < 0, "net_pnl"].sum() if len(trades) else 0
    return {"model": model, "friction_bps_per_side": cost_bps,
            "start": str(daily.session.iloc[0]), "end": str(daily.session.iloc[-1]),
            "sessions": int(len(daily)), "final_equity": float(eq[-1]),
            "total_return_pct": float((eq[-1] / INITIAL - 1) * 100),
            "max_drawdown_pct": float(dd.min() * 100),
            "daily_sharpe_zero_rf": float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(252)) if np.std(ret, ddof=1) else 0,
            "closed_trades": int(len(trades)),
            "win_rate_pct": float((trades.net_pnl > 0).mean() * 100) if len(trades) else 0,
            "profit_factor": float(gain / loss) if loss else None,
            "average_gross_exposure_pct": float(daily.gross_exposure.mean() * 100),
            "peak_positions": int(daily.positions.max()),
            "session_pause_days": int(daily.session_pause_triggered.any()),
            "halted_at_6pct": bool(daily.halt_triggered.any())}


def spy_benchmark(path: Path, friction: float) -> tuple[dict, pd.DataFrame]:
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="raise")
    local = d.time.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    d = d.loc[(minute >= 570) & (minute < 960)].copy()
    local = d.time.dt.tz_convert("America/New_York")
    d["session"] = local.dt.date
    d = d.loc[(local >= START) & (local <= END)].sort_values("time")
    if d.empty:
        raise ValueError("SPY has no bars in the test window")
    entry = float(d.open.iloc[0]) * (1 + friction)
    shares = int(.75 * INITIAL / entry)
    cash = INITIAL - shares * entry
    daily = d.drop_duplicates("session", keep="last")
    values = cash + shares * daily.close.to_numpy(float)
    marks = daily[["session"]].copy()
    marks["equity"] = values
    returns = np.diff(np.r_[INITIAL, values]) / np.r_[INITIAL, values][:-1]
    peaks = np.maximum.accumulate(np.r_[INITIAL, values])[1:]
    result = {"model": "75% SPY / 25% cash", "friction_bps_per_side": friction * 10_000,
            "start": str(daily.session.iloc[0]), "end": str(daily.session.iloc[-1]),
            "sessions": int(len(daily)), "final_equity": float(values[-1]),
            "total_return_pct": float((values[-1] / INITIAL - 1) * 100),
            "max_drawdown_pct": float((values / peaks - 1).min() * 100),
            "daily_sharpe_zero_rf": float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)) if np.std(returns, ddof=1) else 0,
            "closed_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
            "average_gross_exposure_pct": float(shares * daily.close.mean() / values.mean() * 100),
            "peak_positions": 1, "halted_at_6pct": False}
    return result, marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bars_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/declared_portfolio"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.bars_dir.glob("*_30Min.csv")
                   if p.stem.split("_")[0] not in {"SPY", "QQQ"})
    symbols = {p.stem.split("_")[0]: p for p in files}
    unknown = set(symbols) - set(SECTOR)
    if unknown:
        print(f"Ignoring symbols without sector mapping: {sorted(unknown)}")
    symbols = {s: p for s, p in symbols.items() if s in SECTOR}
    if set(symbols) != set(SECTOR):
        raise ValueError(f"Expected fixed 17-name sample; missing={sorted(set(SECTOR)-set(symbols))}; extra={sorted(set(symbols)-set(SECTOR))}")
    summaries = []
    provenance = {"source": "Alpaca SIP 30-minute OHLCV; source bars not redistributed",
                  "test_start": str(START.date()), "test_end": str(END.date()),
                  "regular_session": "09:30-16:00 America/New_York",
                  "warmup_start": "2026-06-01", "symbols": sorted(symbols),
                  "cost_cases_bps_per_side": [5, 10],
                  "adaptive_vs_fixed": "Same signal engine/configuration; adaptation alone differs",
                  "limits": {"max_positions": 5, "max_name_weight": .15,
                             "max_gross_weight": .75, "max_sector_weight": .20,
                             "max_technology_cluster_weight": .25,
                             "planned_name_loss": .005, "aggregate_planned_loss": .02,
                             "stop_atr_multiple": 2},
                  "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list(symbols.values()) + [args.bars_dir / "SPY_30Min.csv"]}}
    for adaptive, label in ((True, "Adaptive"), (False, "Fixed baseline")):
        signal_data = {s: load_symbol(p, adaptive) for s, p in symbols.items()}
        for bps in (5, 10):
            daily, trades = simulate(signal_data, bps / 10_000)
            summaries.append(summarize(daily, trades, label, bps))
            suffix = f"{label.lower().replace(' ', '_')}_{bps}bps"
            daily.to_csv(args.output_dir / f"{suffix}_daily.csv", index=False)
            trades.to_csv(args.output_dir / f"{suffix}_trades.csv", index=False)
            # Compound close-to-close daily returns into calendar-month returns.
            daily["month"] = pd.to_datetime(daily.session).dt.to_period("M").astype(str)
            monthly = daily.groupby("month", sort=True).equity.last()
            prior = pd.concat([pd.Series([INITIAL]), monthly.reset_index(drop=True).iloc[:-1]], ignore_index=True)
            pd.DataFrame({"month": monthly.index, "return_pct": (monthly.to_numpy() / prior.to_numpy() - 1) * 100,
                          "month_end_equity": monthly.to_numpy()}).to_csv(
                              args.output_dir / f"{suffix}_monthly.csv", index=False)
    for bps in (5, 10):
        benchmark, marks = spy_benchmark(args.bars_dir / "SPY_30Min.csv", bps / 10_000)
        summaries.append(benchmark)
        if bps == 5:
            marks.to_csv(args.output_dir / "spy_5bps_daily.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
