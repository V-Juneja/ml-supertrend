"""ML Supertrend research signal engine.

Online parameter adaptation from simulated long/short probes, batches and context memory.
Derived from the Juneja Street research implementation. This module has no price download,
order submission or portfolio execution layer. Pine/Python parity is not established.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================
#  CONFIGURATION  (mirrors every Pine input, same defaults)
# =============================================================


@dataclass
class SupertrendConfig:
    # -- Group 1: Signal Mode ---------------------------------
    signal_mode: str = "Reversal"  # "Reversal" | "Breakout"
    require_new_extreme: bool = True
    min_bars_between_signals: int = 10

    # -- Group 2: Volatility Envelope -------------------------
    sensitivity: int = 30
    atr_period: int = 24
    multiplier: float = 1.4
    source_type: str = "hlcc4"
    use_atr: bool = True

    # -- Group 3: Momentum Filter -----------------------------
    enable_rsi: bool = True
    rsi_len: int = 14
    rsi_lookback_top: int = 50
    rsi_lookback_bot: int = 50

    # -- Group 4: Flow Analysis -------------------------------
    vol_lookback: int = 3
    vol_multiplier: float = 1.2
    require_vol_spike: bool = False

    # -- Group 5: Signal Quality ------------------------------
    enable_major_levels_only: bool = False
    major_level_threshold: float = 4.5

    # -- Group 6: Master Dial ---------------------------------
    dial_k: int = 10
    enable_bin_learning: bool = True
    enable_tick_pressure: bool = True

    # -- Group 7: Auto-Tune Engine ----------------------------
    enable_adaptive: bool = True
    use_faint_cycle: bool = True
    lock_atr_bands: bool = True

    # -- Group 8: Optimizer -----------------------------------
    learn_enabled: bool = True
    learn_rate: float = 0.25
    roll_n: int = 30
    win_hi: float = 0.62
    win_lo: float = 0.38
    use_atr_norm: bool = True
    smooth_alpha: float = 0.35
    param_update_every: int = 3
    deadband_mult: float = 0.01
    deadband_len: float = 0.25
    quant_step_mult: float = 0.05
    quant_step_stop: float = 0.02
    quant_step_tp: float = 0.02
    quant_step_brk: float = 0.01
    revert_every: int = 200
    revert_step: float = 0.01
    pnl_cap_usd: float = 750.0

    # -- Group 9: Risk Guard ----------------------------------
    max_trades_per_session: int = 20
    max_session_loss: float = -1000.0
    base_cooldown_bars: int = 5
    max_consecutive_losses: int = 3
    enable_dynamic_cooldown: bool = True
    apply_risk_to_faint: bool = False

    # -- Group 10: Context Memory -----------------------------
    use_cognitive_map: bool = True
    map_grid_x: int = 8
    map_grid_y: int = 8
    map_sigma: float = 0.8
    map_half_life: int = 50
    map_conf_scale: int = 20
    map_weight_max: float = 0.65
    vol_max_ratio: float = 2.5

    # -- Group 11: Decay Traces -------------------------------
    use_stm: bool = True
    stm_size: int = 30
    stm_decay: float = 0.02
    stm_consolidate_every: int = 200
    stm_mae_tail_thr: float = 0.8
    stm_tail_tighten_cap: float = 0.02

    # -- Group 13: Display & Sizing ---------------------------
    rsi_top: int = 70
    rsi_bot: int = 30
    inp_fast_len: int = 9
    inp_slow_len: int = 21
    inp_breakout_buf: float = 0.20
    inp_stop_mult: float = 1.00
    inp_tp_mult: float = 2.00
    trade_size_usd: float = 1000.0


# =============================================================
#  UTILITY FUNCTIONS
# =============================================================


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def quantize(x: float, step: float) -> float:
    if step == 0:
        return x
    return round(x / step) * step


def rma(series: np.ndarray, length: int) -> np.ndarray:
    """Wilder's RMA — Pine's ta.rma(). alpha = 1/length."""
    length = max(1, length)
    alpha = 1.0 / length
    result = np.full(len(series), np.nan)
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        prev = result[i - 1] if i > 0 else np.nan
        result[i] = series[i] if np.isnan(prev) else prev + alpha * (series[i] - prev)
    return result


def ema(series: np.ndarray, length: int) -> np.ndarray:
    """Standard EMA. alpha = 2/(length+1)."""
    length = max(1, length)
    alpha = 2.0 / (length + 1.0)
    result = np.full(len(series), np.nan)
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        prev = result[i - 1] if i > 0 else np.nan
        result[i] = (
            series[i] if np.isnan(prev) else alpha * series[i] + (1.0 - alpha) * prev
        )
    return result


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.full(len(high), np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    return tr


def atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int,
    use_rma: bool = True,
) -> np.ndarray:
    tr = true_range(high, low, close)
    return rma(tr, length) if use_rma else ema(tr, length)


def rsi_series(close: np.ndarray, length: int) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)

    # Avoid RuntimeWarning from eager evaluation in np.where(avg_gain / avg_loss)
    # by using masked division first, then applying the zero-loss convention.
    rs = np.full_like(avg_gain, np.nan, dtype=float)
    valid = (avg_loss != 0) & np.isfinite(avg_loss) & np.isfinite(avg_gain)
    np.divide(avg_gain, avg_loss, out=rs, where=valid)
    rs = np.where(avg_loss == 0, np.inf, rs)

    return 100.0 - (100.0 / (1.0 + rs))


def get_source(df: pd.DataFrame, source_type: str) -> np.ndarray:
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    src_map = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "hl2": (h + l) / 2,
        "hlc3": (h + l + c) / 3,
        "ohlc4": (o + h + l + c) / 4,
        "hlcc4": (h + l + c + c) / 4,
    }
    return src_map.get(source_type, c)


# =============================================================
#  SUPERTREND ENGINE  (stateful, bar-by-bar)
# =============================================================


class SupertrendEngine:
    """
    Stateful single-bar-at-a-time SuperTrend.
    Mirrors Pine's getSupertrend_var() exactly including
    the ratchet (one-directional band movement) and trend flip.
    """

    def __init__(self, use_atr: bool = True):
        self.use_atr = use_atr
        self.trend: int = 1
        self.upper: float = math.nan
        self.lower: float = math.nan
        self._tr_smooth: float = math.nan

    def step(
        self,
        src: float,
        high: float,
        low: float,
        close_prev: float,
        close: float,
        mult: float,
        length: int,
    ) -> Tuple[int, float, float]:
        """
        Process one bar.
        Returns (trend, upper_band, lower_band)
          upper_band = bullish floor  (active in uptrend)
          lower_band = bearish ceiling (active in downtrend)
        Note: naming follows Pine convention where "upper" is the
        floor that ratchets up, "lower" is the ceiling that ratchets down.
        """
        # True range
        if math.isnan(close_prev):
            tr = high - low
        else:
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))

        # Smooth TR (RMA or EMA)
        L = max(1, length)
        alpha = (1.0 / L) if self.use_atr else (2.0 / (L + 1.0))
        if math.isnan(self._tr_smooth):
            self._tr_smooth = tr
        else:
            self._tr_smooth = self._tr_smooth + alpha * (tr - self._tr_smooth)

        # Raw bands
        raw_upper = src - mult * self._tr_smooth  # bullish floor
        raw_lower = src + mult * self._tr_smooth  # bearish ceiling

        # Ratchet
        prev_upper = self.upper
        prev_lower = self.lower

        if math.isnan(prev_upper):
            new_upper = raw_upper
            new_lower = raw_lower
        else:
            # Pine: upper := close[1] > upper[1] ? max(upper, upper[1]) : upper
            new_upper = (
                max(raw_upper, prev_upper) if close_prev > prev_upper else raw_upper
            )
            # Pine: lower := close[1] < lower[1] ? min(lower, lower[1]) : lower
            new_lower = (
                min(raw_lower, prev_lower) if close_prev < prev_lower else raw_lower
            )

        # Trend flip
        if self.trend == -1 and close > new_lower:
            self.trend = 1
        elif self.trend == 1 and close < new_upper:
            self.trend = -1

        self.upper = new_upper
        self.lower = new_lower
        return self.trend, new_upper, new_lower


# =============================================================
#  REGIME DETECTION
# =============================================================


def calc_hurst(close: np.ndarray, length: int = 64) -> float:
    """Hurst exponent via rescaled range. H>0.5 = trending, H<0.5 = mean-reverting."""
    length = max(length, 16)
    if len(close) < length:
        return 0.5
    seg = close[-length:]
    mean = np.mean(seg)
    dev = seg - mean
    cum = np.cumsum(dev)
    R = cum.max() - cum.min()
    S = np.std(seg)
    if S == 0:
        return 0.5
    return clamp(math.log(R / S) / math.log(length), 0.0, 1.0)


def calc_entropy(close: np.ndarray, length: int = 64) -> float:
    """Shannon entropy of up/down move sequence. High = random, Low = directional."""
    length = max(length, 16)
    if len(close) < length + 1:
        return 0.5
    seg = close[-(length + 1) :]
    diffs = np.diff(seg)
    ups = float(np.sum(diffs > 0))
    dns = float(np.sum(diffs < 0))
    tot = ups + dns
    if tot == 0:
        return 0.0
    p1 = ups / tot
    p2 = dns / tot
    H = 0.0
    if p1 > 0:
        H -= p1 * math.log(p1)
    if p2 > 0:
        H -= p2 * math.log(p2)
    return clamp(H / math.log(2), 0.0, 1.0)


def calc_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> float:
    """Returns scalar ADX for the most recent bar."""
    if len(close) < length * 2 + 1:
        return 25.0
    tr_arr = true_range(high, low, close)
    n = len(close)

    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        dm_plus[i] = up if up > down and up > 0 else 0.0
        dm_minus[i] = down if down > up and down > 0 else 0.0

    atr14 = rma(tr_arr, length)
    di_plus = 100.0 * rma(dm_plus, length) / np.maximum(atr14, 1e-10)
    di_minus = 100.0 * rma(dm_minus, length) / np.maximum(atr14, 1e-10)
    dx = 100.0 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx_arr = rma(dx, length)

    v = adx_arr[-1]
    return float(v) if not np.isnan(v) else 25.0


def detect_market_regime(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """
    Composite regime score in [0, 1].
    1.0 = strongly trending. 0.0 = strongly ranging/choppy.
    Weights: 35% Hurst, 35% inverse-entropy, 30% ADX.
    """
    adx_val = calc_adx(high, low, close, 14)
    trend_strength = clamp(adx_val / 50.0, 0.0, 1.0)
    H = calc_hurst(close, 64)
    Ent = calc_entropy(close, 64)
    score = (
        0.35 * (1 - Ent) + 0.35 * clamp((H - 0.5) * 2, 0.0, 1.0) + 0.30 * trend_strength
    )
    return clamp(score, 0.0, 1.0)


# =============================================================
#  ROLLING WINDOW
# =============================================================


class RollingWindow:
    def __init__(self, maxlen: int):
        self._d: deque = deque(maxlen=maxlen)

    def push(self, v: float):
        self._d.append(v)

    def to_array(self) -> np.ndarray:
        return np.array(self._d, dtype=float)

    def __len__(self) -> int:
        return len(self._d)

    def win_rate_avg(self) -> Tuple[float, float]:
        a = self.to_array()
        if len(a) == 0:
            return 0.0, 0.0
        return float(np.mean(a > 0)), float(np.mean(a))

    def sortino(self) -> Optional[float]:
        a = self.to_array()
        if len(a) == 0:
            return None
        mean = np.mean(a)
        neg = a[a < 0]
        if len(neg) == 0:
            return None
        dd = float(np.std(neg))
        return float(mean / dd) if dd > 0 else None

    def avg(self) -> float:
        a = self.to_array()
        return float(np.mean(a)) if len(a) > 0 else 0.0


# =============================================================
#  SCORECARD
# =============================================================


@dataclass
class Scorecard:
    returns: List[float] = field(default_factory=list)
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    def update(self, ret: float, is_win: bool):
        self.returns.append(ret)
        if len(self.returns) > 100:
            self.returns.pop(0)
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_wins = 0
            self.consecutive_losses += 1
        if len(self.returns) >= 30:
            a = np.array(self.returns)
            mean = float(np.mean(a))
            std = float(np.std(a))
            self.sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
            neg = a[a < 0]
            pos = a[a >= 0]
            if len(neg) > 0:
                dd = float(np.std(neg))
                self.sortino = mean / dd * math.sqrt(252) if dd > 0 else 0.0
                self.avg_loss = float(np.mean(neg))
            if len(pos) > 0:
                self.avg_win = float(np.mean(pos))
            self.profit_factor = (
                abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0
            )
            cum = np.cumsum(a)
            peak = np.maximum.accumulate(cum)
            self.max_drawdown = float(np.max(peak - cum))


# =============================================================
#  DATA CLASSES
# =============================================================


@dataclass
class ProbeEntry:
    entry: float
    entry_bar: int
    direction: int  # +1 long, -1 short
    active: bool
    atr_entry: float


@dataclass
class DecayTrace:
    atr_ret: float
    mae_atr: float
    regime: float
    vol_norm: float
    energy: float
    direction: int


@dataclass
class GridCell:
    mean_atr_ret: float = 0.0
    down_ewm: float = 0.0
    count: int = 0
    conf: float = 0.0
    d_mult: float = 0.0
    d_len: float = 0.0
    d_stop: float = 0.0
    d_tp: float = 0.0
    d_brk: float = 0.0
    last_upd: int = -1


@dataclass
class BatchSample:
    atr_ret: float
    mae_atr: float
    pnl_usd: float
    conf: float


# =============================================================
#  EXECUTION LEDGER
# =============================================================


class ExecLedger:
    def __init__(self):
        self._success: deque = deque(maxlen=100)
        self._slip: deque = deque(maxlen=100)

    def update(
        self, expected: float, actual: float, success: bool
    ) -> Tuple[float, float]:
        slip = abs(actual - expected) / expected if expected != 0 else 0.0
        self._success.append(1.0 if success else 0.0)
        self._slip.append(slip)
        if len(self._success) >= 20:
            sr = float(np.mean(list(self._success)[-20:]))
            avg = float(np.mean(list(self._slip)[-20:]))
            return sr, avg
        return 0.0, 0.0


# =============================================================
#  MASTER DIAL -> all derived constants
# =============================================================


def master_params(K: int) -> Tuple:
    """
    Single meta-knob K in [1,20].
    Returns: N, S, Wmin, w_batch, cap_mult, cap_len, cap_stop,
             cap_tp, cap_brk, db_mult, db_len, alpha_eff
    """
    k = (float(K) - 1.0) / 19.0
    N = clamp(int(round(20.0 - 16.0 * k)), 4, 20)
    S = max(1, int(round(N * (1.0 - 0.9 * k))))
    Wmin = max(0.0, float(N) * (0.80 - 0.40 * k))
    w_batch = 0.25 + 0.35 * k
    cap_m = 0.02 + 0.06 * k
    cap_l = 0.5 + 1.5 * k
    cap_st = 0.01 + 0.04 * k
    cap_tp = 0.01 + 0.03 * k
    cap_bk = 0.005 + 0.025 * k
    db_m = max(0.003, 0.01 * (1.0 - 0.6 * k))
    db_l = max(0.10, 0.25 * (1.0 - 0.6 * k))
    alpha = clamp(0.25 + 0.40 * k, 0.20, 0.65)
    return N, S, Wmin, w_batch, cap_m, cap_l, cap_st, cap_tp, cap_bk, db_m, db_l, alpha


# =============================================================
#  MAIN INDICATOR
# =============================================================


class MLSupertrend:
    """
    Full Python port of Machine Learning Supertrend [Aslan].

    Usage:
        st      = MLSupertrend(SupertrendConfig())
        results = st.run(ohlcv_df)

    Output columns added to the DataFrame:
        trend_plot, upper_band, lower_band
        trend_signal
        buy_signal, sell_signal
        adaptive_multiplier, adaptive_atr_period
        adaptive_stop_mult, adaptive_tp_mult, adaptive_breakout_buf
        regime_score
        long_win_rate, short_win_rate
        long_cum_usd, short_cum_usd
    """

    def __init__(self, cfg: SupertrendConfig = None):
        self.cfg = cfg or SupertrendConfig()
        self._reset()

    # ----------------------------------------------------------
    #  State initialisation
    # ----------------------------------------------------------

    def _reset(self):
        c = self.cfg

        # Adaptive parameters — start at base values
        self.adap_multiplier = float(c.multiplier)
        self.adap_atr_period = float(c.atr_period)
        self.adap_sensitivity = float(c.sensitivity)
        self.adap_stop_mult = float(c.inp_stop_mult)
        self.adap_tp_mult = float(c.inp_tp_mult)
        self.adap_breakout_buf = float(c.inp_breakout_buf)

        # Two SuperTrend engines (plot = visual / signal = adaptive)
        self._st_plot = SupertrendEngine(c.use_atr)
        self._st_signal = SupertrendEngine(c.use_atr)

        # Test matrix probes
        self._probes_long: List[ProbeEntry] = []
        self._probes_short: List[ProbeEntry] = []
        self._long_total = self._long_wins = 0
        self._short_total = self._short_wins = 0
        self._long_cum_usd = self._short_cum_usd = 0.0
        self._long_gp = self._long_gl = 0.0
        self._short_gp = self._short_gl = 0.0
        self._long_best: Optional[float] = None
        self._long_worst: Optional[float] = None
        self._short_best: Optional[float] = None
        self._short_worst: Optional[float] = None

        # Rolling windows
        self._long_roll_usd = RollingWindow(c.roll_n)
        self._short_roll_usd = RollingWindow(c.roll_n)
        self._long_roll_atr = RollingWindow(c.roll_n)
        self._short_roll_atr = RollingWindow(c.roll_n)

        # Scorecards
        self._score_long = Scorecard()
        self._score_short = Scorecard()

        # Execution ledger
        self._exec = ExecLedger()

        # Decay trace buffer
        self._decay_buf: List[DecayTrace] = []

        # Regime grid
        n_cells = c.map_grid_x * c.map_grid_y
        self._grid: List[GridCell] = [GridCell() for _ in range(n_cells)]

        # Micro-batch buffers
        self._batch_l: List[BatchSample] = []
        self._batch_s: List[BatchSample] = []

        # Batch carry-over state
        self._fired_l = self._fired_s = False
        self._mu_bl = self._mu_bs = 0.0
        self._dM_bl = self._dL_bl = self._dSt_bl = self._dTP_bl = self._dBk_bl = 0.0
        self._dM_bs = self._dL_bs = self._dSt_bs = self._dTP_bs = self._dBk_bs = 0.0

        # Optimizer bookkeeping
        self._last_update_bar = -999
        self._bars_since_upd = 0

        # Signal state
        self._top_flag = 0
        self._bot_flag = 0
        self._last_signal_bar = -99999

        # Regime
        self._regime_score = 0.5
        self._last_reg_bar = -1

        # Risk guard
        self._cooldown_until = 0
        self._trades_session = 0
        self._session_pnl = 0.0
        self._consec_losses = 0

        # Pressure log
        self._pressure_log: deque = deque(maxlen=20)

        # Signal history queues (max 500 bars)
        self._high_q: deque = deque(maxlen=500)
        self._low_q: deque = deque(maxlen=500)
        self._rsi_q: deque = deque(maxlen=500)
        self._vol_q: deque = deque(maxlen=500)

    # ----------------------------------------------------------
    #  Grid helpers
    # ----------------------------------------------------------

    def _bucket(self, x: float, bins: int) -> int:
        return clamp(int(math.floor(x * bins)), 0, bins - 1)

    def _cell_idx(self, ix: int, iy: int) -> int:
        return ix * self.cfg.map_grid_y + iy

    @staticmethod
    def _gauss(dx: int, dy: int, sigma: float) -> float:
        return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    def _grid_update(
        self,
        ix: int,
        iy: int,
        atr_ret: float,
        d_mult: float,
        d_len: float,
        d_stop: float,
        d_tp: float,
        d_brk: float,
        bar_idx: int,
    ):
        c = self.cfg
        idx = self._cell_idx(ix, iy)
        if not (0 <= idx < len(self._grid)):
            return
        cell = self._grid[idx]
        alpha = 1.0 - math.exp(-math.log(2.0) / max(1, c.map_half_life))
        cell.mean_atr_ret = cell.mean_atr_ret * (1 - alpha) + atr_ret * alpha
        dn = atr_ret * atr_ret if atr_ret < 0 else 0.0
        cell.down_ewm = cell.down_ewm * (1 - alpha) + dn * alpha
        cell.count += 1
        cell.conf = 1.0 - math.exp(-cell.count / max(1, c.map_conf_scale))
        cell.d_mult = cell.d_mult * (1 - alpha) + d_mult * alpha
        cell.d_len = cell.d_len * (1 - alpha) + d_len * alpha
        cell.d_stop = cell.d_stop * (1 - alpha) + d_stop * alpha
        cell.d_tp = cell.d_tp * (1 - alpha) + d_tp * alpha
        cell.d_brk = cell.d_brk * (1 - alpha) + d_brk * alpha
        cell.last_upd = bar_idx

    def _grid_recommend(self, ix: int, iy: int) -> Tuple:
        c = self.cfg
        wsum = wconf = 0.0
        m_mult = m_len = m_stop = m_tp = m_brk = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = clamp(ix + dx, 0, c.map_grid_x - 1)
                ny = clamp(iy + dy, 0, c.map_grid_y - 1)
                idx = self._cell_idx(nx, ny)
                if not (0 <= idx < len(self._grid)):
                    continue
                cell = self._grid[idx]
                if cell.count == 0:
                    continue
                w = self._gauss(dx, dy, c.map_sigma)
                wsum += w
                wconf += w * cell.conf
                m_mult += w * cell.conf * cell.d_mult
                m_len += w * cell.conf * cell.d_len
                m_stop += w * cell.conf * cell.d_stop
                m_tp += w * cell.conf * cell.d_tp
                m_brk += w * cell.conf * cell.d_brk
        if wsum == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        den = wconf if wconf != 0 else wsum
        grid_conf = clamp(wconf / wsum, 0.0, 1.0)
        return (
            m_mult / den,
            m_len / den,
            m_stop / den,
            m_tp / den,
            m_brk / den,
            grid_conf,
        )

    # ----------------------------------------------------------
    #  Decay trace helpers
    # ----------------------------------------------------------

    def _decay_push(
        self, atr_ret: float, mae_atr: float, regime: float, vol: float, direction: int
    ):
        self._decay_buf.append(
            DecayTrace(atr_ret, mae_atr, regime, vol, 1.0, direction)
        )
        while len(self._decay_buf) > self.cfg.stm_size:
            self._decay_buf.pop(0)

    def _decay_step(self):
        kept = []
        for t in self._decay_buf:
            t.energy *= 1.0 - self.cfg.stm_decay
            if t.energy >= 0.05:
                kept.append(t)
        self._decay_buf = kept

    def _decay_tail_feedback(self) -> float:
        n = len(self._decay_buf)
        if n == 0:
            return 0.0
        thr = self.cfg.stm_mae_tail_thr
        sum_w = sum(t.energy for t in self._decay_buf)
        sum_tail = sum(t.energy for t in self._decay_buf if t.mae_atr > thr)
        frac = sum_tail / sum_w if sum_w > 0 else 0.0
        k = clamp((frac - 0.25) / 0.75, 0.0, 1.0)
        return -self.cfg.stm_tail_tighten_cap * k

    # ----------------------------------------------------------
    #  Probe helpers
    # ----------------------------------------------------------

    def _count_active(self, book: List[ProbeEntry]) -> int:
        return sum(1 for p in book if p.active)

    def _open_probe(
        self,
        book: List[ProbeEntry],
        direction: int,
        entry_px: float,
        atr_entry: float,
        bar_idx: int,
    ):
        for p in book:
            if not p.active:
                p.entry = entry_px
                p.entry_bar = bar_idx
                p.direction = direction
                p.active = True
                p.atr_entry = atr_entry
                return
        book.append(ProbeEntry(entry_px, bar_idx, direction, True, atr_entry))

    def _close_matured(
        self,
        book: List[ProbeEntry],
        direction: int,
        low_win: float,
        high_win: float,
        bar_idx: int,
        close: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Finds oldest active probe with age >= 5 bars, closes it.
        Returns (usd_pnl, atr_return, mae_atr) or (None, None, None).
        """
        HOLD = 5
        best_idx = None
        oldest = int(1e18)
        for i, p in enumerate(book):
            if p.active and (bar_idx - p.entry_bar + 1) >= HOLD:
                if p.entry_bar < oldest:
                    oldest = p.entry_bar
                    best_idx = i
        if best_idx is None:
            return None, None, None

        p = book[best_idx]
        ret = (close - p.entry) * p.direction
        pct = ret / p.entry if p.entry != 0 else 0.0
        usd = pct * self.cfg.trade_size_usd
        atr_ret = ret / p.atr_entry if p.atr_entry != 0 else 0.0

        # MAE: worst adverse move during hold (approximated from high/low window)
        if direction == 1:
            mae_price = max(0.0, p.entry - low_win)
        else:
            mae_price = max(0.0, high_win - p.entry)
        mae_atr = mae_price / p.atr_entry if p.atr_entry != 0 else 0.0

        p.active = False

        # Execution ledger feedback -> adjust params if fill quality is poor
        fill_rate, slippage = self._exec.update(p.entry, close, usd > 0)
        if fill_rate < 0.6 and self.adap_stop_mult < 2.5:
            self.adap_stop_mult *= 1.02
        if slippage > 0.002 and self.adap_multiplier < 4.0:
            self.adap_multiplier *= 1.01

        return usd, atr_ret, mae_atr

    # ----------------------------------------------------------
    #  Optimizer proposals
    # ----------------------------------------------------------

    def _pressure_bias(self) -> float:
        if len(self._pressure_log) >= 5:
            return float(np.mean(list(self._pressure_log)))
        return 0.0

    def _proposals(
        self,
        win_rate: float,
        avg_usd: float,
        avg_atr: float,
        sortino: Optional[float],
        avg_mae_atr: float,
        pf: float,
        avg_win: float,
        avg_loss: float,
        sample_count: int,
        cur_mult: float,
        cur_len: float,
    ) -> Tuple[float, float, float, float, float]:
        c = self.cfg
        dM = dL = dSt = dTP = dBk = 0.0

        if not (c.learn_enabled and c.enable_adaptive and sample_count > 0):
            return dM, dL, dSt, dTP, dBk

        conf = 1.0 / math.sqrt(max(1.0, float(sample_count)))
        step = c.learn_rate * conf
        bias = self._pressure_bias()

        neg = (avg_atr < 0) if c.use_atr_norm else (avg_usd < 0)
        pos = (avg_atr > 0) if c.use_atr_norm else (avg_usd > 0)

        weak = win_rate < c.win_lo and neg
        strong = win_rate > c.win_hi and pos

        if abs(bias) > 0.2:
            step *= 1.0 + abs(bias) * 0.5

        if weak:
            dM += 0.08 * step
            dL += cur_len * 0.05 * step
        if strong:
            dM -= 0.05 * step
            dL -= cur_len * 0.03 * step

        if avg_mae_atr > 0.60:
            dM += 0.02 * step

        if sortino is not None:
            if sortino < 0:
                dM += 0.04 * step
            elif sortino > 1.2:
                dM -= 0.03 * step

        if win_rate > c.win_hi and 0 < pf < 1.2:
            dTP += 0.02 * step
        if win_rate < c.win_lo and avg_loss > avg_win * 1.2:
            dSt -= 0.03 * step
        elif win_rate < c.win_lo:
            dSt += 0.02 * step

        if self._regime_score >= 0.7 and pos:
            dBk -= 0.02 * step
        elif self._regime_score <= 0.3 and neg:
            dBk += 0.02 * step

        score = self._score_long if avg_atr > 0 else self._score_short
        if score.sharpe < 0.5:
            dM += 0.02 * step
        if score.sortino > 1.5:
            dM -= 0.02 * step
        if score.consecutive_losses >= 3:
            dSt += 0.03 * step

        return dM, dL, dSt, dTP, dBk

    # ----------------------------------------------------------
    #  Micro-batch fire
    # ----------------------------------------------------------

    def _batch_fire(
        self,
        buf: List[BatchSample],
        N: int,
        stride: int,
        Wmin: float,
        cap_m: float,
        cap_l: float,
        cap_st: float,
        cap_tp: float,
        cap_bk: float,
    ) -> Tuple[bool, float, float, float, float, float, float]:
        if len(buf) < N:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        sum_w = sum_ret = w_neg = sum_neg_var = 0.0
        sum_mae = gp_w = gl_w = 0.0

        for e in buf[:N]:
            wc = clamp(e.conf, 0.0, 1.0)
            wq = 1.0 / (1.0 + max(0.0, e.mae_atr))
            w = wc * wq
            sum_w += w
            sum_ret += w * e.atr_ret
            sum_mae += w * e.mae_atr
            if e.atr_ret < 0:
                w_neg += w
                sum_neg_var += w * e.atr_ret**2
            if e.pnl_usd > 0:
                gp_w += w * e.pnl_usd
            elif e.pnl_usd < 0:
                gl_w += w * (-e.pnl_usd)

        mu = sum_ret / sum_w if sum_w > 0 else 0.0
        sr = (
            mu / math.sqrt(sum_neg_var / w_neg)
            if w_neg > 0 and sum_neg_var > 0
            else None
        )
        pf = gp_w / gl_w if gl_w > 0 else (9.99 if gp_w > 0 else 0.0)
        conf_f = clamp(sum_w / Wmin, 0.0, 1.5) if Wmin > 0 else 1.0
        avg_mae = sum_mae / sum_w if sum_w > 0 else 0.0

        good = mu > 0 and pf > 1.15 and (sr is None or sr > 0.8)
        bad = mu < 0 or pf < 0.95 or (sr is not None and sr < 0.2)

        dM = dL = dSt = dTP = dBk = 0.0
        if good:
            dM = -cap_m * conf_f
            dL = -cap_l * conf_f
            dSt = -cap_st * conf_f
            dTP = +cap_tp * 0.5 * conf_f
            dBk = -cap_bk * conf_f
        if bad:
            dM = +cap_m * conf_f
            dL = +cap_l * conf_f
            dSt = (-cap_st if avg_mae > 0.6 else +0.5 * cap_st) * conf_f
            dTP = -0.3 * cap_tp * conf_f
            dBk = +cap_bk * conf_f

        for _ in range(min(stride, len(buf))):
            buf.pop(0)

        return True, dM, dL, dSt, dTP, dBk, mu

    # ----------------------------------------------------------
    #  Risk guard
    # ----------------------------------------------------------

    def _can_trade(self, bar_idx: int) -> bool:
        c = self.cfg
        return (
            bar_idx > self._cooldown_until
            and self._trades_session < c.max_trades_per_session
            and self._session_pnl > c.max_session_loss
            and self._consec_losses < c.max_consecutive_losses
        )

    def _apply_risk(self, pnl: float, bar_idx: int):
        c = self.cfg
        self._trades_session += 1
        self._session_pnl += pnl
        if pnl < 0:
            self._consec_losses += 1
            loss_mag = abs(pnl)
            if c.enable_dynamic_cooldown:
                scale = loss_mag / (c.trade_size_usd * 0.02)
                cd_bars = int(min(50, c.base_cooldown_bars * (1 + scale)))
            else:
                cd_bars = c.base_cooldown_bars
            self._cooldown_until = bar_idx + cd_bars
        else:
            self._consec_losses = 0

    # ----------------------------------------------------------
    #  Vol normalisation
    # ----------------------------------------------------------

    def _vol_norm(self, atr_arr: np.ndarray, i: int) -> float:
        c = self.cfg
        if i < 14 or np.isnan(atr_arr[i]):
            return 0.5
        a = atr_arr[i]
        start = max(0, i - 100)
        av = float(np.nanmean(atr_arr[start : i + 1]))
        r = a / av if av > 0 else 0.0
        return clamp(r / c.vol_max_ratio, 0.0, 1.0)

    # ----------------------------------------------------------
    #  RSI lookback helpers
    # ----------------------------------------------------------

    @staticmethod
    def _bars_since_below(val: float, hist: list) -> int:
        for i, v in enumerate(reversed(hist)):
            if v < val:
                return i
        return 9999

    @staticmethod
    def _bars_since_above(val: float, hist: list) -> int:
        for i, v in enumerate(reversed(hist)):
            if v > val:
                return i
        return 9999

    # ==========================================================
    #  MAIN RUN
    # ==========================================================

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full indicator on an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume
            Optionally: datetime (or DatetimeIndex) for session reset.

        Returns
        -------
        pd.DataFrame
            Original df plus indicator columns (see class docstring).
        """
        self._reset()
        c = self.cfg

        opens = df["open"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        n = len(df)

        # Pre-compute full-series arrays
        src_arr = get_source(df, c.source_type)
        rsi_arr = rsi_series(closes, c.rsi_len)
        atr_base = atr_series(highs, lows, closes, c.atr_period, c.use_atr)

        # Date handling for session resets
        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"]).dt.date.values
        elif hasattr(df.index, "date"):
            dates = df.index.date
        else:
            dates = np.arange(n)

        # Output arrays
        trend_plot_arr = np.ones(n, dtype=int)
        upper_band_arr = np.full(n, np.nan)
        lower_band_arr = np.full(n, np.nan)
        trend_sig_arr = np.ones(n, dtype=int)
        buy_arr = np.zeros(n, dtype=bool)
        sell_arr = np.zeros(n, dtype=bool)
        adap_mult_arr = np.full(n, c.multiplier)
        adap_len_arr = np.full(n, float(c.atr_period))
        adap_stop_arr = np.full(n, c.inp_stop_mult)
        adap_tp_arr = np.full(n, c.inp_tp_mult)
        adap_brk_arr = np.full(n, c.inp_breakout_buf)
        regime_arr = np.full(n, 0.5)
        long_wr_arr = np.zeros(n)
        short_wr_arr = np.zeros(n)
        long_cum_arr = np.zeros(n)
        short_cum_arr = np.zeros(n)

        # Master dial constants (fixed for the whole run)
        (
            N_b,
            S_b,
            Wmin_b,
            w_bat,
            cap_m,
            cap_l,
            cap_st,
            cap_tp,
            cap_bk,
            db_m,
            db_l,
            alpha_eff,
        ) = master_params(c.dial_k)

        prev_date = None

        # ======================================================
        #  BAR LOOP
        # ======================================================
        for i in range(n):
            close = closes[i]
            high_i = highs[i]
            low_i = lows[i]
            open_i = opens[i]
            vol_i = volumes[i]
            src_i = src_arr[i]
            rsi_i = rsi_arr[i]
            atr_i = atr_base[i] if not np.isnan(atr_base[i]) else 1e-6
            close_p = closes[i - 1] if i > 0 else math.nan

            # ── Session reset ──────────────────────────────────
            cur_date = dates[i]
            if prev_date is not None and cur_date != prev_date:
                self._trades_session = 0
                self._session_pnl = 0.0
            prev_date = cur_date

            # ── Regime update every 7 bars ─────────────────────
            if i >= 65 and (i - self._last_reg_bar) >= 7:
                self._regime_score = detect_market_regime(
                    highs[: i + 1], lows[: i + 1], closes[: i + 1]
                )
                self._last_reg_bar = i
            regime_arr[i] = self._regime_score

            # ── Vol norm for grid axes ─────────────────────────
            vol_n = self._vol_norm(atr_base, i)

            # ── Plot SuperTrend (visual / locked) ──────────────
            m_plot = (
                c.multiplier
                if c.lock_atr_bands
                else clamp(self.adap_multiplier, 0.5, 5.0)
            )
            l_plot = (
                c.atr_period
                if c.lock_atr_bands
                else int(clamp(self.adap_atr_period, 5, 100))
            )
            tp, up, dn = self._st_plot.step(
                src_i, high_i, low_i, close_p, close, m_plot, l_plot
            )
            trend_plot_arr[i] = tp
            upper_band_arr[i] = up
            lower_band_arr[i] = dn

            # ── Signal SuperTrend (adaptive) ───────────────────
            m_sig = clamp(self.adap_multiplier, 0.5, 5.0)
            l_sig = int(clamp(self.adap_atr_period, 5, 100))
            ts, _, _ = self._st_signal.step(
                src_i, high_i, low_i, close_p, close, m_sig, l_sig
            )
            trend_sig_arr[i] = ts

            # ── Queues for signal filters ──────────────────────
            self._high_q.append(high_i)
            self._low_q.append(low_i)
            self._rsi_q.append(float(rsi_i) if not np.isnan(rsi_i) else 50.0)
            if vol_i > 0:
                self._vol_q.append(vol_i)

            # ── 5-bar hold window ──────────────────────────────
            h_start = max(0, i - 4)
            low_win = float(np.min(lows[h_start : i + 1]))
            high_win = float(np.max(highs[h_start : i + 1]))

            # ==================================================
            #  BACKGROUND TEST MATRIX
            # ==================================================
            if c.enable_adaptive and c.use_faint_cycle:
                # Close matured long probe
                l_usd, l_atr, l_mae = self._close_matured(
                    self._probes_long, +1, low_win, high_win, i, close
                )

                # Close matured short probe
                s_usd, s_atr, s_mae = self._close_matured(
                    self._probes_short, -1, low_win, high_win, i, close
                )

                # ── Process long result ────────────────────────
                if l_usd is not None:
                    cap_usd = clamp(l_usd, -c.pnl_cap_usd, c.pnl_cap_usd)
                    self._long_total += 1
                    if l_usd > 0:
                        self._long_wins += 1
                        self._long_gp += l_usd
                    else:
                        self._long_gl += -l_usd
                    self._long_cum_usd += l_usd
                    self._long_best = max(self._long_best or l_usd, l_usd)
                    self._long_worst = min(self._long_worst or l_usd, l_usd)
                    self._long_roll_usd.push(cap_usd)
                    self._long_roll_atr.push(l_atr)
                    self._score_long.update(l_atr, l_usd > 0)
                    if c.apply_risk_to_faint:
                        self._apply_risk(l_usd, i)
                    else:
                        self._session_pnl += l_usd
                    if c.use_stm:
                        self._decay_push(l_atr, l_mae, self._regime_score, vol_n, +1)
                    if c.enable_bin_learning:
                        ix = self._bucket(self._regime_score, c.map_grid_x)
                        iy = self._bucket(vol_n, c.map_grid_y)
                        gc = self._grid_recommend(ix, iy)[5]
                        self._batch_l.append(
                            BatchSample(
                                l_atr, l_mae, l_usd, gc if c.use_cognitive_map else 1.0
                            )
                        )

                # ── Process short result ───────────────────────
                if s_usd is not None:
                    cap_usd = clamp(s_usd, -c.pnl_cap_usd, c.pnl_cap_usd)
                    self._short_total += 1
                    if s_usd > 0:
                        self._short_wins += 1
                        self._short_gp += s_usd
                    else:
                        self._short_gl += -s_usd
                    self._short_cum_usd += s_usd
                    self._short_best = max(self._short_best or s_usd, s_usd)
                    self._short_worst = min(self._short_worst or s_usd, s_usd)
                    self._short_roll_usd.push(cap_usd)
                    self._short_roll_atr.push(s_atr)
                    self._score_short.update(s_atr, s_usd > 0)
                    if c.apply_risk_to_faint:
                        self._apply_risk(s_usd, i)
                    else:
                        self._session_pnl += s_usd
                    if c.use_stm:
                        self._decay_push(s_atr, s_mae, self._regime_score, vol_n, -1)
                    if c.enable_bin_learning:
                        ix = self._bucket(self._regime_score, c.map_grid_x)
                        iy = self._bucket(vol_n, c.map_grid_y)
                        gc = self._grid_recommend(ix, iy)[5]
                        self._batch_s.append(
                            BatchSample(
                                s_atr, s_mae, s_usd, gc if c.use_cognitive_map else 1.0
                            )
                        )

                # ── Decay traces ───────────────────────────────
                if c.use_stm:
                    self._decay_step()
                    if (
                        i > 0
                        and i % c.stm_consolidate_every == 0
                        and self._decay_buf
                        and c.use_cognitive_map
                    ):
                        d_stop_decay = self._decay_tail_feedback()
                        pos_traces = [t for t in self._decay_buf if t.atr_ret >= 0]
                        neg_traces = [t for t in self._decay_buf if t.atr_ret < 0]
                        if pos_traces:
                            best_pos = max(
                                pos_traces, key=lambda t: abs(t.atr_ret) * t.energy
                            )
                            self._grid_update(
                                self._bucket(best_pos.regime, c.map_grid_x),
                                self._bucket(best_pos.vol_norm, c.map_grid_y),
                                best_pos.atr_ret,
                                0,
                                0,
                                d_stop_decay,
                                0,
                                0,
                                i,
                            )
                        if neg_traces:
                            best_neg = max(
                                neg_traces, key=lambda t: abs(t.atr_ret) * t.energy
                            )
                            self._grid_update(
                                self._bucket(best_neg.regime, c.map_grid_x),
                                self._bucket(best_neg.vol_norm, c.map_grid_y),
                                best_neg.atr_ret,
                                0,
                                0,
                                d_stop_decay,
                                0,
                                0,
                                i,
                            )

                # ── Open new probes ────────────────────────────
                if self._count_active(self._probes_long) < 5:
                    self._open_probe(self._probes_long, +1, open_i, atr_i, i)
                if self._count_active(self._probes_short) < 5:
                    self._open_probe(self._probes_short, -1, open_i, atr_i, i)

                # ── Global optimizer proposals ─────────────────
                lc = len(self._long_roll_atr)
                sc = len(self._short_roll_atr)

                wr_l, avg_usd_l = self._long_roll_usd.win_rate_avg()
                wr_s, avg_usd_s = self._short_roll_usd.win_rate_avg()
                avg_atr_l = self._long_roll_atr.avg()
                avg_atr_s = self._short_roll_atr.avg()
                sort_l = self._long_roll_atr.sortino()
                sort_s = self._short_roll_atr.sortino()

                pf_l = (
                    self._long_gp / self._long_gl
                    if self._long_gl > 0
                    else (9.99 if self._long_gp > 0 else 0.0)
                )
                pf_s = (
                    self._short_gp / self._short_gl
                    if self._short_gl > 0
                    else (9.99 if self._short_gp > 0 else 0.0)
                )

                aw_l = self._long_gp / self._long_wins if self._long_wins > 0 else 0.0
                al_l = self._long_gl / max(1, self._long_total - self._long_wins)
                aw_s = (
                    self._short_gp / self._short_wins if self._short_wins > 0 else 0.0
                )
                al_s = self._short_gl / max(1, self._short_total - self._short_wins)

                dM_l, dL_l, dSt_l, dTP_l, dBk_l = self._proposals(
                    wr_l,
                    avg_usd_l,
                    avg_atr_l,
                    sort_l,
                    0.0,
                    pf_l,
                    aw_l,
                    al_l,
                    lc,
                    self.adap_multiplier,
                    self.adap_atr_period,
                )
                dM_s, dL_s, dSt_s, dTP_s, dBk_s = self._proposals(
                    wr_s,
                    avg_usd_s,
                    avg_atr_s,
                    sort_s,
                    0.0,
                    pf_s,
                    aw_s,
                    al_s,
                    sc,
                    self.adap_multiplier,
                    self.adap_atr_period,
                )

                net_m = (dM_l if lc > 0 else 0) + (dM_s if sc > 0 else 0)
                net_l = (dL_l if lc > 0 else 0) + (dL_s if sc > 0 else 0)
                net_st = (dSt_l if lc > 0 else 0) + (dSt_s if sc > 0 else 0)
                net_tp = (dTP_l if lc > 0 else 0) + (dTP_s if sc > 0 else 0)
                net_bk = (dBk_l if lc > 0 else 0) + (dBk_s if sc > 0 else 0)

                # Grid recommendation
                ix = self._bucket(self._regime_score, c.map_grid_x)
                iy = self._bucket(vol_n, c.map_grid_y)
                gM, gL, gSt, gTP, gBk, g_conf = self._grid_recommend(ix, iy)

                tgt_m_grd = c.multiplier * (1 + gM)
                tgt_l_grd = c.atr_period + gL
                tgt_st_grd = c.inp_stop_mult * (1 + gSt)
                tgt_tp_grd = c.inp_tp_mult * (1 + gTP)
                tgt_bk_grd = c.inp_breakout_buf * (1 + gBk)

                tgt_m_glb = self.adap_multiplier * (1 + net_m)
                tgt_l_glb = self.adap_atr_period + net_l
                tgt_st_glb = self.adap_stop_mult * (1 + net_st)
                tgt_tp_glb = self.adap_tp_mult * (1 + net_tp)
                tgt_bk_glb = self.adap_breakout_buf * (1 + net_bk)

                w_grd = clamp(g_conf, 0.0, c.map_weight_max)
                tgt_m = tgt_m_grd * w_grd + tgt_m_glb * (1 - w_grd)
                tgt_l = tgt_l_grd * w_grd + tgt_l_glb * (1 - w_grd)
                tgt_st = tgt_st_grd * w_grd + tgt_st_glb * (1 - w_grd)
                tgt_tp = tgt_tp_grd * w_grd + tgt_tp_glb * (1 - w_grd)
                tgt_bk = tgt_bk_grd * w_grd + tgt_bk_glb * (1 - w_grd)

                # Micro-batch proposals
                if c.enable_bin_learning:
                    (fL, bM_l, bL_l, bSt_l, bTP_l, bBk_l, mu_l) = self._batch_fire(
                        self._batch_l,
                        N_b,
                        S_b,
                        Wmin_b,
                        cap_m,
                        cap_l,
                        cap_st,
                        cap_tp,
                        cap_bk,
                    )
                    (fS, bM_s, bL_s, bSt_s, bTP_s, bBk_s, mu_s) = self._batch_fire(
                        self._batch_s,
                        N_b,
                        S_b,
                        Wmin_b,
                        cap_m,
                        cap_l,
                        cap_st,
                        cap_tp,
                        cap_bk,
                    )
                    if fL:
                        self._fired_l = True
                        self._mu_bl = mu_l
                    if fS:
                        self._fired_s = True
                        self._mu_bs = mu_s
                    if fL:
                        self._dM_bl = bM_l
                        self._dL_bl = bL_l
                        self._dSt_bl = bSt_l
                        self._dTP_bl = bTP_l
                        self._dBk_bl = bBk_l
                    if fS:
                        self._dM_bs = bM_s
                        self._dL_bs = bL_s
                        self._dSt_bs = bSt_s
                        self._dTP_bs = bTP_s
                        self._dBk_bs = bBk_s

                w_bat_eff = w_bat if (self._fired_l or self._fired_s) else 0.0
                abs_bl = abs(self._mu_bl)
                abs_bs = abs(self._mu_bs)
                if self._fired_l and self._fired_s:
                    wL = 0.7 if abs_bl >= abs_bs else 0.3
                    wS = 1.0 - wL
                elif self._fired_l:
                    wL, wS = 1.0, 0.0
                elif self._fired_s:
                    wL, wS = 0.0, 1.0
                else:
                    wL, wS = 0.0, 0.0

                dM_b = wL * self._dM_bl + wS * self._dM_bs
                dL_b = wL * self._dL_bl + wS * self._dL_bs
                dSt_b = wL * self._dSt_bl + wS * self._dSt_bs
                dTP_b = wL * self._dTP_bl + wS * self._dTP_bs
                dBk_b = wL * self._dBk_bl + wS * self._dBk_bs

                tgt_m_b = self.adap_multiplier * (1 + dM_b)
                tgt_l_b = self.adap_atr_period + dL_b
                tgt_st_b = self.adap_stop_mult * (1 + dSt_b)
                tgt_tp_b = self.adap_tp_mult * (1 + dTP_b)
                tgt_bk_b = self.adap_breakout_buf * (1 + dBk_b)

                tgt_m_f = tgt_m * (1 - w_bat_eff) + tgt_m_b * w_bat_eff
                tgt_l_f = tgt_l * (1 - w_bat_eff) + tgt_l_b * w_bat_eff
                tgt_st_f = tgt_st * (1 - w_bat_eff) + tgt_st_b * w_bat_eff
                tgt_tp_f = tgt_tp * (1 - w_bat_eff) + tgt_tp_b * w_bat_eff
                tgt_bk_f = tgt_bk * (1 - w_bat_eff) + tgt_bk_b * w_bat_eff

                # Apply with all safeguards
                self._bars_since_upd += 1
                can_upd = (i - self._last_update_bar) >= c.param_update_every
                if can_upd and (
                    abs(tgt_m_f - self.adap_multiplier) > db_m
                    or abs(tgt_l_f - self.adap_atr_period) > db_l
                ):
                    new_m = clamp(quantize(tgt_m_f, c.quant_step_mult), 0.5, 5.0)
                    new_l = clamp(round(tgt_l_f), 5, 100)
                    new_st = clamp(quantize(tgt_st_f, c.quant_step_stop), 0.3, 3.0)
                    new_tp = clamp(quantize(tgt_tp_f, c.quant_step_tp), 0.5, 5.0)
                    new_bk = clamp(quantize(tgt_bk_f, c.quant_step_brk), 0.0, 3.0)

                    self.adap_multiplier += (new_m - self.adap_multiplier) * alpha_eff
                    self.adap_atr_period += (new_l - self.adap_atr_period) * alpha_eff
                    self.adap_stop_mult += (new_st - self.adap_stop_mult) * alpha_eff
                    self.adap_tp_mult += (new_tp - self.adap_tp_mult) * alpha_eff
                    self.adap_breakout_buf += (
                        new_bk - self.adap_breakout_buf
                    ) * alpha_eff

                    self._last_update_bar = i
                    self._bars_since_upd = 0

                # Anchor reversion
                if c.revert_every > 0 and i > 0 and i % c.revert_every == 0:
                    self.adap_multiplier = (
                        self.adap_multiplier
                        + (c.multiplier - self.adap_multiplier) * c.revert_step
                    )
                    self.adap_atr_period = (
                        self.adap_atr_period
                        + (c.atr_period - self.adap_atr_period) * c.revert_step
                    )

                # Grid update from closed probes
                if (l_usd is not None or s_usd is not None) and c.use_cognitive_map:
                    ixC = self._bucket(self._regime_score, c.map_grid_x)
                    iyC = self._bucket(vol_n, c.map_grid_y)
                    if l_usd is not None:
                        self._grid_update(
                            ixC, iyC, l_atr, net_m, net_l, net_st, net_tp, net_bk, i
                        )
                    if s_usd is not None:
                        self._grid_update(
                            ixC, iyC, s_atr, net_m, net_l, net_st, net_tp, net_bk, i
                        )

            # ── Write adaptive state to output arrays ──────────
            adap_mult_arr[i] = self.adap_multiplier
            adap_len_arr[i] = self.adap_atr_period
            adap_stop_arr[i] = self.adap_stop_mult
            adap_tp_arr[i] = self.adap_tp_mult
            adap_brk_arr[i] = self.adap_breakout_buf

            long_wr_arr[i] = (
                self._long_wins / self._long_total if self._long_total > 0 else 0.0
            )
            short_wr_arr[i] = (
                self._short_wins / self._short_total if self._short_total > 0 else 0.0
            )
            long_cum_arr[i] = self._long_cum_usd
            short_cum_arr[i] = self._short_cum_usd

            # ==================================================
            #  SIGNAL GENERATION
            # ==================================================
            sens = clamp(int(round(self.adap_sensitivity)), 1, 100)
            lb2 = max(1, int(round(sens / 10.0)))

            high_hist = list(self._high_q)
            low_hist = list(self._low_q)
            rsi_hist = list(self._rsi_q)
            vol_hist = list(self._vol_q)

            # New high detection
            if len(high_hist) >= sens:
                hi_now = max(high_hist[-sens:])
                hi_prev = (
                    max(high_hist[-sens - lb2 : -lb2])
                    if len(high_hist) >= sens + lb2
                    else None
                )
                is_new_high = (
                    hi_prev is not None and hi_now != hi_prev and close > hi_prev
                )
            else:
                is_new_high = False

            # New low detection
            if len(low_hist) >= sens:
                lo_now = min(low_hist[-sens:])
                lo_prev = (
                    min(low_hist[-sens - lb2 : -lb2])
                    if len(low_hist) >= sens + lb2
                    else None
                )
                is_new_low = (
                    lo_prev is not None and lo_now != lo_prev and close < lo_prev
                )
            else:
                is_new_low = False

            # Key levels filter
            atr_levels = atr_i * c.major_level_threshold
            if c.enable_major_levels_only:
                hi_rng = max(high_hist[-sens:]) if len(high_hist) >= sens else high_i
                lo_rng = min(low_hist[-sens:]) if len(low_hist) >= sens else low_i
                final_new_high = is_new_high and (high_i - lo_rng) > atr_levels
                final_new_low = is_new_low and (hi_rng - low_i) > atr_levels
            else:
                final_new_high = is_new_high
                final_new_low = is_new_low

            # RSI filter (lookback-based, not spot)
            if c.enable_rsi and len(rsi_hist) > 0:
                bsc = self._bars_since_below(c.rsi_bot, rsi_hist)
                bsh = self._bars_since_above(c.rsi_top, rsi_hist)
                rsi_cold = bsc < c.rsi_lookback_bot
                rsi_hot = bsh < c.rsi_lookback_top
            else:
                rsi_cold = rsi_hot = True

            # Volume surge filter
            if len(vol_hist) >= c.vol_lookback:
                vol_avg = float(np.mean(vol_hist[-c.vol_lookback :]))
            else:
                vol_avg = vol_i
            vol_surge = vol_i > c.vol_multiplier * vol_avg

            buy_filters = rsi_cold and (not c.require_vol_spike or vol_surge)
            sell_filters = rsi_hot and (not c.require_vol_spike or vol_surge)
            can_sig = (i - self._last_signal_bar) >= c.min_bars_between_signals

            cur_trend = trend_sig_arr[i]
            prev_trend = trend_sig_arr[i - 1] if i > 0 else 1

            buy_sig = sell_sig = False

            if c.signal_mode == "Reversal" and can_sig:
                # Capture previous flag values BEFORE updating (mirrors Pine's topFlag[1]/botFlag[1])
                prev_top_flag = self._top_flag
                prev_bot_flag = self._bot_flag

                # Update flags — mirrors Pine:
                #   topFlag := rTrendSig == -1 ? 0 : finalIsNewHigh and rTrendSig == 1 ? 1 : topFlag[1]
                #   botFlag := rTrendSig ==  1 ? 0 : finalIsNewLow  and rTrendSig == -1 ? 1 : botFlag[1]
                if cur_trend == -1:
                    self._top_flag = 0
                elif final_new_high and cur_trend == 1:
                    self._top_flag = 1
                # else: keep previous value (already stored in self._top_flag)

                if cur_trend == 1:
                    self._bot_flag = 0
                elif final_new_low and cur_trend == -1:
                    self._bot_flag = 1
                # else: keep previous value

                # Reversal sell: flag WAS set last bar (prev==1), now cleared (cur==0) because trend flipped
                # Mirrors Pine: (topFlag[1] == 1 and topFlag == 0)
                rev_sell = (prev_top_flag == 1 and self._top_flag == 0) or (
                    not c.require_new_extreme and prev_trend == 1 and cur_trend == -1
                )

                # Reversal buy: flag WAS set last bar (prev==1), now cleared (cur==0) because trend flipped
                # Mirrors Pine: (botFlag[1] == 1 and botFlag == 0)
                rev_buy = (prev_bot_flag == 1 and self._bot_flag == 0) or (
                    not c.require_new_extreme and prev_trend == -1 and cur_trend == 1
                )

                if rev_sell and sell_filters:
                    sell_sig = True
                    self._last_signal_bar = i
                elif rev_buy and buy_filters:
                    buy_sig = True
                    self._last_signal_bar = i

            elif c.signal_mode == "Breakout" and can_sig:
                if final_new_high and cur_trend == 1 and sell_filters:
                    sell_sig = True
                    self._last_signal_bar = i
                elif final_new_low and cur_trend == -1 and buy_filters:
                    buy_sig = True
                    self._last_signal_bar = i

            buy_arr[i] = buy_sig
            sell_arr[i] = sell_sig

        # ── Assemble output ────────────────────────────────────
        out = df.copy()
        out["trend_plot"] = trend_plot_arr
        out["upper_band"] = upper_band_arr
        out["lower_band"] = lower_band_arr
        out["trend_signal"] = trend_sig_arr
        out["buy_signal"] = buy_arr
        out["sell_signal"] = sell_arr
        out["adaptive_multiplier"] = adap_mult_arr
        out["adaptive_atr_period"] = adap_len_arr
        out["adaptive_stop_mult"] = adap_stop_arr
        out["adaptive_tp_mult"] = adap_tp_arr
        out["adaptive_breakout_buf"] = adap_brk_arr
        out["regime_score"] = regime_arr
        out["long_win_rate"] = long_wr_arr
        out["short_win_rate"] = short_wr_arr
        out["long_cum_usd"] = long_cum_arr
        out["short_cum_usd"] = short_cum_arr
        return out

    # ==========================================================
    #  STATE SNAPSHOT  (serialize / restore)
    # ==========================================================

    def export_snapshot(self, max_cells: int = 40) -> str:
        """Export current learned state as a JSON string."""
        cells = []
        for idx, cell in enumerate(self._grid):
            if cell.count > 0 and len(cells) < max_cells:
                cells.append(
                    {
                        "idx": idx,
                        "count": cell.count,
                        "m_ret": round(cell.mean_atr_ret, 6),
                        "d_ewm": round(cell.down_ewm, 6),
                        "d_mult": round(cell.d_mult, 6),
                        "d_len": round(cell.d_len, 4),
                        "d_stop": round(cell.d_stop, 6),
                        "d_tp": round(cell.d_tp, 6),
                        "d_brk": round(cell.d_brk, 6),
                    }
                )
        snap = {
            "version": "PY1",
            "params": {
                "multiplier": round(self.adap_multiplier, 4),
                "atr_period": round(self.adap_atr_period, 2),
                "stop_mult": round(self.adap_stop_mult, 4),
                "tp_mult": round(self.adap_tp_mult, 4),
                "breakout_buf": round(self.adap_breakout_buf, 4),
            },
            "cells": cells,
        }
        return json.dumps(snap, separators=(",", ":"))

    def restore_snapshot(self, snap_str: str):
        """Restore learned state from a JSON snapshot string."""
        snap = json.loads(snap_str)
        p = snap.get("params", {})
        if "multiplier" in p:
            self.adap_multiplier = float(p["multiplier"])
        if "atr_period" in p:
            self.adap_atr_period = float(p["atr_period"])
        if "stop_mult" in p:
            self.adap_stop_mult = float(p["stop_mult"])
        if "tp_mult" in p:
            self.adap_tp_mult = float(p["tp_mult"])
        if "breakout_buf" in p:
            self.adap_breakout_buf = float(p["breakout_buf"])
        for cd in snap.get("cells", []):
            idx = int(cd["idx"])
            if 0 <= idx < len(self._grid):
                cell = self._grid[idx]
                cell.count = int(cd["count"])
                cell.mean_atr_ret = float(cd["m_ret"])
                cell.down_ewm = float(cd["d_ewm"])
                cell.d_mult = float(cd["d_mult"])
                cell.d_len = float(cd["d_len"])
                cell.d_stop = float(cd["d_stop"])
                cell.d_tp = float(cd["d_tp"])
                cell.d_brk = float(cd["d_brk"])
                cell.conf = 1.0 - math.exp(
                    -cell.count / max(1, self.cfg.map_conf_scale)
                )


