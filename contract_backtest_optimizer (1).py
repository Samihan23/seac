"""
Contract Z-Score Backtester & Optimizer
=========================================

A companion Streamlit + Plotly application to the Seasonality Analysis
Dashboard and Contract Settlement Z-Score Monitor.

Workflow
--------
1. Upload ONE contract's Excel workbook (same "Cleaned Data" sheet format
   exported by the Seasonality Analysis Dashboard -- Year, DayIndex, Date,
   Price columns, one row per trading day per year).
2. Section 1 -- Backtest: define an entry/exit rule using the rolling
   Z-Score and/or Z-Score-of-Std (volatility-of-volatility), and see the
   trade-by-trade and year-by-year results of running that exact rule
   across every year of the contract's history.
3. Section 2 -- Optimizer: instead of guessing thresholds by hand, run a
   systematic parameter search across a grid of entry/exit combinations to
   find the one with the best historical total P&L (or win rate / profit
   factor), then load it straight into the backtest section to inspect.

Strategy definition
--------------------
The sign of the entry Z-Score threshold sets the trade direction:
  - entry_z < 0  -> LONG:  enter when ZScore <= entry_z, exit when ZScore >= exit_z
  - entry_z > 0  -> SHORT: enter when ZScore >= entry_z, exit when ZScore <= exit_z
An optional Z-Score-of-Std volatility filter can (a) block new entries
while |ZScoreOfStd| is already extreme, and (b) force an early exit if
|ZScoreOfStd| spikes above a threshold after entry.

This app is a systematic PARAMETER SEARCH (grid search over the rule's
thresholds), not a black-box machine-learning model -- for a rule this
simple and explainable, that is the correct and fully verifiable tool,
and every candidate combination is backtested with the exact same engine
used in the manual backtest section, so results are directly comparable
and trustworthy.

Run with:
    streamlit run contract_backtest_optimizer.py

Author: Generated for Samihan
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("contract_backtest_optimizer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WINDOW = 20
DARK_BG = "#0E1117"
PANEL_BG = "#151A24"
GRID_COLOR = "rgba(255,255,255,0.08)"
REQUIRED_CLEANED_DATA_COLUMNS = {"Year", "DayIndex", "Date", "Price"}


# ---------------------------------------------------------------------------
# Data loading (self-contained -- same schema as the other two apps)
# ---------------------------------------------------------------------------
@dataclass
class ContractData:
    name: str
    year_data: Dict[str, pd.DataFrame] = field(default_factory=dict)  # year -> DataFrame[Date, Price]
    load_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.load_error is None and bool(self.year_data)


class ContractExcelLoader:
    """Loads a contract's Excel workbook (as produced by the Seasonality
    Analysis Dashboard's "Export Excel" feature) and extracts its per-year
    cleaned price data from the "Cleaned Data" sheet."""

    def __init__(self, contract_name: str, file_bytes: bytes):
        self.contract_name = contract_name
        self.file_bytes = file_bytes

    def load(self) -> ContractData:
        try:
            xls = pd.ExcelFile(io.BytesIO(self.file_bytes))
        except Exception as exc:  # noqa: BLE001
            return ContractData(self.contract_name, {}, load_error=f"Could not open workbook: {exc}")

        sheet_name = self._find_cleaned_data_sheet(xls)
        if sheet_name is None:
            return ContractData(
                self.contract_name, {},
                load_error="No 'Cleaned Data' sheet found. Upload the Excel exported by the "
                           "Seasonality Analysis Dashboard.",
            )
        try:
            raw = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as exc:  # noqa: BLE001
            return ContractData(self.contract_name, {}, load_error=f"Could not read '{sheet_name}': {exc}")

        missing = REQUIRED_CLEANED_DATA_COLUMNS - set(raw.columns)
        if missing:
            return ContractData(
                self.contract_name, {},
                load_error=f"'{sheet_name}' sheet is missing required columns: {sorted(missing)}",
            )

        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
        raw = raw.dropna(subset=["Date", "Price"])
        if raw.empty:
            return ContractData(self.contract_name, {}, load_error="No valid rows after cleaning.")

        year_data: Dict[str, pd.DataFrame] = {}
        for year, group in raw.groupby(raw["Year"].astype(str)):
            g = group.sort_values("Date").reset_index(drop=True)
            year_data[year] = g[["DayIndex", "Date", "Price"]]

        return ContractData(self.contract_name, year_data)

    @staticmethod
    def _find_cleaned_data_sheet(xls: pd.ExcelFile) -> Optional[str]:
        for name in xls.sheet_names:
            if name.strip().lower() == "cleaned data":
                return name
        return None


# ---------------------------------------------------------------------------
# Rolling statistics engine (same math as the other two apps)
# ---------------------------------------------------------------------------
class RollingStatsEngine:
    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = max(2, int(window))
        self.min_periods = max(2, self.window // 2)

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.reset_index(drop=True).copy()
        out["RollingMean"] = out["Price"].rolling(self.window, min_periods=self.min_periods).mean()
        out["RollingStd"] = out["Price"].rolling(self.window, min_periods=self.min_periods).std()
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ZScore"] = (out["Price"] - out["RollingMean"]) / out["RollingStd"]
        out["ZScore"] = out["ZScore"].replace([np.inf, -np.inf], np.nan)

        out["MeanOfStd"] = out["RollingStd"].rolling(self.window, min_periods=self.min_periods).mean()
        out["StdOfStd"] = out["RollingStd"].rolling(self.window, min_periods=self.min_periods).std()
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ZScoreOfStd"] = (out["RollingStd"] - out["MeanOfStd"]) / out["StdOfStd"]
        out["ZScoreOfStd"] = out["ZScoreOfStd"].replace([np.inf, -np.inf], np.nan)
        return out


# ---------------------------------------------------------------------------
# Strategy definition & backtest engine
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """Full strategy definition.

    Direction is an explicit choice (+1 Long, -1 Short) rather than inferred
    from a threshold's sign -- Z-Score-of-Std has no inherent long/short
    polarity of its own (it describes a volatility regime, not a price
    direction), so it can't be used to infer direction the way price
    Z-Score can. Making direction explicit keeps both conditions usable on
    equal footing, alone or combined.

    Entry: at least one of {use_entry_z, use_entry_zs} must be True. If
    both are True, entry_logic ("AND"/"OR") decides how they combine.
      - Z-Score entry compares relative to direction: Long enters when
        ZScore <= entry_z; Short enters when ZScore >= entry_z.
      - Z-Score-of-Std entry uses its own explicit comparison operator
        (entry_zs_op: "<=" or ">="), independent of direction, since "enter
        while vol is calm" vs "enter while vol is already elevated" are
        both legitimate, direction-independent choices.

    Take-profit exit: at least one of {use_exit_z, use_exit_zs} must be
    True. If both are enabled, EITHER firing triggers the exit (OR logic --
    you exit as soon as any take-profit condition is met, not both at once).
    Disabling use_exit_z and using only use_exit_zs gives a strategy whose
    exit (like its entry, if use_entry_zs-only is set too) is driven purely
    by Z-Score-of-Std, with price Z-Score never touched.
      - Z-Score take-profit compares relative to direction: Long exits when
        ZScore >= exit_z; Short exits when ZScore <= exit_z.
      - Z-Score-of-Std take-profit uses its own explicit comparison operator
        (exit_zs_op), independent of direction.

    Stop-loss (sl_z, optional): a separate, further-adverse price Z-Score
    level, checked only if no take-profit condition fired. Long stops out
    when ZScore <= sl_z; Short stops out when ZScore >= sl_z.

    An optional Z-Score-of-Std spike filter (use_vol_exit_filter) can also
    force an exit as a third, independent risk overlay -- distinct from
    use_exit_zs above: this one is direction-agnostic "panic" protection
    (|Z-Score-of-Std| too extreme, regardless of profit), not a take-profit
    signal.
    """

    direction: int  # +1 = Long, -1 = Short

    use_entry_z: bool = True
    entry_z: float = -2.0

    use_entry_zs: bool = False
    entry_zs: float = 2.0
    entry_zs_op: str = "<="  # "<=" or ">="

    entry_logic: str = "AND"  # "AND" or "OR" -- used only if both entry conditions are enabled

    use_exit_z: bool = True
    exit_z: float = 0.0  # take-profit threshold (price Z-Score)

    use_exit_zs: bool = False
    exit_zs: float = 0.0  # take-profit threshold (Z-Score of Std)
    exit_zs_op: str = ">="  # "<=" or ">="

    use_stop_loss: bool = True
    sl_z: float = -3.5  # stop-loss threshold (price Z-Score)

    use_sl_value: bool = False
    sl_value: float = 0.0  # stop-loss threshold (raw unrealized loss in price units, e.g. 5.0 = stop
                             # once the position is down 5.0 or more, regardless of Z-Score)

    use_vol_exit_filter: bool = False  # optional extra risk overlay (distinct from use_exit_zs)
    exit_zs_max: float = 3.0

    use_trade_cost: bool = False
    cost_per_trade: float = 0.0  # flat round-trip cost deducted from each trade's PnL

    @property
    def direction_label(self) -> str:
        return "Long" if self.direction == 1 else "Short"

    @property
    def is_valid(self) -> bool:
        has_entry = self.use_entry_z or self.use_entry_zs
        has_exit = self.use_exit_z or self.use_exit_zs
        return self.direction in (1, -1) and has_entry and has_exit

    def entry_signal(self, z: float, zs: float) -> bool:
        conditions = []
        if self.use_entry_z:
            if pd.isna(z):
                return False
            conditions.append((z <= self.entry_z) if self.direction == 1 else (z >= self.entry_z))
        if self.use_entry_zs:
            if pd.isna(zs):
                return False
            conditions.append((zs <= self.entry_zs) if self.entry_zs_op == "<=" else (zs >= self.entry_zs))
        if not conditions:
            return False
        return all(conditions) if (len(conditions) == 1 or self.entry_logic == "AND") else any(conditions)

    def exit_signal(self, z: float, zs: float, unrealized_pnl: Optional[float] = None) -> Optional[str]:
        """Returns the exit reason string if an exit is triggered, else None.
        Checked in priority order: take-profit (Z-Score and/or Z-Score-of-Std,
        whichever fires first), then stop-loss (Z-Score and/or raw value,
        whichever fires first), then the vol panic filter.

        unrealized_pnl is the position's current mark-to-market P&L (positive
        = profitable, negative = losing) -- required only if use_sl_value is
        enabled; the caller (BacktestEngine) computes it from current price
        vs. entry price."""
        take_profit = False
        if self.use_exit_z and not pd.isna(z):
            tp_z = (z >= self.exit_z) if self.direction == 1 else (z <= self.exit_z)
            take_profit = take_profit or tp_z
        if self.use_exit_zs and not pd.isna(zs):
            tp_zs = (zs <= self.exit_zs) if self.exit_zs_op == "<=" else (zs >= self.exit_zs)
            take_profit = take_profit or tp_zs
        if take_profit:
            return "take_profit"

        if self.use_stop_loss and not pd.isna(z):
            stop_hit = (z <= self.sl_z) if self.direction == 1 else (z >= self.sl_z)
            if stop_hit:
                return "stop_loss_z"

        if self.use_sl_value and unrealized_pnl is not None:
            if unrealized_pnl <= -abs(self.sl_value):
                return "stop_loss_value"

        if self.use_vol_exit_filter and not pd.isna(zs) and abs(zs) >= self.exit_zs_max:
            return "vol_stop"
        return None


@dataclass
class Trade:
    year: str
    direction: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    pnl: float          # net of trade cost (this is what win/loss and all totals use)
    pnl_pct: float
    exit_reason: str
    raw_pnl: float = 0.0   # gross PnL before cost, for reference
    cost: float = 0.0      # cost actually deducted for this trade
    entry_zscore: float = float("nan")
    entry_zscore_of_std: float = float("nan")
    exit_zscore: float = float("nan")
    exit_zscore_of_std: float = float("nan")
    heat: float = 0.0       # Maximum Adverse Excursion, in price units -- how far the price moved
                             # against the position at its WORST point while held (>= 0). Distinct
                             # from equity drawdown, which measures across trades, not within one.
    heat_pct: float = 0.0   # heat relative to entry price


@dataclass
class YearResult:
    year: str
    trades: List[Trade] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> Optional[float]:
        return (self.wins / self.trade_count) if self.trade_count else None

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> Optional[float]:
        return (self.total_pnl / self.trade_count) if self.trade_count else None

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cum = np.cumsum([t.pnl for t in self.trades])
        peak = np.maximum.accumulate(cum)
        drawdown = cum - peak
        return float(drawdown.min())

    @property
    def avg_holding_days(self) -> Optional[float]:
        if not self.trades:
            return None
        days = [(t.exit_date - t.entry_date).days for t in self.trades]
        return float(np.mean(days))

    @property
    def avg_heat(self) -> Optional[float]:
        if not self.trades:
            return None
        return float(np.mean([t.heat for t in self.trades]))


@dataclass
class YearArrays:
    """Plain numpy arrays for one year's rolling stats, with NaN Z-Score rows
    already dropped. Precomputing this ONCE (outside the optimizer's parameter
    loop) and reusing it for every candidate strategy avoids re-doing
    pandas .dropna()/.iterrows() work thousands of times over -- this is the
    dominant cost of the optimizer, and precomputing it is what makes a full
    grid search fast enough to be usable."""
    dates: np.ndarray
    prices: np.ndarray
    zscores: np.ndarray
    zscores_of_std: np.ndarray


def prepare_year_arrays(stats_by_year: Dict[str, pd.DataFrame]) -> Dict[str, YearArrays]:
    prepared: Dict[str, YearArrays] = {}
    for year, df in stats_by_year.items():
        rows = df.dropna(subset=["ZScore"])
        prepared[year] = YearArrays(
            dates=rows["Date"].to_numpy(),
            prices=rows["Price"].to_numpy(dtype=float),
            zscores=rows["ZScore"].to_numpy(dtype=float),
            zscores_of_std=rows["ZScoreOfStd"].to_numpy(dtype=float),
        )
    return prepared


def restrict_to_day_range(
    stats_by_year: Dict[str, pd.DataFrame], day_min: Optional[int], day_max: Optional[int]
) -> Dict[str, pd.DataFrame]:
    """Restrict each year's data to a trading-day-of-year window (using the
    contract's own DayIndex -- day 0, 1, 2, ... from that year's start).

    Applied AFTER rolling stats are already computed on the full year, so
    the Z-Score/Z-Score-of-Std rolling windows still have proper full-year
    context (no artificial NaN gaps from truncating history) -- only which
    days are ELIGIBLE for entry/exit signal checking gets restricted. A
    position still open when the window ends force-closes at the window's
    last available day, just like the existing end-of-year close.
    """
    if day_min is None and day_max is None:
        return stats_by_year
    filtered: Dict[str, pd.DataFrame] = {}
    for year, df in stats_by_year.items():
        if "DayIndex" not in df.columns:
            filtered[year] = df
            continue
        mask = pd.Series(True, index=df.index)
        if day_min is not None:
            mask &= df["DayIndex"] >= day_min
        if day_max is not None:
            mask &= df["DayIndex"] <= day_max
        filtered[year] = df[mask].reset_index(drop=True)
    return filtered


def compute_daily_equity(stats_df: pd.DataFrame, trades: List["Trade"], direction: int) -> pd.DataFrame:
    """Day-by-day mark-to-market equity for one contract/year: cumulative
    REALIZED P&L from closed trades, plus the current UNREALIZED P&L of
    whatever position is open that day (0 if flat). This is what a real
    portfolio combination needs -- not just each trade's final P&L, but
    what the position was actually worth on every day along the way.

    Returns a DataFrame with DayIndex, Date, Price, Equity (all in the same
    net-of-cost units as Trade.pnl).
    """
    df = stats_df.sort_values("DayIndex").reset_index(drop=True)[["DayIndex", "Date", "Price"]].copy()
    if not trades:
        df["Equity"] = 0.0
        return df

    trades_by_entry = sorted(trades, key=lambda t: t.entry_date)
    cum_realized = 0.0
    trade_idx = 0
    current = trades_by_entry[0] if trades_by_entry else None
    equity_vals: List[float] = []

    for _, row in df.iterrows():
        d = row["Date"]
        price = row["Price"]

        # Finalize any trade whose exit already fully passed before this day
        while current is not None and current.exit_date < d:
            cum_realized += current.pnl
            trade_idx += 1
            current = trades_by_entry[trade_idx] if trade_idx < len(trades_by_entry) else None

        if current is not None and current.entry_date <= d < current.exit_date:
            # Open position: mark-to-market using the current day's price
            eq = cum_realized + (price - current.entry_price) * direction
        elif current is not None and d == current.exit_date:
            # Exit day itself: the full net (cost-inclusive) realized PnL lands today
            eq = cum_realized + current.pnl
            cum_realized += current.pnl
            trade_idx += 1
            current = trades_by_entry[trade_idx] if trade_idx < len(trades_by_entry) else None
        else:
            eq = cum_realized

        equity_vals.append(eq)

    df["Equity"] = equity_vals
    return df


def combine_daily_equity(per_contract_equity: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combines multiple contracts' daily equity curves by DayIndex: aligns
    them on a shared DayIndex axis, forward-fills any contract's gaps (e.g.
    a shorter contract's last known value carries flat once its own data
    ends), and sums across contracts to get one true combined portfolio
    equity curve per trading day."""
    if not per_contract_equity:
        return pd.DataFrame(columns=["DayIndex", "Portfolio_Equity"])

    series_list = []
    for name, df in per_contract_equity.items():
        s = df.set_index("DayIndex")["Equity"].rename(name)
        series_list.append(s)

    merged = pd.concat(series_list, axis=1).sort_index()
    merged = merged.ffill().fillna(0.0)
    merged["Portfolio_Equity"] = merged.sum(axis=1)
    return merged.reset_index()


def compute_heat_series(equity: pd.Series) -> pd.Series:
    """Running 'heat' (underwater/drawdown magnitude) of an equity curve at
    every point: how far below its own running peak it currently sits,
    reported as a non-negative magnitude (0 = at a new high)."""
    peak = equity.cummax()
    return (peak - equity).clip(lower=0.0)


class BacktestEngine:
    """Runs a StrategyConfig against one year's rolling-stats data at a time.

    - Enters at the close of the day the entry condition is met, exits at
      the close of the day the exit condition (take-profit, stop-loss, or
      vol-filter stop) is met.
    - Multiple sequential (non-overlapping) trades per year are allowed.
    - A position still open at year-end is force-closed at the last
      available price ("forced_eoy").
    - Rows with NaN Z-Score (not enough history yet for the rolling window)
      are skipped for entry-signal purposes.

    Two equivalent entry points are provided:
      - run_year(df): takes a pandas DataFrame directly -- simplest, used
        for one-off runs (the manual backtest section).
      - run_year_arrays(arrays): takes a pre-extracted YearArrays -- avoids
        repeated pandas overhead, used by the optimizer which calls this
        thousands of times over the same underlying data.
    Both produce identical results; the array path is purely a speed
    optimization, not a different calculation.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def run_year(self, year: str, stats_df: pd.DataFrame) -> YearResult:
        if stats_df.empty:
            return YearResult(year=year)
        rows = stats_df.dropna(subset=["ZScore"])
        arrays = YearArrays(
            dates=rows["Date"].to_numpy(),
            prices=rows["Price"].to_numpy(dtype=float),
            zscores=rows["ZScore"].to_numpy(dtype=float),
            zscores_of_std=rows["ZScoreOfStd"].to_numpy(dtype=float),
        )
        return self.run_year_arrays(year, arrays)

    def run_year_arrays(self, year: str, arrays: YearArrays) -> YearResult:
        result = YearResult(year=year)
        if not self.config.is_valid or len(arrays.prices) == 0:
            return result

        cfg = self.config
        direction = cfg.direction
        position: Optional[Tuple] = None  # (entry_idx, entry_date, entry_price, entry_z, entry_zs)
        n = len(arrays.prices)

        for i in range(n):
            z = arrays.zscores[i]
            zs = arrays.zscores_of_std[i]

            if position is None:
                if cfg.entry_signal(z, zs):
                    position = (i, arrays.dates[i], float(arrays.prices[i]), z, zs)
            else:
                current_price = float(arrays.prices[i])
                entry_price = position[2]
                unrealized_pnl = (current_price - entry_price) * direction
                reason = cfg.exit_signal(z, zs, unrealized_pnl=unrealized_pnl)
                if reason is not None:
                    trade = self._close_trade(
                        year, direction, position, i, arrays.dates[i], current_price,
                        reason, z, zs, arrays.prices,
                    )
                    result.trades.append(trade)
                    position = None

        if position is not None:
            trade = self._close_trade(
                year, direction, position, n - 1, arrays.dates[-1], float(arrays.prices[-1]), "forced_eoy",
                float(arrays.zscores[-1]), float(arrays.zscores_of_std[-1]), arrays.prices,
            )
            result.trades.append(trade)

        return result

    def _close_trade(
        self, year, direction, position, exit_idx, exit_date, exit_price, reason, exit_z, exit_zs, prices,
    ) -> Trade:
        entry_idx, entry_date, entry_price, entry_z, entry_zs = position
        raw_pnl = (exit_price - entry_price) * direction
        cost = self.config.cost_per_trade if self.config.use_trade_cost else 0.0
        pnl = raw_pnl - cost
        pnl_pct = (pnl / abs(entry_price)) if entry_price != 0 else float("nan")

        # Heat (Maximum Adverse Excursion): the worst the price moved AGAINST
        # the position at any point while it was held, regardless of where it
        # ended up. Long -> how far below entry the price dipped; Short -> how
        # far above entry it spiked. Always >= 0.
        path = prices[entry_idx:exit_idx + 1]
        if direction == 1:
            heat = max(0.0, entry_price - float(path.min()))
        else:
            heat = max(0.0, float(path.max()) - entry_price)
        heat_pct = (heat / abs(entry_price)) if entry_price != 0 else float("nan")

        return Trade(
            year=year, direction=direction,
            entry_date=pd.Timestamp(entry_date), entry_price=entry_price,
            exit_date=pd.Timestamp(exit_date), exit_price=exit_price,
            pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            raw_pnl=raw_pnl, cost=cost,
            entry_zscore=entry_z, entry_zscore_of_std=entry_zs,
            exit_zscore=exit_z, exit_zscore_of_std=exit_zs,
            heat=heat, heat_pct=heat_pct,
        )

    def run_all_years(self, stats_by_year: Dict[str, pd.DataFrame]) -> Dict[str, YearResult]:
        return {year: self.run_year(year, df) for year, df in stats_by_year.items()}

    def run_all_years_arrays(self, year_arrays: Dict[str, YearArrays]) -> Dict[str, YearResult]:
        return {year: self.run_year_arrays(year, arrays) for year, arrays in year_arrays.items()}


# ---------------------------------------------------------------------------
# Parameter optimizer (systematic grid search)
# ---------------------------------------------------------------------------
@dataclass
class OptimizerResult:
    config: StrategyConfig
    total_pnl: float
    trade_count: int
    win_rate: Optional[float]
    profit_factor: Optional[float]
    avg_holding_days: Optional[float]
    avg_heat: Optional[float]


class StrategyOptimizer:
    """Systematic grid search over whichever numeric thresholds are enabled
    (entry Z-Score, entry Z-Score-of-Std, take-profit, stop-loss, and/or the
    vol-spike exit filter). Structural choices -- direction, which entry
    conditions to use, AND/OR combination logic, whether to use a stop-loss
    -- are fixed inputs you choose upfront; the search explores threshold
    VALUES within that structure. Every candidate is backtested with the
    exact same BacktestEngine used in the manual backtest section, so
    results are directly comparable and fully reproducible -- not a
    black-box model."""

    def __init__(self, stats_by_year: Dict[str, pd.DataFrame], min_trades: int = 3):
        self.stats_by_year = stats_by_year
        self.year_arrays = prepare_year_arrays(stats_by_year)
        self.min_trades = max(1, min_trades)

    def search(
        self,
        direction: int,
        use_entry_z: bool, entry_z_grid: List[float],
        use_entry_zs: bool, entry_zs_grid: List[float], entry_zs_op: str,
        entry_logic: str,
        use_exit_z: bool, exit_z_grid: List[float],
        use_exit_zs: bool, exit_zs_grid: List[float], exit_zs_op: str,
        use_stop_loss: bool, sl_z_grid: List[float],
        use_sl_value: bool, sl_value_grid: List[float],
        use_vol_exit_filter: bool, exit_zs_max_grid: List[float],
        use_trade_cost: bool = False, cost_per_trade: float = 0.0,
        objective: str = "total_pnl",
    ) -> List[OptimizerResult]:
        if not use_entry_z and not use_entry_zs:
            return []
        if not use_exit_z and not use_exit_zs:
            return []

        entry_z_options = entry_z_grid if use_entry_z else [0.0]
        entry_zs_options = entry_zs_grid if use_entry_zs else [0.0]
        exit_z_options = exit_z_grid if use_exit_z else [0.0]
        exit_zs_options = exit_zs_grid if use_exit_zs else [0.0]
        sl_options = sl_z_grid if use_stop_loss else [0.0]
        sl_value_options = sl_value_grid if use_sl_value else [0.0]
        vol_options = exit_zs_max_grid if use_vol_exit_filter else [0.0]

        results: List[OptimizerResult] = []
        for entry_z in entry_z_options:
            for entry_zs in entry_zs_options:
                for exit_z in exit_z_options:
                    for exit_zs in exit_zs_options:
                        for sl_z in sl_options:
                            for sl_value in sl_value_options:
                                for exit_zs_max in vol_options:
                                    config = StrategyConfig(
                                        direction=direction,
                                        use_entry_z=use_entry_z, entry_z=entry_z,
                                        use_entry_zs=use_entry_zs, entry_zs=entry_zs, entry_zs_op=entry_zs_op,
                                        entry_logic=entry_logic,
                                        use_exit_z=use_exit_z, exit_z=exit_z,
                                        use_exit_zs=use_exit_zs, exit_zs=exit_zs, exit_zs_op=exit_zs_op,
                                        use_stop_loss=use_stop_loss, sl_z=sl_z,
                                        use_sl_value=use_sl_value, sl_value=sl_value,
                                        use_vol_exit_filter=use_vol_exit_filter, exit_zs_max=exit_zs_max,
                                        use_trade_cost=use_trade_cost, cost_per_trade=cost_per_trade,
                                    )
                                    engine = BacktestEngine(config)
                                    year_results = engine.run_all_years_arrays(self.year_arrays)
                                    all_trades = [t for yr in year_results.values() for t in yr.trades]

                                    if len(all_trades) < self.min_trades:
                                        continue

                                    total_pnl = sum(t.pnl for t in all_trades)
                                    wins = sum(1 for t in all_trades if t.pnl > 0)
                                    win_rate = wins / len(all_trades)
                                    gross_win = sum(t.pnl for t in all_trades if t.pnl > 0)
                                    gross_loss = abs(sum(t.pnl for t in all_trades if t.pnl < 0))
                                    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
                                        float("inf") if gross_win > 0 else None
                                    )
                                    avg_holding_days = float(np.mean(
                                        [(t.exit_date - t.entry_date).days for t in all_trades]
                                    ))
                                    avg_heat = float(np.mean([t.heat for t in all_trades]))

                                    results.append(OptimizerResult(
                                        config=config, total_pnl=total_pnl, trade_count=len(all_trades),
                                        win_rate=win_rate, profit_factor=profit_factor,
                                        avg_holding_days=avg_holding_days, avg_heat=avg_heat,
                                    ))

        key_fn = {
            "total_pnl": lambda r: r.total_pnl,
            "win_rate": lambda r: (r.win_rate or -1),
            "profit_factor": lambda r: (r.profit_factor if r.profit_factor is not None else -1),
        }[objective]
        results.sort(key=key_fn, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chart building
# ---------------------------------------------------------------------------
class ChartBuilder:
    def __init__(self):
        self.font_color = "#E6E6E6"

    def _layout(self, title: str, y_title: str, height: int = 420) -> dict:
        return dict(
            title=dict(text=title, font=dict(size=16, color=self.font_color)),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(color=self.font_color, family="Segoe UI, Roboto, sans-serif"),
            hovermode="x unified", height=height, margin=dict(l=60, r=30, t=60, b=50),
            xaxis=dict(title="Date", gridcolor=GRID_COLOR, showspikes=True, spikemode="across"),
            yaxis=dict(title=y_title, gridcolor=GRID_COLOR),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        )

    def price_with_trades(self, stats_df: pd.DataFrame, trades: List[Trade], year: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats_df["Date"], y=stats_df["Price"], mode="lines", name="Price",
            line=dict(color="#4FC3F7", width=1.6),
        ))
        for i, t in enumerate(trades):
            marker_color = "#4CD964" if t.direction == 1 else "#FF9F0A"
            exit_color = "#4CD964" if t.pnl > 0 else "#FF3B30"
            fig.add_trace(go.Scatter(
                x=[t.entry_date], y=[t.entry_price], mode="markers",
                marker=dict(symbol="triangle-up" if t.direction == 1 else "triangle-down",
                            size=12, color=marker_color),
                name=f"Entry ({'Long' if t.direction == 1 else 'Short'})",
                showlegend=(i == 0), hovertemplate=f"Entry: %{{x}}<br>Price: %{{y:.4f}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[t.exit_date], y=[t.exit_price], mode="markers",
                marker=dict(symbol="x", size=10, color=exit_color),
                name="Exit", showlegend=(i == 0),
                hovertemplate=f"Exit ({t.exit_reason}): %{{x}}<br>Price: %{{y:.4f}}<br>PnL: {t.pnl:.4f}<extra></extra>",
            ))
        fig.update_layout(**self._layout(f"{year} -- Price with Entries/Exits", "Price"))
        return fig

    def indicator_with_trades(
        self, stats_df: pd.DataFrame, trades: List[Trade], year: str, column: str, title: str,
    ) -> go.Figure:
        """Z-Score or Z-Score-of-Std over time, with the same entry/exit markers as the price chart."""
        entry_attr = "entry_zscore" if column == "ZScore" else "entry_zscore_of_std"
        exit_attr = "exit_zscore" if column == "ZScore" else "exit_zscore_of_std"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats_df["Date"], y=stats_df[column], mode="lines", name=title,
            line=dict(color="#AB63FA", width=1.6),
        ))
        for i, t in enumerate(trades):
            marker_color = "#4CD964" if t.direction == 1 else "#FF9F0A"
            exit_color = "#4CD964" if t.pnl > 0 else "#FF3B30"
            entry_val, exit_val = getattr(t, entry_attr), getattr(t, exit_attr)
            fig.add_trace(go.Scatter(
                x=[t.entry_date], y=[entry_val], mode="markers",
                marker=dict(symbol="triangle-up" if t.direction == 1 else "triangle-down",
                            size=12, color=marker_color),
                name=f"Entry ({'Long' if t.direction == 1 else 'Short'})",
                showlegend=(i == 0), hovertemplate=f"Entry: %{{x}}<br>{title}: %{{y:.3f}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[t.exit_date], y=[exit_val], mode="markers",
                marker=dict(symbol="x", size=10, color=exit_color),
                name="Exit", showlegend=(i == 0),
                hovertemplate=f"Exit ({t.exit_reason}): %{{x}}<br>{title}: %{{y:.3f}}<br>PnL: {t.pnl:.4f}<extra></extra>",
            ))
        fig.add_hline(y=0, line_color="#888888", line_width=1)
        fig.update_layout(**self._layout(f"{year} -- {title} with Entries/Exits", title))
        return fig

    def equity_curve(self, all_trades: List[Trade]) -> go.Figure:
        fig = go.Figure()
        if not all_trades:
            fig.update_layout(**self._layout("Equity Curve (no trades)", "Cumulative P&L"))
            return fig
        ordered = sorted(all_trades, key=lambda t: (t.year, t.exit_date))
        cum = np.cumsum([t.pnl for t in ordered])
        fig.add_trace(go.Scatter(
            x=[t.exit_date for t in ordered], y=cum, mode="lines+markers",
            line=dict(color="#4CD964", width=2), marker=dict(size=5),
            hovertemplate="%{x}<br>Cumulative PnL: %{y:.4f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="#888888", line_width=1)
        fig.update_layout(**self._layout("Equity Curve (all years, chronological)", "Cumulative P&L"))
        return fig

    def yearly_pnl_bar(self, year_results: Dict[str, YearResult]) -> go.Figure:
        years = sorted(year_results.keys())
        pnls = [year_results[y].total_pnl for y in years]
        colors = ["#4CD964" if p >= 0 else "#FF3B30" for p in pnls]
        fig = go.Figure(go.Bar(x=years, y=pnls, marker_color=colors))
        fig.update_layout(**self._layout("Total P&L by Trading Year", "Total P&L", height=360))
        return fig

    def portfolio_heat_chart(self, combined: pd.DataFrame) -> go.Figure:
        """Combined portfolio equity (top) and Heat/drawdown (bottom, as a
        filled area) over the shared Trading Day # axis -- every contract's
        (and every one of its years') day-by-day mark-to-market exposure,
        summed together, so overlapping positions across your whole
        portfolio show up as compounded risk on the same trading day."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined["DayIndex"], y=combined["Portfolio_Equity"], mode="lines",
            name="Combined Portfolio Equity", line=dict(color="#4CD964", width=2),
            hovertemplate="Day %{x}<br>Equity: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=combined["DayIndex"], y=-combined["Heat"], mode="lines",
            name="Portfolio Heat (underwater)", line=dict(color="#FF3B30", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,59,48,0.25)",
            hovertemplate="Day %{x}<br>Heat: %{customdata:.4f}<extra></extra>",
            customdata=combined["Heat"],
        ))
        fig.add_hline(y=0, line_color="#888888", line_width=1)
        layout = self._layout("Combined Portfolio Equity & Heat by Trading Day", "P&L", height=460)
        layout["xaxis"] = {**layout["xaxis"], "title": "Trading Day # (each contract-year's own day count)"}
        fig.update_layout(**layout)
        return fig


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
class BacktestApp:
    def __init__(self):
        self._init_session_state()
        self.chart_builder = ChartBuilder()

    @staticmethod
    def _init_session_state() -> None:
        defaults = {
            "contract": None,
            "window": DEFAULT_WINDOW,
            "direction": 1,
            "use_entry_z": True,
            "entry_z": -2.0,
            "use_entry_zs": False,
            "entry_zs": 0.0,
            "entry_zs_op": "<=",
            "entry_logic": "AND",
            "use_exit_z": True,
            "exit_z": 0.0,
            "use_exit_zs": False,
            "exit_zs": 0.0,
            "exit_zs_op": "<=",
            "use_stop_loss": True,
            "sl_z": -3.5,
            "use_vol_exit_filter": False,
            "exit_zs_max": 3.0,
            "use_trade_cost": False,
            "cost_per_trade": 0.0,
            "use_day_range": False,
            "day_min": 0,
            "day_max": 0,
            "app_mode": "Single Contract",
            "portfolio_contracts": {},
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    @staticmethod
    def configure_page() -> None:
        st.set_page_config(page_title="Contract Z-Score Backtester", layout="wide", initial_sidebar_state="expanded")
        st.markdown(
            f"""
            <style>
            .stApp {{ background-color: {DARK_BG}; }}
            section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
            div[data-testid="stMetric"] {{
                background-color: {PANEL_BG}; border: 1px solid {GRID_COLOR};
                border-radius: 8px; padding: 10px 14px;
            }}
            h1, h2, h3, h4, p, span, label {{ color: #E6E6E6; }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        st.sidebar.title("⚙️ Controls")

        st.sidebar.subheader("Mode")
        app_mode = st.sidebar.radio(
            "App mode", ["Single Contract", "Multi-Contract Portfolio"],
            index=0 if st.session_state.get("app_mode", "Single Contract") == "Single Contract" else 1,
            key="app_mode_widget",
        )
        st.session_state["app_mode"] = app_mode

        st.sidebar.subheader("Rolling Window")
        st.session_state["window"] = st.sidebar.number_input(
            "Rolling window (days)", min_value=2, max_value=252, value=int(st.session_state["window"]), step=1,
        )

        if app_mode == "Single Contract":
            st.sidebar.subheader("Contract")
            uploaded = st.sidebar.file_uploader(
                "Contract Excel file", type=["xlsx"],
                help="The 'Seasonality_Output.xlsx' exported from the Seasonality Analysis Dashboard.",
            )
            if uploaded is not None:
                contract_name = uploaded.name.rsplit(".", 1)[0]
                if st.session_state["contract"] is None or st.session_state["contract"].name != contract_name:
                    loader = ContractExcelLoader(contract_name, uploaded.getvalue())
                    st.session_state["contract"] = loader.load()
        else:
            st.sidebar.subheader("Contracts")
            uploaded_files = st.sidebar.file_uploader(
                "Contract Excel files (one per contract)", type=["xlsx"], accept_multiple_files=True,
                help="Each file is a 'Seasonality_Output.xlsx' exported from the Seasonality Analysis "
                     "Dashboard, one per contract. Set direction and search ranges for each below.",
            )
            for uf in uploaded_files or []:
                contract_name = uf.name.rsplit(".", 1)[0]
                if contract_name not in st.session_state["portfolio_contracts"]:
                    loader = ContractExcelLoader(contract_name, uf.getvalue())
                    st.session_state["portfolio_contracts"][contract_name] = loader.load()

        if st.sidebar.button("🔄 Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    def render_main(self):
        st.title("📐 Contract Z-Score Backtester & Optimizer")

        if st.session_state.get("app_mode", "Single Contract") == "Multi-Contract Portfolio":
            self._render_portfolio_mode()
            return

        contract: Optional[ContractData] = st.session_state.get("contract")
        if contract is None:
            st.info("Upload a contract's Excel file in the sidebar to begin.")
            return
        if not contract.is_valid:
            st.error(f"Could not load contract: {contract.load_error}")
            return

        window = int(st.session_state["window"])
        engine = RollingStatsEngine(window=window)
        full_stats_by_year = {year: engine.compute_full(df) for year, df in contract.year_data.items()}

        self._render_single_contract_view(full_stats_by_year, prefix="")

    def _ensure_strategy_state(self, prefix: str) -> None:
        """Ensure all per-strategy session-state keys exist under this prefix.
        prefix="" covers the single-contract mode's top-level keys (unchanged
        from before); a non-empty prefix (e.g. "pf_H27_") namespaces a fully
        independent copy of the same state for one contract in Portfolio mode."""
        defaults = {
            "direction": 1,
            "use_entry_z": True, "entry_z": -2.0,
            "use_entry_zs": False, "entry_zs": 0.0, "entry_zs_op": "<=",
            "entry_logic": "AND",
            "use_exit_z": True, "exit_z": 0.0,
            "use_exit_zs": False, "exit_zs": 0.0, "exit_zs_op": "<=",
            "use_stop_loss": True, "sl_z": -3.5,
            "use_sl_value": False, "sl_value": 0.0,
            "use_vol_exit_filter": False, "exit_zs_max": 3.0,
            "use_trade_cost": False, "cost_per_trade": 0.0,
            "use_day_range": False, "day_min": 0, "day_max": 0,
            "optimizer_results": None,
            "included_years": None,
        }
        for k, v in defaults.items():
            key = prefix + k
            if key not in st.session_state:
                st.session_state[key] = v

    def _render_single_contract_view(self, full_stats_by_year: Dict[str, pd.DataFrame], prefix: str = "") -> None:
        """Year-inclusion filter, trading-day-range filter, Backtest section,
        and Optimizer section for one contract's data. prefix="" is the
        single-contract mode's original behavior; Portfolio mode calls this
        once per contract with a unique prefix so every contract gets the
        full, independent feature set with no shared/colliding state."""
        self._ensure_strategy_state(prefix)

        all_years = sorted(full_stats_by_year.keys())
        included_key = prefix + "included_years"
        if not st.session_state[included_key] or not set(st.session_state[included_key]) <= set(all_years):
            st.session_state[included_key] = all_years
        included_years = st.multiselect(
            "Trading years to include (applies to both Backtest and Optimizer below)",
            options=all_years, default=st.session_state[included_key], key=prefix + "included_years_widget",
        )
        st.session_state[included_key] = included_years
        stats_by_year = {y: full_stats_by_year[y] for y in included_years}

        if not stats_by_year:
            st.warning("Select at least one year to include.")
            return

        max_day_available = int(max(df["DayIndex"].max() for df in stats_by_year.values()))
        use_day_range = st.checkbox(
            "Restrict to a trading-day-of-year range", value=st.session_state[prefix + "use_day_range"],
            key=prefix + "use_day_range_cb",
        )
        day_min, day_max = st.session_state[prefix + "day_min"], st.session_state[prefix + "day_max"]
        if use_day_range:
            c1b, c2b = st.columns(2)
            day_min = c1b.number_input("From trading day #", min_value=0, max_value=max_day_available,
                                        value=min(int(day_min), max_day_available), step=1,
                                        key=prefix + "day_min_input")
            day_max = c2b.number_input("To trading day #", min_value=0, max_value=max_day_available,
                                        value=min(int(day_max) if day_max else max_day_available, max_day_available),
                                        step=1, key=prefix + "day_max_input")
            st.caption(
                "Trading day # is each year's own day count from its contract/season start (day 0 = first "
                "trading day of that year) -- so this restricts both Backtest and Optimizer below to only "
                "consider entries/exits within that seasonal window each year, regardless of calendar dates."
            )
        st.session_state[prefix + "use_day_range"] = use_day_range
        st.session_state[prefix + "day_min"], st.session_state[prefix + "day_max"] = day_min, day_max

        if use_day_range:
            stats_by_year = restrict_to_day_range(stats_by_year, int(day_min), int(day_max))

        self._render_backtest_section(stats_by_year, prefix=prefix)
        st.divider()
        self._render_optimizer_section(stats_by_year, prefix=prefix)

    # ------------------------------- Section 1 --------------------------------
    def _render_backtest_section(self, stats_by_year: Dict[str, pd.DataFrame], prefix: str = "") -> None:
        st.header("1️⃣ Backtest")

        direction_choice = st.radio(
            "Direction", ["Long (buy when cheap)", "Short (sell when expensive)"],
            index=0 if st.session_state[prefix + "direction"] == 1 else 1, horizontal=True,
            key=prefix + "bt_direction",
        )
        direction = 1 if direction_choice.startswith("Long") else -1
        st.session_state[prefix + "direction"] = direction

        st.subheader("Entry")
        c1, c2 = st.columns(2)
        with c1:
            use_entry_z = st.checkbox("Use Z-Score for entry", value=st.session_state[prefix + "use_entry_z"],
                                       key=prefix + "bt_use_entry_z")
            entry_z = st.number_input(
                "Entry Z-Score threshold", value=float(st.session_state[prefix + "entry_z"]), step=0.25, format="%.2f",
                disabled=not use_entry_z, key=prefix + "bt_entry_z",
                help="Long enters when ZScore <= this. Short enters when ZScore >= this.",
            )
        with c2:
            use_entry_zs = st.checkbox("Use Z-Score of Std for entry", value=st.session_state[prefix + "use_entry_zs"],
                                        key=prefix + "bt_use_entry_zs")
            entry_zs = st.number_input(
                "Entry Z-Score of Std threshold", value=float(st.session_state[prefix + "entry_zs"]), step=0.25,
                format="%.2f", disabled=not use_entry_zs, key=prefix + "bt_entry_zs",
            )
            entry_zs_op = st.selectbox(
                "Comparison", ["<=", ">="], index=0 if st.session_state[prefix + "entry_zs_op"] == "<=" else 1,
                disabled=not use_entry_zs, key=prefix + "bt_entry_zs_op",
                help="'<=' enters while volatility is calm; '>=' enters while volatility is already elevated.",
            )
        st.session_state.update({
            prefix + "use_entry_z": use_entry_z, prefix + "entry_z": entry_z,
            prefix + "use_entry_zs": use_entry_zs, prefix + "entry_zs": entry_zs, prefix + "entry_zs_op": entry_zs_op,
        })

        entry_logic = st.session_state[prefix + "entry_logic"]
        if use_entry_z and use_entry_zs:
            entry_logic = st.radio(
                "Combine entry conditions using", ["AND", "OR"],
                index=0 if st.session_state[prefix + "entry_logic"] == "AND" else 1, horizontal=True,
                key=prefix + "bt_entry_logic",
                help="AND requires both conditions at once. OR triggers on either one.",
            )
            st.session_state[prefix + "entry_logic"] = entry_logic

        st.subheader("Take-Profit (Exit)")
        c3, c4 = st.columns(2)
        with c3:
            use_exit_z = st.checkbox("Use Z-Score for take-profit", value=st.session_state[prefix + "use_exit_z"],
                                      key=prefix + "bt_use_exit_z")
            exit_z = st.number_input(
                "Take-Profit Z-Score threshold", value=float(st.session_state[prefix + "exit_z"]), step=0.25,
                format="%.2f", disabled=not use_exit_z, key=prefix + "bt_exit_z",
                help="Long exits when ZScore >= this. Short exits when ZScore <= this.",
            )
        with c4:
            use_exit_zs = st.checkbox("Use Z-Score of Std for take-profit", value=st.session_state[prefix + "use_exit_zs"],
                                       key=prefix + "bt_use_exit_zs")
            exit_zs = st.number_input(
                "Take-Profit Z-Score of Std threshold", value=float(st.session_state[prefix + "exit_zs"]), step=0.25,
                format="%.2f", disabled=not use_exit_zs, key=prefix + "bt_exit_zs",
            )
            exit_zs_op = st.selectbox(
                "Comparison", ["<=", ">="], index=0 if st.session_state[prefix + "exit_zs_op"] == "<=" else 1,
                disabled=not use_exit_zs, key=prefix + "bt_exit_zs_op",
            )
        if use_exit_z and use_exit_zs:
            st.caption("Both take-profit conditions are active -- exits as soon as EITHER one fires.")
        st.session_state.update({
            prefix + "use_exit_z": use_exit_z, prefix + "exit_z": exit_z,
            prefix + "use_exit_zs": use_exit_zs, prefix + "exit_zs": exit_zs, prefix + "exit_zs_op": exit_zs_op,
        })

        st.subheader("Stop Loss")
        c_sl1, c_sl2 = st.columns(2)
        with c_sl1:
            use_stop_loss = st.checkbox("Use Stop Loss by Z-Score",
                                         value=st.session_state[prefix + "use_stop_loss"], key=prefix + "bt_use_stop_loss")
            sl_z = st.number_input(
                "Stop-Loss Z-Score threshold", value=float(st.session_state[prefix + "sl_z"]), step=0.25,
                format="%.2f", disabled=not use_stop_loss, key=prefix + "bt_sl_z",
                help="Long stops out when ZScore <= this (should be further negative than entry). "
                     "Short stops out when ZScore >= this (should be further positive than entry).",
            )
        with c_sl2:
            use_sl_value = st.checkbox("Use Stop Loss by Value",
                                        value=st.session_state[prefix + "use_sl_value"], key=prefix + "bt_use_sl_value")
            sl_value = st.number_input(
                "Stop-Loss value (price units, always positive)", value=float(st.session_state[prefix + "sl_value"]),
                step=0.25, min_value=0.0, format="%.2f", disabled=not use_sl_value, key=prefix + "bt_sl_value",
                help="Stops out once the position's unrealized loss reaches this amount, regardless of "
                     "Z-Score -- a hard risk cap independent of the statistical signal.",
            )
        if use_stop_loss and use_sl_value:
            st.caption("Both stop-loss conditions are active -- exits as soon as EITHER one fires.")
        st.session_state.update({
            prefix + "use_stop_loss": use_stop_loss, prefix + "sl_z": sl_z,
            prefix + "use_sl_value": use_sl_value, prefix + "sl_value": sl_value,
        })

        use_vol_exit_filter = st.checkbox(
            "Also force-exit if Z-Score of Std spikes (extra risk overlay)",
            value=st.session_state[prefix + "use_vol_exit_filter"], key=prefix + "bt_use_vol_exit_filter",
        )
        exit_zs_max = st.session_state[prefix + "exit_zs_max"]
        if use_vol_exit_filter:
            exit_zs_max = st.number_input("Force-exit if |Z-Score of Std| >=", value=float(exit_zs_max), step=0.25,
                                           key=prefix + "bt_exit_zs_max")
        st.session_state[prefix + "use_vol_exit_filter"], st.session_state[prefix + "exit_zs_max"] = use_vol_exit_filter, exit_zs_max

        st.subheader("Cost")
        use_trade_cost = st.checkbox("Apply a cost per trade", value=st.session_state[prefix + "use_trade_cost"],
                                      key=prefix + "bt_use_trade_cost")
        cost_per_trade = st.session_state[prefix + "cost_per_trade"]
        if use_trade_cost:
            cost_per_trade = st.number_input(
                "Cost per trade (price units, deducted once per round-trip)",
                value=float(cost_per_trade), step=0.01, min_value=0.0, format="%.4f", key=prefix + "bt_cost_per_trade",
                help="Deducted from each trade's PnL before win/loss and all totals are computed -- "
                     "so it affects the trade log, year summaries, equity curve, and win rate.",
            )
        st.session_state[prefix + "use_trade_cost"], st.session_state[prefix + "cost_per_trade"] = use_trade_cost, cost_per_trade

        config = StrategyConfig(
            direction=direction,
            use_entry_z=use_entry_z, entry_z=entry_z,
            use_entry_zs=use_entry_zs, entry_zs=entry_zs, entry_zs_op=entry_zs_op,
            entry_logic=entry_logic,
            use_exit_z=use_exit_z, exit_z=exit_z,
            use_exit_zs=use_exit_zs, exit_zs=exit_zs, exit_zs_op=exit_zs_op,
            use_stop_loss=use_stop_loss, sl_z=sl_z,
            use_sl_value=use_sl_value, sl_value=sl_value,
            use_vol_exit_filter=use_vol_exit_filter, exit_zs_max=exit_zs_max,
            use_trade_cost=use_trade_cost, cost_per_trade=cost_per_trade,
        )
        if not config.is_valid:
            st.warning("Enable at least one entry condition and at least one take-profit condition "
                       "(each independently Z-Score and/or Z-Score of Std).")
            return

        bt_engine = BacktestEngine(config)
        year_results = bt_engine.run_all_years(stats_by_year)
        all_trades = [t for yr in year_results.values() for t in yr.trades]

        total_pnl = sum(t.pnl for t in all_trades)
        wins = sum(1 for t in all_trades if t.pnl > 0)
        win_rate = (wins / len(all_trades)) if all_trades else None

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", str(len(all_trades)))
        m2.metric("Total P&L", f"{total_pnl:.4f}")
        m3.metric("Win Rate", f"{win_rate:.1%}" if win_rate is not None else "N/A")
        m4.metric("Years Covered", str(len(stats_by_year)))
        avg_holding_overall = (
            float(np.mean([(t.exit_date - t.entry_date).days for t in all_trades])) if all_trades else None
        )
        m5.metric("Avg Holding Days", f"{avg_holding_overall:.1f}" if avg_holding_overall is not None else "N/A")
        avg_heat_overall = float(np.mean([t.heat for t in all_trades])) if all_trades else None
        m6.metric("Avg Heat", f"{avg_heat_overall:.4f}" if avg_heat_overall is not None else "N/A")

        st.subheader("Results by Trading Year")
        year_rows = []
        for year in sorted(year_results.keys()):
            yr = year_results[year]
            year_rows.append({
                "Year": year, "Trades": yr.trade_count,
                "Win Rate": f"{yr.win_rate:.1%}" if yr.win_rate is not None else "N/A",
                "Total P&L": round(yr.total_pnl, 4),
                "Avg P&L/Trade": round(yr.avg_pnl, 4) if yr.avg_pnl is not None else None,
                "Avg Holding Days": round(yr.avg_holding_days, 1) if yr.avg_holding_days is not None else None,
                "Avg Heat": round(yr.avg_heat, 4) if yr.avg_heat is not None else None,
                "Max Drawdown": round(yr.max_drawdown, 4),
            })
        st.dataframe(pd.DataFrame(year_rows), use_container_width=True, hide_index=True)

        if all_trades:
            st.plotly_chart(self.chart_builder.equity_curve(all_trades), use_container_width=True,
                             key=prefix + "bt_chart_equity")
            st.plotly_chart(self.chart_builder.yearly_pnl_bar(year_results), use_container_width=True,
                             key=prefix + "bt_chart_yearly_pnl")

            st.subheader("Trade Log")
            trade_rows = [{
                "Year": t.year, "Direction": "Long" if t.direction == 1 else "Short",
                "Entry Date": t.entry_date.date(), "Entry Price": round(t.entry_price, 4),
                "Entry Z": round(t.entry_zscore, 3), "Entry ZofStd": round(t.entry_zscore_of_std, 3),
                "Exit Date": t.exit_date.date(), "Exit Price": round(t.exit_price, 4),
                "Exit Z": round(t.exit_zscore, 3), "Exit ZofStd": round(t.exit_zscore_of_std, 3),
                "Gross PnL": round(t.raw_pnl, 4), "Cost": round(t.cost, 4),
                "Net PnL": round(t.pnl, 4), "PnL %": round(t.pnl_pct * 100, 2) if not np.isnan(t.pnl_pct) else None,
                "Heat": round(t.heat, 4), "Heat %": round(t.heat_pct * 100, 2) if not np.isnan(t.heat_pct) else None,
                "Exit Reason": t.exit_reason,
            } for t in all_trades]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

            st.subheader("Price Chart with Entries/Exits")
            year_choice = st.selectbox("Year to visualize", sorted(stats_by_year.keys()), index=len(stats_by_year) - 1,
                                        key=prefix + "bt_year_choice")
            st.plotly_chart(
                self.chart_builder.price_with_trades(stats_by_year[year_choice], year_results[year_choice].trades, year_choice),
                use_container_width=True, key=prefix + "bt_chart_price",
            )

            st.subheader("Z-Score Chart with Entries/Exits")
            st.plotly_chart(
                self.chart_builder.indicator_with_trades(
                    stats_by_year[year_choice], year_results[year_choice].trades, year_choice, "ZScore", "Z-Score"
                ),
                use_container_width=True, key=prefix + "bt_chart_zscore",
            )

            st.subheader("Z-Score of Std Chart with Entries/Exits")
            st.plotly_chart(
                self.chart_builder.indicator_with_trades(
                    stats_by_year[year_choice], year_results[year_choice].trades, year_choice,
                    "ZScoreOfStd", "Z-Score of Std",
                ),
                use_container_width=True, key=prefix + "bt_chart_zscoreofstd",
            )
        else:
            st.caption("No trades were triggered with these settings.")

    # ------------------------------- Section 2 --------------------------------
    def _render_optimizer_section(self, stats_by_year: Dict[str, pd.DataFrame], prefix: str = "") -> None:
        st.header("2️⃣ Strategy Optimizer")
        st.caption(
            "Systematic parameter search across a grid of entry/exit thresholds -- every candidate "
            "is backtested with the exact same engine used in Section 1, so results are directly "
            "comparable and reproducible. This is not a black-box model; it's an exhaustive, "
            "verifiable search over the rule's own parameters."
        )

        st.markdown("**Structure** (fixed choices -- the search explores threshold values within these)")
        c1, c2 = st.columns(2)
        with c1:
            opt_direction_choice = st.radio(
                "Direction", ["Long", "Short"], horizontal=True, key=prefix + "opt_direction",
            )
            opt_direction = 1 if opt_direction_choice == "Long" else -1
            opt_use_entry_z = st.checkbox("Search Entry Z-Score", value=True, key=prefix + "opt_use_entry_z")
            opt_use_entry_zs = st.checkbox("Search Entry Z-Score of Std", value=False, key=prefix + "opt_use_entry_zs")
            opt_use_exit_z = st.checkbox("Search Take-Profit Z-Score", value=True, key=prefix + "opt_use_exit_z")
            opt_use_exit_zs = st.checkbox("Search Take-Profit Z-Score of Std", value=False, key=prefix + "opt_use_exit_zs")
        with c2:
            opt_entry_logic = "AND"
            if opt_use_entry_z and opt_use_entry_zs:
                opt_entry_logic = st.radio("Combine entry conditions", ["AND", "OR"], horizontal=True,
                                            key=prefix + "opt_entry_logic")
            opt_entry_zs_op = st.selectbox(
                "Entry Z-Score of Std comparison", ["<=", ">="], key=prefix + "opt_entry_zs_op",
                disabled=not opt_use_entry_zs,
            )
            opt_exit_zs_op = st.selectbox(
                "Take-Profit Z-Score of Std comparison", ["<=", ">="], key=prefix + "opt_exit_zs_op",
                disabled=not opt_use_exit_zs,
            )
            opt_use_sl = st.checkbox("Search Stop-Loss by Z-Score", value=True, key=prefix + "opt_use_sl")
            opt_use_sl_value = st.checkbox("Search Stop-Loss by Value", value=False, key=prefix + "opt_use_sl_value")
            opt_use_vol_exit = st.checkbox("Search vol-spike force-exit filter", value=False, key=prefix + "opt_use_vol_exit")

        if not opt_use_entry_z and not opt_use_entry_zs:
            st.warning("Enable at least one entry condition to search.")
            return
        if not opt_use_exit_z and not opt_use_exit_zs:
            st.warning("Enable at least one take-profit condition to search.")
            return
        st.caption(
            "Tip: to search a strategy driven ENTIRELY by Z-Score of Std (no price Z-Score at all), "
            "uncheck 'Search Entry Z-Score' and 'Search Take-Profit Z-Score', leaving only the "
            "Z-Score of Std boxes checked."
        )

        st.markdown("**Search ranges**")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.caption("Entry Z-Score")
            entry_min = st.number_input("min", value=-3.0, step=0.25, key=prefix + "opt_entry_min", disabled=not opt_use_entry_z)
            entry_max = st.number_input("max", value=-0.5, step=0.25, key=prefix + "opt_entry_max", disabled=not opt_use_entry_z)
            entry_step = st.number_input("step", value=0.25, step=0.05, min_value=0.05, key=prefix + "opt_entry_step", disabled=not opt_use_entry_z)
        with r2:
            st.caption("Take-Profit Z-Score")
            exit_min = st.number_input("min", value=-1.5, step=0.25, key=prefix + "opt_exit_min", disabled=not opt_use_exit_z)
            exit_max = st.number_input("max", value=1.5, step=0.25, key=prefix + "opt_exit_max", disabled=not opt_use_exit_z)
            exit_step = st.number_input("step", value=0.25, step=0.05, min_value=0.05, key=prefix + "opt_exit_step", disabled=not opt_use_exit_z)
        with r3:
            st.caption("Stop-Loss Z-Score")
            sl_min = st.number_input("min", value=-5.0, step=0.25, key=prefix + "opt_sl_min", disabled=not opt_use_sl)
            sl_max = st.number_input("max", value=-3.0, step=0.25, key=prefix + "opt_sl_max", disabled=not opt_use_sl)
            sl_step = st.number_input("step", value=0.25, step=0.05, min_value=0.05, key=prefix + "opt_sl_step", disabled=not opt_use_sl)

        r4, r5, r6, r7 = st.columns(4)
        with r4:
            st.caption("Entry Z-Score of Std")
            entry_zs_min = st.number_input("min", value=0.0, step=0.25, key=prefix + "opt_entry_zs_min", disabled=not opt_use_entry_zs)
            entry_zs_max_ = st.number_input("max", value=2.0, step=0.25, key=prefix + "opt_entry_zs_max", disabled=not opt_use_entry_zs)
            entry_zs_step = st.number_input("step", value=0.5, step=0.05, min_value=0.05, key=prefix + "opt_entry_zs_step", disabled=not opt_use_entry_zs)
        with r5:
            st.caption("Take-Profit Z-Score of Std")
            exit_zs_min = st.number_input("min", value=-1.0, step=0.25, key=prefix + "opt_exit_zs_min", disabled=not opt_use_exit_zs)
            exit_zs_max_ = st.number_input("max", value=1.0, step=0.25, key=prefix + "opt_exit_zs_max", disabled=not opt_use_exit_zs)
            exit_zs_step = st.number_input("step", value=0.5, step=0.05, min_value=0.05, key=prefix + "opt_exit_zs_step", disabled=not opt_use_exit_zs)
        with r6:
            st.caption("Stop-Loss by Value")
            sl_value_min = st.number_input("min", value=1.0, step=0.25, min_value=0.0, key=prefix + "opt_sl_value_min", disabled=not opt_use_sl_value)
            sl_value_max = st.number_input("max", value=5.0, step=0.25, min_value=0.0, key=prefix + "opt_sl_value_max", disabled=not opt_use_sl_value)
            sl_value_step = st.number_input("step", value=0.5, step=0.05, min_value=0.05, key=prefix + "opt_sl_value_step", disabled=not opt_use_sl_value)
        with r7:
            st.caption("Vol-spike force-exit threshold")
            vol_min = st.number_input("min", value=2.5, step=0.25, key=prefix + "opt_vol_min", disabled=not opt_use_vol_exit)
            vol_max = st.number_input("max", value=4.0, step=0.25, key=prefix + "opt_vol_max", disabled=not opt_use_vol_exit)
            vol_step = st.number_input("step", value=0.5, step=0.05, min_value=0.05, key=prefix + "opt_vol_step", disabled=not opt_use_vol_exit)

        c_cost1, c_cost2 = st.columns(2)
        opt_use_trade_cost = c_cost1.checkbox("Apply cost per trade", value=st.session_state[prefix + "use_trade_cost"],
                                               key=prefix + "opt_use_cost")
        opt_cost_per_trade = c_cost2.number_input(
            "Cost per trade", value=float(st.session_state[prefix + "cost_per_trade"]), step=0.01, min_value=0.0,
            format="%.4f", disabled=not opt_use_trade_cost, key=prefix + "opt_cost_value",
        )

        objective = st.selectbox("Optimize for", ["total_pnl", "win_rate", "profit_factor"], index=0,
                                  key=prefix + "opt_objective")
        min_trades = st.number_input("Minimum trades required to consider a combo valid", value=3, min_value=1, step=1,
                                      key=prefix + "opt_min_trades")

        if st.button("🚀 Run Optimization", use_container_width=True, key=prefix + "opt_run_button"):
            def build_grid(lo, hi, step):
                return [round(x, 4) for x in np.arange(lo, hi + step / 2, step)]

            entry_grid = build_grid(entry_min, entry_max, entry_step) if opt_use_entry_z else [0.0]
            exit_grid = build_grid(exit_min, exit_max, exit_step) if opt_use_exit_z else [0.0]
            sl_grid = build_grid(sl_min, sl_max, sl_step) if opt_use_sl else [0.0]
            sl_value_grid = build_grid(sl_value_min, sl_value_max, sl_value_step) if opt_use_sl_value else [0.0]
            entry_zs_grid = build_grid(entry_zs_min, entry_zs_max_, entry_zs_step) if opt_use_entry_zs else [0.0]
            exit_zs_grid = build_grid(exit_zs_min, exit_zs_max_, exit_zs_step) if opt_use_exit_zs else [0.0]
            vol_grid = build_grid(vol_min, vol_max, vol_step) if opt_use_vol_exit else [0.0]

            total_combos = (
                len(entry_grid) * len(entry_zs_grid) * len(exit_grid) * len(exit_zs_grid)
                * len(sl_grid) * len(sl_value_grid) * len(vol_grid)
            )
            # Measured throughput is roughly 1-2 ms/combo per ~2,500 price rows (varies with
            # trade frequency and years included); shown as a rough heads-up only.
            est_seconds = total_combos * 0.0017 * (sum(len(df) for df in stats_by_year.values()) / 2500)
            if total_combos > 5000:
                st.caption(f"⏱️ Roughly {total_combos:,} combinations -- estimated ~{est_seconds:.0f}s. "
                           f"Narrow the ranges or increase step sizes to speed this up.")

            optimizer = StrategyOptimizer(stats_by_year, min_trades=int(min_trades))
            with st.spinner(f"Evaluating up to {total_combos:,} combinations..."):
                results = optimizer.search(
                    direction=opt_direction,
                    use_entry_z=opt_use_entry_z, entry_z_grid=entry_grid,
                    use_entry_zs=opt_use_entry_zs, entry_zs_grid=entry_zs_grid, entry_zs_op=opt_entry_zs_op,
                    entry_logic=opt_entry_logic,
                    use_exit_z=opt_use_exit_z, exit_z_grid=exit_grid,
                    use_exit_zs=opt_use_exit_zs, exit_zs_grid=exit_zs_grid, exit_zs_op=opt_exit_zs_op,
                    use_stop_loss=opt_use_sl, sl_z_grid=sl_grid,
                    use_sl_value=opt_use_sl_value, sl_value_grid=sl_value_grid,
                    use_vol_exit_filter=opt_use_vol_exit, exit_zs_max_grid=vol_grid,
                    use_trade_cost=opt_use_trade_cost, cost_per_trade=opt_cost_per_trade,
                    objective=objective,
                )
            st.session_state[prefix + "optimizer_results"] = results

        results: Optional[List[OptimizerResult]] = st.session_state.get(prefix + "optimizer_results")
        if results:
            st.subheader(f"Top Results (by {objective})")
            top_rows = []
            for r in results[:20]:
                top_rows.append({
                    "Direction": r.config.direction_label,
                    "Entry Z": r.config.entry_z if r.config.use_entry_z else "off",
                    "Entry ZofStd": f"{r.config.entry_zs_op}{r.config.entry_zs}" if r.config.use_entry_zs else "off",
                    "Entry Logic": r.config.entry_logic if (r.config.use_entry_z and r.config.use_entry_zs) else "-",
                    "TP Z": r.config.exit_z if r.config.use_exit_z else "off",
                    "TP ZofStd": f"{r.config.exit_zs_op}{r.config.exit_zs}" if r.config.use_exit_zs else "off",
                    "Stop-Loss Z": r.config.sl_z if r.config.use_stop_loss else "off",
                    "Stop-Loss Value": r.config.sl_value if r.config.use_sl_value else "off",
                    "Vol Exit": r.config.exit_zs_max if r.config.use_vol_exit_filter else "off",
                    "Trades": r.trade_count,
                    "Avg Holding Days": round(r.avg_holding_days, 1) if r.avg_holding_days is not None else None,
                    "Avg Heat": round(r.avg_heat, 4) if r.avg_heat is not None else None,
                    "Total P&L": round(r.total_pnl, 4),
                    "Win Rate": f"{r.win_rate:.1%}" if r.win_rate is not None else "N/A",
                    "Profit Factor": round(r.profit_factor, 2) if r.profit_factor not in (None, float("inf")) else (
                        "∞" if r.profit_factor == float("inf") else "N/A"
                    ),
                })
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

            if st.button("✅ Load Best Result into Backtest (Section 1)", use_container_width=True,
                         key=prefix + "opt_load_best_button"):
                best = results[0].config
                st.session_state.update({
                    prefix + "direction": best.direction,
                    prefix + "use_entry_z": best.use_entry_z, prefix + "entry_z": best.entry_z,
                    prefix + "use_entry_zs": best.use_entry_zs, prefix + "entry_zs": best.entry_zs,
                    prefix + "entry_zs_op": best.entry_zs_op,
                    prefix + "entry_logic": best.entry_logic,
                    prefix + "use_exit_z": best.use_exit_z, prefix + "exit_z": best.exit_z,
                    prefix + "use_exit_zs": best.use_exit_zs, prefix + "exit_zs": best.exit_zs,
                    prefix + "exit_zs_op": best.exit_zs_op,
                    prefix + "use_stop_loss": best.use_stop_loss, prefix + "sl_z": best.sl_z,
                    prefix + "use_sl_value": best.use_sl_value, prefix + "sl_value": best.sl_value,
                    prefix + "use_vol_exit_filter": best.use_vol_exit_filter, prefix + "exit_zs_max": best.exit_zs_max,
                    prefix + "use_trade_cost": best.use_trade_cost, prefix + "cost_per_trade": best.cost_per_trade,
                })
                st.rerun()
        elif st.session_state.get(prefix + "optimizer_results") is not None:
            st.warning("No combination met the minimum trade count with these settings. Try widening the search "
                       "ranges or lowering the minimum trades required.")

    # ------------------------------ Portfolio mode ------------------------------
    # ------------------------------ Portfolio mode ------------------------------
    def _render_portfolio_mode(self) -> None:
        contracts: Dict[str, ContractData] = st.session_state["portfolio_contracts"]
        if not contracts:
            st.info(
                "Upload multiple contract Excel files in the sidebar to begin. Each contract gets its "
                "own full Backtest and Optimizer section below -- set its own direction, entry/exit "
                "conditions, and search ranges, exactly like Single Contract mode. Run the Optimizer "
                "within each contract's section to find ITS OWN best individual entry/exit Z-Score "
                "combination -- the Combined Overview above automatically stacks each contract's own "
                "best result into one overall portfolio profit figure."
            )
            return

        invalid = {name: c for name, c in contracts.items() if not c.is_valid}
        if invalid:
            with st.expander(f"\u26a0\ufe0f {len(invalid)} file(s) could not be loaded", expanded=True):
                for name, c in invalid.items():
                    st.error(f"**{name}**: {c.load_error}")
        valid_contracts = {name: c for name, c in contracts.items() if c.is_valid}
        if not valid_contracts:
            return

        window = int(st.session_state["window"])
        engine = RollingStatsEngine(window=window)

        self._render_portfolio_overview(valid_contracts, engine)

        st.divider()
        st.caption(
            f"{len(valid_contracts)} contract(s) loaded. Set up and run the Optimizer within each "
            f"contract's own section below (its own direction, entry/exit conditions, and search "
            f"ranges) to find ITS OWN best individual combination -- the Combined Overview above "
            f"updates automatically once a contract has a result."
        )
        for name in sorted(valid_contracts):
            st.divider()
            st.header(f"\U0001F4C4 {name}")
            full_stats_by_year = {y: engine.compute_full(df) for y, df in valid_contracts[name].year_data.items()}
            self._render_single_contract_view(full_stats_by_year, prefix=f"pf_{name}_")

    def _render_portfolio_overview(self, valid_contracts: Dict[str, ContractData], engine: "RollingStatsEngine") -> None:
        """Combines whichever contracts have already had their OWN Optimizer
        run (in their own section below, with their own direction and search
        ranges) into one overall portfolio result -- each contract keeps its
        own independently-best entry/exit combination; this just stacks them
        together. Maximizing each contract's own P&L independently and
        summing is mathematically the same as maximizing the portfolio total,
        since the contracts don't interact."""
        overview = []
        per_series_equity: Dict[str, pd.DataFrame] = {}
        for name, c in valid_contracts.items():
            prefix = f"pf_{name}_"
            results = st.session_state.get(prefix + "optimizer_results")
            if not results:
                continue
            best = results[0]

            full_stats_by_year = {y: engine.compute_full(df) for y, df in c.year_data.items()}
            all_years = sorted(full_stats_by_year.keys())
            included_years = st.session_state.get(prefix + "included_years") or all_years
            stats_by_year = {y: full_stats_by_year[y] for y in included_years if y in full_stats_by_year}
            if st.session_state.get(prefix + "use_day_range"):
                stats_by_year = restrict_to_day_range(
                    stats_by_year,
                    int(st.session_state.get(prefix + "day_min", 0) or 0),
                    int(st.session_state.get(prefix + "day_max", 0) or 0),
                )

            bt_engine = BacktestEngine(best.config)
            year_results = bt_engine.run_all_years(stats_by_year)
            trades = [t for yr in year_results.values() for t in yr.trades]
            overview.append({"contract": name, "result": best, "trades": trades})

            # Day-by-day mark-to-market equity for EVERY (contract, year) instance, on that
            # contract-year's own Trading Day # axis -- this is what the combined portfolio
            # Heat chart below sums across, so overlapping exposure across your whole
            # portfolio (and across each contract's own historical cycles) compounds
            # correctly on a shared "trading day" timeline, not just as separate summary
            # numbers.
            for year, yr_result in year_results.items():
                if year not in stats_by_year:
                    continue
                per_series_equity[f"{name} {year}"] = compute_daily_equity(
                    stats_by_year[year], yr_result.trades, best.config.direction
                )

        st.header("\U0001F4CA Combined Portfolio Overview")
        st.caption(
            "Each contract below finds ITS OWN best individual entry/exit Z-Score combination "
            "(using its own search ranges you set in its own section). This combines those "
            "independent results into one overall portfolio profit figure -- run the Optimizer in "
            "any contract's section below to add it here."
        )

        if not overview:
            st.info(
                "No contract has Optimizer results yet. Scroll to any contract's section below, set "
                "its direction and search ranges, and click Run Optimization -- it will appear here "
                "automatically."
            )
            return

        all_trades = [t for r in overview for t in r["trades"]]
        overall_pnl = sum(t.pnl for t in all_trades)
        overall_wins = sum(1 for t in all_trades if t.pnl > 0)
        overall_win_rate = (overall_wins / len(all_trades)) if all_trades else None
        overall_avg_heat = float(np.mean([t.heat for t in all_trades])) if all_trades else None
        overall_avg_holding = (
            float(np.mean([(t.exit_date - t.entry_date).days for t in all_trades])) if all_trades else None
        )

        combined_daily = combine_daily_equity(per_series_equity) if per_series_equity else pd.DataFrame()
        if not combined_daily.empty:
            combined_daily["Heat"] = compute_heat_series(combined_daily["Portfolio_Equity"])
            portfolio_max_heat = float(combined_daily["Heat"].max())
        else:
            portfolio_max_heat = None

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Contracts Included", f"{len(overview)} / {len(valid_contracts)}")
        m2.metric("Total Trades", str(len(all_trades)))
        m3.metric("Overall Total P&L", f"{overall_pnl:.4f}")
        m4.metric("Overall Win Rate", f"{overall_win_rate:.1%}" if overall_win_rate is not None else "N/A")
        m5.metric("Overall Avg Heat", f"{overall_avg_heat:.4f}" if overall_avg_heat is not None else "N/A")
        m6.metric("Overall Avg Holding Days", f"{overall_avg_holding:.1f}" if overall_avg_holding is not None else "N/A")
        m7.metric("Portfolio Max Heat (combined, worst trading day)",
                   f"{portfolio_max_heat:.4f}" if portfolio_max_heat is not None else "N/A")

        st.subheader("Each Contract's Own Best Individual Entry/Exit")
        rows = []
        for r in overview:
            cfg = r["result"].config
            rows.append({
                "Contract": r["contract"], "Direction": cfg.direction_label,
                "Entry Z": cfg.entry_z if cfg.use_entry_z else "off",
                "Entry ZofStd": f"{cfg.entry_zs_op}{cfg.entry_zs}" if cfg.use_entry_zs else "off",
                "TP Z": cfg.exit_z if cfg.use_exit_z else "off",
                "TP ZofStd": f"{cfg.exit_zs_op}{cfg.exit_zs}" if cfg.use_exit_zs else "off",
                "Stop-Loss Z": cfg.sl_z if cfg.use_stop_loss else "off",
                "Trades": r["result"].trade_count,
                "Total P&L": round(r["result"].total_pnl, 4),
                "Win Rate": f"{r['result'].win_rate:.1%}" if r["result"].win_rate is not None else "N/A",
                "Avg Heat": round(r["result"].avg_heat, 4) if r["result"].avg_heat is not None else None,
                "Avg Holding Days": round(r["result"].avg_holding_days, 1) if r["result"].avg_holding_days is not None else None,
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button(
            "\u2b07\ufe0f Export Combined Results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="Portfolio_Combined_Results.csv",
            mime="text/csv",
        )

        if not combined_daily.empty:
            st.subheader("Combined Portfolio Equity & Heat by Trading Day")
            st.caption(
                "Every included (contract, year) instance's day-by-day mark-to-market exposure, summed "
                "together on the shared Trading Day # axis -- this is what shows overlapping risk across "
                "your whole portfolio at any point, not just each trade's own separate Heat number."
            )
            st.plotly_chart(self.chart_builder.portfolio_heat_chart(combined_daily), use_container_width=True,
                             key="portfolio_combined_heat_chart")

        if all_trades:
            st.subheader("Equity Curve (trade-by-trade, chronological)")
            st.plotly_chart(self.chart_builder.equity_curve(all_trades), use_container_width=True,
                             key="portfolio_overview_equity_chart")

    def run(self) -> None:
        self.configure_page()
        try:
            self.render_sidebar()
            self.render_main()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled application error")
            st.error(f"An unexpected error occurred: {exc}")


def main() -> None:
    app = BacktestApp()
    app.run()


if __name__ == "__main__":
    main()

