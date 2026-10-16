"""
Seasonality Analysis Dashboard
===============================

A production-quality Streamlit + Plotly dashboard for analyzing seasonal
price patterns across an arbitrary number of years.

Features
--------
- Automatic detection of an unlimited number of "Timestamp.<YEAR>" / "<YEAR>"
  column pairs in the source CSV (no hardcoded years).
- Per-year data cleaning (numeric coercion, NA/blank handling, row drops).
- Three interactive Plotly charts: raw seasonality, rolling standard
  deviation, and rolling z-score.
- A live statistics panel and a searchable/sortable/filterable data table.
- One-click Excel (.xlsx) and standalone HTML export.

Run with:
    streamlit run seasonality_dashboard.py

Author: Generated for Samihan
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seasonality_dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH = r"C:\Users\samihan.joshi\Downloads\Data csv.csv"
DEFAULT_ROLLING_WINDOW = 20
COLOR_PALETTE = [
    "#00CC96", "#EF553B", "#636EFA", "#FFA15A", "#AB63FA",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#00A08B", "#C97B84", "#7A9E9F", "#FF7F50", "#6A5ACD",
]
CURRENT_YEAR_COLOR = "#FFFFFF"
DARK_BG = "#0E1117"
PANEL_BG = "#151A24"
GRID_COLOR = "rgba(255,255,255,0.08)"


# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------
class SeasonalityDataLoader:
    """Loads a wide-format seasonality CSV and cleans each year independently.

    The CSV is expected to contain repeating column pairs of the form:
        Timestamp.<YEAR>, <YEAR>, Timestamp.<YEAR>, <YEAR>, ...
    The number of years is not known ahead of time and is detected
    automatically via column-name pattern matching.
    """

    YEAR_COL_PATTERN = re.compile(r"^\d{4}$")
    TIMESTAMP_COL_PATTERN = re.compile(r"^Timestamp\.(\d{4})$", re.IGNORECASE)

    def __init__(self, source):
        """
        Parameters
        ----------
        source : str | io.BytesIO | io.StringIO
            A filesystem path or an in-memory buffer (e.g. from
            st.file_uploader) pointing to the CSV data.
        """
        self.source = source
        self.raw_df: Optional[pd.DataFrame] = None
        self.year_data: Dict[str, pd.DataFrame] = {}

    def load(self) -> Dict[str, pd.DataFrame]:
        """Read the CSV, detect year columns, and clean each year's series."""
        self.raw_df = self._read_csv()

        year_pairs = self._detect_year_columns(self.raw_df.columns.tolist())
        if not year_pairs:
            raise ValueError(
                "No valid 'Timestamp.<YEAR>' / '<YEAR>' column pairs were "
                "detected in the CSV header. Please check the file format."
            )

        for year, (ts_col, price_col) in year_pairs.items():
            try:
                cleaned = self._clean_year(self.raw_df, ts_col, price_col)
            except Exception as exc:  # noqa: BLE001 - surface per-year failure
                logger.error("Failed to clean year %s: %s", year, exc)
                continue

            if not cleaned.empty:
                self.year_data[year] = cleaned
            else:
                logger.warning("Year %s produced no valid rows; skipping.", year)

        if not self.year_data:
            raise ValueError("Every detected year was empty after cleaning.")

        # Sort dict by year for deterministic downstream ordering
        self.year_data = dict(sorted(self.year_data.items(), key=lambda kv: kv[0]))
        return self.year_data

    def _read_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.source)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"CSV file not found: {self.source}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV file is empty: {self.source}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV could not be parsed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected error reading CSV: {exc}") from exc

    def _detect_year_columns(self, columns: List[str]) -> Dict[str, Tuple[str, str]]:
        """Pair up Timestamp.<YEAR> and <YEAR> columns by matching year label."""
        price_cols: Dict[str, str] = {}
        ts_cols: Dict[str, str] = {}

        for col in columns:
            col_str = str(col).strip()
            if self.YEAR_COL_PATTERN.match(col_str):
                price_cols[col_str] = col
                continue
            match = self.TIMESTAMP_COL_PATTERN.match(col_str)
            if match:
                ts_cols[match.group(1)] = col

        pairs: Dict[str, Tuple[str, str]] = {}
        for year, price_col in price_cols.items():
            ts_col = ts_cols.get(year)
            if ts_col is not None:
                pairs[year] = (ts_col, price_col)
            else:
                logger.warning("No Timestamp column matched for year %s", year)

        if not pairs:
            pairs = self._positional_fallback(columns)

        return pairs

    @staticmethod
    def _positional_fallback(columns: List[str]) -> Dict[str, Tuple[str, str]]:
        """Fallback: assume columns alternate (timestamp, price) pairs."""
        pairs: Dict[str, Tuple[str, str]] = {}
        idx = 0
        anon_counter = 0
        while idx < len(columns) - 1:
            ts_col, price_col = columns[idx], columns[idx + 1]
            label_match = re.search(r"(\d{4})", str(price_col))
            year_label = label_match.group(1) if label_match else f"Series{anon_counter}"
            pairs[year_label] = (ts_col, price_col)
            anon_counter += 1
            idx += 2
        return pairs

    @staticmethod
    def _clean_year(df: pd.DataFrame, ts_col: str, price_col: str) -> pd.DataFrame:
        """Clean a single year's Timestamp/Price columns.

        - Coerces price to numeric, turning NA/blank/invalid entries into NaN.
        - Drops rows where price is NaN (and the paired timestamp along with it).
        - Parses dates and derives helper columns for seasonal alignment.
        """
        sub = df[[ts_col, price_col]].copy()
        sub.columns = ["Date", "Price"]

        # Normalize and coerce price to numeric; invalid tokens -> NaN
        price_str = sub["Price"].astype(str).str.strip()
        price_str = price_str.replace({"NA": np.nan, "N/A": np.nan, "": np.nan, "nan": np.nan})
        sub["Price"] = pd.to_numeric(price_str, errors="coerce")

        # Parse dates; unparseable dates become NaT
        sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")

        # Drop rows with missing price (timestamp drops along with it)
        sub = sub.dropna(subset=["Price"])
        # Drop rows with unparseable dates
        sub = sub.dropna(subset=["Date"])

        sub = sub.sort_values("Date").reset_index(drop=True)

        # Helper columns for seasonal alignment across years.
        # DayIndex = trading-day offset from THIS series' own first date (0, 1, 2, ...).
        # This is the correct alignment axis for contract years that don't start
        # on Jan 1 (e.g. a "2026" contract that begins trading in Jul 2025) --
        # calendar month-day alignment would fold such a series back on itself.
        sub["DayIndex"] = range(len(sub))
        sub["DayOfYear"] = sub["Date"].dt.dayofyear
        sub["MonthDay"] = sub["Date"].dt.strftime("%m-%d")
        sub["DateLabel"] = sub["Date"].dt.strftime("%Y-%m-%d")

        return sub


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------
class SeasonalityAnalyzer:
    """Computes rolling standard deviation and rolling z-score per year."""

    def __init__(self, year_data: Dict[str, pd.DataFrame], window: int = DEFAULT_ROLLING_WINDOW):
        self.year_data = year_data
        self.window = max(2, int(window))

    def compute(self) -> Dict[str, pd.DataFrame]:
        """Return a dict of year -> DataFrame with rolling level-1 and level-2 stats added.

        Level 1 (on Price):   RollingMean, RollingStd, ZScore
        Level 2 (on RollingStd, i.e. "volatility of volatility"):
                               StdOfStd, MeanOfStd, ZScoreOfStd
        """
        results: Dict[str, pd.DataFrame] = {}
        min_periods = max(2, self.window // 2)

        for year, df in self.year_data.items():
            try:
                out = df.copy()
                out["RollingMean"] = out["Price"].rolling(
                    window=self.window, min_periods=min_periods
                ).mean()
                out["RollingStd"] = out["Price"].rolling(
                    window=self.window, min_periods=min_periods
                ).std()
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["ZScore"] = (out["Price"] - out["RollingMean"]) / out["RollingStd"]
                out["ZScore"] = out["ZScore"].replace([np.inf, -np.inf], np.nan)

                # --- Level 2: standard deviation of the standard deviation ---
                out["MeanOfStd"] = out["RollingStd"].rolling(
                    window=self.window, min_periods=min_periods
                ).mean()
                out["StdOfStd"] = out["RollingStd"].rolling(
                    window=self.window, min_periods=min_periods
                ).std()
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["ZScoreOfStd"] = (out["RollingStd"] - out["MeanOfStd"]) / out["StdOfStd"]
                out["ZScoreOfStd"] = out["ZScoreOfStd"].replace([np.inf, -np.inf], np.nan)

                results[year] = out
            except Exception as exc:  # noqa: BLE001
                logger.error("Rolling stats failed for year %s: %s", year, exc)

        return results


# ---------------------------------------------------------------------------
# Chart Building
# ---------------------------------------------------------------------------
@dataclass
class ChartTheme:
    """Central place for dashboard color/theme constants."""

    bg: str = DARK_BG
    panel_bg: str = PANEL_BG
    grid: str = GRID_COLOR
    font_color: str = "#E6E6E6"
    palette: List[str] = field(default_factory=lambda: list(COLOR_PALETTE))
    current_color: str = CURRENT_YEAR_COLOR


class ChartBuilder:
    """Builds the three dashboard Plotly figures with a shared dark theme."""

    def __init__(self, theme: Optional[ChartTheme] = None):
        self.theme = theme or ChartTheme()

    def _base_layout(self, title: str, y_title: str, height: int = 520) -> dict:
        return dict(
            title=dict(text=title, font=dict(size=18, color=self.theme.font_color)),
            paper_bgcolor=self.theme.bg,
            plot_bgcolor=self.theme.bg,
            font=dict(color=self.theme.font_color, family="Segoe UI, Roboto, sans-serif"),
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            height=height,
            margin=dict(l=60, r=30, t=70, b=50),
            xaxis=dict(
                title="Date (aligned to current contract's calendar)",
                # Real date axis. Each year's series is shifted forward/back by
                # whole calendar years so it lands exactly on the CURRENT
                # contract's calendar timeline (see _aligned_dates). E.g. if the
                # current contract is "2027" and today is 2026-07-09, a "2026"
                # contract's price from 2025-07-09 shifts to 2026-07-09 -- the
                # same x-position as today's current-contract point. This keeps
                # every contract-year "parallel" by real date, and naturally
                # supports contracts of any length (multi-year, non-Jan-start)
                # without folding a series back on itself.
                type="date",
                tickformat="%b %Y",
                gridcolor=self.theme.grid,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#888888",
                spikethickness=1,
                showline=True,
                linecolor=self.theme.grid,
            ),
            yaxis=dict(
                title=y_title,
                gridcolor=self.theme.grid,
                showspikes=True,
                spikemode="across",
                spikecolor="#888888",
                spikethickness=1,
                showline=True,
                linecolor=self.theme.grid,
            ),
        )

    def _color_for(self, idx: int, year: str, current_year: str) -> Tuple[str, float]:
        if year == current_year:
            return self.theme.current_color, 3.0
        return self.theme.palette[idx % len(self.theme.palette)], 1.6

    @staticmethod
    def _aligned_dates(df: pd.DataFrame, year_label: str, current_year: str) -> pd.Series:
        """Shift this series' real dates by whole calendar years so it lines
        up on the CURRENT contract's calendar timeline.

        Example: current contract is "2027", today is 2026-07-09. For the
        "2026" contract series, shift = 2027 - 2026 = +1 year, so its price
        from 2025-07-09 is plotted at x = 2026-07-09 -- directly parallel
        with the current contract's most recent (today's) point.
        """
        try:
            shift_years = int(current_year) - int(year_label)
        except (TypeError, ValueError):
            shift_years = 0  # non-numeric year labels (positional fallback) -- no shift
        if shift_years == 0:
            return df["Date"]
        return df["Date"] + pd.DateOffset(years=shift_years)

    def build_raw_chart(
        self, year_data: Dict[str, pd.DataFrame], years_to_plot: List[str], current_year: str
    ) -> go.Figure:
        """Chart 1: Raw seasonality — price vs. date, aligned to the current contract's calendar."""
        fig = go.Figure()
        for idx, year in enumerate(years_to_plot):
            df = year_data.get(year)
            if df is None or df.empty:
                continue
            color, width = self._color_for(idx, year, current_year)
            fig.add_trace(
                go.Scatter(
                    x=self._aligned_dates(df, year, current_year),
                    y=df["Price"],
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=width),
                    connectgaps=True,  # draw straight through non-trading days (weekends/holidays)
                    customdata=df["DateLabel"],
                    hovertemplate="<b>" + str(year) + "</b><br>Actual date: %{customdata}"
                    "<br>Price: %{y:.4f}<extra></extra>",
                )
            )
        fig.update_layout(**self._base_layout("Raw Seasonality", "Price"))
        return fig

    def build_std_chart(
        self, rolling_data: Dict[str, pd.DataFrame], years_to_plot: List[str],
        current_year: str, window: int,
    ) -> go.Figure:
        """Chart 2: Rolling standard deviation seasonality, with range slider."""
        fig = go.Figure()
        for idx, year in enumerate(years_to_plot):
            df = rolling_data.get(year)
            if df is None or df.empty:
                continue
            color, width = self._color_for(idx, year, current_year)
            fig.add_trace(
                go.Scatter(
                    x=self._aligned_dates(df, year, current_year),
                    y=df["RollingStd"],
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=width),
                    connectgaps=True,
                    customdata=df["DateLabel"],
                    hovertemplate="<b>" + str(year) + "</b><br>Actual date: %{customdata}"
                    "<br>Std: %{y:.4f}<extra></extra>",
                )
            )
        layout = self._base_layout(f"Rolling Std Dev Seasonality (window={window})", "Rolling Std Dev")
        layout["xaxis"] = {**layout["xaxis"], "rangeslider": dict(visible=True, bgcolor=self.theme.panel_bg)}
        fig.update_layout(**layout)
        return fig

    def build_zscore_chart(
        self, rolling_data: Dict[str, pd.DataFrame], years_to_plot: List[str],
        current_year: str, window: int,
    ) -> go.Figure:
        """Chart 3: Rolling z-score seasonality with +/-1, +/-2, +/-3 reference lines."""
        fig = go.Figure()
        for idx, year in enumerate(years_to_plot):
            df = rolling_data.get(year)
            if df is None or df.empty:
                continue
            color, width = self._color_for(idx, year, current_year)
            fig.add_trace(
                go.Scatter(
                    x=self._aligned_dates(df, year, current_year),
                    y=df["ZScore"],
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=width),
                    connectgaps=True,
                    customdata=df["DateLabel"],
                    hovertemplate="<b>" + str(year) + "</b><br>Actual date: %{customdata}"
                    "<br>Z-Score: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(**self._base_layout(f"Rolling Z-Score Seasonality (window={window})", "Z-Score"))
        self._add_zscore_reference_lines(fig)
        return fig

    def build_std_of_std_chart(
        self, rolling_data: Dict[str, pd.DataFrame], years_to_plot: List[str],
        current_year: str, window: int,
    ) -> go.Figure:
        """Chart 4: Rolling Std-of-Std ("volatility of volatility") seasonality.

        This is the rolling standard deviation of the rolling standard
        deviation series itself -- it measures how stable/unstable the
        volatility regime is, independent of the price level.
        """
        fig = go.Figure()
        for idx, year in enumerate(years_to_plot):
            df = rolling_data.get(year)
            if df is None or df.empty:
                continue
            color, width = self._color_for(idx, year, current_year)
            fig.add_trace(
                go.Scatter(
                    x=self._aligned_dates(df, year, current_year),
                    y=df["StdOfStd"],
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=width),
                    connectgaps=True,
                    customdata=df["DateLabel"],
                    hovertemplate="<b>" + str(year) + "</b><br>Actual date: %{customdata}"
                    "<br>Std of Std: %{y:.4f}<extra></extra>",
                )
            )
        layout = self._base_layout(
            f"Rolling Std-of-Std (Vol-of-Vol) Seasonality (window={window})", "Std of Rolling Std"
        )
        layout["xaxis"] = {**layout["xaxis"], "rangeslider": dict(visible=True, bgcolor=self.theme.panel_bg)}
        fig.update_layout(**layout)
        return fig

    def build_zscore_of_std_chart(
        self, rolling_data: Dict[str, pd.DataFrame], years_to_plot: List[str],
        current_year: str, window: int,
    ) -> go.Figure:
        """Chart 5: Z-score of the rolling standard deviation ("vol-of-vol z-score").

        Measures how extreme the current volatility level is relative to its
        own recent rolling mean/std -- i.e. is the market unusually calm or
        unusually turbulent right now, seasonally speaking.
        """
        fig = go.Figure()
        for idx, year in enumerate(years_to_plot):
            df = rolling_data.get(year)
            if df is None or df.empty:
                continue
            color, width = self._color_for(idx, year, current_year)
            fig.add_trace(
                go.Scatter(
                    x=self._aligned_dates(df, year, current_year),
                    y=df["ZScoreOfStd"],
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=width),
                    connectgaps=True,
                    customdata=df["DateLabel"],
                    hovertemplate="<b>" + str(year) + "</b><br>Actual date: %{customdata}"
                    "<br>Z-Score of Std: %{y:.3f}<extra></extra>",
                )
            )
        fig.update_layout(
            **self._base_layout(f"Z-Score of Rolling Std (Vol-of-Vol) Seasonality (window={window})", "Z-Score of Std")
        )
        self._add_zscore_reference_lines(fig)
        return fig

    @staticmethod
    def _add_zscore_reference_lines(fig: go.Figure) -> None:
        """Draw 0, +/-1, +/-2, +/-3 sigma horizontal reference lines on a z-score chart."""
        for level, dash, color in [
            (0, "solid", "#AAAAAA"),
            (1, "dash", "#4CD964"), (-1, "dash", "#4CD964"),
            (2, "dot", "#FFCC00"), (-2, "dot", "#FFCC00"),
            (3, "dashdot", "#FF3B30"), (-3, "dashdot", "#FF3B30"),
        ]:
            fig.add_hline(
                y=level, line_dash=dash, line_color=color, line_width=1,
                annotation_text=f"{level:+d}σ" if level != 0 else "0",
                annotation_position="right",
                annotation_font=dict(color=color, size=10),
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class StatsCalculator:
    """Derives the summary statistics shown in the dashboard's stats panel."""

    def __init__(self, rolling_data: Dict[str, pd.DataFrame], current_year: str):
        self.rolling_data = rolling_data
        self.current_year = current_year

    def compute(self) -> Dict[str, Optional[float]]:
        stats: Dict[str, Optional[float]] = {}

        current_df = self.rolling_data.get(self.current_year)
        if current_df is not None and not current_df.empty:
            latest_valid = current_df.dropna(subset=["ZScore"])
            stats["latest_zscore"] = (
                float(latest_valid["ZScore"].iloc[-1]) if not latest_valid.empty else None
            )
            latest_std = current_df.dropna(subset=["RollingStd"])
            stats["latest_sd"] = (
                float(latest_std["RollingStd"].iloc[-1]) if not latest_std.empty else None
            )
            latest_stdofstd = current_df.dropna(subset=["StdOfStd"])
            stats["latest_stdofstd"] = (
                float(latest_stdofstd["StdOfStd"].iloc[-1]) if not latest_stdofstd.empty else None
            )
            latest_zscoreofstd = current_df.dropna(subset=["ZScoreOfStd"])
            stats["latest_zscoreofstd"] = (
                float(latest_zscoreofstd["ZScoreOfStd"].iloc[-1]) if not latest_zscoreofstd.empty else None
            )
            stats["current_year_trading_days"] = int(len(current_df))
        else:
            stats["latest_zscore"] = None
            stats["latest_sd"] = None
            stats["latest_stdofstd"] = None
            stats["latest_zscoreofstd"] = None
            stats["current_year_trading_days"] = 0

        all_z = pd.concat(
            [df["ZScore"] for df in self.rolling_data.values() if not df.empty],
            ignore_index=True,
        ).dropna()
        all_std = pd.concat(
            [df["RollingStd"] for df in self.rolling_data.values() if not df.empty],
            ignore_index=True,
        ).dropna()
        all_zscoreofstd = pd.concat(
            [df["ZScoreOfStd"] for df in self.rolling_data.values() if not df.empty],
            ignore_index=True,
        ).dropna()
        all_stdofstd = pd.concat(
            [df["StdOfStd"] for df in self.rolling_data.values() if not df.empty],
            ignore_index=True,
        ).dropna()

        stats["highest_zscore"] = float(all_z.max()) if not all_z.empty else None
        stats["lowest_zscore"] = float(all_z.min()) if not all_z.empty else None
        stats["avg_volatility"] = float(all_std.mean()) if not all_std.empty else None
        stats["max_volatility"] = float(all_std.max()) if not all_std.empty else None
        stats["highest_zscoreofstd"] = float(all_zscoreofstd.max()) if not all_zscoreofstd.empty else None
        stats["lowest_zscoreofstd"] = float(all_zscoreofstd.min()) if not all_zscoreofstd.empty else None
        stats["avg_stdofstd"] = float(all_stdofstd.mean()) if not all_stdofstd.empty else None
        stats["max_stdofstd"] = float(all_stdofstd.max()) if not all_stdofstd.empty else None
        stats["total_trading_days"] = int(
            sum(len(df) for df in self.rolling_data.values())
        )
        stats["num_years"] = len(self.rolling_data)

        return stats


# ---------------------------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------------------------
class ExportManager:
    """Builds downloadable Excel and standalone HTML artifacts."""

    @staticmethod
    def build_excel_bytes(rolling_data: Dict[str, pd.DataFrame]) -> bytes:
        """Build Seasonality_Output.xlsx with Cleaned Data / Rolling SD / Rolling Z-score /
        Rolling SDofSD / Rolling ZScore of SD sheets."""
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                # --- Cleaned Data sheet (long format, all years stacked) ---
                cleaned_frames = []
                for year, df in rolling_data.items():
                    tmp = df[["DayIndex", "Date", "Price"]].copy()
                    tmp.insert(0, "Year", year)
                    cleaned_frames.append(tmp)
                cleaned_all = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else pd.DataFrame()
                cleaned_all.to_excel(writer, sheet_name="Cleaned Data", index=False)

                # --- Rolling SD sheet (wide format: DayIndex, one col per year) ---
                sd_wide = ExportManager._pivot_metric(rolling_data, "RollingStd")
                sd_wide.to_excel(writer, sheet_name="Rolling SD", index=False)

                # --- Rolling Z-score sheet ---
                z_wide = ExportManager._pivot_metric(rolling_data, "ZScore")
                z_wide.to_excel(writer, sheet_name="Rolling Z-score", index=False)

                # --- Rolling Std-of-Std (volatility of volatility) sheet ---
                sdofsd_wide = ExportManager._pivot_metric(rolling_data, "StdOfStd")
                sdofsd_wide.to_excel(writer, sheet_name="Rolling SDofSD", index=False)

                # --- Z-score of Std sheet ---
                zofsd_wide = ExportManager._pivot_metric(rolling_data, "ZScoreOfStd")
                zofsd_wide.to_excel(writer, sheet_name="Rolling ZScore of SD", index=False)

                # Basic formatting: header styling
                workbook = writer.book
                header_fmt = workbook.add_format(
                    {"bold": True, "bg_color": "#1F2430", "font_color": "#FFFFFF", "border": 1}
                )
                for sheet_name, frame in [
                    ("Cleaned Data", cleaned_all), ("Rolling SD", sd_wide), ("Rolling Z-score", z_wide),
                    ("Rolling SDofSD", sdofsd_wide), ("Rolling ZScore of SD", zofsd_wide),
                ]:
                    ws = writer.sheets[sheet_name]
                    for col_idx, col_name in enumerate(frame.columns):
                        ws.write(0, col_idx, col_name, header_fmt)
                        ws.set_column(col_idx, col_idx, 14)
        except Exception as exc:  # noqa: BLE001
            logger.error("Excel export failed: %s", exc)
            raise

        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def _pivot_metric(rolling_data: Dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
        frames = []
        for year, df in rolling_data.items():
            tmp = df[["DayIndex", metric]].copy()
            tmp = tmp.rename(columns={metric: str(year)})
            tmp = tmp.set_index("DayIndex")
            frames.append(tmp)
        if not frames:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
        wide = wide.reset_index()
        wide = wide.rename(columns={"DayIndex": "Trading Day #"})
        return wide

    @staticmethod
    def build_html_bytes(figures: List[Tuple[str, go.Figure]]) -> bytes:
        """Combine multiple Plotly figures into one standalone, offline-capable HTML file."""
        parts = [
            "<html><head><meta charset='utf-8'>",
            f"<title>Seasonality Dashboard</title>",
            f"<style>body{{background:{DARK_BG};margin:0;padding:20px;"
            f"font-family:'Segoe UI',Roboto,sans-serif;color:#E6E6E6;}}"
            f"h2{{border-bottom:1px solid {GRID_COLOR};padding-bottom:8px;}}</style>",
            "</head><body>",
            "<h1>Seasonality Analysis Dashboard</h1>",
        ]
        for i, (title, fig) in enumerate(figures):
            include_js = "cdn" if i == 0 else False
            parts.append(f"<h2>{title}</h2>")
            parts.append(
                fig.to_html(full_html=False, include_plotlyjs=include_js, config={"responsive": True})
            )
        parts.append("</body></html>")
        html_str = "\n".join(parts)
        return html_str.encode("utf-8")


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------
class SeasonalityDashboardApp:
    """Orchestrates the Streamlit UI, session state, and rendering pipeline."""

    def __init__(self):
        self.chart_builder = ChartBuilder()
        self._init_session_state()

    @staticmethod
    def _init_session_state() -> None:
        defaults = {
            "csv_path": DEFAULT_CSV_PATH,
            "window": DEFAULT_ROLLING_WINDOW,
            "selected_years": None,   # None => not yet initialized
            "current_year": None,
            "year_data": None,
            "load_error": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ---------------------------- Setup / Styling ----------------------------
    @staticmethod
    def configure_page() -> None:
        st.set_page_config(
            page_title="Seasonality Analysis Dashboard",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(
            f"""
            <style>
            .stApp {{ background-color: {DARK_BG}; }}
            section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
            div[data-testid="stMetric"] {{
                background-color: {PANEL_BG};
                border: 1px solid {GRID_COLOR};
                border-radius: 8px;
                padding: 10px 14px;
            }}
            h1, h2, h3, h4, p, span, label {{ color: #E6E6E6; }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------- Data load -------------------------------
    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_and_clean(path: str, file_bytes: Optional[bytes]) -> Dict[str, pd.DataFrame]:
        source = io.BytesIO(file_bytes) if file_bytes is not None else path
        loader = SeasonalityDataLoader(source)
        return loader.load()

    def _handle_load(self, path: str, uploaded_file) -> None:
        try:
            file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
            year_data = self._load_and_clean(path, file_bytes)
            st.session_state["year_data"] = year_data
            st.session_state["load_error"] = None
            all_years = sorted(year_data.keys())
            st.session_state["selected_years"] = all_years
            st.session_state["current_year"] = all_years[-1] if all_years else None
        except Exception as exc:  # noqa: BLE001
            st.session_state["year_data"] = None
            st.session_state["load_error"] = str(exc)
            logger.error("Data load failed: %s", exc)

    # -------------------------------- Sidebar --------------------------------
    def render_sidebar(self) -> None:
        st.sidebar.title("⚙️ Controls")

        st.sidebar.subheader("Data Source")
        source_mode = st.sidebar.radio(
            "CSV source", ["File path", "Upload file"], horizontal=True, label_visibility="collapsed"
        )

        uploaded_file = None
        csv_path = st.session_state["csv_path"]
        if source_mode == "File path":
            csv_path = st.sidebar.text_input("CSV file path", value=st.session_state["csv_path"])
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

        if st.sidebar.button("📂 Load Data", use_container_width=True):
            st.session_state["csv_path"] = csv_path
            with st.spinner("Loading and cleaning data..."):
                self._handle_load(csv_path, uploaded_file)

        if st.session_state["load_error"]:
            st.sidebar.error(f"Load failed: {st.session_state['load_error']}")

        year_data = st.session_state["year_data"]
        if year_data:
            all_years = sorted(year_data.keys())

            st.sidebar.subheader("Year Selection")
            current_year = st.sidebar.selectbox(
                "Current year (highlighted)",
                options=all_years,
                index=all_years.index(st.session_state["current_year"])
                if st.session_state["current_year"] in all_years else len(all_years) - 1,
            )
            st.session_state["current_year"] = current_year

            selected_years = st.sidebar.multiselect(
                "Historical years to plot",
                options=all_years,
                default=st.session_state["selected_years"] or all_years,
            )
            st.session_state["selected_years"] = selected_years

            st.sidebar.subheader("Rolling Window")
            window = st.sidebar.number_input(
                "Rolling window (days)", min_value=2, max_value=252,
                value=int(st.session_state["window"]), step=1,
            )
            st.session_state["window"] = window

            st.sidebar.subheader("Export")
            # Export buttons are rendered lazily in main() once figures/data exist,
            # via placeholders so we can inject the actual bytes.
            st.session_state["_export_excel_slot"] = st.sidebar.empty()
            st.session_state["_export_html_slot"] = st.sidebar.empty()

        st.sidebar.subheader("Reset")
        if st.sidebar.button("🔄 Reset Dashboard", use_container_width=True):
            for key in ["csv_path", "window", "selected_years", "current_year", "year_data", "load_error"]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()

    # -------------------------------- Main body -------------------------------
    def render_main(self) -> None:
        st.title("📈 Seasonality Analysis Dashboard")

        year_data = st.session_state.get("year_data")
        if not year_data:
            st.info(
                "Enter a CSV path or upload a file in the sidebar, then click "
                "**Load Data** to begin. Years are detected automatically — "
                "no configuration needed."
            )
            return

        selected_years = st.session_state.get("selected_years") or sorted(year_data.keys())
        current_year = st.session_state.get("current_year")
        window = int(st.session_state.get("window", DEFAULT_ROLLING_WINDOW))

        # Ensure current year is always included in the plotted set
        years_to_plot = list(dict.fromkeys(selected_years + ([current_year] if current_year else [])))
        years_to_plot = [y for y in sorted(years_to_plot) if y in year_data]

        if not years_to_plot:
            st.warning("No years selected. Please select at least one year in the sidebar.")
            return

        analyzer = SeasonalityAnalyzer(year_data, window=window)
        rolling_data = analyzer.compute()

        # --- Charts ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Raw Seasonality", "Rolling Std Dev", "Rolling Z-Score",
             "Std of Std Dev", "Z-Score of Std Dev"]
        )

        with tab1:
            fig_raw = self.chart_builder.build_raw_chart(year_data, years_to_plot, current_year)
            st.plotly_chart(fig_raw, use_container_width=True, config={"responsive": True, "displaylogo": False})

        with tab2:
            fig_std = self.chart_builder.build_std_chart(rolling_data, years_to_plot, current_year, window)
            st.plotly_chart(fig_std, use_container_width=True, config={"responsive": True, "displaylogo": False})

        with tab3:
            fig_z = self.chart_builder.build_zscore_chart(rolling_data, years_to_plot, current_year, window)
            st.plotly_chart(fig_z, use_container_width=True, config={"responsive": True, "displaylogo": False})

        with tab4:
            fig_stdofstd = self.chart_builder.build_std_of_std_chart(
                rolling_data, years_to_plot, current_year, window
            )
            st.plotly_chart(
                fig_stdofstd, use_container_width=True, config={"responsive": True, "displaylogo": False}
            )

        with tab5:
            fig_zofstd = self.chart_builder.build_zscore_of_std_chart(
                rolling_data, years_to_plot, current_year, window
            )
            st.plotly_chart(
                fig_zofstd, use_container_width=True, config={"responsive": True, "displaylogo": False}
            )

        # --- Statistics panel ---
        st.subheader("📊 Statistics")
        stats = StatsCalculator(rolling_data, current_year).compute()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{current_year} Latest Z-Score", self._fmt(stats["latest_zscore"], "{:.3f}"))
        c2.metric(f"{current_year} Latest SD({window})", self._fmt(stats["latest_sd"], "{:.4f}"))
        c3.metric("Highest Z-Score (all years)", self._fmt(stats["highest_zscore"], "{:.3f}"))
        c4.metric("Lowest Z-Score (all years)", self._fmt(stats["lowest_zscore"], "{:.3f}"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Average Volatility (all years)", self._fmt(stats["avg_volatility"], "{:.4f}"))
        c6.metric("Maximum Volatility (all years)", self._fmt(stats["max_volatility"], "{:.4f}"))
        c7.metric("Total Trading Days (all years)", str(stats["total_trading_days"]))
        c8.metric("Years Loaded", str(stats["num_years"]))

        st.caption("Volatility-of-Volatility (Std-of-Std)")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric(f"{current_year} Latest Std-of-Std", self._fmt(stats["latest_stdofstd"], "{:.4f}"))
        c10.metric(f"{current_year} Latest Z-Score of Std", self._fmt(stats["latest_zscoreofstd"], "{:.3f}"))
        c11.metric("Highest Z-Score of Std (all years)", self._fmt(stats["highest_zscoreofstd"], "{:.3f}"))
        c12.metric("Lowest Z-Score of Std (all years)", self._fmt(stats["lowest_zscoreofstd"], "{:.3f}"))

        c13, c14, _, _ = st.columns(4)
        c13.metric("Average Std-of-Std (all years)", self._fmt(stats["avg_stdofstd"], "{:.4f}"))
        c14.metric("Maximum Std-of-Std (all years)", self._fmt(stats["max_stdofstd"], "{:.4f}"))

        # --- Data table ---
        st.subheader("🔎 Data Table")
        combined = self._build_combined_table(rolling_data, years_to_plot)
        search_term = st.text_input("Search table (matches any column)", value="")
        filtered = combined
        if search_term:
            mask = combined.apply(
                lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1
            )
            filtered = combined[mask]

        st.dataframe(filtered, use_container_width=True, height=400)
        st.download_button(
            "⬇️ Export Table as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="Seasonality_Table.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # --- Exports (Excel / HTML) wired into sidebar placeholders ---
        excel_bytes = ExportManager.build_excel_bytes(rolling_data)
        st.session_state["_export_excel_slot"].download_button(
            "📗 Export Excel",
            data=excel_bytes,
            file_name="Seasonality_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        html_bytes = ExportManager.build_html_bytes(
            [("Raw Seasonality", fig_raw), ("Rolling Std Dev Seasonality", fig_std),
             ("Rolling Z-Score Seasonality", fig_z),
             ("Rolling Std-of-Std (Vol-of-Vol) Seasonality", fig_stdofstd),
             ("Z-Score of Rolling Std (Vol-of-Vol) Seasonality", fig_zofstd)]
        )
        st.session_state["_export_html_slot"].download_button(
            "🌐 Export HTML",
            data=html_bytes,
            file_name="Seasonality_Dashboard.html",
            mime="text/html",
            use_container_width=True,
        )

    @staticmethod
    def _fmt(value: Optional[float], fmt_str: str) -> str:
        return fmt_str.format(value) if value is not None and not np.isnan(value) else "N/A"

    @staticmethod
    def _build_combined_table(rolling_data: Dict[str, pd.DataFrame], years_to_plot: List[str]) -> pd.DataFrame:
        cols = ["DayIndex", "Date", "Price", "RollingMean", "RollingStd", "ZScore", "StdOfStd", "ZScoreOfStd"]
        frames = []
        for year in years_to_plot:
            df = rolling_data.get(year)
            if df is None or df.empty:
                continue
            tmp = df[cols].copy()
            tmp = tmp.rename(columns={"DayIndex": "Trading Day #"})
            tmp.insert(0, "Year", year)
            frames.append(tmp)
        if not frames:
            return pd.DataFrame(columns=["Year", "Trading Day #"] + cols[1:])
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["Year", "Date"]).reset_index(drop=True)
        return combined

    def run(self) -> None:
        self.configure_page()
        try:
            self.render_sidebar()
            self.render_main()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled dashboard error")
            st.error(f"An unexpected error occurred: {exc}")


def main() -> None:
    app = SeasonalityDashboardApp()
    app.run()


if __name__ == "__main__":
    main()
