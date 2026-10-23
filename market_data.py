"""Yahoo Finance access, with batch downloads and graceful failures."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from scanner import FIELDS


def _extract_ticker(raw: pd.DataFrame, symbol: str, many: bool) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance may return either ticker-first or field-first columns.
            if symbol in raw.columns.get_level_values(0):
                frame = raw.xs(symbol, axis=1, level=0).copy()
            elif symbol in raw.columns.get_level_values(1):
                frame = raw.xs(symbol, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        elif not many:
            frame = raw.copy()
        else:
            return pd.DataFrame()
        frame.columns = [str(k).title() for k in frame.columns]
        if not all(col in frame.columns for col in FIELDS):
            return pd.DataFrame()
        return frame.loc[:, FIELDS].dropna().sort_index()
    except Exception:
        return pd.DataFrame()


def download_daily(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    raw = yf.download(
        tickers=list(symbols), period="2y", interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True, timeout=20,
    )
    return {s: _extract_ticker(raw, s, len(symbols) > 1) for s in symbols}


def download_intraday(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    raw = yf.download(
        tickers=list(symbols), period="5d", interval="5m", group_by="ticker",
        auto_adjust=True, progress=False, threads=True, timeout=20,
    )
    return {s: _extract_ticker(raw, s, len(symbols) > 1) for s in symbols}
