"""Read-only Yahoo Finance stock and option-chain data adapter.

A missing bid/ask stays missing. Never substitute lastPrice as an executable quote.
"""
from __future__ import annotations
from datetime import date
import pandas as pd
import yfinance as yf

FIELDS = ("contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "lastTradeDate")


def get_expirations(symbol: str) -> tuple[str, ...]:
    expirations = tuple(yf.Ticker(symbol).options)
    return tuple(e for e in expirations if date.fromisoformat(e) >= date.today())


def get_chain(symbol: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obj = yf.Ticker(symbol).option_chain(expiration)
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=FIELDS)
        frame = df.copy()
        for c in FIELDS:
            if c not in frame.columns:
                frame[c] = pd.NA
        for c in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        return frame.loc[:, FIELDS].sort_values("strike").reset_index(drop=True)
    return clean(obj.calls), clean(obj.puts)


def get_last_underlying_price(symbol: str) -> float | None:
    d = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
    if d is None or d.empty:
        return None
    c = pd.to_numeric(d["Close"], errors="coerce").dropna()
    if c.empty:
        return None
    return float(c.iloc[-1])
