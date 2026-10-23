"""Pure, educational swing-signal calculations; no trading or broker functions."""
from __future__ import annotations

import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
FIELDS = ("Open", "High", "Low", "Close", "Volume")


def completed_daily_bars(data: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Do not score a candle still forming during the regular US equity session.

    A 4:15pm ET buffer gives daily bars time to settle; the feed can still lag.
    """
    if data is None or data.empty:
        return pd.DataFrame()
    now_et = (now or datetime.now(ET)).astimezone(ET)
    bars = data.copy()
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is not None:
        dates = bars.index.tz_convert(ET).date
    else:
        dates = bars.index.date
    if len(bars) and dates[-1] == now_et.date() and now_et.time() < time(16, 15):
        bars = bars.iloc[:-1]
    return bars.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def market_regime(data_map: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    """Favorable if SPY or QQQ has close > SMA50 > SMA200."""
    outcomes: list[bool] = []
    for symbol in ("SPY", "QQQ"):
        bars = data_map.get(symbol, pd.DataFrame())
        if len(bars) < 200:
            continue
        close = bars["Close"]
        sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
        outcomes.append(bool(close.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]))
    if not outcomes:
        return False, "Benchmarks unavailable"
    return any(outcomes), "Favorable" if any(outcomes) else "Cautious"


def analyze_symbol(
    symbol: str,
    bars: pd.DataFrame,
    favorable: bool,
    account_value: float,
    risk_percent: float,
    max_capital_percent: float,
) -> dict:
    """Align with the user's existing simple trend/pullback/breakout rubric.

    This score is rules-based, NOT a probability of profit. All levels are
    illustrative reference prices and do not imply an actual order.
    """
    basic = {"Ticker": symbol, "Setup": "Data unavailable", "Score": 0,
             "Category": "C — Low priority", "Reason": "Fewer than 205 completed daily bars"}
    if bars is None or len(bars) < 205:
        return basic
    c, h, l, v = (bars[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    ma20, ma50, ma200 = (c.rolling(w).mean() for w in (20, 50, 200))
    true_range = pd.concat((h - l, (h - c.shift()).abs(), (l - c.shift()).abs()), axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    price, atr_now = float(c.iloc[-1]), float(atr.iloc[-1])
    if not math.isfinite(price) or not math.isfinite(atr_now) or price <= 0 or atr_now <= 0:
        basic["Reason"] = "Price or ATR unavailable"
        return basic
    prior_20_volume = float(v.iloc[-21:-1].mean())
    relative_volume = float(v.iloc[-1] / prior_20_volume) if prior_20_volume > 0 else 0.0
    uptrend = price > float(ma50.iloc[-1]) > float(ma200.iloc[-1])
    ma20_rising = float(ma20.iloc[-1]) > float(ma20.iloc[-6])
    pullback = (
        uptrend and ma20_rising
        and float(l.iloc[-5:].min()) <= float(ma20.iloc[-1]) + 0.5 * atr_now
        and price > float(c.iloc[-2])
        and price >= float(ma20.iloc[-1])
    )
    breakout = uptrend and price > float(h.iloc[-21:-1].max()) and relative_volume >= 1.2
    setup = "Potential pullback" if pullback else "Potential breakout" if breakout else "No setup"
    score = (
        (25 if uptrend else 0)
        + (10 if price > float(ma20.iloc[-1]) else 0)
        + (10 if ma20_rising else 0)
        + (25 if setup != "No setup" else 0)
        + (10 if relative_volume >= 1.2 else 0)
        + (15 if favorable else 0)
        + (5 if atr_now / price < 0.08 else 0)
    )
    category = "A — High priority" if score >= 80 else "B — Watchlist" if score >= 65 else "C — Low priority"
    entry = price
    stop = max(0.01, entry - 2 * atr_now)
    target = entry + 4 * atr_now
    one_share_risk = entry - stop
    risk_dollars = account_value * risk_percent / 100
    max_position_dollars = account_value * max_capital_percent / 100
    size_by_risk = math.floor(risk_dollars / one_share_risk) if one_share_risk > 0 else 0
    size_by_cash = math.floor(max_position_dollars / entry)
    shares = max(0, min(size_by_risk, size_by_cash))
    last_date = pd.Timestamp(bars.index[-1])
    if last_date.tzinfo is not None:
        last_date = last_date.tz_convert(ET)
    reason = (
        "Rising trend with a pullback/rebound"
        if pullback else "20-day high breakout with elevated volume"
        if breakout else "No qualifying pattern"
    )
    return {
        "Ticker": symbol, "Signal session": last_date.date().isoformat(),
        "Price": round(price, 2), "Setup": setup,
        "Rel. volume": round(relative_volume, 2),
        "Score": score, "Category": category,
        "Market filter": "Favorable" if favorable else "Cautious",
        "Entry ref.": round(entry, 2),
        "Stop ref.": round(stop, 2),
        "Target ref.": round(target, 2),
        "Shares est.": shares,
        "Planned risk ($)": round(shares * one_share_risk, 2),
        "Illustrative upside ($)": round(shares * (target - entry), 2),
        "ATR": round(atr_now, 3), "Reason": reason,
    }
