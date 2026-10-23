"""Pure, read-only educational screening logic for *long* stock options.

No orders. All numbers are rule alignment, not expected profits or win rates.
Contracts with invalid/missing executable-side quotes are excluded, not guessed.
"""
from __future__ import annotations

from datetime import date
import math
from typing import Any

import pandas as pd


def market_direction(bars_map: dict[str, pd.DataFrame]) -> str:
    """Broad-market read: bullish/bearish when any of SPY/QQQ is aligned;
    conflicting benchmarks show Mixed, unavailable shows Unavailable.
    """
    bullish, bearish = [], []
    for symbol in ("SPY", "QQQ"):
        d = bars_map.get(symbol)
        if d is None or len(d) < 200:
            continue
        c = pd.to_numeric(d["Close"], errors="coerce")
        ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
        bullish.append(bool(c.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]))
        bearish.append(bool(c.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]))
    if not bullish:
        return "Unavailable"
    if any(bullish) and not any(bearish):
        return "Bullish"
    if any(bearish) and not any(bullish):
        return "Bearish"
    return "Mixed"


def directional_stock_signal(symbol: str, bars: pd.DataFrame, regime: str) -> dict[str, Any] | None:
    """Symmetric trend + pullback/breakout long-call and long-put research leads.

    Derived from original Swing Desk moving-average and ATR logic. Does not
    assert directional forecasting ability; no trade if no pattern qualifies.
    """
    if bars is None or len(bars) < 205:
        return None
    c, h, l, v = (pd.to_numeric(bars[k], errors="coerce") for k in ("Close", "High", "Low", "Volume"))
    ma20, ma50, ma200 = (c.rolling(w).mean() for w in (20, 50, 200))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    p, a = float(c.iloc[-1]), float(atr.iloc[-1])
    if not (math.isfinite(p) and math.isfinite(a) and p > 0 and a > 0):
        return None
    previous_volume = float(v.iloc[-21:-1].mean())
    relative_volume = float(v.iloc[-1] / previous_volume) if previous_volume > 0 else 0
    ma20_rises = float(ma20.iloc[-1]) > float(ma20.iloc[-6])
    ma20_falls = float(ma20.iloc[-1]) < float(ma20.iloc[-6])
    bull_trend = p > float(ma50.iloc[-1]) > float(ma200.iloc[-1])
    bear_trend = p < float(ma50.iloc[-1]) < float(ma200.iloc[-1])
    bull_pullback = (bull_trend and ma20_rises
                     and float(l.iloc[-5:].min()) <= float(ma20.iloc[-1]) + 0.5*a
                     and p > float(c.iloc[-2]) and p >= float(ma20.iloc[-1]))
    bull_breakout = (bull_trend and p > float(h.iloc[-21:-1].max()) and relative_volume >= 1.2)
    bear_pullback = (bear_trend and ma20_falls
                     and float(h.iloc[-5:].max()) >= float(ma20.iloc[-1]) - 0.5*a
                     and p < float(c.iloc[-2]) and p <= float(ma20.iloc[-1]))
    bear_breakdown = (bear_trend and p < float(l.iloc[-21:-1].min()) and relative_volume >= 1.2)

    if bull_pullback or bull_breakout:
        side = "CALL"
        setup = "Bullish pullback" if bull_pullback else "Bullish breakout"
        trending, above_ma20, sloped = bull_trend, p > float(ma20.iloc[-1]), ma20_rises
    elif bear_pullback or bear_breakdown:
        side = "PUT"
        setup = "Bearish pullback" if bear_pullback else "Bearish breakdown"
        trending, above_ma20, sloped = bear_trend, p < float(ma20.iloc[-1]), ma20_falls
    else:
        return None

    market_supports = regime == ("Bullish" if side == "CALL" else "Bearish")
    score = (25 if trending else 0) + (10 if above_ma20 else 0) + (10 if sloped else 0) + 25
    score += (10 if relative_volume >= 1.2 else 0) + (15 if market_supports else 0)
    score += 5 if a / p < 0.08 else 0
    ts = pd.Timestamp(bars.index[-1])
    return {
        "Ticker": symbol, "Direction": side, "Stock setup": setup, "Stock score": min(100, score),
        "Signal close": round(p, 2), "Signal session": ts.date().isoformat(),
        "Relative volume": round(relative_volume, 2), "Market": regime,
    }


def _num(value: Any) -> float | None:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def rank_contracts(
    chain: pd.DataFrame,
    stock: dict[str, Any],
    expiry: str,
    underlying_reference: float,
    max_debit_dollars: float,
    min_oi: int = 100,
    min_volume: int = 10,
    max_spread_pct: float = 15.0,
    min_dte: int = 30,
    max_dte: int = 60,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rank standard 100-share long calls/puts. Pay ask; never use last price.

    Returns accepted candidates and rejection diagnostics. A 'score' is
    rule-compliance (underlying, liquidity, time), not a win probability.
    """
    counts = {"invalid_quote": 0, "liquidity": 0, "moneyness": 0,
              "over_budget": 0, "expiry": 0, "adjusted": 0}
    if chain is None or chain.empty:
        return [], counts
    try:
        dte = (date.fromisoformat(expiry) - (today or date.today())).days
    except ValueError:
        counts["expiry"] = len(chain)
        return [], counts
    if not min_dte <= dte <= max_dte or underlying_reference <= 0:
        counts["expiry"] = len(chain)
        return [], counts

    results = []
    direction = stock["Direction"]
    for _, row in chain.iterrows():
        size = row.get("contractSize", 100)
        # Reject known nonstandard/adjusted contracts; "REGULAR" from Yahoo = 100.
        if str(size).upper() not in ("100", "100.0", "REGULAR"):
            counts["adjusted"] += 1
            continue
        strike, bid, ask = (_num(row.get(k)) for k in ("strike", "bid", "ask"))
        if strike is None or strike <= 0 or bid is None or ask is None or ask <= bid or bid <= 0:
            counts["invalid_quote"] += 1
            continue
        midpoint = (bid+ask) / 2
        spread_pct = 100 * (ask-bid) / midpoint
        oi, vol = _num(row.get("openInterest")), _num(row.get("volume"))
        if spread_pct > max_spread_pct or oi is None or oi < min_oi or vol is None or vol < min_volume:
            counts["liquidity"] += 1
            continue
        delta = _num(row.get("delta"))
        moneyness = abs(strike / underlying_reference - 1)
        # No far out-of-money lottery-like contracts. Where delta is present,
        # require moderate absolute delta, and avoid inconsistent extreme strikes.
        if moneyness > 0.10 or (delta is not None and not (0.30 <= abs(delta) <= 0.75)):
            counts["moneyness"] += 1
            continue
        debit = ask * 100
        if debit > max_debit_dollars or debit <= 0:
            counts["over_budget"] += 1
            continue
        stock_score = min(100, max(0, int(stock["Stock score"])))
        points = round(stock_score * 0.4)
        points += 10 if oi >= 500 else 5
        points += 10 if vol >= 100 else 5 if vol >= 20 else 2
        points += 15 if spread_pct <= 5 else 10 if spread_pct <= 10 else 5
        points += 10 if (delta is not None and .35 <= abs(delta) <= .65) or (delta is None and moneyness <= .03) else 5
        points += 10 if 35 <= dte <= 50 else 5
        points += 5 if debit <= 0.75*max_debit_dollars else 0
        points = min(100, points)
        symbol = str(row.get("contractSymbol") or "").strip()
        if not symbol:
            counts["invalid_quote"] += 1
            continue
        reason = (f"{stock['Stock setup']}; OI {int(oi):,}; vol {int(vol):,}; "
                  f"bid/ask spread {spread_pct:.1f}%; {dte} DTE")
        results.append({
            "Ticker": stock["Ticker"], "Direction": direction,
            "Contract": symbol, "Expiration": expiry, "DTE": dte,
            "Strike": round(strike, 2), "Bid": round(bid, 2), "Ask": round(ask, 2),
            "Cost / max loss ($)": round(debit, 2),
            "Expiry break-even": round(strike + ask if direction == "CALL" else strike - ask, 2),
            "Bid/ask spread %": round(spread_pct, 1), "Open interest": int(oi), "Volume": int(vol),
            "Delta": round(delta, 2) if delta is not None else None,
            "Stock setup": stock["Stock setup"], "Stock score": stock_score,
            "Screen score": points, "Why it passed": reason,
            "Signal session": stock["Signal session"], "Signal close": stock["Signal close"],
            "Underlying reference": round(underlying_reference, 2),
        })
    return sorted(results, key=lambda x: (-x["Screen score"], x["Bid/ask spread %"], -x["Open interest"])), counts
