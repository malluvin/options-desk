"""Read-only Yahoo and Tradier sandbox option-market adapters.

All Tradier requests are GET-only, sandbox-only, with an explicit path allowlist.
No fallback from unavailable bid/ask to last traded price.
"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
FIELDS = ("contractSymbol", "contractSize", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "delta", "lastTradeDate")
BASE_URL = "https://sandbox.tradier.com/v1"

class OptionsDataError(RuntimeError):
    pass


def _clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(FIELDS))
    f = df.copy()
    for col in FIELDS:
        if col not in f.columns:
            f[col] = pd.NA
    # Missing multiplier on standard yfinance chain is recorded as "REGULAR".
    f["contractSize"] = f["contractSize"].fillna("REGULAR")
    for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "delta"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    return f.loc[:, FIELDS].sort_values("strike").reset_index(drop=True)


def _tradier_get(path: str, token: str, params: dict[str, Any]) -> dict:
    allowed = {"/markets/options/expirations", "/markets/options/chains", "/markets/quotes"}
    if path not in allowed:
        raise OptionsDataError("Unsupported market-data path")
    if not token or not token.strip():
        raise OptionsDataError("Tradier sandbox token not configured")
    try:
        r = requests.get(BASE_URL + path, headers={"Authorization": "Bearer " + token.strip(), "Accept": "application/json"}, params=params, timeout=15)
        if r.status_code == 401:
            raise OptionsDataError("Tradier sandbox rejected the token (401). Check your sandbox token, not a live token.")
        if r.status_code == 429:
            raise OptionsDataError("Tradier rate limit (429). Retry after a cooldown.")
        if r.status_code >= 400:
            raise OptionsDataError(f"Tradier returned HTTP {r.status_code}.")
        payload = r.json()
        if not isinstance(payload, dict) or payload.get("fault") or payload.get("errors"):
            raise OptionsDataError("Tradier returned an unexpected error payload.")
        return payload
    except OptionsDataError:
        raise
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise OptionsDataError(f"Tradier request failed ({type(exc).__name__}).") from exc


def get_expirations(symbol: str, provider: str = "Yahoo Finance", token: str = "") -> tuple[str, ...]:
    if provider == "Tradier sandbox":
        data = _tradier_get("/markets/options/expirations", token, {"symbol": symbol, "includeAllRoots": "false"})
        dates = (data.get("expirations") or {}).get("date") if isinstance(data.get("expirations"), dict) else None
        dates = [dates] if isinstance(dates, str) else dates
        if not isinstance(dates, list):
            return ()
    else:
        try:
            import yfinance as yf
            dates = list(yf.Ticker(symbol).options)
        except Exception as exc:
            raise OptionsDataError(f"Yahoo Finance options lookup failed ({type(exc).__name__}). Cloud-hosted requests are sometimes blocked or throttled.") from exc
    from datetime import date
    today = datetime.now(ET).date()
    result = []
    for item in dates:
        try:
            val = str(item)
            if date.fromisoformat(val) >= today:
                result.append(val)
        except (ValueError, TypeError):
            pass
    return tuple(sorted(set(result)))


def get_chain(symbol: str, expiry: str, provider: str = "Yahoo Finance", token: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    if provider == "Tradier sandbox":
        data = _tradier_get("/markets/options/chains", token, {"symbol": symbol, "expiration": expiry, "greeks": "true"})
        options = data.get("options")
        values = options.get("option") if isinstance(options, dict) else None
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            return _clean_chain(pd.DataFrame()), _clean_chain(pd.DataFrame())
        rows = []
        for o in values:
            if not isinstance(o, dict):
                continue
            greeks = o.get("greeks") if isinstance(o.get("greeks"), dict) else {}
            rows.append({
                "optionType": str(o.get("option_type") or "").lower(),
                "contractSymbol": o.get("symbol"),
                "contractSize": o.get("contract_size", 100),
                "strike": o.get("strike"), "bid": o.get("bid"), "ask": o.get("ask"),
                "lastPrice": o.get("last"), "volume": o.get("volume"),
                "openInterest": o.get("open_interest"), "delta": greeks.get("delta"),
                "impliedVolatility": greeks.get("mid_iv"), "lastTradeDate": o.get("trade_date"),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _clean_chain(frame), _clean_chain(frame)
        return (_clean_chain(frame[frame["optionType"] == "call"]),
                _clean_chain(frame[frame["optionType"] == "put"]))
    try:
        import yfinance as yf
        data = yf.Ticker(symbol).option_chain(expiry)
        return _clean_chain(data.calls), _clean_chain(data.puts)
    except Exception as exc:
        raise OptionsDataError(f"Yahoo Finance option chain failed ({type(exc).__name__}).") from exc


def get_last_underlying_price(symbol: str, provider: str = "Yahoo Finance", token: str = "") -> float | None:
    if provider == "Tradier sandbox":
        data = _tradier_get("/markets/quotes", token, {"symbols": symbol})
        quotes = data.get("quotes")
        quote = quotes.get("quote") if isinstance(quotes, dict) else None
        if isinstance(quote, list):
            quote = quote[0] if quote else None
        if not isinstance(quote, dict):
            return None
        for key in ("last", "close"):
            try:
                p = float(quote.get(key))
                if p > 0:
                    return p
            except (TypeError, ValueError):
                pass
        return None
    try:
        import yfinance as yf
        d = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
        c = pd.to_numeric(d["Close"], errors="coerce").dropna() if d is not None and not d.empty else pd.Series(dtype=float)
        return float(c.iloc[-1]) if not c.empty else None
    except Exception as exc:
        raise OptionsDataError(f"Yahoo Finance underlying lookup failed ({type(exc).__name__}).") from exc
