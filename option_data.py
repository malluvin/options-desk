"""Read-only options market data adapters (Yahoo Finance and Tradier sandbox).

Trade placement is intentionally absent. Missing executable-side bid/ask stays missing;
never substitute lastPrice as if it were a live executable quote.
"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
FIELDS = ("contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "lastTradeDate")
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"


class OptionsDataError(RuntimeError):
    """Market-data lookup failed; no chain or quote should be inferred."""


def _clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(FIELDS))
    f = df.copy()
    for col in FIELDS:
        if col not in f.columns:
            f[col] = pd.NA
    for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    return f.loc[:, FIELDS].sort_values("strike").reset_index(drop=True)


def _tradier_get(path: str, token: str, params: dict[str, Any]) -> dict:
    # Deliberate read-only allowlist. Never send order requests or use live API base URL.
    allowed = {"/markets/options/expirations", "/markets/options/chains", "/markets/quotes"}
    if path not in allowed:
        raise OptionsDataError("Unsupported market-data endpoint")
    if not token or not token.strip():
        raise OptionsDataError("Tradier sandbox token is missing")
    try:
        response = requests.get(
            TRADIER_SANDBOX + path,
            headers={"Authorization": f"Bearer {token.strip()}", "Accept": "application/json"},
            params=params, timeout=12,
        )
        if response.status_code == 401:
            raise OptionsDataError("Tradier rejected the sandbox token (401). Check that it is a sandbox token, not a production token.")
        if response.status_code == 429:
            raise OptionsDataError("Tradier rate limited the market-data lookup (429). Try again later.")
        if response.status_code >= 400:
            raise OptionsDataError(f"Tradier API returned HTTP {response.status_code}. Check API access/status.")
        payload = response.json()
        if not isinstance(payload, dict):
            raise OptionsDataError("Tradier returned an unexpected response format.")
        if payload.get("fault") or payload.get("errors"):
            raise OptionsDataError("Tradier returned an API error for this lookup.")
        return payload
    except OptionsDataError:
        raise
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise OptionsDataError(f"Tradier connection/response failed ({type(exc).__name__}).") from exc


def get_expirations(symbol: str, provider: str = "Yahoo Finance", token: str = "") -> tuple[str, ...]:
    if provider == "Tradier sandbox":
        raw = _tradier_get("/markets/options/expirations", token, {"symbol": symbol, "includeAllRoots": "true"})
        dates = (raw.get("expirations") or {}).get("date") if isinstance(raw.get("expirations"), dict) else None
        if isinstance(dates, str):
            dates = [dates]
        if not isinstance(dates, list):
            return ()
    else:
        try:
            import yfinance as yf
            dates = list(yf.Ticker(symbol).options)
        except Exception as exc:
            raise OptionsDataError(f"Yahoo Finance options request failed ({type(exc).__name__}). This often happens because of provider throttling or data blocking on cloud-hosted apps.") from exc
    today = datetime.now(ET).date()
    output = []
    for day in dates:
        try:
            from datetime import date
            if date.fromisoformat(str(day)) >= today:
                output.append(str(day))
        except (ValueError, TypeError):
            continue
    return tuple(sorted(set(output)))


def get_chain(symbol: str, expiration: str, provider: str = "Yahoo Finance", token: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    if provider == "Tradier sandbox":
        data = _tradier_get("/markets/options/chains", token, {"symbol": symbol, "expiration": expiration, "greeks": "false"})
        options = data.get("options")
        contracts = options.get("option") if isinstance(options, dict) else None
        if isinstance(contracts, dict):
            contracts = [contracts]
        if not isinstance(contracts, list):
            return _clean_chain(pd.DataFrame()), _clean_chain(pd.DataFrame())
        rows = []
        for option in contracts:
            if not isinstance(option, dict):
                continue
            greeks = option.get("greeks") or {}
            if not isinstance(greeks, dict):
                greeks = {}
            rows.append({
                "optionType": str(option.get("option_type") or "").lower(),
                "contractSymbol": option.get("symbol"),
                "strike": option.get("strike"),
                "bid": option.get("bid"),
                "ask": option.get("ask"),
                "lastPrice": option.get("last"),
                "volume": option.get("volume"),
                "openInterest": option.get("open_interest"),
                "impliedVolatility": greeks.get("mid_iv"),
                "lastTradeDate": option.get("trade_date"),
            })
        f = pd.DataFrame(rows)
        if f.empty:
            return _clean_chain(f), _clean_chain(f)
        return (_clean_chain(f[f["optionType"] == "call"]),
                _clean_chain(f[f["optionType"] == "put"]))
    try:
        import yfinance as yf
        obj = yf.Ticker(symbol).option_chain(expiration)
        return _clean_chain(obj.calls), _clean_chain(obj.puts)
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
                value = float(quote.get(key))
                if value > 0:
                    return value
            except (ValueError, TypeError):
                pass
        return None
    try:
        import yfinance as yf
        d = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
    except Exception as exc:
        raise OptionsDataError(f"Yahoo Finance stock quote failed ({type(exc).__name__}).") from exc
    if d is None or d.empty:
        return None
    c = pd.to_numeric(d["Close"], errors="coerce").dropna()
    return float(c.iloc[-1]) if not c.empty else None
