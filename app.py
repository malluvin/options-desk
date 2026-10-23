"""Options Desk: mobile-friendly, read-only options analysis + local CSV paper journal."""
from __future__ import annotations

import math
import re
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from option_data import get_chain, get_expirations, get_last_underlying_price
from options_math import create_payoff, expiration_pnl, suggested_contracts
from scanner import analyze_symbol, completed_daily_bars, market_regime
from market_data import download_daily

ET = ZoneInfo("America/New_York")
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA",
    "PLTR", "COIN", "MU", "AVGO", "CRM", "ADBE", "UBER", "QQQ", "SPY",
]
SCAN_SYMBOLS = [s for s in DEFAULT_SYMBOLS if s not in ("SPY", "QQQ")]
STRATEGIES = ["Long call", "Long put", "Bull call debit spread", "Bear put debit spread"]
JOURNAL_COLUMNS = [
    "Logged ET", "Ticker", "Strategy", "Expiration", "Long strike", "Short strike",
    "Contracts", "Premium per share", "Simulated cost ($)", "Max loss est. ($)",
    "Max gain est. ($)", "Expiration breakeven", "Notes"
]

st.set_page_config(page_title="Options Desk", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container {padding-top: 1rem; padding-bottom: 3rem; max-width: 1160px}
[data-testid="stMetric"] {background: rgba(115,158,170,.10); padding: 10px; border-radius: 12px}
</style>""", unsafe_allow_html=True)
st.title("📊 Options Desk")
st.caption("Options strategy simulator · Stock-signal cross-check · Paper journal · No brokerage connection")

with st.sidebar:
    st.subheader("Account settings")
    account_value = st.number_input("Simulated account ($)", 100.0, 1000000.0, 4000.0, 250.0)
    risk_pct = st.slider("Maximum modeled risk per position (%)", 0.1, 5.0, 0.5, 0.1)
    cap_pct = st.slider("Maximum cash allocation (%)", 10, 100, 95, 5)
    st.caption("These controls estimate affordable contracts. They do not provide trade recommendations.")
    st.divider()
    st.caption("Market data from a free public data source may be delayed, inaccurate, limited, or unavailable. No execution prices are guaranteed.")

risk_cap = account_value * risk_pct / 100
lab, sandbox, signals, journal, guide = st.tabs(["Options lab", "Manual simulator", "Stock signals", "Paper journal", "Guide & risks"])

@st.cache_data(ttl=1800, show_spinner=False)
def cached_expirations(symbol: str):
    return get_expirations(symbol)

@st.cache_data(ttl=300, show_spinner=False)
def cached_chain(symbol: str, expiration: str):
    return get_chain(symbol, expiration)

@st.cache_data(ttl=180, show_spinner=False)
def cached_underlying(symbol: str):
    return get_last_underlying_price(symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_daily(symbols: tuple[str, ...]):
    return download_daily(symbols)


def is_number(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def row_for_strike(frame: pd.DataFrame, strike: float) -> pd.Series:
    return frame.loc[(frame["strike"] - strike).abs().idxmin()]


def strikes_for(frame: pd.DataFrame) -> list[float]:
    return sorted({float(x) for x in frame["strike"].dropna() if float(x) > 0})


def strike_picker(label: str, available: list[float], target: float, key: str) -> float:
    closest = min(range(len(available)), key=lambda i: abs(available[i] - target))
    return st.selectbox(label, available, index=closest, format_func=lambda k: f"${k:g}", key=key)


def quote_card(row: pd.Series, leg: str):
    ask = row.get("ask")
    bid = row.get("bid")
    volume = row.get("volume")
    oi = row.get("openInterest")
    iv = row.get("impliedVolatility")
    st.markdown(f"**{leg}: {row.get('contractSymbol', 'Option')}**")
    p1, p2, p3 = st.columns(3)
    p1.metric("Bid", f"${float(bid):.2f}" if is_number(bid) else "—")
    p2.metric("Ask", f"${float(ask):.2f}" if is_number(ask) else "—")
    p3.metric("Open interest", f"{int(oi):,}" if is_number(oi) else "—")
    iv_text = f"{float(iv)*100:.1f}%" if is_number(iv) else "N/A"
    vol_text = f"{int(volume):,}" if is_number(volume) else "N/A"
    st.caption(f"Volume: {vol_text} · IV: {iv_text} · Last trade (not quote time): {row.get('lastTradeDate', 'Unknown')}")


with lab:
    st.subheader("Build a paper options trade")
    st.caption("Choose an underlying, expiration, and strategy. Calculations below are modeled at expiration, not forecasts of the option's price before expiration.")
    col_symbol, col_strategy = st.columns([1, 2])
    with col_symbol:
        ticker_raw = st.text_input("Stock / ETF ticker", value="AAPL", max_chars=10).strip().upper()
    with col_strategy:
        strategy = st.selectbox("Strategy", STRATEGIES, index=0)

    ticker_valid = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker_raw))
    if not ticker_valid:
        st.error("Enter a valid stock ticker, such as AAPL, SPY, or MSFT.")
    else:
        try:
            expiration_options = cached_expirations(ticker_raw)
            stock_price = cached_underlying(ticker_raw)
        except Exception as exc:
            expiration_options = ()
            stock_price = None
            st.error("Could not retrieve the option list or underlying quote from the free provider. Try another ticker or refresh later.")
        if not expiration_options:
            st.info("No listed option expirations were returned. Try another stock or check again later.")
        else:
            today_et = datetime.now(ET).date()
            nearest_45 = min(range(len(expiration_options)), key=lambda i: abs((date.fromisoformat(expiration_options[i]) - today_et).days - 45))
            expiry = st.selectbox("Option expiration", expiration_options, index=nearest_45)
            days_to_expiry = max(0, (date.fromisoformat(expiry) - today_et).days)
            q1, q2 = st.columns(2)
            q1.metric("Underlying reference", f"${stock_price:,.2f}" if stock_price else "Unavailable")
            q2.metric("Calendar days until expiry", days_to_expiry)

            try:
                calls, puts = cached_chain(ticker_raw, expiry)
            except Exception:
                calls, puts = pd.DataFrame(), pd.DataFrame()
                st.error("Option-chain data couldn't be loaded. Yahoo Finance sometimes rate-limits or omits chains.")

            is_call = strategy in ("Long call", "Bull call debit spread")
            is_spread = strategy in ("Bull call debit spread", "Bear put debit spread")
            chain = calls if is_call else puts
            usable = chain.dropna(subset=["strike"]).copy() if not chain.empty else pd.DataFrame()
            strikes = strikes_for(usable) if not usable.empty else []
            if not strikes:
                st.warning("No contracts for that expiration and strategy. Select another expiration or ticker.")
            else:
                st.markdown("#### Pick your strikes")
                aim = stock_price if stock_price else float(strikes[len(strikes)//2])
                long_strike = strike_picker("Buy (long) strike", strikes, aim, key=f"long_{ticker_raw}_{expiry}_{strategy}")
                long_leg = row_for_strike(usable, long_strike)
                short_leg = None
                short_strike = None
                valid_short = True
                if is_spread:
                    short_choices = [x for x in strikes if x > long_strike] if is_call else [x for x in strikes if x < long_strike]
                    if short_choices:
                        desired = long_strike + 5 if is_call else long_strike - 5
                        short_strike = strike_picker("Sell (short) strike", short_choices, desired,
                                                     key=f"short_{ticker_raw}_{expiry}_{strategy}_{long_strike}")
                        short_leg = row_for_strike(usable, short_strike)
                    else:
                        valid_short = False
                        st.warning("This long strike has no compatible short strike. Choose a different long strike.")

                if valid_short:
                    quote_card(long_leg, "Buy")
                    if short_leg is not None:
                        quote_card(short_leg, "Sell")
                    st.markdown("#### Cost estimate")
                    ask_long = float(long_leg["ask"]) if is_number(long_leg["ask"]) else None
                    bid_short = float(short_leg["bid"]) if short_leg is not None and is_number(short_leg["bid"]) else None
                    # Strict: non-zero executable side required. Never use lastPrice as a proxy.
                    quote_ok = (ask_long is not None and ask_long > 0 and
                                (not is_spread or (bid_short is not None and bid_short > 0)))
                    indicative_debit = ask_long - (bid_short or 0) if quote_ok else None
                    manual = st.checkbox("Use a hypothetical premium instead of provider bid/ask", value=not quote_ok)
                    if manual:
                        suggested_default = float(indicative_debit) if indicative_debit is not None and indicative_debit > 0 else 1.00
                        per_share_debit = st.number_input("Hypothetical net debit per share ($)",
                                                          min_value=0.01, value=round(suggested_default, 2), step=0.05)
                        st.info("This is a manually entered hypothetical premium—not a market quote or evidence of an available fill.")
                    elif quote_ok:
                        per_share_debit = indicative_debit
                        st.caption("Estimated debit uses the long leg's ASK minus the short leg's BID. Actual execution may differ, and even these quotes can be stale.")
                    else:
                        per_share_debit = None
                        st.warning("Bid/ask data is missing or nonpositive. Enable the hypothetical-premium option to simulate a scenario.")

                    model = None
                    if per_share_debit is not None:
                        try:
                            model = create_payoff(strategy, float(long_strike), float(per_share_debit),
                                                  float(short_strike) if short_strike is not None else None)
                        except ValueError as exc:
                            st.warning(f"Invalid modeled position: {exc}. Choose different strikes or premium.")
                    if model is not None:
                        max_allowed = suggested_contracts(model.max_loss, account_value, risk_pct, cap_pct)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Max modeled loss / contract", f"${model.max_loss:,.2f}")
                        m2.metric("Max gain / contract", "Uncapped*" if model.max_gain is None else f"${model.max_gain:,.2f}")
                        m3.metric("Expiration break-even", f"${model.breakeven:,.2f}")
                        r1, r2 = st.columns(2)
                        r1.metric("Risk budget", f"${risk_cap:,.2f}")
                        r2.metric("Contracts within budget", max_allowed)
                        if max_allowed == 0:
                            st.warning("Even one standard contract/spread exceeds your selected modeled risk or allocation limit. Zero is the correct output under those settings.")
                        if model.max_gain is None:
                            st.caption("*A long call has theoretically unlimited upside, not a prediction of likely returns.")

                        st.markdown("#### Expiration payoff (one contract / spread)")
                        if stock_price is None:
                            chart_center = float(long_strike)
                        else:
                            chart_center = float(stock_price)
                        low = max(0.01, min(chart_center, float(long_strike),
                                            float(short_strike) if short_strike is not None else chart_center) * 0.65)
                        high = max(chart_center, float(long_strike),
                                   float(short_strike) if short_strike is not None else chart_center) * 1.35
                        outcomes = np.linspace(low, high, 101)
                        df_payoff = pd.DataFrame({
                            "Stock at expiration ($)": outcomes,
                            "Net P/L ($)": [expiration_pnl(model, float(s)) for s in outcomes],
                        }).set_index("Stock at expiration ($)")
                        st.line_chart(df_payoff, y="Net P/L ($)", height=260)
                        st.caption("At-expiration payoff, one standard 100-share contract or spread, excluding fees, slippage and assignment/exercise complications. Not a forecast of pre-expiration P/L.")

                        st.markdown("#### Record a simulated paper trade")
                        with st.form("log_trade", clear_on_submit=False):
                            lots = st.number_input("Number of paper contracts", 1, 100, value=max(1, min(max_allowed, 1)), step=1)
                            notes = st.text_input("Paper-trade thesis or notes (optional)", max_chars=280)
                            submit = st.form_submit_button("Add to paper journal", use_container_width=True)
                        if submit:
                            if "journal_rows" not in st.session_state:
                                st.session_state.journal_rows = []
                            st.session_state.journal_rows.append({
                                "Logged ET": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
                                "Ticker": ticker_raw,
                                "Strategy": strategy,
                                "Expiration": expiry,
                                "Long strike": float(long_strike),
                                "Short strike": float(short_strike) if short_strike is not None else "",
                                "Contracts": int(lots),
                                "Premium per share": round(float(per_share_debit), 4),
                                "Simulated cost ($)": round(float(model.debit) * 100 * int(lots), 2),
                                "Max loss est. ($)": round(float(model.max_loss) * int(lots), 2),
                                "Max gain est. ($)": round(float(model.max_gain) * int(lots), 2) if model.max_gain is not None else "Unlimited theoretical",
                                "Expiration breakeven": round(float(model.breakeven), 2),
                                "Notes": notes,
                            })
                            if int(lots) > max_allowed:
                                st.warning("Saved as a hypothetical paper trade, but this exceeds the risk budget you configured.")
                            st.success("Added to this browser session's paper journal. Download its CSV to keep it after the session ends.")

                with st.expander("View nearby option-chain quotes"):
                    if stock_price:
                        preview = usable.assign(distance=(usable["strike"] - stock_price).abs()).nsmallest(18, "distance").sort_values("strike")
                    else:
                        preview = usable.head(18)
                    view_cols = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "lastTradeDate"]
                    st.dataframe(preview[view_cols], use_container_width=True, hide_index=True)
            if st.button("Refresh option-chain data", use_container_width=True):
                cached_chain.clear()
                cached_expirations.clear()
                cached_underlying.clear()
                st.rerun()


with sandbox:
    st.subheader("Manual options simulator — works without market data")
    st.caption("Enter your own illustrative stock price, strikes, and premium. Every value here is hypothetical, not a live quote or a recommended trade.")
    sim_ticker = st.text_input("Symbol for your journal", "DEMO", key="sim_ticker").strip().upper()
    sim_strategy = st.selectbox("Options strategy", STRATEGIES, key="sim_strategy")
    sim_expiry = st.date_input("Hypothetical expiration", value=date.today(), min_value=date.today(), key="sim_expiry")
    sim_stock = st.number_input("Hypothetical stock price ($)", 0.01, 100000.0, 100.0, 1.0, key="sim_stock")
    sim_long = st.number_input("Bought option strike ($)", 0.01, 100000.0, 100.0, 1.0, key="sim_long")
    sim_spread = sim_strategy in ("Bull call debit spread", "Bear put debit spread")
    sim_short = None
    if sim_spread:
        sim_short = st.number_input("Sold option strike ($)", 0.01, 100000.0,
                                    105.0 if sim_strategy == "Bull call debit spread" else 95.0,
                                    1.0, key=f"sim_short_{sim_strategy}")
    sim_debit = st.number_input("Hypothetical net debit per share ($)", 0.01, 10000.0, 1.50, 0.05, key="sim_debit")
    try:
        simulated = create_payoff(sim_strategy, sim_long, sim_debit, sim_short)
    except ValueError as e:
        st.warning(f"Fix the example: {e}")
    else:
        s1,s2,s3 = st.columns(3)
        s1.metric("Max modeled loss / contract", f"${simulated.max_loss:,.2f}")
        s2.metric("Max gain / contract", "Uncapped*" if simulated.max_gain is None else f"${simulated.max_gain:,.2f}")
        s3.metric("Break-even at expiration", f"${simulated.breakeven:,.2f}")
        allowed = suggested_contracts(simulated.max_loss, account_value, risk_pct, cap_pct)
        st.caption(f"Under your ${risk_cap:,.2f} risk budget: **{allowed} contract(s)**. A paper scenario above that limit is for learning only.")
        if allowed == 0:
            st.warning("One standard contract/spread exceeds your configured loss or cash allocation limit.")
        low = max(0.01, min(float(sim_stock), float(sim_long), float(sim_short) if sim_short else float(sim_long)) * 0.6)
        high = max(float(sim_stock), float(sim_long), float(sim_short) if sim_short else float(sim_long)) * 1.4
        prices = np.linspace(low, high, 101)
        sim_curve = pd.DataFrame({"Stock at expiration ($)": prices,
                                 "Net P/L ($)": [expiration_pnl(simulated, float(s)) for s in prices]}).set_index("Stock at expiration ($)")
        st.line_chart(sim_curve, y="Net P/L ($)", height=260)
        st.caption("Expiration only · one standard 100-share contract/spread · no assignment, fees or slippage modeled")
        with st.form("manual_paper_form"):
            sim_lots = st.number_input("Paper contracts", 1, 100, 1, 1)
            sim_notes = st.text_input("Notes (optional)", max_chars=280, key="sim_notes")
            add_sim = st.form_submit_button("Save hypothetical paper trade", use_container_width=True)
        if add_sim:
            if "journal_rows" not in st.session_state:
                st.session_state.journal_rows = []
            st.session_state.journal_rows.append({
                "Logged ET": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": sim_ticker or "DEMO",
                "Strategy": sim_strategy,
                "Expiration": sim_expiry.isoformat(),
                "Long strike": float(sim_long),
                "Short strike": float(sim_short) if sim_short is not None else "",
                "Contracts": int(sim_lots),
                "Premium per share": round(float(sim_debit), 4),
                "Simulated cost ($)": round(float(sim_debit) * 100 * int(sim_lots), 2),
                "Max loss est. ($)": round(float(simulated.max_loss) * int(sim_lots), 2),
                "Max gain est. ($)": round(float(simulated.max_gain) * int(sim_lots), 2) if simulated.max_gain is not None else "Unlimited theoretical",
                "Expiration breakeven": round(float(simulated.breakeven), 2),
                "Notes": "[MANUAL SCENARIO] " + sim_notes,
            })
            st.success("Hypothetical scenario saved in this session. Download the CSV in the Paper journal tab to keep it.")

with signals:
    st.subheader("Underlying stock setups")
    st.caption("Reuses Swing Desk's completed-candle pullback/breakout scoring. Bullish stock setups are not proof that a call will be profitable. Puts are not rated by these bullish rules.")
    raw_watch = st.text_area("Scan these symbols", ", ".join(SCAN_SYMBOLS), height=92)
    watch = [s for s in dict.fromkeys(x.strip().upper() for x in raw_watch.replace("\n", ",").split(","))
             if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s)]
    if len(watch) > 50:
        st.warning("For free data reliability, scanning is limited to the first 50 symbols.")
        watch = watch[:50]
    if st.button("Run daily stock scan", type="primary", use_container_width=True):
        with st.spinner("Downloading completed daily bars and ranking setups…"):
            try:
                symbols = tuple(dict.fromkeys(watch + ["SPY", "QQQ"]))
                raw = cached_daily(symbols)
                completed = {s: completed_daily_bars(raw.get(s, pd.DataFrame())) for s in symbols}
                favorable, regime_label = market_regime(completed)
                result_rows = [analyze_symbol(s, completed.get(s, pd.DataFrame()), favorable,
                                              account_value, risk_pct, cap_pct) for s in watch]
                st.session_state.scan_output = (result_rows, regime_label)
            except Exception:
                st.error("Daily scan failed due to a network or market-data problem. Please try again later.")
    if "scan_output" in st.session_state:
        result_rows, regime_label = st.session_state.scan_output
        frame = pd.DataFrame(result_rows)
        if not frame.empty:
            valid = frame[frame["Setup"].isin(["Potential pullback", "Potential breakout"])].copy()
            st.markdown(f"**Market filter:** {regime_label} · **Scanned:** {len(frame)} · **Potential bullish setups:** {len(valid)}")
            if valid.empty:
                st.info("No qualifying bullish setup in this scan. That is a valid result.")
            else:
                st.dataframe(valid.sort_values(["Score", "Rel. volume"], ascending=False), use_container_width=True, hide_index=True)
            with st.expander("All scan results"):
                st.dataframe(frame, use_container_width=True, hide_index=True)
            st.caption("Stock signals are independent of the options chain. Always check liquidity, time decay, volatility, and earnings before simulating an options position.")
    if st.button("Clear daily scan cache"):
        cached_daily.clear()
        st.session_state.pop("scan_output", None)
        st.rerun()

with journal:
    st.subheader("Paper-trade journal")
    st.info("This journal exists only in your current Streamlit browser session. It is NOT a cloud database. Download your CSV before closing or resetting the app, then upload it on your next visit. On a public app, don't enter private account information.")
    if "journal_rows" not in st.session_state:
        st.session_state.journal_rows = []
    file = st.file_uploader("Import a previously downloaded paper journal CSV", type=["csv"])
    if file is not None and st.button("Import uploaded journal"):
        try:
            loaded = pd.read_csv(file).fillna("")
            if not all(col in loaded.columns for col in JOURNAL_COLUMNS):
                st.error("This file does not have all required Options Desk journal columns.")
            elif len(loaded) > 10000:
                st.error("Journal is too large for this starter app (10,000 rows maximum).")
            else:
                st.session_state.journal_rows = loaded[JOURNAL_COLUMNS].to_dict("records")
                st.success(f"Imported {len(loaded)} rows. Existing in-session entries were replaced.")
        except Exception:
            st.error("Could not read that journal CSV.")
    if st.session_state.journal_rows:
        history = pd.DataFrame(st.session_state.journal_rows)
        for col in JOURNAL_COLUMNS:
            if col not in history:
                history[col] = ""
        st.dataframe(history[JOURNAL_COLUMNS], use_container_width=True, hide_index=True)
        st.download_button("Download paper journal CSV", history[JOURNAL_COLUMNS].to_csv(index=False).encode("utf-8"),
                           file_name="options_desk_paper_journal.csv", mime="text/csv", use_container_width=True)
        st.caption("These are opening-plan snapshots, not marks-to-market or verified fills. P/L after an exit is not automatically tracked.")
        if st.button("Clear journal from this session"):
            st.session_state.journal_rows = []
            st.rerun()
    else:
        st.caption("No paper trades logged in this browser session yet. Start in Options lab.")

with guide:
    st.subheader("How to use Options Desk")
    st.markdown("""
1. **Start with a stock idea.** Run the Stock signals tab or enter a ticker directly in Options lab.
2. **Choose a strategy.** Long calls and bull call debit spreads are bullish; long puts and bear put debit spreads are bearish.
3. **Review expiration and the option chain.** An option can lose money even if the stock moves in the expected direction because of time decay, implied volatility changes, and/or paying too much premium.
4. **Check strikes and liquidity.** Large bid/ask spreads, low open interest, stale quotes, and upcoming earnings all matter.
5. **Check the real risk.** A standard equity option normally covers 100 shares. The app models the upfront net debit as the maximum expiration loss **while the selected spread remains intact**; short legs can be assigned and create unexpected stock positions.
6. **Simulate manually.** Use the Manual simulator when the free provider is unavailable; add a paper trade to the journal and export the CSV to retain it.
""")
    st.warning("No order placement, no broker connection, no automatic monitoring, and no guaranteed live quotes. The chart displays terminal expiration P/L only, not mark-to-market before expiration. Exercise/assignment (particularly near expiration), transaction fees, and slippage are not modeled. Scores are not probabilities of profit. This is a learning tool, not personalized investment advice.")
    st.markdown("**Options education:** [Options Industry Council](https://www.optionseducation.org/) · **Deployment:** [Streamlit Community Cloud](https://share.streamlit.io/)")
