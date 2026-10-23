"""Options Desk 2.0: scanner-first, read-only, mobile-friendly research app.

There are NO trade placement APIs in this project. Scanner rankings only show
alignment with explicitly published filters; they are not trade recommendations.
"""
from __future__ import annotations

import hmac
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from market_data import download_daily
from option_data import OptionsDataError, get_chain, get_expirations, get_last_underlying_price
from options_math import create_payoff, expiration_pnl
from options_scanner import market_direction, directional_stock_signal, rank_contracts
from scanner import completed_daily_bars

ET = ZoneInfo("America/New_York")
DEFAULT_WATCHLIST = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "TSLA",
    "NFLX", "AVGO", "MU", "PLTR", "CRM", "ADBE", "UBER", "COIN",
    "JPM", "GS", "V", "MA", "COST", "WMT", "LLY", "ISRG",
)
JOURNAL_COLS = ["Logged ET", "Ticker", "Contract", "Direction", "Entry premium", "Contracts", "Reason", "Outcome / exit notes"]

st.set_page_config(page_title="Options Desk", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {max-width:1180px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stMetric"] {background:rgba(112,144,153,.11);border-radius:12px;padding:10px}
@media (max-width:640px) {h1{font-size:2rem !important}.block-container{padding-left:.8rem;padding-right:.8rem}}
</style>
""", unsafe_allow_html=True)


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError, OSError):
        return ""


passcode = _secret("APP_PASSCODE")
if passcode and not st.session_state.get("unlocked", False):
    st.title("🔐 Options Desk")
    st.caption("Private research dashboard · Manual trading only")
    with st.form("unlock_form"):
        attempt = st.text_input("Enter access code", type="password")
        submit = st.form_submit_button("Unlock", use_container_width=True)
    if submit:
        if hmac.compare_digest(attempt, passcode):
            st.session_state.unlocked = True
            st.rerun()
        else:
            st.error("Incorrect access code")
    st.stop()

st.title("📈 Options Desk")
st.caption("Find call & put research candidates · Check costs and liquidity · Paper trade manually")

sandbox_token = _secret("TRADIER_SANDBOX_TOKEN")
if sandbox_token and not passcode:
    st.warning("Tradier token is configured, but APP_PASSCODE is missing. Tradier is disabled to avoid exposing personal-use market data to visitors. Configure a passcode in Streamlit Secrets.")
    sandbox_token = ""
provider = "Tradier sandbox" if sandbox_token else "Yahoo Finance"

with st.sidebar:
    st.subheader("Scanner controls")
    watchlist_text = st.text_area("Stocks to monitor", ", ".join(DEFAULT_WATCHLIST), height=126)
    direction_filter = st.selectbox("Show", ["Calls and puts", "Calls only", "Puts only"])
    account_value = st.number_input("Paper account ($)", min_value=100.0, max_value=1_000_000.0, value=4000.0, step=250.0)
    max_premium = st.number_input("Max loss on one contract ($)", min_value=5.0, max_value=float(account_value), value=min(200.0, float(account_value)), step=5.0)
    st.caption(f"One long contract may lose its entire premium. Your limit is {max_premium/account_value*100:.1f}% of your paper account, before fees.")
    dte_min, dte_max = st.slider("Days until expiration", min_value=7, max_value=120, value=(30, 60), step=1)
    min_oi = st.select_slider("Min open interest", options=[0, 25, 50, 100, 250, 500], value=100)
    min_vol = st.select_slider("Min session volume", options=[0, 5, 10, 25, 50, 100], value=10)
    max_spread = st.slider("Max bid/ask spread (%)", min_value=3, max_value=35, value=15, step=1)
    max_underlyings = st.slider("Max stock setups to check", 1, 10, 5)
    st.divider()
    st.caption(f"Options source: **{provider}**")
    if provider == "Tradier sandbox":
        st.caption("Sandbox quotes approximately 15 minutes delayed; Greeks can be updated less frequently. Personal-use connection only.")
    else:
        st.caption("No token required, but Yahoo option-chain requests can fail from cloud hosting. Tradier sandbox recommended for a working options feed.")
    if passcode and st.button("Lock dashboard", use_container_width=True):
        st.session_state.unlocked = False
        st.rerun()

symbols = list(dict.fromkeys(x.strip().upper() for x in re.split(r"[,\s]+", watchlist_text) if x.strip()))
symbols = [x for x in symbols if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", x)]
symbols = [x for x in symbols if x not in ("SPY", "QQQ", "IWM")]
symbols = symbols[:40]  # keep free-data requests reasonably bounded

@st.cache_data(ttl=1800, show_spinner=False)
def history_cached(symbols_to_scan: tuple[str, ...]):
    return download_daily(symbols_to_scan)

@st.cache_data(ttl=1800, show_spinner=False)
def expirations_cached(symbol: str, src: str, token: str):
    return get_expirations(symbol, src, token)

@st.cache_data(ttl=300, show_spinner=False)
def chain_cached(symbol: str, expiration: str, src: str, token: str):
    return get_chain(symbol, expiration, src, token)

@st.cache_data(ttl=300, show_spinner=False)
def underlying_cached(symbol: str, src: str, token: str):
    return get_last_underlying_price(symbol, src, token)


def find_expirations(values: tuple[str, ...], min_days: int, max_days: int, today: date) -> list[str]:
    chosen = []
    for expiry in values:
        try:
            dte = (date.fromisoformat(expiry) - today).days
            if min_days <= dte <= max_days:
                chosen.append(expiry)
        except ValueError:
            continue
    chosen.sort(key=lambda v: abs((date.fromisoformat(v)-today).days-(min_days+max_days)/2))
    return chosen[:2]  # limit requests for the first version


def perform_scan() -> dict:
    today_et = datetime.now(ET).date()
    output = {"timestamp": datetime.now(ET).strftime("%b %d, %Y %I:%M %p ET"),
              "provider": provider, "symbols_checked": len(symbols), "market": "Unavailable",
              "signals": [], "contracts": [], "errors": [], "counts": {},
              "underlyings_checked": 0, "expiries_checked": 0}
    if not symbols:
        output["errors"].append("Watchlist is empty; add stock symbols in the sidebar.")
        return output
    try:
        data = history_cached(tuple(dict.fromkeys(symbols + ["SPY", "QQQ"])))
    except Exception as exc:
        output["errors"].append(f"Historical stock prices could not be downloaded ({type(exc).__name__}).")
        return output
    bars = {symbol: completed_daily_bars(data.get(symbol, pd.DataFrame()))
            for symbol in (symbols + ["SPY", "QQQ"])}
    output["market"] = market_direction(bars)
    leads = []
    for symbol in symbols:
        try:
            lead = directional_stock_signal(symbol, bars[symbol], output["market"])
            if lead:
                if (today_et - date.fromisoformat(lead["Signal session"])).days > 7:
                    output["errors"].append(f"{symbol}: last completed daily bar is more than seven calendar days old. Skipping stale signal.")
                    continue
                if direction_filter == "Calls only" and lead["Direction"] != "CALL":
                    continue
                if direction_filter == "Puts only" and lead["Direction"] != "PUT":
                    continue
                leads.append(lead)
        except Exception:
            output["errors"].append(f"{symbol}: historical analysis failed.")
    leads.sort(key=lambda x: -x["Stock score"])
    output["signals"] = leads
    for lead in leads[:max_underlyings]:
        ticker = lead["Ticker"]
        try:
            exp = expirations_cached(ticker, provider, sandbox_token)
            selected = find_expirations(exp, dte_min, dte_max, today_et)
            if not selected:
                output["errors"].append(f"{ticker}: no expirations in your {dte_min}–{dte_max} day range.")
                continue
            try:
                underlying = underlying_cached(ticker, provider, sandbox_token)
            except OptionsDataError:
                underlying = None
            # If quote unavailable, use labeled last completed stock close only for proximity filters.
            reference = float(underlying) if underlying else float(lead["Signal close"])
            output["underlyings_checked"] += 1
            for e in selected:
                output["expiries_checked"] += 1
                calls, puts = chain_cached(ticker, e, provider, sandbox_token)
                selected_chain = calls if lead["Direction"] == "CALL" else puts
                ranked, counts = rank_contracts(
                    selected_chain, lead, e, reference, max_premium,
                    min_oi, min_vol, max_spread, dte_min, dte_max, today_et,
                )
                for k, v in counts.items():
                    output["counts"][k] = output["counts"].get(k, 0) + v
                for contract in ranked:
                    contract["Underlying quote type"] = "provider reference" if underlying else "last completed close"
                output["contracts"].extend(ranked)
        except OptionsDataError as exc:
            output["errors"].append(f"{ticker}: {exc}")
        except Exception as exc:
            output["errors"].append(f"{ticker}: option lookup failed ({type(exc).__name__}).")
    output["contracts"].sort(key=lambda x: (-x["Screen score"], x["Bid/ask spread %"], -x["Open interest"]))
    # Keep a useful cross-stock shortlist instead of dozens of strikes in one name.
    per_ticker = {}
    shortlist = []
    for item in output["contracts"]:
        symbol = item["Ticker"]
        if per_ticker.get(symbol, 0) >= 3:
            continue
        per_ticker[symbol] = per_ticker.get(symbol, 0) + 1
        shortlist.append(item)
    output["contracts"] = shortlist[:30]
    return output


main_tab, browse_tab, journal_tab, help_tab = st.tabs(["🔎 Ranked scanner", "🧮 Option calculator", "📝 Paper journal", "ℹ️ Setup & risks"])

with main_tab:
    st.subheader("Top screened call & put contracts")
    st.caption("Screens the latest completed daily stock candles and then checks listed options for liquidity, expiration, and maximum premium. It does not predict which contracts will win.")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        run = st.button("🔎 Scan for option candidates", type="primary", use_container_width=True)
    with col_b:
        if st.button("Clear cached market data", use_container_width=True):
            history_cached.clear()
            expirations_cached.clear()
            chain_cached.clear()
            underlying_cached.clear()
            st.session_state.pop("scan_result", None)
            st.session_state.pop("scan_key", None)
            st.rerun()
    key = (tuple(symbols), provider, direction_filter, max_premium, dte_min, dte_max,
           min_oi, min_vol, max_spread, max_underlyings)
    if run:
        with st.spinner("Checking stock setups and option chains. Some data providers may block requests or return delayed quotes…"):
            st.session_state.scan_result = perform_scan()
            st.session_state.scan_key = key
    if "scan_result" not in st.session_state:
        st.info("Select your filters in the sidebar, then tap **Scan for option candidates**. Nothing is bought or sold.")
    elif st.session_state.get("scan_key") != key:
        st.warning("Your filters have changed. Run a new scan to update the results.")
    else:
        output = st.session_state.scan_result
        metrics = st.columns(3)
        metrics[0].metric("Stock setups", len(output["signals"]))
        metrics[1].metric("Qualified contracts", len(output["contracts"]))
        metrics[2].metric("Market trend", output["market"])
        st.caption(f"Scanned {output['symbols_checked']} stocks · Checked {output['underlyings_checked']} optionable setups and {output['expiries_checked']} expiration chains · {output['timestamp']} · {output['provider']}")
        if output["provider"] == "Tradier sandbox":
            st.info("Tradier sandbox option quotes are delayed about 15 minutes. Greeks/IV may refresh less often; a data-fetch timestamp is not a quote timestamp.")
        else:
            st.info("Yahoo Finance chain data may be delayed or inaccessible from Streamlit Cloud. A missing chain is not evidence that the stock has no options.")
        results = output["contracts"]
        if results:
            df = pd.DataFrame(results)
            for i, row in enumerate(results, 1):
                row["Rank"] = i
            table = pd.DataFrame(results)[[
                "Rank", "Ticker", "Direction", "Expiration", "Strike", "Ask", "Cost / max loss ($)",
                "Screen score", "DTE", "Bid/ask spread %", "Open interest", "Volume", "Contract",
            ]]
            st.markdown("#### Contracts passing your filters")
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("Download screened contracts (CSV)", data=df.to_csv(index=False).encode(),
                               file_name=f"options_screen_{datetime.now(ET).date()}.csv",
                               mime="text/csv", use_container_width=True)
            row_index = st.selectbox("Inspect a contract", range(len(results)),
                                     format_func=lambda i: f"#{i+1} — {results[i]['Ticker']} {results[i]['Expiration']} ${results[i]['Strike']:.2f} {results[i]['Direction']} · ${results[i]['Cost / max loss ($)']:.0f} max premium")
            chosen = results[row_index]
            st.markdown("#### Selected contract")
            top = st.columns(3)
            top[0].metric("Ask (per share)", f"${chosen['Ask']:.2f}")
            top[1].metric("One contract premium", f"${chosen['Cost / max loss ($)']:.2f}")
            top[2].metric("Rule score", f"{chosen['Screen score']}/100")
            st.write(f"**{chosen['Contract']}** · {chosen['Direction']} · {chosen['Expiration']} ({chosen['DTE']} DTE)")
            st.write(f"**Why shown:** {chosen['Why it passed']}. Underlying stock setup score: {chosen['Stock score']}/100.")
            st.write(f"**Expiry break-even:** ${chosen['Expiry break-even']:.2f} (plus fees). **Maximum modeled loss:** ${chosen['Cost / max loss ($)']:.2f} for one long standard contract, excluding fees.")
            st.caption(f"Underlying price reference: ${chosen['Underlying reference']:.2f} ({chosen['Underlying quote type']}); stock signal session: {chosen['Signal session']}. Quotes may not be executable at displayed prices.")
            if chosen["Delta"] is not None:
                st.caption(f"Reported delta: {chosen['Delta']:+.2f} (may be stale). No earnings filter is currently applied; check earnings and news yourself.")
        else:
            if not output["signals"]:
                st.info("No qualifying stock setups were detected. The scanner won't invent an options trade on a quiet day.")
            elif output["errors"]:
                st.warning("Stock setups were detected, but no contracts passed or market data could not be retrieved. Check provider messages and filters below.")
            else:
                st.info("No option contracts passed your liquidity, expiration, and premium filters. You can adjust your paper-trading budget or review rejected-contract reasons, but don't force an entry.")
        with st.expander("Underlying stock setups"):
            if output["signals"]:
                st.dataframe(pd.DataFrame(output["signals"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No stock signals available.")
        with st.expander("Why were contracts rejected?"):
            if output["counts"]:
                st.dataframe(pd.DataFrame([{"Reason": k.replace("_", " ").title(), "Contracts": v} for k,v in output["counts"].items()]), hide_index=True, use_container_width=True)
            else:
                st.caption("No option contracts were received for filtering.")
        if output["errors"]:
            with st.expander(f"Market-data notices ({len(output['errors'])})", expanded=not results):
                for message in output["errors"][:30]:
                    st.write("• " + message)
    st.caption("Screen score measures rule alignment, NOT win probability or expected return. Long options can expire worthless; maximum premium can be lost. This app cannot place orders.")

with browse_tab:
    st.subheader("Manual option payoff calculator")
    st.caption("Use this when market-data APIs are unavailable or when evaluating a quote from your paper brokerage. All prices are hypothetical unless you enter an actual quoted premium.")
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Strategy", ["Long call", "Long put", "Bull call debit spread", "Bear put debit spread"])
        ticker = st.text_input("Underlying ticker", value="AAPL").strip().upper()
        strike = st.number_input("Long strike ($)", min_value=0.01, value=200.0, step=1.0)
    with col2:
        premium = st.number_input("Premium / net debit per share ($)", min_value=0.01, value=2.0, step=.25)
        contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1)
        other_strike = None
        if "spread" in strategy.lower():
            other_strike = st.number_input("Short strike ($)", min_value=0.01,
                                            value=(strike+5 if strategy.startswith("Bull") else max(0.01, strike-5)), step=1.0)
    try:
        payoff = create_payoff(strategy, strike, premium, other_strike)
        s1,s2,s3 = st.columns(3)
        s1.metric("Modeled max loss", f"${payoff.max_loss*contracts:,.2f}")
        s2.metric("Expiration break-even", f"${payoff.breakeven:,.2f}")
        s3.metric("Max gain", "Not capped" if payoff.max_gain is None else f"${payoff.max_gain*contracts:,.2f}")
        samples = [max(0, strike*(0.75 + .005*i)) for i in range(101)]
        curve = pd.DataFrame({"Stock at expiration": samples,
                              "Modeled P/L ($)": [expiration_pnl(payoff, p, int(contracts)) for p in samples]})
        st.line_chart(curve, x="Stock at expiration", y="Modeled P/L ($)")
        st.caption("This chart shows expiration payoff only, not price changes before expiration. Excludes commissions, slippage, assignment/exercise complications, and taxes.")
    except ValueError as exc:
        st.warning(str(exc))

with journal_tab:
    st.subheader("Paper-trading journal")
    st.caption("Journal lives in your current browser session only. Download CSV before closing or clearing the session. On public apps, never enter personal account details.")
    if "journal_rows" not in st.session_state:
        st.session_state.journal_rows = []
    imported = st.file_uploader("Import an earlier journal CSV", type="csv")
    if imported is not None and st.button("Load imported journal"):
        try:
            frame = pd.read_csv(imported).fillna("")
            if not all(col in frame.columns for col in JOURNAL_COLS):
                st.error("CSV does not have the expected journal columns.")
            else:
                st.session_state.journal_rows = frame[JOURNAL_COLS].to_dict(orient="records")
                st.success("Journal loaded into this session.")
        except Exception:
            st.error("Could not read that journal CSV.")
    with st.form("paper_journal"):
        j1,j2 = st.columns(2)
        with j1:
            journal_ticker = st.text_input("Stock symbol", value="AAPL")
            contract_symbol = st.text_input("Option contract symbol (optional)")
            journal_direction = st.selectbox("Position", ["Long call", "Long put", "Debit spread"])
        with j2:
            journal_premium = st.number_input("Paper entry premium/share", min_value=0.01, value=2.0, step=.25)
            journal_contracts = st.number_input("Paper contracts", min_value=1, max_value=100, value=1)
            journal_reason = st.text_input("Reason for the practice trade")
        journal_notes = st.text_area("Exit/outcome notes (can fill in later when you export/edit)")
        add = st.form_submit_button("Add paper-trade note", use_container_width=True)
    if add:
        st.session_state.journal_rows.append({
            "Logged ET": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
            "Ticker": journal_ticker.strip().upper(), "Contract": contract_symbol.strip().upper(),
            "Direction": journal_direction, "Entry premium": float(journal_premium),
            "Contracts": int(journal_contracts), "Reason": journal_reason, "Outcome / exit notes": journal_notes,
        })
        st.success("Added to this session's journal. Export to save it.")
    if st.session_state.journal_rows:
        journal_df = pd.DataFrame(st.session_state.journal_rows, columns=JOURNAL_COLS)
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        st.download_button("Save journal CSV", journal_df.to_csv(index=False).encode(), "options_paper_journal.csv", mime="text/csv", use_container_width=True)
        if st.button("Clear session journal"):
            st.session_state.journal_rows = []
            st.rerun()

with help_tab:
    st.subheader("What the scanner does")
    st.markdown("""
1. **Finds a bullish or bearish stock setup** from completed daily candles (moving averages, pullbacks and breakouts).
2. **Looks up real listed contracts** in your options-market-data source for expirations in the selected range.
3. **Rejects unsuitable contracts**: missing or crossed bid/ask, wide spreads, low volume/open interest, strikes very far from spot, nonstandard multipliers, or premium beyond your budget.
4. **Ranks what remains** based on the underlying setup, liquidity, strike proximity/delta when available, time until expiration, and premium affordability.

**What the score is not:** a probability of winning, a predicted return, an earnings forecast, or a recommendation to trade. It is a transparent rules-based filter. Every contract still needs review for earnings, news, IV, spreads, and current market conditions.

**Buying options:** Long calls (bullish) and long puts (bearish) can lose **100% of the premium**. The calculator's maximum loss excludes fees. Contracts are usually for 100 shares, but adjusted contracts exist and are excluded by the scanner.

**Selling options:** This version does **not** recommend selling to open uncovered options. You can sell to close a long option in your separate paper-trading platform. Defined-risk credit spreads may be a future addition after testing.

**Data:** Yahoo Finance may refuse option chains from Streamlit Cloud. Tradier sandbox, if connected privately, provides delayed (about 15 minute) option quotes. Greeks can refresh hourly. The app makes read-only market-data requests and has no trading API calls.
""")
    if not sandbox_token:
        st.markdown("#### Connect Tradier for options contracts")
        st.write("1. Obtain a **sandbox/paper** token from your personal Tradier account. 2. In Streamlit Community Cloud, open **Options Desk → ⋮ → Settings → Secrets**. 3. Paste the following, using your own confidential values:")
        st.code('APP_PASSCODE = "YOUR_LONG_PRIVATE_PASSCODE"\nTRADIER_SANDBOX_TOKEN = "YOUR_SANDBOX_TOKEN"', language="toml")
        st.write("4. Save secrets and reboot the app if requested. Never put these secrets into GitHub. Restrict access to yourself: Tradier describes API access as personal-use unless you are a partner. A passcode is only a basic gate, not full authentication.")
    st.warning("Research and paper practice only. No automatic execution, no accurate guarantee of quote freshness or fills, and no earnings-event filter yet. Backtest with realistic costs and paper trade before using money.")
