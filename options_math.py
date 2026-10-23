"""Pure expiration payoff formulas for standard 100-share equity option contracts.

Educational models only: fees, assignment, corporate actions, margin, exercise,
slippage, dividend and broker-specific mechanics are deliberately excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

MULTIPLIER = 100


@dataclass(frozen=True)
class Payoff:
    strategy: str
    long_strike: float
    debit: float                 # Premium paid per underlying share
    short_strike: float | None
    max_loss: float              # Dollars for one standard contract/spread
    max_gain: float | None       # None = theoretically unlimited
    breakeven: float


def create_payoff(strategy: str, long_strike: float, debit: float,
                  short_strike: float | None = None) -> Payoff:
    if not all(math.isfinite(float(v)) for v in (long_strike, debit)):
        raise ValueError("Invalid strike or premium")
    if long_strike <= 0 or debit <= 0:
        raise ValueError("Strike and debit must be greater than zero")
    if strategy == "Long call":
        return Payoff(strategy, long_strike, debit, None, debit * MULTIPLIER,
                      None, long_strike + debit)
    if strategy == "Long put":
        return Payoff(strategy, long_strike, debit, None, debit * MULTIPLIER,
                      max(0.0, long_strike - debit) * MULTIPLIER,
                      long_strike - debit)
    if short_strike is None or not math.isfinite(short_strike) or short_strike <= 0:
        raise ValueError("Choose a valid short-leg strike")
    if strategy == "Bull call debit spread":
        if not short_strike > long_strike:
            raise ValueError("A bull call spread sells a higher-strike call")
        width = short_strike - long_strike
        if not debit < width:
            raise ValueError("Net debit must be smaller than the strike width")
        return Payoff(strategy, long_strike, debit, short_strike,
                      debit * MULTIPLIER, (width - debit) * MULTIPLIER,
                      long_strike + debit)
    if strategy == "Bear put debit spread":
        if not short_strike < long_strike:
            raise ValueError("A bear put spread sells a lower-strike put")
        width = long_strike - short_strike
        if not debit < width:
            raise ValueError("Net debit must be smaller than the strike width")
        return Payoff(strategy, long_strike, debit, short_strike,
                      debit * MULTIPLIER, (width - debit) * MULTIPLIER,
                      long_strike - debit)
    raise ValueError(f"Unknown strategy: {strategy}")


def expiration_pnl(option: Payoff, underlying_at_expiry: float, contracts: int = 1) -> float:
    if not math.isfinite(underlying_at_expiry) or underlying_at_expiry < 0 or contracts < 1:
        raise ValueError("Invalid stock price or number of contracts")
    k = option.long_strike
    if option.strategy == "Long call":
        spread_value = max(underlying_at_expiry - k, 0)
    elif option.strategy == "Long put":
        spread_value = max(k - underlying_at_expiry, 0)
    elif option.strategy == "Bull call debit spread":
        spread_value = min(max(underlying_at_expiry - k, 0), option.short_strike - k)
    elif option.strategy == "Bear put debit spread":
        spread_value = min(max(k - underlying_at_expiry, 0), k - option.short_strike)
    else:
        raise ValueError("Unknown strategy")
    return round((spread_value - option.debit) * MULTIPLIER * contracts, 2)


def suggested_contracts(max_loss_per_combo: float, account_value: float,
                        risk_percent: float, max_capital_percent: float = 95.0) -> int:
    """For debit positions, upfront debit equals risk estimate in this simple model.

    Returns zero when even one standard 100-share contract exceeds the limit.
    """
    if max_loss_per_combo <= 0 or account_value <= 0 or risk_percent <= 0:
        return 0
    by_risk = math.floor((account_value * risk_percent / 100) / max_loss_per_combo)
    by_cash = math.floor((account_value * max_capital_percent / 100) / max_loss_per_combo)
    return max(0, min(by_risk, by_cash))
