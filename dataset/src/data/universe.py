"""Build liquid-stock universes from Massive grouped daily summaries."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .massive_rest import MassiveRestClient

# Skip preferred/warrant-style symbols; keep common stocks (incl. BRK.B)
_COMMON_STOCK = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")


def is_common_equity(ticker: str) -> bool:
    if not _COMMON_STOCK.match(ticker):
        return False
    # Drop preferred share patterns like KIMpL, BACpB (lowercase letter class)
    if re.search(r"[a-z]", ticker):
        return False
    return True


def build_top_liquid_universe(
    client: MassiveRestClient,
    n: int = 300,
    lookback_days: int = 20,
    min_price: float = 5.0,
    always_include: list[str] | None = None,
) -> list[str]:
    """Rank by average dollar volume over recent trading days."""
    always_include = always_include or ["SPY"]
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=lookback_days)
    dollar_vol: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for d in dates:
        day = d.strftime("%Y-%m-%d")
        try:
            g = client.get_grouped_daily(day)
        except Exception:
            continue
        if g.empty:
            continue
        g = g[g["close"] >= min_price]
        g = g[g["ticker"].map(is_common_equity)]
        for _, row in g.iterrows():
            t = row["ticker"]
            dv = float(row["close"]) * float(row["volume"])
            dollar_vol[t] += dv
            counts[t] += 1

    # Average dollar volume where seen on at least half the sample days
    min_days = max(lookback_days // 2, 5)
    ranked = [
        t
        for t, _ in sorted(
            ((t, dollar_vol[t] / counts[t]) for t in dollar_vol if counts[t] >= min_days),
            key=lambda x: x[1],
            reverse=True,
        )
    ]
    top = ranked[:n]
    for sym in always_include:
        if sym not in top:
            top.insert(0, sym)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in top:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[: n + len(always_include)]


def save_universe(tickers: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + "\n")


def load_universe(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
