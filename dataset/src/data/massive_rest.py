"""Massive.com REST client for daily stock aggregates."""

from __future__ import annotations

import os
import time
from typing import Iterator

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = os.getenv("MASSIVE_API_BASE", "https://api.massive.com")


class MassiveRestClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE,
        request_delay: float = 0.35,
        max_retries: int = 6,
    ):
        self.api_key = api_key or os.environ["MASSIVE_API_KEY"]
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get_with_retry(self, url: str, params: dict | None) -> requests.Response:
        for attempt in range(self.max_retries):
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 60))
                continue
            resp.raise_for_status()
            time.sleep(self.request_delay)
            return resp
        resp.raise_for_status()
        raise RuntimeError("unreachable")

    def get_daily_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for one ticker between YYYY-MM-DD dates."""
        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start}/{end}"
        )
        params: dict[str, str | int | bool] = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
        }
        rows: list[dict] = []
        while url:
            resp = self._get_with_retry(
                url, params if "aggs/ticker" in url else None
            )
            payload = resp.json()
            if payload.get("status") not in ("OK", "DELAYED"):
                raise RuntimeError(f"{ticker}: unexpected status {payload.get('status')}: {payload}")
            for bar in payload.get("results") or []:
                rows.append(
                    {
                        "date": pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_convert("America/New_York").normalize().tz_localize(None),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                        "vwap": bar.get("vw"),
                        "transactions": bar.get("n"),
                    }
                )
            url = payload.get("next_url")
            params = {}  # next_url is fully qualified
            if url and "apiKey=" not in url:
                # Some next_url responses omit key; session auth covers it
                pass
        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        df["ticker"] = ticker
        return df.reset_index(drop=True)

    def iter_daily_bars(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        for ticker in tickers:
            yield ticker, self.get_daily_bars(ticker, start, end)


def smoke_test() -> None:
    client = MassiveRestClient()
    df = client.get_daily_bars("AAPL", "2025-01-02", "2025-01-15")
    assert len(df) >= 5, f"expected bars, got {len(df)}"
    print(f"OK: AAPL {len(df)} bars, {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    smoke_test()
