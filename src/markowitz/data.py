"""Historical market-data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class HistoricalMarketData:
    """Aligned returns and annualized inputs for portfolio analysis."""

    asset_returns: pd.DataFrame
    market_returns: pd.Series
    expected_returns: np.ndarray
    covariance: np.ndarray
    market_expected_return: float
    risk_free_rate: float


def download_adjusted_close(
    tickers: Sequence[str] | str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted closing prices using yfinance."""

    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            'Historical data requires: pip install -e ".[market]"'
        ) from exc

    requested = [tickers] if isinstance(tickers, str) else list(tickers)
    if not requested:
        raise ValueError("at least one ticker is required")

    raw = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no observations")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame(name=requested[0])
    elif len(requested) == 1:
        close.columns = requested

    available = [ticker for ticker in requested if ticker in close.columns]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"missing price history for: {', '.join(missing)}")

    prices = close[available].sort_index().ffill().dropna()
    if len(prices) < 3:
        raise ValueError("insufficient overlapping price observations")
    return prices


def estimate_annualized_statistics(
    prices: pd.DataFrame,
    *,
    trading_days: int = 252,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate annualized arithmetic returns and covariance from prices."""

    if trading_days <= 0:
        raise ValueError("trading_days must be positive")
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError("at least two return observations are required")
    expected_returns = returns.mean().to_numpy(dtype=float) * trading_days
    covariance = returns.cov().to_numpy(dtype=float) * trading_days
    return expected_returns, covariance, returns


def download_historical_market_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    *,
    market_ticker: str = "^GSPC",
    risk_free_ticker: str = "^IRX",
    trading_days: int = 252,
) -> HistoricalMarketData:
    """Download aligned assets, a market benchmark, and a risk-free proxy."""

    import pandas as pd

    assets_and_market = list(tickers) + [market_ticker]
    prices = download_adjusted_close(assets_and_market, start, end)
    expected_returns, covariance, all_returns = estimate_annualized_statistics(
        prices,
        trading_days=trading_days,
    )

    asset_count = len(tickers)
    asset_returns = all_returns[list(tickers)]
    market_returns = all_returns[market_ticker]
    market_expected_return = float(market_returns.mean() * trading_days)

    risk_free_prices = download_adjusted_close(risk_free_ticker, start, end)
    risk_free_values = pd.to_numeric(risk_free_prices.iloc[:, 0], errors="coerce").dropna()
    if risk_free_values.empty:
        raise ValueError("risk-free series contains no usable observations")
    risk_free_rate = float(risk_free_values.mean() / 100.0)

    return HistoricalMarketData(
        asset_returns=asset_returns,
        market_returns=market_returns,
        expected_returns=expected_returns[:asset_count],
        covariance=covariance[:asset_count, :asset_count],
        market_expected_return=market_expected_return,
        risk_free_rate=risk_free_rate,
    )
