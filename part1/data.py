"""Data utilities for Part 1 financial forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM"]
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-01-01"
OHLC = ["Open", "High", "Low", "Close"]


def _clean_download_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """Download adjusted OHLCV data, caching one CSV per ticker."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = cache_path / f"{ticker}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            df = _clean_download_frame(df)
            df.to_csv(csv_path)
        df = _clean_download_frame(df)
        missing = [col for col in OHLC if col not in df.columns]
        if missing:
            raise ValueError(f"{ticker} data is missing required columns: {missing}")
        out[ticker] = df
    return out


def chronological_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test splits without crossing calendar boundaries."""
    train = df.loc["2020-01-01":"2024-07-31"].copy()
    val = df.loc["2024-08-01":"2024-12-31"].copy()
    test = df.loc["2025-01-01":"2025-12-31"].copy()
    return train, val, test


def fit_scaler(df_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df_train[OHLC].to_numpy(dtype=np.float32))
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = df.copy()
    scaled.loc[:, OHLC] = scaler.transform(df[OHLC].to_numpy(dtype=np.float32))
    return scaled


def make_windows(
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    T: int = 20,
    D: int = 5,
    mode: str = "returns",
    l: int = 3,
    gamma: float = 1.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build split-local sliding windows.

    The feature windows use scaled OHLC values. Targets are always computed
    from raw adjusted prices so return ratios remain meaningful.
    """
    if mode not in {"returns", "rolling", "turning"}:
        raise ValueError(f"unknown mode: {mode}")
    if len(df_scaled) != len(df_raw):
        raise ValueError("scaled and raw frames must have the same length")

    features = df_scaled[OHLC].to_numpy(dtype=np.float32)
    close = df_raw["Close"].to_numpy(dtype=np.float64)
    high = df_raw["High"].to_numpy(dtype=np.float64)
    max_lag = l if mode == "rolling" else 0
    n = len(df_raw)
    X, y = [], []

    for start in range(0, n - T - D + 1):
        t = start + T - 1
        if t + D >= n:
            continue
        if mode == "rolling" and t + 1 - max_lag < 0:
            continue

        p_t = close[t]
        if not np.isfinite(p_t) or p_t == 0:
            continue

        window = features[start : start + T]
        if not np.isfinite(window).all():
            continue

        if mode == "returns":
            target = np.array([(close[t + d] - p_t) / p_t for d in range(1, D + 1)])
        elif mode == "rolling":
            target = []
            for d in range(1, D + 1):
                values = close[t + d - l : t + d + 1]
                target.append((values.mean() - p_t) / p_t)
            target = np.array(target)
        else:
            future_high = high[t + 1 : t + D + 1].max()
            target = float((future_high - p_t) / p_t > gamma)

        if np.isfinite(target).all():
            X.append(window)
            y.append(target)

    X_arr = np.asarray(X, dtype=np.float32)
    if mode == "turning":
        y_arr = np.asarray(y, dtype=np.float32)
    else:
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, D)
    return X_arr, y_arr


def _concat(parts: list[np.ndarray], tail_shape: tuple[int, ...]) -> np.ndarray:
    if parts:
        return np.concatenate(parts, axis=0)
    return np.empty((0, *tail_shape), dtype=np.float32)


def make_dataset(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    mode: str = "returns",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path = "data",
    T: int = 20,
    D: int = 5,
    l: int = 3,
    gamma: float = 1.1,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Build train/val/test TensorDatasets concatenated across tickers."""
    raw_by_ticker = download_data(tickers, start, end, cache_dir)
    split_X: list[list[np.ndarray]] = [[], [], []]
    split_y: list[list[np.ndarray]] = [[], [], []]

    for df_raw in raw_by_ticker.values():
        train_raw, val_raw, test_raw = chronological_splits(df_raw)
        scaler = fit_scaler(train_raw)
        split_raw = [train_raw, val_raw, test_raw]
        split_scaled = [apply_scaler(part, scaler) for part in split_raw]

        for idx, (scaled, raw) in enumerate(zip(split_scaled, split_raw)):
            X, y = make_windows(scaled, raw, T=T, D=D, mode=mode, l=l, gamma=gamma)
            split_X[idx].append(X)
            split_y[idx].append(y)

    datasets = []
    y_shape = () if mode == "turning" else (D,)
    for xs, ys in zip(split_X, split_y):
        X_all = _concat(xs, (T, len(OHLC)))
        y_all = _concat(ys, y_shape)
        datasets.append(TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all)))
    return tuple(datasets)  # type: ignore[return-value]


def make_recent_return_baseline(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    mode: str = "returns",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path = "data",
    T: int = 20,
    D: int = 5,
    l: int = 3,
) -> np.ndarray:
    """Return the raw-price recent-return baseline for test regression windows."""
    if mode not in {"returns", "rolling"}:
        raise ValueError("recent-return baseline is only defined for regression modes")
    raw_by_ticker = download_data(tickers, start, end, cache_dir)
    baselines = []
    for df_raw in raw_by_ticker.values():
        _, _, test_raw = chronological_splits(df_raw)
        close = test_raw["Close"].to_numpy(dtype=np.float64)
        n = len(test_raw)
        for win_start in range(0, n - T - D + 1):
            t = win_start + T - 1
            if t + D >= n:
                continue
            if mode == "rolling" and t + 1 - l < 0:
                continue
            prev = close[t - 1]
            current = close[t]
            if not np.isfinite(prev) or not np.isfinite(current) or prev == 0:
                continue
            recent = (current - prev) / prev
            baselines.append(np.full(D, recent, dtype=np.float32))
    if not baselines:
        return np.empty((0, D), dtype=np.float32)
    return np.stack(baselines).astype(np.float32)


if __name__ == "__main__":
    train, val, test = make_dataset(["AAPL"], mode="returns")
    print(f"train X/y: {train.tensors[0].shape} {train.tensors[1].shape}")
    print(f"val X/y:   {val.tensors[0].shape} {val.tensors[1].shape}")
    print(f"test X/y:  {test.tensors[0].shape} {test.tensors[1].shape}")
