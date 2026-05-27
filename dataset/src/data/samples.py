"""Build [N, 120, F] samples with metadata and rolling splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import ROUTER_COLS, feature_columns


@dataclass
class DatasetBundle:
    X: np.ndarray
    router_features: np.ndarray
    y: np.ndarray
    asset_id: np.ndarray
    dates: np.ndarray
    split_id: np.ndarray
    feature_names: list[str]


def _rolling_split_id(dates: pd.Series, cfg: dict) -> np.ndarray:
    """Assign split_id by rolling monthly scheme from config."""
    first = pd.Timestamp(cfg["splits"].get("first_label_date", dates.min()))
    train_m = int(cfg["splits"]["train_months"])
    val_m = int(cfg["splits"]["val_months"])
    test_m = int(cfg["splits"]["test_months"])
    block = train_m + val_m + test_m

    split_ids = np.full(len(dates), -1, dtype=np.int32)
    for i, d in enumerate(dates):
        if d < first:
            continue
        months_since = (d.year - first.year) * 12 + (d.month - first.month)
        if months_since < 0:
            continue
        pos_in_cycle = months_since % block
        if pos_in_cycle < train_m:
            split_ids[i] = 0  # train
        elif pos_in_cycle < train_m + val_m:
            split_ids[i] = 1  # val
        else:
            split_ids[i] = 2  # test
    return split_ids


def build_samples(
    panel: pd.DataFrame,
    cfg: dict,
    fit_scaler_on_train: bool = True,
) -> DatasetBundle:
    """
    panel: long format with columns ticker, date, features..., target_fwd_5d_log
    """
    lookback = int(cfg["sample"]["lookback_days"])
    forward = int(cfg["sample"]["forward_days"])
    first_label = pd.Timestamp(cfg["splits"]["first_label_date"])
    feat_cols = feature_columns(include_raw=True)
    router_cols = [c for c in ROUTER_COLS if c in feat_cols]

    xs, routers, ys, assets, dates, splits = [], [], [], [], [], []

    for ticker, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        values = g[feat_cols].to_numpy(dtype=np.float64)
        target = g["target_fwd_5d_log"].to_numpy(dtype=np.float64)
        router = g[router_cols].to_numpy(dtype=np.float64)
        n = len(g)

        for end_idx in range(lookback - 1, n - forward):
            label_date = g.loc[end_idx, "date"]
            if label_date < first_label:
                continue
            start_idx = end_idx - lookback + 1
            window = values[start_idx : end_idx + 1]
            if window.shape[0] != lookback:
                continue
            if np.any(~np.isfinite(window)) or not np.isfinite(target[end_idx]):
                continue
            xs.append(window)
            routers.append(router[end_idx])
            ys.append(target[end_idx])
            assets.append(ticker)
            dates.append(label_date)
            splits.append(-1)  # filled below

    if not xs:
        raise ValueError("No samples produced; check date range and warmup.")

    X = np.stack(xs, axis=0)
    router_features = np.stack(routers, axis=0)
    y = np.array(ys, dtype=np.float64)[:, None]
    asset_id = np.array(assets)
    date_arr = pd.to_datetime(dates)
    split_id = _rolling_split_id(date_arr, cfg)

    # Z-score normalize using train split only
    train_mask = split_id == 0
    if fit_scaler_on_train and train_mask.any():
        mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
        std = X[train_mask].std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X = (X - mean) / std
        r_mean = router_features[train_mask].mean(axis=0, keepdims=True)
        r_std = router_features[train_mask].std(axis=0, keepdims=True)
        r_std = np.where(r_std < 1e-8, 1.0, r_std)
        router_features = (router_features - r_mean) / r_std

    return DatasetBundle(
        X=X.astype(np.float32),
        router_features=router_features.astype(np.float32),
        y=y.astype(np.float32),
        asset_id=asset_id,
        dates=date_arr.to_numpy(),
        split_id=split_id,
        feature_names=feat_cols,
    )


def save_dataset(bundle: DatasetBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", bundle.X)
    np.save(out_dir / "router_features.npy", bundle.router_features)
    np.save(out_dir / "y.npy", bundle.y)
    pd.DataFrame(
        {
            "asset_id": bundle.asset_id,
            "date": bundle.dates,
            "split_id": bundle.split_id,
        }
    ).to_parquet(out_dir / "metadata.parquet", index=False)
    (out_dir / "feature_names.txt").write_text("\n".join(bundle.feature_names))


def load_dataset(processed_dir: Path) -> DatasetBundle:
    meta = pd.read_parquet(processed_dir / "metadata.parquet")
    names = (processed_dir / "feature_names.txt").read_text().strip().split("\n")
    return DatasetBundle(
        X=np.load(processed_dir / "X.npy"),
        router_features=np.load(processed_dir / "router_features.npy"),
        y=np.load(processed_dir / "y.npy"),
        asset_id=meta["asset_id"].to_numpy(),
        dates=meta["date"].to_numpy(),
        split_id=meta["split_id"].to_numpy(),
        feature_names=names,
    )


def sample_count_table(bundle: DatasetBundle) -> pd.DataFrame:
    meta = pd.DataFrame({"split_id": bundle.split_id})
    labels = {0: "train", 1: "val", 2: "test", -1: "warmup/unassigned"}
    meta["split"] = meta["split_id"].map(labels)
    return meta.groupby("split").size().rename("count").reset_index()
