from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.analysis.ablation_common import ordered_methods
from src.analysis.results_analysis import normalize_method_names


def load_confounding_ablation_metrics(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "ablation_metrics.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Confounding ablation metrics file not found: {candidate}")
    df = pd.read_csv(candidate)
    df = normalize_method_names(df)
    numeric_columns = [
        "seed",
        "confounding_strength",
        "outcome_confounding_scale_reference",
        "nonadditive_gamma_reference",
        "outcome_confounding_scale",
        "nonadditive_confounding_gamma",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def confounding_metric_frame(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "regret",
    split: str = "test",
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    if metric not in metrics_df.columns:
        raise KeyError(f"Metric {metric!r} is not present in the confounding ablation metrics table.")
    required = {"confounding_strength", "method", "split", "seed"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise KeyError(f"Missing required confounding ablation columns: {missing}")

    df = metrics_df.copy()
    df = df[df["split"].astype(str) == str(split)].copy()
    if methods is not None:
        allowed = {str(method) for method in methods}
        df = df[df["method"].astype(str).isin(allowed)].copy()
    if seeds is not None:
        allowed_seeds = {int(seed) for seed in seeds}
        df = df[df["seed"].astype(int).isin(allowed_seeds)].copy()

    df["metric_value"] = pd.to_numeric(df[metric], errors="coerce")
    df["confounding_strength"] = pd.to_numeric(df["confounding_strength"], errors="coerce")
    df = df[df["metric_value"].notna() & df["confounding_strength"].notna()].copy()
    return df


