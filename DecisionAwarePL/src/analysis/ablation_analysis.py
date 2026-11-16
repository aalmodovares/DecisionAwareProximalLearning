from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.ablation_common import ordered_methods
from src.analysis.results_analysis import normalize_method_names


def load_ablation_metrics(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "ablation_metrics.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Ablation metrics file not found: {candidate}")
    df = pd.read_csv(candidate)
    df = normalize_method_names(df)
    for column in ("seed", "ablation_value", "weighting_tau", "weighting_lam"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "weighting_n_rounds" in df.columns:
        df["weighting_n_rounds"] = pd.to_numeric(df["weighting_n_rounds"], errors="coerce").astype("Int64")
    return df


def ablation_metric_frame(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "regret",
    split: str = "test",
    parameter: str,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    if metric not in metrics_df.columns:
        raise KeyError(f"Metric {metric!r} is not present in the ablation metrics table.")
    required = {"ablation_parameter", "ablation_value", "method", "split", "seed"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise KeyError(f"Missing required ablation columns: {missing}")

    df = metrics_df.copy()
    df = df[df["split"].astype(str) == str(split)].copy()
    df = df[df["ablation_parameter"].astype(str) == str(parameter)].copy()
    if methods is not None:
        allowed = {str(method) for method in methods}
        df = df[df["method"].astype(str).isin(allowed)].copy()
    if seeds is not None:
        allowed_seeds = {int(seed) for seed in seeds}
        df = df[df["seed"].astype(int).isin(allowed_seeds)].copy()

    df["metric_value"] = pd.to_numeric(df[metric], errors="coerce")
    df["ablation_value"] = pd.to_numeric(df["ablation_value"], errors="coerce")
    df = df[df["metric_value"].notna() & df["ablation_value"].notna()].copy()
    return df


def plot_ablation_metric_ci(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "regret",
    split: str = "test",
    parameter: str,
    methods: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    seeds: Sequence[int] | None = None,
    confidence: float = 0.95,
    n_boot: int = 3000,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    data = ablation_metric_frame(
        metrics_df,
        metric=metric,
        split=split,
        parameter=parameter,
        methods=methods,
        seeds=seeds,
    )
    if data.empty:
        raise ValueError(f"No ablation data for parameter={parameter!r}, split={split!r}, metric={metric!r}.")

    order = ordered_methods(data["method"].dropna().unique(), method_order)
    palette = {method: colors[method] for method in order if colors and method in colors}
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sns.lineplot(
        data=data,
        x="ablation_value",
        y="metric_value",
        hue="method",
        hue_order=order,
        palette=palette or None,
        estimator="mean",
        errorbar=("ci", 100.0 * float(confidence)),
        n_boot=int(n_boot),
        marker="o",
        linewidth=2.0,
        err_style="band",
        ax=ax,
    )
    xlabel = {"tau": "tau", "lam": "lambda", "n_rounds": "Number of weighting rounds"}.get(parameter, parameter)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {xlabel}")
    ax.grid(axis="y", alpha=0.25)
    if parameter == "n_rounds":
        values = sorted(data["ablation_value"].dropna().unique())
        ax.set_xticks(values)
        ax.set_xticklabels([str(int(value)) for value in values])

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        mapped_labels = [labels.get(label, label) if labels else label for label in legend_labels]
        ax.legend(handles, mapped_labels, title=None, frameon=False)
    return fig
