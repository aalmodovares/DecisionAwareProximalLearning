from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import seaborn as sns
from tabulate import tabulate

from src.utils import ensure_dir


ID_COLUMNS = {
    "dataset",
    "seed",
    "split",
    "model_key",
    "method",
    "run_dir",
    "raw_method",
    "patient_rank",
    "patient_id",
    "predictor_type",
    "curve_type",
    "associational_method",
}

METHOD_NAME_MAP = {
    "PMMR": "PMMR",
    "Local_PMMR": "Local PMMR",
    "KPV": "KPV",
    "Local_KPV": "Local KPV",
    "Weighted_KPV": "Weighted KPV",
    "DFPV": "DFPV",
    "Local_DFPV": "Local DFPV",
    "Weighted_DFPV": "Weighted DFPV",
    "DR_KPV": "DR-KPV",
    "DR_PMMR_Marginal": "DR-PMMR marginal",
    "DR_PMMR_Conditional": "DR-PMMR conditional",
    "NMMR": "NMMR",
    "NMMR_U": "NMMR-U",
    "NMMR_V": "NMMR-V",
    "Weighted_PMMR": "Weighted PMMR",
    "Weighted_NMMR": "Weighted NMMR",
    "Weighted_NMMR_U": "Weighted NMMR-U",
    "Weighted_NMMR_V": "Weighted NMMR-V",
    "S_Learner": "S-Learner",
    "Oracle_S_Learner": "Oracle S-Learner",
    "Naive_OOSR": "Naive OOSR",
    "Oracle_Naive_OOSR": "Oracle Naive OOSR",
    "local_pmmr": "Local PMMR",
    "local_kpv": "Local KPV",
    "local_dfpv": "Local DFPV",
    "weighted_nmmr_u": "Weighted NMMR-U",
    "weighted_nmmr_v": "Weighted NMMR-V",
    "weighted_kpv": "Weighted KPV",
    "weighted_dfpv": "Weighted DFPV",
}

ORACLE_METHOD_TOKENS = ("oracle",)

DEFAULT_METRIC_DIRECTIONS = {
    "policy_value": True,
    "policy_value_causal_marginal": True,
    "policy_value_oracle": True,
    "policy_value_structural": True,
    "optimal_policy_value": True,
    "optimal_policy_value_causal_marginal": True,
    "optimal_policy_value_oracle": True,
    "optimal_policy_value_structural": True,
    "regret": False,
    "regret_causal_marginal": False,
    "regret_oracle": False,
    "regret_structural": False,
    "policy_mse": False,
    "policy_mse_causal_marginal": False,
    "policy_mse_oracle": False,
    "policy_mse_structural": False,
    "global_rmse": False,
    "local_rmse": False,
    "observed_rmse": False,
    "factual_mean_rmse": False,
    "factual_mean_mae": False,
    "mean_curve_rmse": False,
    "mean_curve_mae": False,
    "mean_curve_bias": False,
    "mean_curve_argmax_abs_error": False,
    "mean_curve_direct_rmse": False,
    "mean_curve_direct_mae": False,
    "mean_curve_direct_bias": False,
    "mean_curve_direct_argmax_abs_error": False,
}

_METRIC_SPLIT_PREFIXES = (
    ("validation_", "validation"),
    ("val_", "validation"),
    ("test_", "test"),
)


def _resolve_path(path: str | Path, default_file: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_dir() and default_file is not None:
        candidate = candidate / default_file
    return candidate


def normalize_method_name(method: object) -> str:
    text = str(method)
    return METHOD_NAME_MAP.get(text, text)


def normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method" in out.columns:
        if "raw_method" not in out.columns:
            out["raw_method"] = out["method"]
        out["method"] = out["method"].map(normalize_method_name)
    return out


def non_oracle_methods(methods: Iterable[str]) -> list[str]:
    return [
        str(method)
        for method in methods
        if not any(token in str(method).lower() for token in ORACLE_METHOD_TOKENS)
    ]


def resolve_metric_spec(metric: str, default_split: str = "test") -> tuple[str, str, str]:
    metric = str(metric)
    for prefix, split in _METRIC_SPLIT_PREFIXES:
        if metric.startswith(prefix):
            return metric, split, metric[len(prefix) :]
    return metric, default_split, metric


def _holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    n = p.shape[0]
    order = np.argsort(p)
    adjusted_sorted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * p[idx]
        running = max(running, adjusted)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    reject_sorted = np.zeros(n, dtype=bool)
    still_reject = True
    for rank, idx in enumerate(order):
        if still_reject and p[idx] <= alpha / (n - rank):
            reject_sorted[rank] = True
        else:
            still_reject = False
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject, adjusted


def _friedman_test(all_data: np.ndarray, comp_index: int, alpha: float, higher_is_better: bool) -> dict[str, np.ndarray | float]:
    table = np.asarray(all_data, dtype=float)
    n_blocks, n_methods = table.shape
    if n_methods < 2:
        raise ValueError("Friedman test requires at least two methods.")
    if higher_is_better:
        ranks = np.array([scipy_stats.rankdata(-row) for row in table])
    else:
        ranks = np.array([scipy_stats.rankdata(row) for row in table])
    average_ranks = np.mean(ranks, axis=0)
    statistic, p_value = scipy_stats.friedmanchisquare(*[table[:, i] for i in range(n_methods)])

    baseline = average_ranks[comp_index]
    z_values = []
    p_values = []
    denom = math.sqrt(n_methods * (n_methods + 1) / (6 * n_blocks))
    for idx in range(n_methods):
        z = (average_ranks[idx] - baseline) / denom
        z_values.append(z)
        p_values.append(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    reject_corr, p_values_corr = _holm_adjust(p_values, alpha=alpha)
    reject_unc = np.asarray(p_values, dtype=float) < alpha
    return {
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "average_ranks": np.asarray(average_ranks, dtype=float),
        "z_values": np.asarray(z_values, dtype=float),
        "p_values_post_hoc_unc": np.asarray(p_values, dtype=float),
        "p_values_post_hoc_corr": np.asarray(p_values_corr, dtype=float),
        "reject_post_hoc_unc": reject_unc,
        "reject_post_hoc_corr": reject_corr,
    }


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_method_names(df)
    for column in out.columns:
        if column in ID_COLUMNS:
            continue
        converted = pd.to_numeric(out[column], errors="coerce")
        if converted.notna().any():
            out[column] = converted
    if "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    if "patient_rank" in out.columns:
        out["patient_rank"] = pd.to_numeric(out["patient_rank"], errors="coerce").astype("Int64")
    if "patient_id" in out.columns:
        out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype("Int64")
    return out


def load_results_metrics(path: str | Path) -> pd.DataFrame:
    metrics_path = _resolve_path(path, "results_metrics.csv")
    df = pd.read_csv(metrics_path)
    return _coerce_numeric_columns(df)


def _seed_from_run_dir(path: Path) -> int | None:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return None


def seed_run_dirs(
    results_dir: str | Path,
    seeds: Sequence[int] | None = None,
    *,
    include_append: bool = True,
) -> list[Path]:
    results_path = Path(results_dir)
    dirs: list[Path] = []

    runs_dir = results_path / "runs"
    if runs_dir.exists():
        dirs.extend(sorted(path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")))

    append_dir = results_path / "runs_append"
    if include_append and append_dir.exists():
        append_runs = sorted(path for path in append_dir.iterdir() if path.is_dir() and path.name.startswith("append_"))
        for append_run in append_runs:
            dirs.extend(sorted(path for path in append_run.iterdir() if path.is_dir() and path.name.startswith("seed_")))

    if not dirs:
        searched = [str(runs_dir)]
        if include_append:
            searched.append(str(append_dir / "append_*"))
        raise FileNotFoundError(f"No seed run directories found below: {searched}")

    if seeds is None:
        return dirs
    seed_set = {int(seed) for seed in seeds}
    out = []
    for path in dirs:
        seed = _seed_from_run_dir(path)
        if seed is None:
            continue
        if seed in seed_set:
            out.append(path)
    return out


def _dedupe_seed_table(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    key_by_file = {
        "global_curves.csv": [
            "dataset",
            "seed",
            "split",
            "method",
            "predictor_type",
            "curve_type",
            "treatment",
        ],
        "patient_curves.csv": [
            "dataset",
            "seed",
            "split",
            "patient_rank",
            "patient_id",
            "method",
            "predictor_type",
            "curve_type",
            "treatment",
        ],
        "patient_metadata.csv": [
            "dataset",
            "seed",
            "split",
            "patient_rank",
            "patient_id",
        ],
    }
    keys = [key for key in key_by_file.get(filename, []) if key in df.columns]
    if not keys:
        return df
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def _load_seed_tables(
    results_dir: str | Path,
    filename: str,
    seeds: Sequence[int] | None = None,
    *,
    include_append: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    frames = []
    for source_order, run_dir in enumerate(seed_run_dirs(results_dir, seeds=seeds, include_append=include_append)):
        path = run_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            seed = _seed_from_run_dir(run_dir)
            frame["seed"] = np.nan if seed is None else seed
        frame["run_dir"] = str(run_dir)
        frame["_source_order"] = source_order
        frames.append(frame)
    if not frames:
        searched = [str(Path(results_dir) / "runs")]
        if include_append:
            searched.append(str(Path(results_dir) / "runs_append" / "append_*"))
        raise FileNotFoundError(f"No {filename!r} files found below: {searched}")
    out = _coerce_numeric_columns(pd.concat(frames, ignore_index=True))
    out = out.sort_values("_source_order", kind="stable")
    if deduplicate:
        out = _dedupe_seed_table(out, filename)
    return out.drop(columns=["_source_order"], errors="ignore")


def load_global_curves(
    results_dir: str | Path,
    seeds: Sequence[int] | None = None,
    *,
    include_append: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    return _load_seed_tables(
        results_dir,
        "global_curves.csv",
        seeds=seeds,
        include_append=include_append,
        deduplicate=deduplicate,
    )


def load_patient_curves(
    results_dir: str | Path,
    seeds: Sequence[int] | None = None,
    *,
    include_append: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    return _load_seed_tables(
        results_dir,
        "patient_curves.csv",
        seeds=seeds,
        include_append=include_append,
        deduplicate=deduplicate,
    )


def load_patient_metadata(
    results_dir: str | Path,
    seeds: Sequence[int] | None = None,
    *,
    include_append: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    return _load_seed_tables(
        results_dir,
        "patient_metadata.csv",
        seeds=seeds,
        include_append=include_append,
        deduplicate=deduplicate,
    )


def available_metrics(metrics_df: pd.DataFrame, *, split: str = "test") -> list[str]:
    df = metrics_df[metrics_df["split"].astype(str) == split] if "split" in metrics_df.columns else metrics_df
    candidates = []
    for column in df.columns:
        if column in ID_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            candidates.append(column)
    return candidates


def metric_spec_available(metrics_df: pd.DataFrame, metric: str, *, split: str = "test") -> bool:
    _, metric_split, column = resolve_metric_spec(metric, default_split=split)
    if column not in metrics_df.columns:
        return False
    df = metrics_df
    if "split" in df.columns:
        df = df[df["split"].astype(str) == metric_split]
    if df.empty:
        return False
    values = pd.to_numeric(df[column], errors="coerce")
    return bool(values.notna().any())


def metric_higher_is_better(metric: str, directions: dict[str, bool] | None = None) -> bool:
    merged = dict(DEFAULT_METRIC_DIRECTIONS)
    if directions:
        merged.update(directions)
    if metric in merged:
        return bool(merged[metric])
    _, _, base_metric = resolve_metric_spec(metric)
    if base_metric in merged:
        return bool(merged[base_metric])
    lower_tokens = ("rmse", "mae", "mse", "regret", "error", "bias", "loss")
    if any(token in base_metric.lower() for token in lower_tokens):
        return False
    return True


def summarize_metric_ci(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split: str = "test",
    group_cols: Sequence[str] = ("dataset", "model_key", "method"),
    confidence: float = 0.95,
) -> pd.DataFrame:
    df = metrics_df.copy()
    rows = []
    for metric in metrics:
        metric_name, metric_split, metric_column = resolve_metric_spec(metric, default_split=split)
        metric_df = df
        if "split" in metric_df.columns:
            metric_df = metric_df[metric_df["split"].astype(str) == metric_split]
        if metric_column not in metric_df.columns:
            continue
        values_df = metric_df[list(group_cols) + ["seed", metric_column]].copy()
        values_df["metric_value"] = pd.to_numeric(values_df[metric_column], errors="coerce")
        values_df = values_df[np.isfinite(values_df["metric_value"])]
        for keys, group in values_df.groupby(list(group_cols), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = group["metric_value"].to_numpy(dtype=float)
            n = int(values.shape[0])
            mean = float(np.mean(values)) if n else np.nan
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            sem = float(std / math.sqrt(n)) if n > 1 else 0.0
            if n > 1:
                tcrit = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
                ci_half_width = tcrit * sem
            else:
                ci_half_width = 0.0
            row = {col: value for col, value in zip(group_cols, keys)}
            row.update(
                {
                    "metric": metric,
                    "source_split": metric_split,
                    "source_metric": metric_column,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci_low": mean - ci_half_width,
                    "ci_high": mean + ci_half_width,
                    "ci_half_width": ci_half_width,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_publication_table(
    summary_df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    method_col: str = "method",
    value_format: str = "{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]",
) -> pd.DataFrame:
    df = summary_df.copy()
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
        df["metric"] = pd.Categorical(df["metric"], categories=list(metrics), ordered=True)
        df = df.sort_values("metric")
    df["value_ci"] = [
        value_format.format(
            mean=float(row["mean"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            std=float(row["std"]),
            sem=float(row["sem"]),
            n=int(row["n"]),
        )
        for _, row in df.iterrows()
    ]
    table = df.pivot_table(index=method_col, columns="metric", values="value_ci", aggfunc="first", observed=False)
    if metrics is not None:
        table = table.reindex(columns=[metric for metric in metrics if metric in table.columns])
    return table


def plot_method_metric_ci(
    summary_df: pd.DataFrame,
    metric: str,
    *,
    method_order: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (9.0, 4.8),
    ax: plt.Axes | None = None,
    rotate_labels: int = 30,
) -> plt.Figure:
    data = summary_df[summary_df["metric"] == metric].copy()
    data = data[data["method"].notna()].copy()
    data["method"] = data["method"].astype(str)
    if method_order is not None:
        observed = set(data["method"])
        order = [str(method) for method in method_order if str(method) in observed]
        order.extend(sorted(method for method in observed if method not in set(order)))
        data = data[data["method"].isin(order)].copy()
        data["method"] = pd.Categorical(data["method"], categories=order, ordered=True)
        data = data.sort_values("method")
    else:
        data = data.sort_values("mean")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    methods = data["method"].astype(str).tolist()
    x = np.arange(len(methods))
    means = data["mean"].to_numpy(dtype=float)
    yerr = data["ci_half_width"].to_numpy(dtype=float)
    bar_colors = [colors.get(method, "tab:blue") if colors else "tab:blue" for method in methods]
    ax.bar(x, means, yerr=yerr, capsize=4, color=bar_colors, alpha=0.82)
    tick_labels = [labels.get(method, method) if labels else method for method in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=rotate_labels, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_method_metrics_ci_grid(
    summary_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    method_order: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    metric_labels: dict[str, str] | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.4, 4.2),
    title: str = "Method metrics with confidence intervals",
) -> plt.Figure:
    metrics = [metric for metric in metrics if metric in set(summary_df["metric"])]
    if not metrics:
        raise ValueError("None of the requested metrics are present in the summary table.")
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, metric in zip(axes.ravel(), metrics):
        plot_method_metric_ci(
            summary_df,
            metric,
            method_order=method_order,
            labels=labels,
            colors=colors,
            title=metric_labels.get(metric, metric.replace("_", " ")) if metric_labels else metric.replace("_", " "),
            ylabel=metric_labels.get(metric, metric) if metric_labels else metric,
            ax=ax,
        )
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def summarize_curves_ci(
    curves_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("dataset", "method", "predictor_type", "curve_type", "treatment"),
    confidence: float = 0.95,
) -> pd.DataFrame:
    df = curves_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[np.isfinite(df["value"])]
    rows = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["value"].to_numpy(dtype=float)
        n = int(values.shape[0])
        mean = float(np.mean(values)) if n else np.nan
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        if n > 1:
            tcrit = float(scipy_stats.t.ppf(0.975 if confidence == 0.95 else 0.5 + confidence / 2.0, df=n - 1))
            ci_half_width = tcrit * sem
        else:
            ci_half_width = 0.0
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_low": mean - ci_half_width,
                "ci_high": mean + ci_half_width,
                "ci_half_width": ci_half_width,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_global_curves(
    global_curves_df: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    predictor_types: Sequence[str] | None = ("mean_or_primary",),
    separate_predictor_types: bool = False,
    references: Sequence[str] = ("ground_truth_oracle_conditioned", "causal_marginal", "proxy_conditioned"),
    reference_visibility: dict[str, bool] | None = None,
    method_order: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
    reference_styles: dict[str, dict[str, object]] | None = None,
    confidence: float = 0.95,
    show_ci: bool = True,
    mark_curve_maxima: bool = False,
    max_marker: str = "*",
    max_marker_size: float = 8.0,
    figsize: tuple[float, float] = (9.5, 5.4),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    summary = summarize_curves_ci(global_curves_df, confidence=confidence)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    reference_styles = reference_styles or {
        "ground_truth_oracle_conditioned": {"color": "black", "linewidth": 2.5, "linestyle": "-"},
        "causal_marginal": {"color": "tab:green", "linewidth": 1.8, "linestyle": "--"},
        "proxy_conditioned": {"color": "0.45", "linewidth": 1.8, "linestyle": ":"},
        "structure_term": {"color": "tab:purple", "linewidth": 1.4, "linestyle": "-."},
    }
    def _mark_curve_max(x_values: pd.Series, y_values: pd.Series, *, color: str | None) -> None:
        if not mark_curve_maxima:
            return
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return
        if not np.isfinite(y_arr).any():
            return
        idx = int(np.nanargmax(y_arr))
        ax.plot(
            [x_arr[idx]],
            [y_arr[idx]],
            marker=max_marker,
            markersize=float(max_marker_size),
            linestyle="None",
            color=color or "black",
            zorder=6,
        )

    for curve_type in references:
        if reference_visibility is not None and not bool(reference_visibility.get(str(curve_type), True)):
            continue
        ref = summary[(summary["method"] == "reference") & (summary["curve_type"] == curve_type)]
        if ref.empty:
            continue
        ref = ref.sort_values("treatment")
        style = reference_styles.get(curve_type, {})
        line = ax.plot(
            ref["treatment"],
            ref["mean"],
            label=labels.get(curve_type, curve_type) if labels else curve_type,
            **style,
        )[0]
        _mark_curve_max(ref["treatment"], ref["mean"], color=line.get_color())

    pred = summary[summary["curve_type"] == "prediction"].copy()
    if methods is not None:
        pred = pred[pred["method"].isin(methods)]
    if predictor_types is not None:
        predictor_order = [str(name) for name in predictor_types]
        pred = pred[pred["predictor_type"].astype(str).isin(set(predictor_order))]
        if pred.empty:
            raise ValueError(
                "No prediction curves available after filtering predictor_types="
                f"{predictor_order}."
            )
    if method_order is None:
        method_order = sorted(pred["method"].dropna().astype(str).unique())
    predictor_style_cycle = ["-", "--", ":", "-."]
    for method in method_order:
        method_df = pred[pred["method"].astype(str) == str(method)].copy()
        if method_df.empty:
            continue
        color = colors.get(method, None) if colors else None
        base_linestyle = linestyles.get(method, "-") if linestyles else "-"
        base_label = labels.get(method, method) if labels else method

        if separate_predictor_types:
            if predictor_types is None:
                local_predictor_order = sorted(method_df["predictor_type"].dropna().astype(str).unique())
            else:
                local_predictor_order = [
                    name for name in predictor_order if name in set(method_df["predictor_type"].astype(str))
                ]
            for idx, predictor_name in enumerate(local_predictor_order):
                curve = method_df[method_df["predictor_type"].astype(str) == predictor_name].sort_values("treatment")
                if curve.empty:
                    continue
                linestyle = base_linestyle if idx == 0 else predictor_style_cycle[idx % len(predictor_style_cycle)]
                label = f"{base_label} [{predictor_name}]"
                line = ax.plot(
                    curve["treatment"],
                    curve["mean"],
                    label=label,
                    linewidth=1.8,
                    color=color,
                    linestyle=linestyle,
                )[0]
                line_color = line.get_color()
                if show_ci:
                    ax.fill_between(
                        curve["treatment"].to_numpy(dtype=float),
                        curve["ci_low"].to_numpy(dtype=float),
                        curve["ci_high"].to_numpy(dtype=float),
                        color=line_color,
                        alpha=0.12,
                        linewidth=0,
                    )
                _mark_curve_max(curve["treatment"], curve["mean"], color=line_color)
            continue

        curve = method_df.sort_values("treatment")
        if curve.empty:
            continue
        line = ax.plot(
            curve["treatment"],
            curve["mean"],
            label=base_label,
            linewidth=1.8,
            color=color,
            linestyle=base_linestyle,
        )[0]
        line_color = line.get_color()
        if show_ci:
            ax.fill_between(
                curve["treatment"].to_numpy(dtype=float),
                curve["ci_low"].to_numpy(dtype=float),
                curve["ci_high"].to_numpy(dtype=float),
                color=line_color,
                alpha=0.16,
                linewidth=0,
            )
        _mark_curve_max(curve["treatment"], curve["mean"], color=line_color)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Mean potential outcome")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_patient_surface_from_table(
    patient_curves_df: pd.DataFrame,
    *,
    seed: int,
    patient_rank: int | None = None,
    patient_id: int | None = None,
    methods: Sequence[str] | None = None,
    references: Sequence[str] = ("ground_truth_causal", "associational", "proxy_conditioned", "oracle_conditioned"),
    reference_visibility: dict[str, bool] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
    reference_styles: dict[str, dict[str, object]] | None = None,
    mark_curve_maxima: bool = False,
    max_marker: str = "*",
    max_marker_size: float = 7.0,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if patient_rank is None and patient_id is None:
        raise ValueError("Pass either patient_rank or patient_id.")
    df = patient_curves_df[patient_curves_df["seed"].astype(int) == int(seed)].copy()
    if patient_rank is not None:
        df = df[df["patient_rank"].astype(int) == int(patient_rank)]
    if patient_id is not None:
        df = df[df["patient_id"].astype(int) == int(patient_id)]
    if df.empty:
        raise ValueError("No patient curves found for the requested seed/patient.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    reference_styles = reference_styles or {
        "ground_truth_causal": {"color": "black", "linewidth": 2.4, "linestyle": "-"},
        "associational": {"color": "tab:brown", "linewidth": 1.5, "linestyle": ":"},
        "proxy_conditioned": {"color": "0.45", "linewidth": 1.6, "linestyle": "--"},
        "oracle_conditioned": {"color": "tab:purple", "linewidth": 1.5, "linestyle": "-."},
    }
    def _mark_curve_max(x_values: pd.Series, y_values: pd.Series, *, color: str | None) -> None:
        if not mark_curve_maxima:
            return
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return
        if not np.isfinite(y_arr).any():
            return
        idx = int(np.nanargmax(y_arr))
        ax.plot(
            [x_arr[idx]],
            [y_arr[idx]],
            marker=max_marker,
            markersize=float(max_marker_size),
            linestyle="None",
            color=color or "black",
            zorder=6,
        )

    for curve_type in references:
        if reference_visibility is not None and not bool(reference_visibility.get(str(curve_type), True)):
            continue
        curve = df[(df["method"] == "reference") & (df["curve_type"] == curve_type)].sort_values("treatment")
        if curve.empty:
            continue
        style = reference_styles.get(curve_type, {})
        line = ax.plot(
            curve["treatment"],
            curve["value"],
            label=labels.get(curve_type, curve_type) if labels else curve_type,
            **style,
        )[0]
        _mark_curve_max(curve["treatment"], curve["value"], color=line.get_color())
    pred = df[df["curve_type"] == "prediction"].copy()
    if methods is not None:
        pred = pred[pred["method"].isin(methods)]
        order = [str(method) for method in methods if str(method) in set(pred["method"].astype(str))]
        pred["method"] = pd.Categorical(pred["method"].astype(str), categories=order, ordered=True)
        pred = pred.sort_values(["method", "treatment"])
    for method, curve in pred.groupby("method", sort=False, observed=True):
        curve = curve.sort_values("treatment")
        color = colors.get(method, None) if colors else None
        linestyle = linestyles.get(method, "-") if linestyles else "-"
        line = ax.plot(
            curve["treatment"],
            curve["value"],
            linewidth=1.4,
            label=labels.get(method, method) if labels else method,
            color=color,
            linestyle=linestyle,
        )[0]
        _mark_curve_max(curve["treatment"], curve["value"], color=line.get_color())
    patient_label = f"patient_rank={patient_rank}" if patient_rank is not None else f"patient_id={patient_id}"
    ax.set_title(f"Seed {seed} | {patient_label}")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Outcome")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_patient_surface_gallery_from_table(
    patient_curves_df: pd.DataFrame,
    *,
    seed: int,
    patient_ranks: Sequence[int] = tuple(range(6)),
    methods: Sequence[str] | None = None,
    reference_visibility: dict[str, bool] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
    mark_curve_maxima: bool = False,
    max_marker: str = "*",
    max_marker_size: float = 7.0,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (6.5, 4.0),
) -> plt.Figure:
    ranks = list(patient_ranks)
    nrows = int(math.ceil(len(ranks) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, rank in zip(axes.ravel(), ranks):
        plot_patient_surface_from_table(
            patient_curves_df,
            seed=seed,
            patient_rank=int(rank),
            methods=methods,
            reference_visibility=reference_visibility,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            mark_curve_maxima=mark_curve_maxima,
            max_marker=max_marker,
            max_marker_size=max_marker_size,
            ax=ax,
        )
        ax.legend_.remove() if ax.legend_ else None
    for ax in axes.ravel()[len(ranks):]:
        ax.axis("off")
    handles, legend_labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(f"Patient-specific surfaces | seed={seed}", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def build_metric_block_table(
    metrics_df: pd.DataFrame,
    metric: str,
    *,
    split: str = "test",
    methods: Sequence[str] | None = None,
    block_cols: Sequence[str] = ("dataset", "seed"),
    method_col: str = "method",
) -> pd.DataFrame:
    df = metrics_df.copy()
    metric_name, metric_split, metric_column = resolve_metric_spec(metric, default_split=split)
    if "split" in df.columns:
        df = df[df["split"].astype(str) == metric_split]
    if methods is not None:
        df = df[df[method_col].isin(methods)]
    if metric_column not in df.columns:
        raise KeyError(f"Metric {metric_name!r} maps to {metric_column!r}, which is not present in metrics_df.")
    df["metric_value"] = pd.to_numeric(df[metric_column], errors="coerce")
    df = df[np.isfinite(df["metric_value"])]
    table = df.pivot_table(index=list(block_cols), columns=method_col, values="metric_value", aggfunc="mean")
    table = table.dropna(axis=0, how="any")
    return table


def _best_method_from_block_table(block_table: pd.DataFrame, higher_is_better: bool) -> str:
    means = block_table.mean(axis=0)
    return str(means.idxmax() if higher_is_better else means.idxmin())


def run_friedman_by_metric(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split: str = "test",
    methods: Sequence[str] | None = None,
    metric_directions: dict[str, bool] | None = None,
    alpha: float = 0.05,
    block_cols: Sequence[str] = ("dataset", "seed"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    posthoc_rows = []
    for metric in metrics:
        higher = metric_higher_is_better(metric, metric_directions)
        block_table = build_metric_block_table(
            metrics_df,
            metric,
            split=split,
            methods=methods,
            block_cols=block_cols,
        )
        if block_table.shape[1] < 3 or block_table.shape[0] < 2:
            continue
        method_names = list(block_table.columns)
        baseline = _best_method_from_block_table(block_table, higher)
        comp_index = method_names.index(baseline)
        result = _friedman_test(block_table.to_numpy(dtype=float), comp_index, alpha, higher)
        omnibus_rows.append(
            {
                "metric": metric,
                "n_blocks": int(block_table.shape[0]),
                "n_methods": int(block_table.shape[1]),
                "baseline": baseline,
                "higher_is_better": bool(higher),
                "friedman_statistic": float(result["friedman_statistic"]),
                "friedman_p_value": float(result["friedman_p_value"]),
            }
        )
        average_ranks = np.asarray(result["average_ranks"], dtype=float)
        for idx, method in enumerate(method_names):
            posthoc_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "higher_is_better": bool(higher),
                    "average_rank": float(average_ranks[idx]),
                    "z_value": float(np.asarray(result["z_values"])[idx]),
                    "p_value_uncorrected": float(np.asarray(result["p_values_post_hoc_unc"])[idx]),
                    "p_value_holm": float(np.asarray(result["p_values_post_hoc_corr"])[idx]),
                    "reject_uncorrected": bool(np.asarray(result["reject_post_hoc_unc"])[idx]),
                    "reject_holm": bool(np.asarray(result["reject_post_hoc_corr"])[idx]),
                    "is_baseline": method == baseline,
                }
            )
    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


def run_wilcoxon_vs_best_by_metric(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split: str = "test",
    methods: Sequence[str] | None = None,
    metric_directions: dict[str, bool] | None = None,
    alpha: float = 0.05,
    block_cols: Sequence[str] = ("dataset", "seed"),
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        higher = metric_higher_is_better(metric, metric_directions)
        block_table = build_metric_block_table(
            metrics_df,
            metric,
            split=split,
            methods=methods,
            block_cols=block_cols,
        )
        if block_table.shape[1] < 2 or block_table.shape[0] < 2:
            continue
        baseline = _best_method_from_block_table(block_table, higher)
        baseline_values = block_table[baseline].to_numpy(dtype=float)
        p_values = []
        raw_rows = []
        for method in block_table.columns:
            values = block_table[method].to_numpy(dtype=float)
            if method == baseline:
                statistic = 0.0
                p_value = 1.0
                alternative = "baseline"
            else:
                alternative = "less" if higher else "greater"
                try:
                    statistic, p_value = scipy_stats.wilcoxon(
                        values,
                        baseline_values,
                        alternative=alternative,
                        zero_method=zero_method,
                    )
                except ValueError:
                    statistic, p_value = np.nan, 1.0
            p_values.append(float(p_value))
            raw_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "higher_is_better": bool(higher),
                    "n_blocks": int(block_table.shape[0]),
                    "method_mean": float(np.mean(values)),
                    "baseline_mean": float(np.mean(baseline_values)),
                    "mean_difference_vs_baseline": float(np.mean(values - baseline_values)),
                    "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "alternative": alternative,
                    "p_value_uncorrected": float(p_value),
                    "is_baseline": method == baseline,
                }
            )
        reject, corrected = _holm_adjust(p_values, alpha=alpha)
        for idx, row in enumerate(raw_rows):
            row["p_value_holm"] = float(corrected[idx])
            row["reject_holm"] = bool(reject[idx])
            row["reject_uncorrected"] = bool(row["p_value_uncorrected"] < alpha)
            rows.append(row)
    return pd.DataFrame(rows)


def run_statistical_tests(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split: str = "test",
    methods: Sequence[str] | None = None,
    metric_directions: dict[str, bool] | None = None,
    alpha: float = 0.05,
    block_cols: Sequence[str] = ("dataset", "seed"),
) -> dict[str, pd.DataFrame]:
    friedman_omnibus, friedman_posthoc = run_friedman_by_metric(
        metrics_df,
        metrics,
        split=split,
        methods=methods,
        metric_directions=metric_directions,
        alpha=alpha,
        block_cols=block_cols,
    )
    wilcoxon = run_wilcoxon_vs_best_by_metric(
        metrics_df,
        metrics,
        split=split,
        methods=methods,
        metric_directions=metric_directions,
        alpha=alpha,
        block_cols=block_cols,
    )
    return {
        "friedman_omnibus": friedman_omnibus,
        "friedman_posthoc": friedman_posthoc,
        "wilcoxon": wilcoxon,
    }


def p_value_marker(p_value: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p_value):
        return "nan"
    text = "<1e-3" if p_value < 1e-3 else f"{p_value:.3f}"
    return text if p_value < alpha else f"{text}*"


def statistical_test_table(
    test_df: pd.DataFrame,
    *,
    value_col: str = "p_value_holm",
    alpha: float = 0.05,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame(columns=list(methods) if methods is not None else None)
    df = test_df.copy()
    df["p_value_text"] = [p_value_marker(float(value), alpha=alpha) for value in df[value_col]]
    df.loc[df["is_baseline"].astype(bool), "p_value_text"] = df.loc[
        df["is_baseline"].astype(bool), "p_value_text"
    ] + " (baseline)"
    table = df.pivot_table(index="metric", columns="method", values="p_value_text", aggfunc="first")
    if methods is not None:
        table = table.reindex(columns=list(methods))
    return table


def print_statistical_tables(
    tests: dict[str, pd.DataFrame],
    *,
    alpha: float = 0.05,
    tablefmt: str = "grid",
    methods: Sequence[str] | None = None,
) -> None:
    friedman_omnibus = tests.get("friedman_omnibus", pd.DataFrame())
    if not friedman_omnibus.empty:
        print("\nFriedman omnibus tests")
        print(tabulate(friedman_omnibus, headers="keys", tablefmt=tablefmt, showindex=False))
    friedman_table = statistical_test_table(tests.get("friedman_posthoc", pd.DataFrame()), alpha=alpha, methods=methods)
    if not friedman_table.empty:
        print("\nFriedman post-hoc Holm p-values")
        print(tabulate(friedman_table.reset_index(), headers="keys", tablefmt=tablefmt, showindex=False))
    wilcoxon_table = statistical_test_table(tests.get("wilcoxon", pd.DataFrame()), alpha=alpha, methods=methods)
    if not wilcoxon_table.empty:
        print("\nPaired Wilcoxon Holm p-values")
        print(tabulate(wilcoxon_table.reset_index(), headers="keys", tablefmt=tablefmt, showindex=False))


def export_analysis_tables(
    output_dir: str | Path,
    *,
    metric_summary: pd.DataFrame | None = None,
    publication_table: pd.DataFrame | None = None,
    tests: dict[str, pd.DataFrame] | None = None,
    test_methods: Sequence[str] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(Path(output_dir))
    written: dict[str, Path] = {}
    if metric_summary is not None:
        path = output_dir / "metric_summary_ci.csv"
        metric_summary.to_csv(path, index=False)
        written["metric_summary_ci"] = path
    if publication_table is not None:
        path = output_dir / "publication_metric_table.csv"
        publication_table.to_csv(path)
        written["publication_metric_table"] = path
    if tests is not None:
        for name, table in tests.items():
            path = output_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            written[name] = path
        friedman_table = statistical_test_table(tests.get("friedman_posthoc", pd.DataFrame()), methods=test_methods)
        if not friedman_table.empty:
            path = output_dir / "friedman_posthoc_holm_table.csv"
            friedman_table.to_csv(path)
            written["friedman_posthoc_holm_table"] = path
        wilcoxon_table = statistical_test_table(tests.get("wilcoxon", pd.DataFrame()), methods=test_methods)
        if not wilcoxon_table.empty:
            path = output_dir / "wilcoxon_holm_table.csv"
            wilcoxon_table.to_csv(path)
            written["wilcoxon_holm_table"] = path
    return written


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 220) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def set_plot_theme(style: str = "whitegrid", context: str = "notebook", font_scale: float = 1.0) -> None:
    sns.set_theme(style=style, context=context, font_scale=font_scale)
