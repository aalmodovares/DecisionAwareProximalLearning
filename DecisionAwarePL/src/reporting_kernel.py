from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label(row: dict[str, Any]) -> str:
    round_value = row.get("round", "")
    round_part = "" if round_value in ("", None) else f" r{int(float(round_value))}"
    weighted = " weighted" if str(row.get("weighted", "False")).lower() in {"true", "1"} else ""
    return f"{row.get('model', '')}{round_part} {row.get('kernel', '')}{weighted}".strip()


def _base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if np.isclose(float(row.get("bandwidth_multiplier", 1.0)), 1.0)]


def _plot_kernel_summary_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = _base_rows(rows)
    if not rows:
        return
    metrics = [
        ("offdiag_mean", "Offdiag Mean"),
        ("offdiag_std", "Offdiag Std"),
        ("near_zero_rate", "Near-Zero Rate"),
        ("near_one_rate", "Near-One Rate"),
        ("effective_rank_ratio", "Effective Rank Ratio"),
        ("avg_neighbors_0p1", "Avg Neighbors K>0.1"),
    ]
    labels = [_label(row) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = [float(row.get(metric, np.nan)) for row in rows]
        colors = [
            "tab:red" if row.get("status") == "TOO_DIAGONAL" else "tab:orange" if row.get("status") == "TOO_CONSTANT" else "tab:blue"
            for row in rows
        ]
        ax.bar(x, values, color=colors, alpha=0.82)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Kernel Scalar Diagnostics at Current Bandwidth", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_bandwidth_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    metrics = [
        ("near_zero_rate", "Near-Zero Rate"),
        ("near_one_rate", "Near-One Rate"),
        ("effective_rank_ratio", "Effective Rank Ratio"),
        ("avg_neighbors_0p1", "Avg Neighbors K>0.1"),
    ]
    groups = {}
    for row in rows:
        key = (row.get("model", ""), row.get("round", ""), row.get("kernel", ""), row.get("weighted", False))
        groups.setdefault(key, []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for key, group_rows in groups.items():
            group_rows = sorted(group_rows, key=lambda row: float(row.get("bandwidth_multiplier", 1.0)))
            x = [float(row["bandwidth_multiplier"]) for row in group_rows]
            y = [float(row.get(metric, np.nan)) for row in group_rows]
            label = _label(group_rows[0])
            ax.plot(x, y, marker="o", linewidth=1.4, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Bandwidth multiplier")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("Kernel Bandwidth Sweep", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_offdiag_distribution(path: Path, offdiag_rows: list[dict[str, Any]]) -> None:
    if not offdiag_rows:
        return
    groups = {}
    for row in offdiag_rows:
        key = (row.get("model", ""), row.get("round", ""), row.get("kernel", ""), row.get("weighted", False))
        groups.setdefault(key, []).append(float(row["offdiag_value"]))
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0.0, 1.0, 41)
    for idx, (key, values) in enumerate(groups.items()):
        if idx >= 16:
            break
        sample_row = {
            "model": key[0],
            "round": key[1],
            "kernel": key[2],
            "weighted": key[3],
        }
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=_label(sample_row))
    ax.set_xlabel("Off-diagonal kernel value")
    ax.set_ylabel("Density")
    ax.set_title("Off-Diagonal Kernel Value Distributions")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_kernel_spectrum(path: Path, spectrum_rows: list[dict[str, Any]]) -> None:
    if not spectrum_rows:
        return
    groups = {}
    for row in spectrum_rows:
        key = (row.get("model", ""), row.get("round", ""), row.get("kernel", ""), row.get("weighted", False))
        groups.setdefault(key, []).append(row)
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, (key, group_rows) in enumerate(groups.items()):
        if idx >= 16:
            break
        group_rows = sorted(group_rows, key=lambda row: int(row["eigen_index"]))
        x = [int(row["eigen_index"]) + 1 for row in group_rows]
        y = [float(row["eigen_share"]) for row in group_rows]
        sample_row = {
            "model": key[0],
            "round": key[1],
            "kernel": key[2],
            "weighted": key[3],
        }
        ax.plot(x, y, linewidth=1.5, label=_label(sample_row))
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue share")
    ax.set_yscale("log")
    ax.set_title("Kernel Eigenvalue Spectrum")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_status_grid(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = _base_rows(rows)
    if not rows:
        return
    columns = ["model", "round", "kernel", "weighted", "status", "offdiag_q50", "near_zero_rate", "near_one_rate", "effective_rank_ratio"]
    cell_text = []
    for row in rows:
        row_values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3g}"
            row_values.append(str(value))
        cell_text.append(row_values)
    fig_height = max(3.0, 0.38 * len(cell_text) + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title("Kernel Diagnostic Status at Current Bandwidth", pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_kernel_diagnostics(
    *,
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    offdiag_rows: list[dict[str, Any]] | None = None,
    spectrum_rows: list[dict[str, Any]] | None = None,
    save_offdiag_values: bool = False,
    save_figures: bool = True,
) -> list[str]:
    output_dir = Path(output_dir)
    created: list[str] = []
    if not rows:
        return created
    _write_rows(output_dir / "kernel_diagnostics.csv", rows)
    created.append(str(output_dir / "kernel_diagnostics.csv"))
    if spectrum_rows:
        _write_rows(output_dir / "kernel_spectrum.csv", spectrum_rows)
        created.append(str(output_dir / "kernel_spectrum.csv"))
    if save_offdiag_values and offdiag_rows:
        _write_rows(output_dir / "kernel_offdiag_values.csv", offdiag_rows)
        created.append(str(output_dir / "kernel_offdiag_values.csv"))
    if not save_figures:
        return created

    figure_dir = ensure_dir(output_dir / "figures" / "kernel_diagnostics")
    figure_specs = [
        ("kernel_summary_bars.png", lambda path: _plot_kernel_summary_bars(path, rows)),
        ("bandwidth_sweep.png", lambda path: _plot_bandwidth_sweep(path, rows)),
        ("offdiag_distribution.png", lambda path: _plot_offdiag_distribution(path, offdiag_rows or [])),
        ("kernel_spectrum.png", lambda path: _plot_kernel_spectrum(path, spectrum_rows or [])),
        ("kernel_status_grid.png", lambda path: _plot_status_grid(path, rows)),
    ]
    for filename, plotter in figure_specs:
        path = figure_dir / filename
        plotter(path)
        if path.exists():
            created.append(str(path))
    return created
