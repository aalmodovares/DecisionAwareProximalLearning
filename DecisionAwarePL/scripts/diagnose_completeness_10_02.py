from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sensitivity_utils import load_results_config, resolve_seeds
from src.data.simulator import Wu2024Simulator


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "results" / "default_synthetic_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "completeness_10_02_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose latent recoverability under the 10_02 completeness DGP."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-version", default="wu2024_nonadditive")
    parser.add_argument("--structure-type", default="single_peaked_heterogeneous")
    parser.add_argument("--violation-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--z-threshold-max", type=float, default=2.0)
    parser.add_argument("--w-threshold-max", type=float, default=2.0)
    parser.add_argument("--noise-scale-max", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    return parser.parse_args()


def _design_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    intercept = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([intercept, x], axis=1)


def _fit_linear_predictor(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(_design_matrix(train_x), np.asarray(train_y, dtype=float), rcond=None)
    return beta


def _predict_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _design_matrix(x) @ beta


def _columnwise_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    sse = np.sum(np.square(truth - pred), axis=0)
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    sst = np.sum(np.square(centered), axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


def _joint_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum(np.square(truth - pred)))
    centered = truth - np.mean(truth, axis=0, keepdims=True)
    sst = float(np.sum(np.square(centered)))
    return 1.0 - sse / max(sst, 1e-12)


def _proxy_variance(z: np.ndarray, w: np.ndarray) -> float:
    return float(np.mean(np.var(np.concatenate([z, w], axis=1), axis=0)))


def _write_rows_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict[str, float | int | str]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric_rows = [row for row in rows if row["metric"] == metric]
    alphas = sorted({float(row["alpha"]) for row in metric_rows})
    means: list[float] = []
    stds: list[float] = []
    for alpha in alphas:
        values = np.asarray([float(row["value"]) for row in metric_rows if float(row["alpha"]) == alpha], dtype=float)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return np.asarray(alphas, dtype=float), np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _plot_metric_frame(rows: list[dict[str, float | int | str]], *, output_path: Path) -> None:
    metric_order = [
        ("latent_uz_r2", "R^2(U_z | Z, W)"),
        ("latent_uw_r2", "R^2(U_w | Z, W)"),
        ("joint_latent_r2", "Joint latent R^2"),
        ("proxy_variance", "Mean proxy variance"),
    ]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(16, 3.8), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metric_order, strict=True):
        alphas, means, stds = _metric_summary(rows, metric_key)
        ax.plot(alphas, means, marker="o", linewidth=2.0)
        ax.fill_between(alphas, means - stds, means + stds, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("alpha")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _diagnostic_rows_for_run(*, simulator: Wu2024Simulator, seed: int, split: str, alpha: float) -> list[dict[str, float | int | str]]:
    train_sample, test_sample = simulator.sample_train_test(seed)
    eval_sample = train_sample if split == "train" else test_sample

    train_x = np.concatenate([train_sample.treatment_proxy, train_sample.outcome_proxy], axis=1)
    eval_x = np.concatenate([eval_sample.treatment_proxy, eval_sample.outcome_proxy], axis=1)
    train_latents = np.concatenate([train_sample.latent_uz, train_sample.latent_uw], axis=1)
    eval_latents = np.concatenate([eval_sample.latent_uz, eval_sample.latent_uw], axis=1)

    beta = _fit_linear_predictor(train_x, train_latents)
    eval_pred = _predict_linear(beta, eval_x)
    per_latent_r2 = _columnwise_r2(eval_latents, eval_pred)
    joint_r2 = _joint_r2(eval_latents, eval_pred)
    variance = _proxy_variance(eval_sample.treatment_proxy, eval_sample.outcome_proxy)
    return [
        {"seed": int(seed), "split": split, "alpha": float(alpha), "metric": "latent_uz_r2", "value": float(per_latent_r2[0])},
        {"seed": int(seed), "split": split, "alpha": float(alpha), "metric": "latent_uw_r2", "value": float(per_latent_r2[1])},
        {"seed": int(seed), "split": split, "alpha": float(alpha), "metric": "joint_latent_r2", "value": float(joint_r2)},
        {"seed": int(seed), "split": split, "alpha": float(alpha), "metric": "proxy_variance", "value": float(variance)},
    ]


def main() -> None:
    args = parse_args()
    base_config, _per_model_configs, _configured_models = load_results_config(args.config)
    base_config.data.version = str(args.dataset_version)
    base_config.data.structure_type = str(args.structure_type)
    base_config.data.proxy_independence_violation_strength = 0.0
    seeds = resolve_seeds(
        seeds=args.seeds,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        default_seed=int(base_config.seed),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for alpha in [float(value) for value in args.violation_grid]:
        config = copy.deepcopy(base_config)
        config.data.completeness_violation_strength = alpha
        config.data.completeness_violation_mode = "central_band_dropout_noise"
        config.data.completeness_relu_max_threshold_z = float(args.z_threshold_max)
        config.data.completeness_relu_max_threshold_w = float(args.w_threshold_max)
        config.data.completeness_noise_scale_max = float(args.noise_scale_max)
        simulator = Wu2024Simulator(config.data)
        for seed in seeds:
            rows.extend(
                _diagnostic_rows_for_run(
                    simulator=simulator,
                    seed=int(seed),
                    split=str(args.split),
                    alpha=alpha,
                )
            )

    _write_rows_csv(output_dir / "completeness_10_02_latent_diagnostics.csv", rows)
    _plot_metric_frame(rows, output_path=output_dir / "completeness_10_02_latent_diagnostics.png")


if __name__ == "__main__":
    main()
