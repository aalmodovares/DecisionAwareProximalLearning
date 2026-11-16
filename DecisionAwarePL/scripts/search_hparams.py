from __future__ import annotations

import argparse
import csv
import concurrent.futures
import contextlib
import sys
import time
from pathlib import Path

import matplotlib
import optuna
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_to_dict
from src.experiment import MODEL_DISPLAY_NAMES, run_experiment
from src.hparam_search import (
    apply_overrides,
    create_base_config,
    ensure_search_output_dir,
    extract_objective_value,
    get_objective_settings,
    load_search_config,
    requested_models_from_args,
    resolve_trial_config,
    save_bundle,
    save_trial_rows,
    trial_metrics_row,
    trial_row,
)
from src.utils import save_json


PAIRWISE_METRIC_KEYS = ("regret", "policy_mse", "local_mse", "factual_mse", "global_mse")
FAST_FACTUAL_OBJECTIVES = {"validation_factual_mean_rmse", "validation_factual_rmse", "factual_mean_rmse", "factual_rmse"}
FAST_FACTUAL_AUTO_MODELS = {"dr_kpv", "dr_pmmr", "dr_pmmr_conditional"}


def _save_global_metrics_csv(group_dir: Path, models: list[str]) -> Path | None:
    rows: list[dict[str, object]] = []
    for model_key in models:
        model_metrics_path = group_dir / model_key / "metrics.csv"
        if not model_metrics_path.exists():
            continue
        with model_metrics_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
    if not rows:
        return None
    output_path = group_dir / "metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        for stream in self._streams:
            isatty_fn = getattr(stream, "isatty", None)
            if callable(isatty_fn):
                try:
                    if bool(isatty_fn()):
                        return True
                except Exception:
                    continue
        return False

    def __getattr__(self, name: str):
        for stream in self._streams:
            if hasattr(stream, name):
                return getattr(stream, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def _safe_set_user_attr(
    trial: optuna.Trial,
    key: str,
    value,
    *,
    max_retries: int = 8,
    initial_sleep: float = 0.25,
) -> None:
    """Best-effort user-attr write with retries for transient SQLite lock errors."""
    sleep_s = float(initial_sleep)
    for attempt in range(max_retries + 1):
        try:
            trial.set_user_attr(key, value)
            return
        except optuna.exceptions.StorageInternalError as exc:
            message = str(exc).lower()
            locked = "database is locked" in message or "operationalerror" in message
            if (not locked) or attempt >= max_retries:
                raise
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2.0, 4.0)


def _with_sqlite_timeout(storage_url: str, timeout_sec: int = 120) -> str:
    if not storage_url.startswith("sqlite:///"):
        return storage_url
    if "?" in storage_url:
        if "timeout=" in storage_url:
            return storage_url
        return f"{storage_url}&timeout={int(timeout_sec)}"
    return f"{storage_url}?timeout={int(timeout_sec)}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _objective_from_validation_row(row: dict[str, str], metric_name: str) -> float:
    if metric_name == "validation_policy_value":
        return float(row["policy_value"])
    if metric_name == "validation_regret":
        return float(row["regret"])
    if metric_name == "validation_policy_mse":
        return float(row["policy_mse"])
    if metric_name == "validation_local_rmse":
        return float(row["local_rmse"])
    if metric_name == "validation_global_rmse":
        return float(row["global_rmse"])
    if metric_name == "validation_observed_rmse":
        return float(row["observed_rmse"])
    if metric_name == "validation_observational_loss":
        return float(row["observational_loss"])
    if metric_name in FAST_FACTUAL_OBJECTIVES:
        return float(row["factual_mean_rmse"])
    raise KeyError(f"Unsupported validation metric {metric_name!r}.")


def _load_cached_trial_result(trial_dir: Path, model_key: str, metric_name: str) -> tuple[float, list[dict[str, str]], list[dict[str, str]]] | None:
    validation_rows = _read_csv_rows(trial_dir / "validation_metrics.csv")
    test_rows = _read_csv_rows(trial_dir / "metrics.csv")
    if not validation_rows:
        return None
    display_name = MODEL_DISPLAY_NAMES[model_key]
    validation_row = next((row for row in validation_rows if row.get("method") == display_name), None)
    if validation_row is None:
        return None
    value = _objective_from_validation_row(validation_row, metric_name)
    return value, test_rows, validation_rows


def _to_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _extract_eval_metrics_from_validation_row(row: dict[str, object]) -> dict[str, float]:
    regret = _to_float(row.get("regret_causal_marginal", row.get("regret")))
    policy_mse = _to_float(row.get("policy_mse_causal_marginal", row.get("policy_mse")))
    local_rmse = _to_float(row.get("local_rmse"))
    factual_rmse = _to_float(row.get("factual_mean_rmse", row.get("factual_rmse")))
    global_rmse = _to_float(row.get("global_rmse"))
    return {
        "regret": regret,
        "policy_mse": policy_mse,
        "local_mse": local_rmse * local_rmse if np.isfinite(local_rmse) else float("nan"),
        "factual_mse": factual_rmse * factual_rmse if np.isfinite(factual_rmse) else float("nan"),
        "global_mse": global_rmse * global_rmse if np.isfinite(global_rmse) else float("nan"),
    }


def _collect_trial_eval_rows(study: optuna.study.Study, model_key: str) -> list[dict[str, object]]:
    display_name = MODEL_DISPLAY_NAMES[model_key]
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        validation_rows = list(trial.user_attrs.get("validation_metrics", []))
        validation_row = next((row for row in validation_rows if row.get("method") == display_name), None)
        if validation_row is None:
            continue
        eval_metrics = _extract_eval_metrics_from_validation_row(validation_row)
        rows.append(
            {
                "trial_number": int(trial.number),
                "state": str(trial.state),
                "overrides": dict(trial.user_attrs.get("overrides", {})),
                **eval_metrics,
            }
        )
    return rows


def _best_trials_by_metric(trial_eval_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for metric_key in PAIRWISE_METRIC_KEYS:
        candidates = [row for row in trial_eval_rows if np.isfinite(_to_float(row.get(metric_key)))]
        if not candidates:
            out[metric_key] = {
                "direction": "minimize",
                "best_value": None,
                "trial_number": None,
                "hyperparameters": {},
            }
            continue
        best = min(candidates, key=lambda row: float(row[metric_key]))
        out[metric_key] = {
            "direction": "minimize",
            "best_value": float(best[metric_key]),
            "trial_number": int(best["trial_number"]),
            "hyperparameters": dict(best.get("overrides", {})),
        }
    return out


def _save_pairwise_metrics_figure(model_dir: Path, trial_eval_rows: list[dict[str, object]]) -> Path | None:
    if not trial_eval_rows:
        return None
    matrices = {key: np.asarray([_to_float(row.get(key)) for row in trial_eval_rows], dtype=float) for key in PAIRWISE_METRIC_KEYS}
    if all(np.all(~np.isfinite(values)) for values in matrices.values()):
        return None

    n = len(PAIRWISE_METRIC_KEYS)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), squeeze=False)
    for i, y_key in enumerate(PAIRWISE_METRIC_KEYS):
        y = matrices[y_key]
        for j, x_key in enumerate(PAIRWISE_METRIC_KEYS):
            ax = axes[i, j]
            x = matrices[x_key]
            mask = np.isfinite(x) & np.isfinite(y)
            if i == j:
                if np.any(mask):
                    ax.hist(x[mask], bins=20, alpha=0.8, color="tab:blue")
                ax.set_ylabel("count")
            else:
                if np.any(mask):
                    ax.scatter(x[mask], y[mask], s=10, alpha=0.6, color="tab:blue")
            if i == n - 1:
                ax.set_xlabel(x_key)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(y_key)
            else:
                ax.set_yticklabels([])
    fig.suptitle("Pairwise metric distributions (validation)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = model_dir / "pairwise_metrics_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _run_model_study(job: dict) -> dict:
    search_config = load_search_config(Path(job["search_config"]))
    model_key = str(job["model_key"])
    model_dir = Path(job["model_dir"])
    metric_name, direction = get_objective_settings(search_config)
    # Keep fast factual evaluation only for DR models in factual-objective HPO.
    fast_factual_eval = (metric_name in FAST_FACTUAL_OBJECTIVES) and (model_key in FAST_FACTUAL_AUTO_MODELS)
    base_config = create_base_config(
        search_config,
        output_dir=model_dir,
        seed=int(job["seed"]),
        verbose=int(job["verbose"]),
        plot_losses=False,
    )
    base_config.fast_factual_eval = fast_factual_eval
    base_config.save_outputs = bool(job["save_trial_artifacts"])
    base_config.save_figures = bool(job["save_trial_artifacts"]) and not fast_factual_eval
    base_config.plot_losses = bool(job["trial_plot_losses"])
    study_name = f"{job['study_group']}__{model_key}"
    sampler = optuna.samplers.TPESampler(seed=int(job["seed"]))
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=str(job["optuna_storage"]),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_config, overrides = resolve_trial_config(base_config, search_config, model_key, trial)
        trial_dir = Path(job["model_dir"]) / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        if bool(job.get("resume_trials", True)):
            cached = _load_cached_trial_result(trial_dir, model_key, metric_name)
            if cached is not None:
                cached_value, cached_test_rows, cached_validation_rows = cached
                _safe_set_user_attr(trial, "overrides", overrides)
                _safe_set_user_attr(trial, "test_metrics", cached_test_rows)
                _safe_set_user_attr(trial, "validation_metrics", cached_validation_rows)
                _safe_set_user_attr(trial, "trial_output_dir", str(trial_dir))
                _safe_set_user_attr(trial, "cached_trial_reused", True)
                return float(cached_value)
        trial_config.output_dir = trial_dir
        trial_config.fast_factual_eval = fast_factual_eval
        trial_config.save_outputs = bool(job["save_trial_artifacts"])
        trial_config.save_figures = bool(job["save_trial_artifacts"]) and not fast_factual_eval
        trial_config.plot_losses = bool(job["trial_plot_losses"])
        log_path = trial_dir / "trial.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            tee = _Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                result = run_experiment(
                    trial_config,
                    selected_models=[model_key],
                    model_configs={model_key: trial_config},
                )
        value = extract_objective_value(result, model_key, metric_name)
        _safe_set_user_attr(trial, "overrides", overrides)
        _safe_set_user_attr(trial, "test_metrics", result["metrics"])
        _safe_set_user_attr(trial, "validation_metrics", result["validation_metrics"])
        _safe_set_user_attr(trial, "trial_output_dir", str(trial_dir))
        return value

    try:
        study.optimize(
            objective,
            n_trials=int(job["max_trials"]),
            n_jobs=max(1, int(job.get("trial_jobs", 1))),
        )
    except ValueError as exc:
        if "CategoricalDistribution does not support dynamic value space" in str(exc):
            raise ValueError(
                "Optuna rejected this run because a categorical search space changed for an existing study. "
                f"study_group={job['study_group']!r}, model={model_key!r}. "
                "Use a new study.name or --study-group, or remove the old optuna_studies.db if you do not "
                "need the previous trials."
            ) from exc
        raise

    trial_rows = [
        trial_row(trial, model_key, metric_name, dict(trial.user_attrs.get("overrides", {})))
        for trial in study.trials
    ]
    metrics_rows = [
        trial_metrics_row(trial, model_key, metric_name, dict(trial.user_attrs.get("overrides", {})))
        for trial in study.trials
    ]
    trial_eval_rows = _collect_trial_eval_rows(study, model_key)
    best_by_metric = _best_trials_by_metric(trial_eval_rows)
    pairwise_path = _save_pairwise_metrics_figure(model_dir, trial_eval_rows)
    best_overrides = dict(study.best_trial.user_attrs.get("overrides", {}))
    best_config = create_base_config(
        search_config,
        output_dir=model_dir,
        seed=int(job["seed"]),
        verbose=int(job["verbose"]),
        plot_losses=False,
    )
    fixed_model = search_config.get("fixed_per_model", {}).get(model_key, {})
    apply_overrides(best_config, fixed_model)
    apply_overrides(best_config, best_overrides)
    best_config.save_outputs = True
    best_config.save_figures = True

    save_trial_rows(trial_rows, model_dir / "trials.csv")
    save_trial_rows(metrics_rows, model_dir / "metrics.csv")
    save_json(
        {
            "study_name": study_name,
            "model": model_key,
            "direction": direction,
            "objective_metric": metric_name,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "best_trials_by_metric": best_by_metric,
            "pairwise_metrics_figure": str(pairwise_path) if pairwise_path is not None else None,
            "pairwise_metrics_used": list(PAIRWISE_METRIC_KEYS),
        },
        model_dir / "study_summary.json",
    )
    save_json(best_overrides, model_dir / "best_params.json")
    save_json(config_to_dict(best_config), model_dir / "best_config.json")

    return {
        "model": model_key,
        "study_name": study_name,
        "direction": direction,
        "objective_metric": metric_name,
        "best_value": study.best_value,
        "best_params": best_overrides,
        "best_config": config_to_dict(best_config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for weighted proximal OOSR models.")
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "search_results")
    parser.add_argument("--study-group", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--trial-jobs",
        type=int,
        default=None,
        help="Parallel Optuna trials per model. If omitted and only one model is searched, --jobs is reused.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optuna storage URL. Default: sqlite file in the study-group directory.",
    )
    parser.add_argument(
        "--resume-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse finished trial folders and avoid re-running completed trials when possible.",
    )
    parser.add_argument("--save-trial-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trial-plot-losses", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_config = load_search_config(args.search_config)
    models = requested_models_from_args(args.models)
    trial_jobs = args.trial_jobs if args.trial_jobs is not None else (args.jobs if len(models) == 1 else 1)
    default_group = search_config.get("study", {}).get("name", args.search_config.stem)
    study_group = args.study_group or default_group
    max_trials = args.max_trials or int(search_config.get("study", {}).get("max_trials", 50))
    group_dir = ensure_search_output_dir(args.output_dir, study_group)
    storage_url = args.optuna_storage
    if storage_url is None:
        storage_path = (group_dir / "optuna_studies.db").resolve().as_posix()
        storage_url = f"sqlite:///{storage_path}"
    storage_url = _with_sqlite_timeout(str(storage_url), timeout_sec=120)

    jobs = []
    for model_key in models:
        model_dir = group_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "search_config": str(args.search_config),
                "model_key": model_key,
                "study_group": study_group,
                "model_dir": str(model_dir),
                "max_trials": max_trials,
                "seed": args.seed,
                "verbose": args.verbose,
                "trial_jobs": trial_jobs,
                "optuna_storage": storage_url,
                "resume_trials": bool(args.resume_trials),
                "save_trial_artifacts": args.save_trial_artifacts,
                "trial_plot_losses": args.trial_plot_losses,
            }
        )

    results: dict[str, dict] = {}
    if args.jobs == 1 or len(jobs) == 1:
        for job in jobs:
            result = _run_model_study(job)
            results[result["model"]] = result
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as executor:
            futures = [executor.submit(_run_model_study, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results[result["model"]] = result

    base_config = create_base_config(
        search_config,
        output_dir=group_dir / "best_model_eval",
        seed=args.seed,
        verbose=args.verbose,
        plot_losses=False,
    )
    bundle_path = save_bundle(group_dir, base_config, results)
    global_metrics_path = _save_global_metrics_csv(group_dir, models)
    save_json(
        {
            "study_group": study_group,
            "models": models,
            "max_trials": max_trials,
            "bundle_path": str(bundle_path),
            "global_metrics_csv": str(global_metrics_path) if global_metrics_path is not None else None,
        },
        group_dir / "bundle_summary.json",
    )


if __name__ == "__main__":
    main()
