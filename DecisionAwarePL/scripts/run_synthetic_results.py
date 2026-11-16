from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, all_mean_configs, all_nmmr_configs, all_weighting_configs, config_to_dict
from src.experiment import ALL_MODEL_KEYS, MODEL_DISPLAY_NAMES, VALID_MODEL_KEYS, run_experiment
from src.analysis.sensitivity_utils import apply_mapping as _apply_mapping
from src.analysis.sensitivity_utils import set_treatment_grid_size as _set_treatment_grid_size
from src.analysis.runner_utils import _prepare_config_for_seed, _resolve_seeds
from src.results_export import ResultsExportConfig
from src.utils import ensure_dir, save_json


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "results" / "default_synthetic_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed synthetic evaluations and save article-ready raw result tables."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "synthetic_results")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model keys, or 'all'. Overrides the JSON models field. Valid keys: {','.join(VALID_MODEL_KEYS)}.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of seeds to run in parallel.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument(
        "--model-progress-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one-line start/done messages for each seed/model without enabling training progress bars.",
    )
    parser.add_argument("--surface-patients", type=int, default=50)
    parser.add_argument("--treatment-grid-size", type=int, default=51)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Merge this run into an existing results_metrics.csv instead of replacing it.",
    )
    parser.add_argument(
        "--existing-results-policy",
        choices=["skip", "overwrite"],
        default="skip",
        help=(
            "Only used with --append-results. 'skip' does not rerun complete existing "
            "model/seed results; 'overwrite' reruns requested models and replaces their rows."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Use tiny data and short training schedules for a quick check.")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        help="Write a default results JSON to this path and exit.",
    )
    return parser.parse_args()








def _apply_smoke(config: ExperimentConfig) -> None:
    config.data.n_train = 80
    config.data.n_val = 0
    config.data.n_test = 90
    config.data.support_pilot_size = 1000
    config.data.causal_mc_samples = 64
    config.data.causal_mc_batch_size = 64
    config.data.associational_mc_samples = 500
    config.data.tcga_normalization_pilot_size = 500
    config.data.tcga_causal_mc_samples_per_x = 32
    config.data.tcga_policy_value_mc_samples = 16
    for flow_config in (config.propensity_flow, config.w_flow):
        flow_config.epochs = 4
        flow_config.min_epochs = 1
        flow_config.early_stop_patience = 1
        flow_config.hidden_features = (8,)
    for mean_config in all_mean_configs(config):
        mean_config.epochs = 4
        mean_config.min_epochs = 1
        mean_config.early_stop_patience = 1
        mean_config.hidden_dims = (8,)
        mean_config.batch_size = 64
        mean_config.crossfit_folds = min(int(mean_config.crossfit_folds), 2)
    for nmmr_config in all_nmmr_configs(config):
        nmmr_config.epochs = 4
        nmmr_config.min_epochs = 1
        nmmr_config.early_stop_patience = 1
        nmmr_config.hidden_dims = (8,)
        nmmr_config.batch_size = 64
    for weighting_config in all_weighting_configs(config):
        weighting_config.n_rounds = 1
        weighting_config.early_stop_patience = 1
    config.kpv.max_train_size = 80
    config.kpv.n_lambda1_search = min(int(config.kpv.n_lambda1_search), 2)
    config.kpv.n_lambda2_search = min(int(config.kpv.n_lambda2_search), 2)
    config.dfpv.epochs = 4
    config.dfpv.min_epochs = 1
    config.dfpv.early_stop_patience = 1
    config.dfpv.hidden_dims = (8,)
    config.dfpv.feature_dim = min(int(config.dfpv.feature_dim), 4)
    config.dfpv.max_train_size = 80
    config.dr_kpv.max_train_size = 80
    config.dr_kpv.regularization_grid_points = min(int(config.dr_kpv.regularization_grid_points), 2)
    config.dr_pmmr.max_train_size = 80
    config.dr_pmmr.regularization_grid_points = min(int(config.dr_pmmr.regularization_grid_points), 2)
    config.dr_pmmr_conditional.max_train_size = 80
    config.dr_pmmr_conditional.regularization_grid_points = min(int(config.dr_pmmr_conditional.regularization_grid_points), 2)
    config.s_learner.epochs = 4
    config.s_learner.min_epochs = 1
    config.s_learner.early_stop_patience = 1
    config.s_learner.hidden_dims = (8,)
    config.s_learner.batch_size = 64
    config.decaflow.epochs = 4
    config.decaflow.min_epochs = 1
    config.decaflow.early_stop_patience = 1
    config.decaflow.encoder_hidden_features = (8,)
    config.decaflow.decoder_hidden_features = (8,)
    config.decaflow.conditioner_hidden_features = (4,)
    config.decaflow.batch_size = 64
    config.decaflow.prediction_batch_size = 64
    config.decaflow.interventional_mc_samples = 4


def _write_default_results_config(path: Path) -> None:
    config = ExperimentConfig()
    config.save_figures = False
    config.plot_losses = False
    _set_treatment_grid_size(config, 51)
    payload = {
        "models": "all",
        "config": config_to_dict(config),
        "per_model_configs": {},
    }
    ensure_dir(path.parent)
    save_json(payload, path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results config not found: {path}. Use --write-default-config {path} to create a default file."
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_results_config(path: Path) -> tuple[ExperimentConfig, dict[str, ExperimentConfig], Any]:
    raw = _load_json(path)
    base_config = ExperimentConfig()
    if isinstance(raw.get("config"), dict):
        _apply_mapping(base_config, raw["config"])
    if isinstance(raw.get("fixed"), dict):
        _apply_mapping(base_config, raw["fixed"])

    per_model_configs: dict[str, ExperimentConfig] = {}
    for section_name in ("per_model_configs", "fixed_per_model"):
        section = raw.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for model_key, overrides in section.items():
            if model_key not in VALID_MODEL_KEYS:
                raise ValueError(f"Unknown model key in {section_name}: {model_key!r}")
            local_config = copy.deepcopy(base_config)
            if not isinstance(overrides, dict):
                raise ValueError(f"{section_name}.{model_key} must be an object.")
            _apply_mapping(local_config, overrides)
            per_model_configs[model_key] = local_config

    return base_config, per_model_configs, raw.get("models", "all")


def _resolve_models(cli_models: str | None, config_models: Any) -> list[str]:
    raw = cli_models if cli_models is not None else config_models
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return list(ALL_MODEL_KEYS)
    if isinstance(raw, str):
        selected = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        selected = [str(part).strip() for part in raw if str(part).strip()]
    else:
        raise ValueError("models must be either 'all', a comma-separated string, or a list.")
    unknown = sorted(set(selected) - set(VALID_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown model keys requested: {unknown}. Valid keys: {VALID_MODEL_KEYS}")
    return selected






def _metric_rows_from_result(
    result: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    run_dir: Path,
) -> list[dict[str, Any]]:
    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    rows: list[dict[str, Any]] = []
    for split, metric_rows in (
        ("test", result.get("metrics", [])),
        ("validation", result.get("validation_metrics", [])),
    ):
        for metric_row in metric_rows:
            method = str(metric_row.get("method", ""))
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": int(seed),
                "split": split,
                "model_key": display_to_key.get(method, ""),
                "method": method,
                "run_dir": str(run_dir),
            }
            for key, value in metric_row.items():
                if key != "method":
                    row[key] = value
            rows.append(row)
    return rows


def _read_metric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_metric_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_metric_csv(path)


def _append_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _model_key_from_metric_row(row: dict[str, Any]) -> str:
    model_key = str(row.get("model_key", "") or "").strip()
    if model_key:
        return model_key
    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    method = str(row.get("method", "") or "").strip()
    return display_to_key.get(method, method)


def _metric_row_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    seed_raw = row.get("seed", 0)
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError):
        seed = 0
    return (
        str(row.get("dataset", "") or ""),
        seed,
        str(row.get("split", "") or ""),
        _model_key_from_metric_row(row),
    )


def _completed_models_for_seed(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> set[str]:
    splits_by_model: dict[str, set[str]] = {}
    for row in rows:
        if str(row.get("dataset", "") or "") != str(dataset):
            continue
        try:
            row_seed = int(row.get("seed", 0))
        except (TypeError, ValueError):
            continue
        if row_seed != int(seed):
            continue
        model_key = _model_key_from_metric_row(row)
        if not model_key:
            continue
        split = str(row.get("split", "") or "")
        splits_by_model.setdefault(model_key, set()).add(split)
    return {model_key for model_key, splits in splits_by_model.items() if {"test", "validation"}.issubset(splits)}


def _merge_metric_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str, str]] = []
    for row in [*existing_rows, *new_rows]:
        key = _metric_row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def _backup_existing_results(output_dir: Path, run_id: str, *, mode: str) -> dict[str, Any]:
    backup_dir = ensure_dir(output_dir / "backups")
    copied: list[dict[str, str]] = []
    for filename in ("results_metrics.csv", "results_run_config.json", "failures.csv"):
        src = output_dir / filename
        if not src.exists():
            continue
        dst = backup_dir / f"{src.stem}_{run_id}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "backup": str(dst)})

    manifest = {
        "mode": mode,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "copied": copied,
    }
    manifest_path = backup_dir / f"{mode}_manifest_{run_id}.json"
    save_json(manifest, manifest_path)
    return {"backup_dir": str(backup_dir), "manifest": str(manifest_path), "copied": copied}


def _load_existing_seed_metrics(seed_dir: Path, *, dataset: str, seed: int) -> list[dict[str, Any]]:
    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    rows: list[dict[str, Any]] = []
    for split, filename in (("test", "metrics.csv"), ("validation", "validation_metrics.csv")):
        for metric_row in _read_metric_csv(seed_dir / filename):
            method = str(metric_row.get("method", ""))
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": int(seed),
                "split": split,
                "model_key": display_to_key.get(method, ""),
                "method": method,
                "run_dir": str(seed_dir),
            }
            for key, value in metric_row.items():
                if key != "method":
                    row[key] = value
            rows.append(row)
    return rows


def _seed_outputs_complete(seed_dir: Path) -> bool:
    required = [
        "metrics.csv",
        "validation_metrics.csv",
        "global_curves.csv",
        "patient_curves.csv",
        "patient_metadata.csv",
    ]
    return all((seed_dir / name).exists() for name in required)


def _run_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    seed_dir = Path(job["seed_dir"])
    base_config: ExperimentConfig = job["base_config"]
    model_configs: dict[str, ExperimentConfig] = job["model_configs"]
    selected_models: list[str] = job["selected_models"]
    verbose: int | None = job["verbose"]
    treatment_grid_size = int(job["treatment_grid_size"])
    surface_patients = int(job["surface_patients"])
    resume = bool(job["resume"])
    model_progress_logs = bool(job["model_progress_logs"])

    dataset = str(base_config.data.version)
    if not selected_models:
        return {
            "seed": seed,
            "status": "skipped-existing",
            "metrics": [],
            "run_dir": str(seed_dir),
        }

    if resume and _seed_outputs_complete(seed_dir):
        return {
            "seed": seed,
            "status": "skipped",
            "metrics": _load_existing_seed_metrics(seed_dir, dataset=dataset, seed=seed),
            "run_dir": str(seed_dir),
        }

    run_config = _prepare_config_for_seed(
        base_config,
        seed=seed,
        output_dir=seed_dir,
        verbose=verbose,
        treatment_grid_size=treatment_grid_size,
    )
    run_model_configs = {
        key: _prepare_config_for_seed(
            value,
            seed=seed,
            output_dir=seed_dir,
            verbose=verbose,
            treatment_grid_size=treatment_grid_size,
        )
        for key, value in model_configs.items()
    }
    export_config = ResultsExportConfig(
        enabled=True,
        n_surface_patients=surface_patients,
        treatment_grid_size=treatment_grid_size,
        split="test",
    )
    result = run_experiment(
        run_config,
        selected_models=selected_models,
        model_configs=run_model_configs,
        results_export=export_config,
        model_progress=model_progress_logs,
    )
    return {
        "seed": seed,
        "status": "completed",
        "metrics": _metric_rows_from_result(
            result,
            dataset=run_config.data.version,
            seed=seed,
            run_dir=seed_dir,
        ),
        "run_dir": str(seed_dir),
    }


def _metric_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["dataset", "seed", "split", "model_key", "method", "run_dir"]
    keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    return fieldnames


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = _metric_fieldnames(rows) if rows else ["dataset", "seed", "split", "model_key", "method", "run_dir"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row.get("seed", 0)), str(row.get("split", "")), str(row.get("method", "")))))


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "run_dir", "error_type", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.write_default_config is not None:
        _write_default_results_config(args.write_default_config)
        print(f"Wrote default results config to {args.write_default_config}")
        return

    base_config, model_configs, config_models = _load_results_config(args.config)
    selected_models = _resolve_models(args.models, config_models)
    seeds = _resolve_seeds(args, base_config)
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.surface_patients <= 0:
        raise ValueError("--surface-patients must be positive.")
    if args.treatment_grid_size <= 2:
        raise ValueError("--treatment-grid-size must be at least 3.")

    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "results_metrics.csv"
    run_id = _append_run_id()
    append_id = run_id if args.append_results else None
    backup_info: dict[str, Any] | None = None
    existing_metric_rows = _read_metric_csv_if_exists(metrics_path) if args.append_results else []
    backup_needed = any((output_dir / filename).exists() for filename in ("results_metrics.csv", "results_run_config.json", "failures.csv"))
    if args.append_results:
        backup_info = _backup_existing_results(output_dir, str(append_id), mode="append")
        runs_dir = ensure_dir(output_dir / "runs_append" / f"append_{append_id}")
        failures_path = output_dir / f"failures_append_{append_id}.csv"
        run_config_path = output_dir / f"results_run_config_append_{append_id}.json"
    else:
        if backup_needed:
            backup_info = _backup_existing_results(output_dir, f"replace_{run_id}", mode="replace")
        runs_dir = ensure_dir(output_dir / "runs")
        failures_path = output_dir / "failures.csv"
        run_config_path = output_dir / "results_run_config.json"

    _set_treatment_grid_size(base_config, args.treatment_grid_size)
    for model_config in model_configs.values():
        _set_treatment_grid_size(model_config, args.treatment_grid_size)
    if args.smoke:
        _apply_smoke(base_config)
        for model_config in model_configs.values():
            _apply_smoke(model_config)
        _set_treatment_grid_size(base_config, args.treatment_grid_size)
        for model_config in model_configs.values():
            _set_treatment_grid_size(model_config, args.treatment_grid_size)

    save_json(
        {
            "config_path": str(args.config),
            "selected_models": selected_models,
            "requested_models": selected_models,
            "seeds": seeds,
            "jobs": int(args.jobs),
            "surface_patients": int(args.surface_patients),
            "treatment_grid_size": int(args.treatment_grid_size),
            "model_progress_logs": bool(args.model_progress_logs),
            "append_results": bool(args.append_results),
            "append_id": append_id,
            "existing_results_policy": args.existing_results_policy,
            "backup": backup_info,
            "base_config": config_to_dict(base_config),
            "per_model_configs": {key: config_to_dict(value) for key, value in model_configs.items()},
        },
        run_config_path,
    )

    jobs = []
    dataset = str(base_config.data.version)
    for seed in seeds:
        models_for_seed = list(selected_models)
        if args.append_results and args.existing_results_policy == "skip":
            completed_models = _completed_models_for_seed(existing_metric_rows, dataset=dataset, seed=int(seed))
            models_for_seed = [model_key for model_key in selected_models if model_key not in completed_models]
        jobs.append(
            {
                "seed": seed,
                "seed_dir": runs_dir / f"seed_{int(seed):04d}",
                "base_config": base_config,
                "model_configs": model_configs,
                "selected_models": models_for_seed,
                "verbose": args.verbose,
                "treatment_grid_size": args.treatment_grid_size,
                "surface_patients": args.surface_patients,
                "model_progress_logs": args.model_progress_logs,
                "resume": args.resume,
            }
        )

    all_metric_rows: list[dict[str, Any]] = list(existing_metric_rows) if args.append_results else []
    failure_rows: list[dict[str, Any]] = []
    script_start = time.perf_counter()
    print(
        (
            f"[run-start] seeds={len(seeds)} models={len(selected_models)} jobs={int(args.jobs)} "
            f"append={bool(args.append_results)} policy={args.existing_results_policy} output_dir={output_dir}"
        ),
        flush=True,
    )

    if args.jobs == 1:
        for job in jobs:
            try:
                result = _run_seed_job(job)
                if args.append_results:
                    all_metric_rows = _merge_metric_rows(all_metric_rows, result["metrics"])
                else:
                    all_metric_rows.extend(result["metrics"])
                print(f"[{result['status']}] seed={result['seed']} run_dir={result['run_dir']}")
                _write_metrics(metrics_path, all_metric_rows)
            except Exception as exc:
                failure = {
                    "seed": int(job["seed"]),
                    "run_dir": str(job["seed_dir"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failure_rows.append(failure)
                _write_failures(failures_path, failure_rows)
                if not args.continue_on_error:
                    raise
                print(f"[failed] seed={job['seed']} error={type(exc).__name__}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            futures = {pool.submit(_run_seed_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    if args.append_results:
                        all_metric_rows = _merge_metric_rows(all_metric_rows, result["metrics"])
                    else:
                        all_metric_rows.extend(result["metrics"])
                    print(f"[{result['status']}] seed={result['seed']} run_dir={result['run_dir']}")
                    _write_metrics(metrics_path, all_metric_rows)
                except Exception as exc:
                    failure = {
                        "seed": int(job["seed"]),
                        "run_dir": str(job["seed_dir"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failure_rows.append(failure)
                    _write_failures(failures_path, failure_rows)
                    if not args.continue_on_error:
                        raise
                    print(f"[failed] seed={job['seed']} error={type(exc).__name__}: {exc}")

    _write_metrics(metrics_path, all_metric_rows)
    _write_failures(failures_path, failure_rows)
    print(f"[done] metrics={metrics_path} elapsed={time.perf_counter() - script_start:.2f}s")


if __name__ == "__main__":
    main()
