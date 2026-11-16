from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, all_mean_configs, all_nmmr_configs, config_to_dict
from src.experiment import MODEL_DISPLAY_NAMES, VALID_MODEL_KEYS, run_experiment
from src.analysis.sensitivity_utils import apply_mapping as _apply_mapping
from src.analysis.sensitivity_utils import set_treatment_grid_size as _set_treatment_grid_size
from src.analysis.runner_utils import _prepare_config_for_seed, _resolve_seeds
from src.results_export import ResultsExportConfig
from src.utils import ensure_dir, save_json


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "results" / "default_synthetic_results.json"

WEIGHTING_CONFIG_BY_MODEL = {
    "weighted_pmmr": "weighting_pmmr",
    "weighted_kpv": "weighting_kpv",
    "weighted_dfpv": "weighting_dfpv",
    "weighted_nmmr_u": "weighting_nmmr_u",
    "weighted_nmmr_v": "weighting_nmmr_v",
    "naive_oosr": "weighting_naive_oosr",
    "oracle_naive_oosr": "weighting_oracle_naive_oosr",
}


@dataclass(frozen=True)
class AblationPoint:
    parameter: str
    value: float | int

    @property
    def value_label(self) -> str:
        if self.parameter == "n_rounds":
            return str(int(self.value))
        return f"{float(self.value):.8g}".replace("-", "m").replace(".", "p")

    @property
    def run_name(self) -> str:
        return f"{self.parameter}_{self.value_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-dimensional ablations over OOSR weighting tau, lam, and n_rounds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "weight_ablation")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=(
            "Comma-separated weighted model keys. Valid keys: "
            f"{','.join(sorted(WEIGHTING_CONFIG_BY_MODEL))}."
        ),
    )
    parser.add_argument("--dataset-version", type=str, default=None)
    parser.add_argument("--structure-type", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes.")
    parser.add_argument(
        "--parallelism",
        choices=["seed", "seed_point"],
        default="seed_point",
        help=(
            "'seed' runs one worker per seed and sweeps points serially. "
            "'seed_point' runs one worker per (seed, ablation point)."
        ),
    )
    parser.add_argument("--tau-grid", type=float, nargs="*", default=None)
    parser.add_argument("--lam-grid", type=float, nargs="*", default=None)
    parser.add_argument("--n-rounds-grid", type=int, nargs="*", default=None)
    parser.add_argument("--surface-patients", type=int, default=50)
    parser.add_argument("--treatment-grid-size", type=int, default=51)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument(
        "--model-progress-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print model start/done messages inside each ablation run.",
    )
    parser.add_argument(
        "--aggregate-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also concatenate global/patient curve CSVs into top-level ablation CSVs.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use tiny data and short schedules for a quick check.")
    return parser.parse_args()






def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
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
            if not isinstance(overrides, dict):
                raise ValueError(f"{section_name}.{model_key} must be an object.")
            local_config = copy.deepcopy(base_config)
            _apply_mapping(local_config, overrides)
            per_model_configs[model_key] = local_config
    return base_config, per_model_configs, raw.get("models", [])




def _apply_smoke(config: ExperimentConfig) -> None:
    config.data.n_train = 80
    config.data.n_val = 0
    config.data.n_test = 90
    config.data.support_pilot_size = 1000
    config.data.causal_mc_samples = min(int(config.data.causal_mc_samples), 64)
    config.data.causal_mc_batch_size = min(int(config.data.causal_mc_batch_size), 64)
    config.data.associational_mc_samples = min(int(config.data.associational_mc_samples), 500)
    config.data.tcga_normalization_pilot_size = min(int(config.data.tcga_normalization_pilot_size), 500)
    config.data.tcga_causal_mc_samples_per_x = min(int(config.data.tcga_causal_mc_samples_per_x), 32)
    config.data.tcga_policy_value_mc_samples = min(int(config.data.tcga_policy_value_mc_samples), 16)
    for mean_config in all_mean_configs(config):
        mean_config.epochs = min(int(mean_config.epochs), 4)
        mean_config.min_epochs = min(int(mean_config.min_epochs), 1)
        mean_config.early_stop_patience = min(int(mean_config.early_stop_patience), 1)
        mean_config.hidden_dims = (8,)
        mean_config.batch_size = min(int(mean_config.batch_size), 64)
        mean_config.crossfit_folds = min(int(mean_config.crossfit_folds), 2)
    for nmmr_config in all_nmmr_configs(config):
        nmmr_config.epochs = min(int(nmmr_config.epochs), 4)
        nmmr_config.min_epochs = min(int(nmmr_config.min_epochs), 1)
        nmmr_config.early_stop_patience = min(int(nmmr_config.early_stop_patience), 1)
        nmmr_config.hidden_dims = (8,)
        nmmr_config.batch_size = 64
    config.kpv.max_train_size = 80 if config.kpv.max_train_size is None else min(int(config.kpv.max_train_size), 80)
    config.dfpv.epochs = min(int(config.dfpv.epochs), 4)
    config.dfpv.min_epochs = min(int(config.dfpv.min_epochs), 1)
    config.dfpv.early_stop_patience = min(int(config.dfpv.early_stop_patience), 1)
    config.dfpv.hidden_dims = (8,)
    config.dfpv.feature_dim = min(int(config.dfpv.feature_dim), 4)
    config.dfpv.max_train_size = 80 if config.dfpv.max_train_size is None else min(int(config.dfpv.max_train_size), 80)
    config.s_learner.epochs = min(int(config.s_learner.epochs), 4)
    config.s_learner.min_epochs = min(int(config.s_learner.min_epochs), 1)
    config.s_learner.early_stop_patience = min(int(config.s_learner.early_stop_patience), 1)
    config.s_learner.hidden_dims = (8,)
    config.s_learner.batch_size = 64


def _resolve_models(raw_models: str) -> list[str]:
    selected = [part.strip() for part in raw_models.split(",") if part.strip()]
    if not selected:
        raise ValueError("--models must include at least one weighted model.")
    unknown = sorted(set(selected) - set(VALID_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown model keys requested: {unknown}. Valid keys: {VALID_MODEL_KEYS}")
    not_weighted = sorted(set(selected) - set(WEIGHTING_CONFIG_BY_MODEL))
    if not_weighted:
        raise ValueError(
            "run_ablation_weights.py only accepts weighted/OOSR models. "
            f"Unsupported keys: {not_weighted}. Valid keys: {sorted(WEIGHTING_CONFIG_BY_MODEL)}"
        )
    return selected




def _resolve_ablation_points(args: argparse.Namespace) -> list[AblationPoint]:
    points: list[AblationPoint] = []
    for value in args.tau_grid or []:
        points.append(AblationPoint("tau", float(value)))
    for value in args.lam_grid or []:
        points.append(AblationPoint("lam", float(value)))
    for value in args.n_rounds_grid or []:
        if int(value) <= 0:
            raise ValueError("--n-rounds-grid values must be positive.")
        points.append(AblationPoint("n_rounds", int(value)))
    if not points:
        raise ValueError("Provide at least one of --tau-grid, --lam-grid, or --n-rounds-grid.")
    return points




def _weighting_config(config: ExperimentConfig, model_key: str):
    return getattr(config, WEIGHTING_CONFIG_BY_MODEL[model_key])


def _apply_ablation_to_config(config: ExperimentConfig, selected_models: list[str], point: AblationPoint) -> None:
    for model_key in selected_models:
        weighting = _weighting_config(config, model_key)
        if point.parameter == "n_rounds":
            weighting.n_rounds = int(point.value)
        elif point.parameter == "tau":
            weighting.tau = float(point.value)
        elif point.parameter == "lam":
            weighting.lam = float(point.value)
        else:
            raise ValueError(f"Unknown ablation parameter: {point.parameter!r}")


def _model_ablation_values(config: ExperimentConfig, model_key: str) -> dict[str, Any]:
    weighting = _weighting_config(config, model_key)
    return {
        "weighting_section": WEIGHTING_CONFIG_BY_MODEL[model_key],
        "weighting_tau": float(weighting.tau),
        "weighting_lam": float(weighting.lam),
        "weighting_n_rounds": int(weighting.n_rounds),
    }


def _metric_rows_from_result(
    result: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    run_dir: Path,
    point: AblationPoint,
    model_configs: dict[str, ExperimentConfig],
) -> list[dict[str, Any]]:
    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    rows: list[dict[str, Any]] = []
    for split, metric_rows in (
        ("test", result.get("metrics", [])),
        ("validation", result.get("validation_metrics", [])),
    ):
        for metric_row in metric_rows:
            method = str(metric_row.get("method", ""))
            model_key = display_to_key.get(method, "")
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": int(seed),
                "split": split,
                "model_key": model_key,
                "method": method,
                "run_dir": str(run_dir),
                "ablation_parameter": point.parameter,
                "ablation_value": point.value,
                "ablation_run": point.run_name,
            }
            if model_key in model_configs:
                row.update(_model_ablation_values(model_configs[model_key], model_key))
            for key, value in metric_row.items():
                if key != "method":
                    row[key] = value
            rows.append(row)
    return rows


def _ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    ensure_dir(path.parent)
    fieldnames = _ordered_fieldnames(rows, preferred) if rows else preferred
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _annotate_csv_file(path: Path, annotations: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv_if_exists(path)
    if not rows:
        return []
    out = []
    for row in rows:
        annotated = dict(row)
        annotated.update(annotations)
        out.append(annotated)
    _write_csv(path, out, list(annotations.keys()))
    return out


def _backup_existing_output(output_dir: Path) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = ensure_dir(output_dir / "backups")
    copied: list[dict[str, str]] = []
    for filename in (
        "ablation_metrics.csv",
        "ablation_global_curves.csv",
        "ablation_patient_curves.csv",
        "ablation_patient_metadata.csv",
        "ablation_run_config.json",
        "failures.csv",
    ):
        src = output_dir / filename
        if not src.exists():
            continue
        dst = backup_dir / f"{src.stem}_{run_id}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "backup": str(dst)})
    manifest = {
        "mode": "ablation",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "copied": copied,
    }
    manifest_path = backup_dir / f"ablation_manifest_{run_id}.json"
    save_json(manifest, manifest_path)
    return {"backup_dir": str(backup_dir), "manifest": str(manifest_path), "copied": copied}


def _run_seed_point_job(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    seed_dir = Path(job["seed_dir"])
    base_config: ExperimentConfig = job["base_config"]
    base_model_configs: dict[str, ExperimentConfig] = job["model_configs"]
    selected_models: list[str] = job["selected_models"]
    point: AblationPoint = job["point"]
    point_index = int(job.get("point_index", 1))
    n_points = int(job.get("n_points", 1))
    verbose: int | None = job["verbose"]
    treatment_grid_size = int(job["treatment_grid_size"])
    surface_patients = int(job["surface_patients"])
    model_progress_logs = bool(job["model_progress_logs"])
    aggregate_curves = bool(job["aggregate_curves"])

    run_dir = seed_dir / point.run_name
    run_config = _prepare_config_for_seed(
        base_config,
        seed=seed,
        output_dir=run_dir,
        verbose=verbose,
        treatment_grid_size=treatment_grid_size,
    )
    run_model_configs: dict[str, ExperimentConfig] = {}
    for model_key in selected_models:
        source_config = base_model_configs.get(model_key, base_config)
        model_config = _prepare_config_for_seed(
            source_config,
            seed=seed,
            output_dir=run_dir,
            verbose=verbose,
            treatment_grid_size=treatment_grid_size,
        )
        _apply_ablation_to_config(model_config, [model_key], point)
        run_model_configs[model_key] = model_config
    _apply_ablation_to_config(run_config, selected_models, point)

    print(
        (
            f"[ablation-start] seed={seed} point={point_index}/{n_points} "
            f"{point.parameter}={point.value} models={','.join(selected_models)}"
        ),
        flush=True,
    )
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
    metrics = _metric_rows_from_result(
        result,
        dataset=run_config.data.version,
        seed=seed,
        run_dir=run_dir,
        point=point,
        model_configs=run_model_configs,
    )

    annotations = {
        "ablation_parameter": point.parameter,
        "ablation_value": point.value,
        "ablation_run": point.run_name,
    }
    if aggregate_curves:
        global_curves = _annotate_csv_file(run_dir / "global_curves.csv", annotations)
        patient_curves = _annotate_csv_file(run_dir / "patient_curves.csv", annotations)
        patient_metadata = _annotate_csv_file(run_dir / "patient_metadata.csv", annotations)
    else:
        _annotate_csv_file(run_dir / "global_curves.csv", annotations)
        _annotate_csv_file(run_dir / "patient_curves.csv", annotations)
        _annotate_csv_file(run_dir / "patient_metadata.csv", annotations)
        global_curves = []
        patient_curves = []
        patient_metadata = []

    print(
        f"[ablation-done] seed={seed} {point.parameter}={point.value} run_dir={run_dir}",
        flush=True,
    )

    return {
        "seed": seed,
        "point": point.run_name,
        "ablation_parameter": point.parameter,
        "ablation_value": point.value,
        "status": "completed",
        "metrics": metrics,
        "global_curves": global_curves,
        "patient_curves": patient_curves,
        "patient_metadata": patient_metadata,
        "run_dir": str(run_dir),
    }


def _run_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    seed_dir = Path(job["seed_dir"])
    points: list[AblationPoint] = job["points"]
    metrics: list[dict[str, Any]] = []
    global_curves: list[dict[str, Any]] = []
    patient_curves: list[dict[str, Any]] = []
    patient_metadata: list[dict[str, Any]] = []
    for point_index, point in enumerate(points, start=1):
        point_job = dict(job)
        point_job["point"] = point
        point_job["point_index"] = point_index
        point_job["n_points"] = len(points)
        result = _run_seed_point_job(point_job)
        metrics.extend(result.get("metrics", []))
        global_curves.extend(result.get("global_curves", []))
        patient_curves.extend(result.get("patient_curves", []))
        patient_metadata.extend(result.get("patient_metadata", []))
    return {
        "seed": seed,
        "status": "completed",
        "metrics": metrics,
        "global_curves": global_curves,
        "patient_curves": patient_curves,
        "patient_metadata": patient_metadata,
        "run_dir": str(seed_dir),
    }


def _sort_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        try:
            seed = int(row.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0
        try:
            value = float(row.get("ablation_value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return (
            seed,
            str(row.get("ablation_parameter", "")),
            value,
            str(row.get("model_key", "")),
            str(row.get("split", "")),
        )

    return sorted(rows, key=key)


def _sort_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def numeric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
        try:
            return float(row.get(name, default))
        except (TypeError, ValueError):
            return default

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(numeric(row, "seed")),
            str(row.get("ablation_parameter", "")),
            numeric(row, "ablation_value"),
            str(row.get("method", "")),
            str(row.get("predictor_type", "")),
            str(row.get("curve_type", "")),
            int(numeric(row, "patient_rank", -1)),
            numeric(row, "treatment"),
        )

    return sorted(rows, key=key)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.surface_patients <= 0:
        raise ValueError("--surface-patients must be positive.")
    if args.treatment_grid_size <= 1:
        raise ValueError("--treatment-grid-size must be greater than 1.")

    base_config, per_model_configs, _ = _load_results_config(args.config)
    selected_models = _resolve_models(args.models)
    if args.dataset_version is not None:
        base_config.data.version = str(args.dataset_version)
        for cfg in per_model_configs.values():
            cfg.data.version = str(args.dataset_version)
    if args.structure_type is not None:
        base_config.data.structure_type = str(args.structure_type)
        for cfg in per_model_configs.values():
            cfg.data.structure_type = str(args.structure_type)
    if args.verbose is not None:
        base_config.verbose = int(args.verbose)
        for cfg in per_model_configs.values():
            cfg.verbose = int(args.verbose)
    _set_treatment_grid_size(base_config, int(args.treatment_grid_size))
    for cfg in per_model_configs.values():
        _set_treatment_grid_size(cfg, int(args.treatment_grid_size))
    if args.smoke:
        _apply_smoke(base_config)
        for cfg in per_model_configs.values():
            _apply_smoke(cfg)

    seeds = _resolve_seeds(args, base_config)
    points = _resolve_ablation_points(args)

    output_dir = ensure_dir(args.output_dir)
    backup = _backup_existing_output(output_dir)
    runs_dir = ensure_dir(output_dir / "runs")
    metrics_path = output_dir / "ablation_metrics.csv"
    failures_path = output_dir / "failures.csv"

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "models": selected_models,
        "dataset_version": base_config.data.version,
        "structure_type": base_config.data.structure_type,
        "seeds": seeds,
        "jobs": int(args.jobs),
        "parallelism": str(args.parallelism),
        "n_seed_jobs": len(seeds),
        "n_seed_point_jobs": len(seeds) * len(points),
        "surface_patients": int(args.surface_patients),
        "treatment_grid_size": int(args.treatment_grid_size),
        "ablation_points": [{"parameter": p.parameter, "value": p.value, "run": p.run_name} for p in points],
        "aggregate_curves": bool(args.aggregate_curves),
        "backup": backup,
        "base_config": config_to_dict(base_config),
        "per_model_configs": {key: config_to_dict(value) for key, value in per_model_configs.items()},
    }
    save_json(run_config, output_dir / "ablation_run_config.json")

    if args.parallelism == "seed":
        jobs = [
            {
                "seed": seed,
                "seed_dir": runs_dir / f"seed_{seed:03d}",
                "base_config": base_config,
                "model_configs": per_model_configs,
                "selected_models": selected_models,
                "points": points,
                "verbose": args.verbose,
                "treatment_grid_size": args.treatment_grid_size,
                "surface_patients": args.surface_patients,
                "model_progress_logs": args.model_progress_logs,
                "aggregate_curves": args.aggregate_curves,
            }
            for seed in seeds
        ]
        job_runner = _run_seed_job
    else:
        jobs = [
            {
                "seed": seed,
                "seed_dir": runs_dir / f"seed_{seed:03d}",
                "base_config": base_config,
                "model_configs": per_model_configs,
                "selected_models": selected_models,
                "point": point,
                "point_index": point_index,
                "n_points": len(points),
                "verbose": args.verbose,
                "treatment_grid_size": args.treatment_grid_size,
                "surface_patients": args.surface_patients,
                "model_progress_logs": args.model_progress_logs,
                "aggregate_curves": args.aggregate_curves,
            }
            for seed in seeds
            for point_index, point in enumerate(points, start=1)
        ]
        job_runner = _run_seed_point_job

    all_metrics: list[dict[str, Any]] = []
    all_global_curves: list[dict[str, Any]] = []
    all_patient_curves: list[dict[str, Any]] = []
    all_patient_metadata: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        (
            f"[ablation-run-start] seeds={len(seeds)} points={len(points)} "
            f"models={len(selected_models)} jobs={int(args.jobs)} parallelism={args.parallelism} "
            f"work_items={len(jobs)} output_dir={output_dir}"
        ),
        flush=True,
    )

    def consume_result(result: dict[str, Any]) -> None:
        all_metrics.extend(result.get("metrics", []))
        all_global_curves.extend(result.get("global_curves", []))
        all_patient_curves.extend(result.get("patient_curves", []))
        all_patient_metadata.extend(result.get("patient_metadata", []))
        _write_csv(
            metrics_path,
            _sort_metric_rows(all_metrics),
            [
                "dataset",
                "seed",
                "split",
                "model_key",
                "method",
                "ablation_parameter",
                "ablation_value",
                "ablation_run",
                "weighting_section",
                "weighting_tau",
                "weighting_lam",
                "weighting_n_rounds",
                "local_rmse",
                "regret",
                "run_dir",
            ],
        )
        if args.aggregate_curves:
            _write_csv(output_dir / "ablation_global_curves.csv", _sort_curve_rows(all_global_curves), [])
            _write_csv(output_dir / "ablation_patient_curves.csv", _sort_curve_rows(all_patient_curves), [])
            _write_csv(output_dir / "ablation_patient_metadata.csv", _sort_curve_rows(all_patient_metadata), [])

    if args.jobs == 1:
        for job in jobs:
            try:
                result = job_runner(job)
                consume_result(result)
                print(f"[{result['status']}] seed={result['seed']} run_dir={result['run_dir']}", flush=True)
            except Exception as exc:
                failure = {
                    "seed": int(job["seed"]),
                    "run_dir": str(job["seed_dir"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                _write_csv(failures_path, failures, ["seed", "run_dir", "error_type", "error"])
                if not args.continue_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            futures = {pool.submit(job_runner, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    consume_result(result)
                    print(f"[{result['status']}] seed={result['seed']} run_dir={result['run_dir']}", flush=True)
                except Exception as exc:
                    failure = {
                        "seed": int(job["seed"]),
                        "run_dir": str(job["seed_dir"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    _write_csv(failures_path, failures, ["seed", "run_dir", "error_type", "error"])
                    if not args.continue_on_error:
                        raise

    _write_csv(
        metrics_path,
        _sort_metric_rows(all_metrics),
        [
            "dataset",
            "seed",
            "split",
            "model_key",
            "method",
            "ablation_parameter",
            "ablation_value",
            "ablation_run",
            "weighting_section",
            "weighting_tau",
            "weighting_lam",
            "weighting_n_rounds",
            "local_rmse",
            "regret",
            "run_dir",
        ],
    )
    if failures:
        _write_csv(failures_path, failures, ["seed", "run_dir", "error_type", "error"])
    print(f"[done] metrics={metrics_path} elapsed={time.perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
