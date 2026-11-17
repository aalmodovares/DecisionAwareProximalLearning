from __future__ import annotations

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sensitivity_utils import (
    resolve_sensitivity_models,
    apply_smoke,
    load_results_config,
    prepare_config_for_run,
    prepare_per_model_configs,
    read_metric_csv_if_exists,
    resolve_seeds,
    run_outputs_complete,
    write_rows_csv,
)
from src.experiment import MODEL_DISPLAY_NAMES, VALID_MODEL_KEYS, run_experiment
from src.results_export import ResultsExportConfig
from src.utils import ensure_dir, save_json


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "results" / "default_synthetic_results.json"
DEFAULT_MODELS = "pmmr,weighted_pmmr,kpv,weighted_kpv,dfpv,weighted_dfpv"
ALLOWED_MODELS = {
    "pmmr",
    "weighted_pmmr",
    "kpv",
    "weighted_kpv",
    "dfpv",
    "weighted_dfpv",
}
PROXY_DEPENDENCE_MODE = "gaussian_copula_correlated_disturbances_structural"
RHO_DESCRIPTION = (
    "rho is the latent Gaussian-copula correlation between aligned Z and W disturbances"
)


@dataclass(frozen=True)
class ProxyDependencePoint:
    strength: float

    @property
    def value_label(self) -> str:
        return f"{float(self.strength):.8g}".replace("-", "m").replace(".", "p")

    @property
    def run_name(self) -> str:
        return f"proxy_dependence_copula_structural_{self.value_label}"

    def annotations(self) -> dict[str, Any]:
        return {
            "assumption_family": "negative_control_proxy_dependence",
            "proxy_independence_violation_strength": float(self.strength),
            "proxy_independence_violation_mode": PROXY_DEPENDENCE_MODE,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sensitivity analysis using correlated treatment- and "
            "outcome-proxy disturbances."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "sensitivity_proxy_dependence_copula_structural",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help=f"Comma-separated model keys. Allowed keys: {','.join(sorted(ALLOWED_MODELS))}.",
    )
    parser.add_argument("--dataset-version", type=str, default="wu2024_nonadditive")
    parser.add_argument("--structure-type", type=str, default="single_peaked_heterogeneous")
    parser.add_argument(
        "--violation-grid",
        type=float,
        nargs="+",
        required=True,
        help="Grid of proxy-dependence strengths in [0, 1]. Strength 0 reproduces the original DGP.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--surface-patients", type=int, default=50)
    parser.add_argument("--treatment-grid-size", type=int, default=51)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument(
        "--model-progress-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print model start/done messages inside each sensitivity run.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip point/seed runs whose output directory already contains a complete finished run.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolve_models(raw_models: str) -> list[str]:
    return resolve_sensitivity_models(raw_models, VALID_MODEL_KEYS, ALLOWED_MODELS)


def _resolve_points(values: list[float]) -> list[ProxyDependencePoint]:
    points = [ProxyDependencePoint(float(value)) for value in values]
    for point in points:
        if not 0.0 <= point.strength <= 1.0:
            raise ValueError(
                f"Proxy-dependence strengths must lie in [0, 1]. Received {point.strength!r}."
            )
    return points


def _apply_point(config, point: ProxyDependencePoint) -> None:
    config.data.proxy_independence_violation_mode = PROXY_DEPENDENCE_MODE
    config.data.proxy_independence_violation_strength = float(point.strength)
    config.data.completeness_violation_strength = 0.0


def _metric_rows_from_result(
    result: dict[str, Any],
    *,
    dataset: str,
    structure_type: str,
    seed: int,
    run_dir: Path,
    point: ProxyDependencePoint,
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
                "structure_type": structure_type,
                "seed": int(seed),
                "split": split,
                "model_key": display_to_key.get(method, ""),
                "method": method,
                "run_dir": str(run_dir),
                **point.annotations(),
            }
            for key, value in metric_row.items():
                if key != "method":
                    row[key] = value
            rows.append(row)
    return rows


def _metric_rows_from_saved_run(
    *,
    dataset: str,
    structure_type: str,
    seed: int,
    run_dir: Path,
    point: ProxyDependencePoint,
) -> list[dict[str, Any]]:
    display_to_key = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    rows: list[dict[str, Any]] = []
    for split, filename in (("test", "metrics.csv"), ("validation", "validation_metrics.csv")):
        for metric_row in read_metric_csv_if_exists(run_dir / filename):
            method = str(metric_row.get("method", ""))
            row: dict[str, Any] = {
                "dataset": dataset,
                "structure_type": structure_type,
                "seed": int(seed),
                "split": split,
                "model_key": display_to_key.get(method, ""),
                "method": method,
                "run_dir": str(run_dir),
                **point.annotations(),
            }
            for key, value in metric_row.items():
                if key != "method":
                    row[key] = value
            rows.append(row)
    return rows


def _run_single_task(
    *,
    base_config,
    per_model_configs,
    selected_models: list[str],
    point: ProxyDependencePoint,
    seed: int,
    output_root: Path,
    treatment_grid_size: int,
    surface_patients: int,
    verbose: int | None,
    model_progress_logs: bool,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    run_dir = output_root / point.run_name / f"seed_{int(seed):04d}"
    if resume and run_outputs_complete(run_dir):
        rows = _metric_rows_from_saved_run(
            dataset=str(base_config.data.version),
            structure_type=str(base_config.data.structure_type),
            seed=seed,
            run_dir=run_dir,
            point=point,
        )
        return {
            "rows": rows,
            "manifest_entry": {
                "seed": int(seed),
                "point": point.annotations(),
                "run_dir": str(run_dir),
                "status": "skipped",
            },
        }
    config = prepare_config_for_run(
        base_config,
        seed=seed,
        output_dir=run_dir,
        verbose=verbose,
        treatment_grid_size=treatment_grid_size,
    )
    local_model_configs = prepare_per_model_configs(
        per_model_configs,
        seed=seed,
        output_dir=run_dir,
        verbose=verbose,
        treatment_grid_size=treatment_grid_size,
    )
    if smoke:
        apply_smoke(config)
        for local in local_model_configs.values():
            apply_smoke(local)
    _apply_point(config, point)
    for local in local_model_configs.values():
        _apply_point(local, point)

    result = run_experiment(
        config,
        selected_models=selected_models,
        model_configs=local_model_configs,
        results_export=ResultsExportConfig(
            enabled=True,
            n_surface_patients=int(surface_patients),
            treatment_grid_size=int(treatment_grid_size),
            split="test",
        ),
        model_progress=bool(model_progress_logs),
    )
    rows = _metric_rows_from_result(
        result,
        dataset=str(config.data.version),
        structure_type=str(config.data.structure_type),
        seed=seed,
        run_dir=run_dir,
        point=point,
    )
    return {
        "rows": rows,
        "manifest_entry": {
            "seed": int(seed),
            "point": point.annotations(),
            "run_dir": str(run_dir),
            "status": "completed",
        },
    }


def main() -> None:
    args = parse_args()
    selected_models = _resolve_models(args.models)
    points = _resolve_points(list(args.violation_grid))
    base_config, per_model_configs, _configured_models = load_results_config(args.config)
    base_config.data.version = str(args.dataset_version)
    base_config.data.structure_type = str(args.structure_type)
    base_config.data.proxy_independence_violation_mode = PROXY_DEPENDENCE_MODE
    for local in per_model_configs.values():
        local.data.proxy_independence_violation_mode = PROXY_DEPENDENCE_MODE
    seeds = resolve_seeds(
        seeds=args.seeds,
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        default_seed=int(base_config.seed),
    )
    output_root = ensure_dir(Path(args.output_dir))

    tasks = [
        {
            "base_config": copy.deepcopy(base_config),
            "per_model_configs": copy.deepcopy(per_model_configs),
            "selected_models": list(selected_models),
            "point": point,
            "seed": int(seed),
            "output_root": output_root,
            "treatment_grid_size": int(args.treatment_grid_size),
            "surface_patients": int(args.surface_patients),
            "verbose": args.verbose,
            "model_progress_logs": bool(args.model_progress_logs),
            "resume": bool(args.resume),
            "smoke": bool(args.smoke),
        }
        for point in points
        for seed in seeds
    ]

    all_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    if int(args.jobs) <= 1:
        for task in tasks:
            outcome = _run_single_task(**task)
            all_rows.extend(outcome["rows"])
            manifest_entries.append(outcome["manifest_entry"])
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_run_single_task, **task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    outcome = future.result()
                except Exception:
                    if not args.continue_on_error:
                        raise
                    manifest_entries.append(
                        {
                            "seed": int(task["seed"]),
                            "point": task["point"].annotations(),
                            "run_dir": str(Path(task["output_root"]) / task["point"].run_name / f"seed_{int(task['seed']):04d}"),
                            "error": True,
                        }
                    )
                    continue
                all_rows.extend(outcome["rows"])
                manifest_entries.append(outcome["manifest_entry"])

    preferred_columns = [
        "dataset",
        "structure_type",
        "seed",
        "split",
        "model_key",
        "method",
        "assumption_family",
        "proxy_independence_violation_strength",
        "proxy_independence_violation_mode",
        "run_dir",
    ]
    write_rows_csv(output_root / "results_metrics.csv", all_rows, preferred_columns)
    save_json(
        {
            "created_at": datetime.now().isoformat(),
            "config_path": str(args.config),
            "output_dir": str(output_root),
            "selected_models": selected_models,
            "dataset_version": str(args.dataset_version),
            "structure_type": str(args.structure_type),
            "seeds": [int(seed) for seed in seeds],
            "violation_grid": [float(point.strength) for point in points],
            "proxy_independence_violation_mode": PROXY_DEPENDENCE_MODE,
            "parameter_description": RHO_DESCRIPTION,
            "manifest_entries": manifest_entries,
        },
        output_root / "run_manifest.json",
    )


if __name__ == "__main__":
    main()
