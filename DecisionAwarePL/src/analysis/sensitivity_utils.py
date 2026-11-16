from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from src.config import (
    ExperimentConfig,
    all_mean_configs,
    all_nmmr_configs,
    all_weighting_configs,
)
from src.hparam_search import set_config_value
from src.utils import ensure_dir


def _flatten_mapping(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_mapping(next_prefix, child, out)
        return
    out[prefix] = value


def apply_mapping(config: ExperimentConfig, values: dict[str, Any]) -> None:
    flat: dict[str, Any] = {}
    _flatten_mapping("", values, flat)
    for dotted_path, value in flat.items():
        set_config_value(config, dotted_path, value)


def load_results_config(path: Path) -> tuple[ExperimentConfig, dict[str, ExperimentConfig], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    base_config = ExperimentConfig()
    if isinstance(raw.get("config"), dict):
        apply_mapping(base_config, raw["config"])
    if isinstance(raw.get("fixed"), dict):
        apply_mapping(base_config, raw["fixed"])

    per_model_configs: dict[str, ExperimentConfig] = {}
    for section_name in ("per_model_configs", "fixed_per_model"):
        section = raw.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for model_key, overrides in section.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"{section_name}.{model_key} must be an object.")
            local_config = copy.deepcopy(base_config)
            apply_mapping(local_config, overrides)
            per_model_configs[model_key] = local_config
    configured_models = raw.get("models", [])
    if not isinstance(configured_models, list):
        configured_models = []
    return base_config, per_model_configs, list(configured_models)


def set_treatment_grid_size(config: ExperimentConfig, grid_size: int) -> None:
    config.data.policy_grid_size = int(grid_size)
    for mean_config in all_mean_configs(config):
        mean_config.treatment_grid_size = int(grid_size)
        mean_config.evaluation_grid_size = int(grid_size)


def _optional_min(value: int | None, limit: int) -> int:
    if value is None:
        return int(limit)
    return min(int(value), int(limit))


def apply_smoke(config: ExperimentConfig) -> None:
    config.data.n_train = 80
    config.data.n_val = 0
    config.data.n_test = 90
    config.data.support_pilot_size = min(int(config.data.support_pilot_size), 1000)
    config.data.causal_mc_samples = min(int(config.data.causal_mc_samples), 64)
    config.data.causal_mc_batch_size = min(int(config.data.causal_mc_batch_size), 64)
    config.data.associational_mc_samples = min(int(config.data.associational_mc_samples), 500)

    for flow_config in (config.propensity_flow, config.w_flow):
        flow_config.epochs = min(int(flow_config.epochs), 4)
        flow_config.min_epochs = min(int(flow_config.min_epochs), 1)
        flow_config.early_stop_patience = min(int(flow_config.early_stop_patience), 1)
        flow_config.hidden_features = (8,)

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

    for weighting_config in all_weighting_configs(config):
        weighting_config.n_rounds = min(int(weighting_config.n_rounds), 1)
        weighting_config.early_stop_patience = min(int(weighting_config.early_stop_patience), 1)

    config.kpv.max_train_size = _optional_min(config.kpv.max_train_size, 80)
    config.kpv.n_lambda1_search = min(int(config.kpv.n_lambda1_search), 2)
    config.kpv.n_lambda2_search = min(int(config.kpv.n_lambda2_search), 2)
    config.dfpv.epochs = min(int(config.dfpv.epochs), 4)
    config.dfpv.min_epochs = min(int(config.dfpv.min_epochs), 1)
    config.dfpv.early_stop_patience = min(int(config.dfpv.early_stop_patience), 1)
    config.dfpv.hidden_dims = (8,)
    config.dfpv.feature_dim = min(int(config.dfpv.feature_dim), 4)
    config.dfpv.max_train_size = _optional_min(config.dfpv.max_train_size, 80)
    config.s_learner.epochs = min(int(config.s_learner.epochs), 4)
    config.s_learner.min_epochs = min(int(config.s_learner.min_epochs), 1)
    config.s_learner.early_stop_patience = min(int(config.s_learner.early_stop_patience), 1)
    config.s_learner.hidden_dims = (8,)
    config.s_learner.batch_size = 64


def prepare_config_for_run(
    config: ExperimentConfig,
    *,
    seed: int,
    output_dir: Path,
    verbose: int | None,
    treatment_grid_size: int,
) -> ExperimentConfig:
    run_config = copy.deepcopy(config)
    run_config.seed = int(seed)
    run_config.output_dir = output_dir
    run_config.save_outputs = True
    run_config.save_figures = False
    run_config.plot_losses = False
    run_config.fast_factual_eval = False
    if verbose is not None:
        run_config.verbose = int(verbose)
    set_treatment_grid_size(run_config, treatment_grid_size)
    return run_config


def prepare_per_model_configs(
    model_configs: dict[str, ExperimentConfig],
    *,
    seed: int,
    output_dir: Path,
    verbose: int | None,
    treatment_grid_size: int,
) -> dict[str, ExperimentConfig]:
    prepared: dict[str, ExperimentConfig] = {}
    for model_key, config in model_configs.items():
        prepared[model_key] = prepare_config_for_run(
            config,
            seed=seed,
            output_dir=output_dir,
            verbose=verbose,
            treatment_grid_size=treatment_grid_size,
        )
    return prepared


def resolve_seeds(
    *,
    seeds: list[int] | None,
    n_seeds: int | None,
    seed_start: int,
    default_seed: int,
) -> list[int]:
    if seeds:
        return [int(seed) for seed in seeds]
    if n_seeds is not None:
        if int(n_seeds) <= 0:
            raise ValueError("--n-seeds must be positive.")
        return list(range(int(seed_start), int(seed_start) + int(n_seeds)))
    return [int(default_seed)]


def read_metric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_metric_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_metric_csv(path)


def run_outputs_complete(run_dir: Path) -> bool:
    required = [
        "metrics.csv",
        "validation_metrics.csv",
        "global_curves.csv",
        "patient_curves.csv",
        "patient_metadata.csv",
    ]
    return all((run_dir / name).exists() for name in required)


def write_rows_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    ensure_dir(path.parent)
    keys = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_sensitivity_models(raw_models: str, valid_models, allowed_models) -> list[str]:
    selected = [part.strip() for part in raw_models.split(",") if part.strip()]
    if not selected:
        raise ValueError("--models must include at least one model.")
    unknown = sorted(set(selected) - set(valid_models))
    if unknown:
        raise ValueError(f"Unknown model keys requested: {unknown}. Valid keys: {valid_models}")
    unsupported = sorted(set(selected) - set(allowed_models))
    if unsupported:
        raise ValueError(
            "This sensitivity script only supports PMMR/KPV/DFPV with their weighted variants. "
            f"Unsupported keys: {unsupported}"
        )
    return selected
