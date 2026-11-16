from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.config import ExperimentConfig, config_to_dict
from src.experiment import ALL_MODEL_KEYS, MODEL_DISPLAY_NAMES, VALID_MODEL_KEYS
from src.utils import ensure_dir, save_json

if TYPE_CHECKING:
    import optuna


FLOW_ALIAS_PREFIX = "flow."
NMMR_ALIAS_PREFIX = "nmmr."


def load_search_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple):
        return tuple(value)
    if isinstance(current, Path):
        return Path(value)
    return value


def _set_single_path(config: ExperimentConfig, dotted_path: str, value: Any) -> None:
    target: Any = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"Unknown config section {part!r} in path {dotted_path!r}.")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise AttributeError(f"Unknown config field {leaf!r} in path {dotted_path!r}.")
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce_value(current, value))


def set_config_value(config: ExperimentConfig, dotted_path: str, value: Any) -> None:
    if dotted_path.startswith(FLOW_ALIAS_PREFIX):
        suffix = dotted_path[len(FLOW_ALIAS_PREFIX) :]
        _set_single_path(config, f"propensity_flow.{suffix}", value)
        _set_single_path(config, f"w_flow.{suffix}", value)
        return
    if dotted_path.startswith(NMMR_ALIAS_PREFIX):
        suffix = dotted_path[len(NMMR_ALIAS_PREFIX) :]
        _set_single_path(config, f"nmmr.{suffix}", value)
        _set_single_path(config, f"nmmr_u.{suffix}", value)
        _set_single_path(config, f"nmmr_v.{suffix}", value)
        return
    _set_single_path(config, dotted_path, value)


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    for dotted_path, value in overrides.items():
        set_config_value(config, dotted_path, value)
    return config


def _flatten_nested_config(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_nested_config(next_prefix, child, out)
    else:
        out[prefix] = value


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig()
    flat: dict[str, Any] = {}
    _flatten_nested_config("", data, flat)
    for dotted_path, value in flat.items():
        set_config_value(config, dotted_path, value)
    return config


def create_base_config(search_config: dict[str, Any], *, output_dir: Path, seed: int, verbose: int, plot_losses: bool) -> ExperimentConfig:
    config = ExperimentConfig()
    config.output_dir = output_dir
    config.seed = seed
    config.verbose = verbose
    config.plot_losses = plot_losses
    fixed = search_config.get("fixed", {})
    apply_overrides(config, fixed)
    return config


def _sample_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    spec_type = spec["type"]
    if spec_type == "categorical":
        choices = spec["choices"]
        if any(isinstance(choice, (list, dict)) for choice in choices):
            index = trial.suggest_int(f"{name}__idx", 0, len(choices) - 1)
            return choices[index]
        return trial.suggest_categorical(name, choices)
    if spec_type == "float":
        step = spec.get("step")
        return trial.suggest_float(name, spec["low"], spec["high"], step=step, log=bool(spec.get("log", False)))
    if spec_type == "logfloat":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if spec_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
    raise ValueError(f"Unsupported search space type {spec_type!r} for {name}.")


def sample_trial_overrides(
    trial: optuna.Trial,
    search_config: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    common_space = search_config.get("search", {}).get("common", {})
    per_model_space = search_config.get("search", {}).get("per_model", {}).get(model_key, {})
    overrides: dict[str, Any] = {}
    for name, spec in common_space.items():
        overrides[name] = _sample_from_spec(trial, name, spec)
    for name, spec in per_model_space.items():
        overrides[name] = _sample_from_spec(trial, f"{model_key}.{name}", spec)
    return overrides


def resolve_trial_config(
    base_config: ExperimentConfig,
    search_config: dict[str, Any],
    model_key: str,
    trial: optuna.Trial,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    config = copy.deepcopy(base_config)
    fixed_model = search_config.get("fixed_per_model", {}).get(model_key, {})
    apply_overrides(config, fixed_model)
    overrides = sample_trial_overrides(trial, search_config, model_key)
    apply_overrides(config, overrides)
    return config, overrides


def get_objective_settings(search_config: dict[str, Any]) -> tuple[str, str]:
    study_cfg = search_config.get("study", {})
    metric = study_cfg.get("metric", "validation_policy_value")
    direction = study_cfg.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported optimization direction {direction!r}.")
    return metric, direction


def extract_objective_value(result: dict[str, Any], model_key: str, metric: str) -> float:
    display_name = MODEL_DISPLAY_NAMES[model_key]
    rows = [row for row in result["validation_metrics"] if row["method"] == display_name]
    if not rows:
        raise KeyError(f"Validation metrics for model {display_name!r} were not returned.")
    row = rows[0]
    if metric == "validation_policy_value":
        return float(row["policy_value"])
    if metric == "validation_regret":
        return float(row["regret"])
    if metric == "validation_policy_mse":
        return float(row["policy_mse"])
    if metric == "validation_local_rmse":
        return float(row["local_rmse"])
    if metric == "validation_global_rmse":
        return float(row["global_rmse"])
    if metric == "validation_observed_rmse":
        return float(row["observed_rmse"])
    if metric == "validation_observational_loss":
        return float(row["observational_loss"])
    if metric in {"validation_factual_mean_rmse", "validation_factual_rmse", "factual_mean_rmse", "factual_rmse"}:
        return float(row["factual_mean_rmse"])
    raise KeyError(f"Unsupported validation metric {metric!r}.")


def trial_row(
    trial: optuna.trial.FrozenTrial,
    model_key: str,
    metric_name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_key,
        "trial_number": trial.number,
        "trial_id": f"{model_key}_trial_{trial.number:04d}",
        "objective_metric": metric_name,
        "objective_value": trial.value,
        "state": str(trial.state),
    }
    for key, value in overrides.items():
        row[key] = value
    return row


def _metric_row_for_model(rows: list[dict[str, Any]], display_name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == display_name:
            return row
    return {}


def _prefixed_metrics(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "method":
            continue
        out[f"{prefix}{key}"] = value
    return out


def trial_metrics_row(
    trial: optuna.trial.FrozenTrial,
    model_key: str,
    metric_name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    display_name = MODEL_DISPLAY_NAMES[model_key]
    validation_rows = list(trial.user_attrs.get("validation_metrics", []))
    test_rows = list(trial.user_attrs.get("test_metrics", []))
    validation_row = _metric_row_for_model(validation_rows, display_name)
    test_row = _metric_row_for_model(test_rows, display_name)

    row = trial_row(trial, model_key, metric_name, overrides)
    row.update(_prefixed_metrics("val_", validation_row))
    row.update(_prefixed_metrics("test_", test_row))
    return row


def save_trial_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def requested_models_from_args(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list(ALL_MODEL_KEYS)
    unknown = sorted(set(requested) - set(VALID_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown model keys requested for HPO: {unknown}")
    return requested


def save_bundle(
    output_dir: Path,
    base_config: ExperimentConfig,
    per_model_results: dict[str, dict[str, Any]],
) -> Path:
    bundle = {
        "base_config": config_to_dict(base_config),
        "models": {
            model_key: {
                "best_value": result["best_value"],
                "best_params": result["best_params"],
                "best_config": result["best_config"],
                "objective_metric": result["objective_metric"],
                "direction": result["direction"],
                "study_name": result["study_name"],
            }
            for model_key, result in per_model_results.items()
        },
    }
    bundle_path = output_dir / "bundle_best_configs.json"
    save_json(bundle, bundle_path)
    return bundle_path


def load_bundle(path: Path) -> tuple[ExperimentConfig, dict[str, ExperimentConfig]]:
    bundle = json.loads(Path(path).read_text())
    base_config = experiment_config_from_dict(bundle["base_config"])
    model_configs = {
        model_key: experiment_config_from_dict(entry["best_config"])
        for model_key, entry in bundle["models"].items()
    }
    return base_config, model_configs


def ensure_search_output_dir(root: Path, study_group: str) -> Path:
    return ensure_dir(root / study_group)
