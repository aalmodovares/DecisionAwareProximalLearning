from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.data.simulator import FactualSample, Wu2024Simulator
from src.reporting import build_patient_curve_table
from src.utils import ensure_dir, verbose_print


@dataclass(frozen=True)
class ResultsExportConfig:
    enabled: bool = True
    n_surface_patients: int = 50
    treatment_grid_size: int = 51
    split: str = "test"


def _ordered_fieldnames(rows: list[dict[str, object]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_global_curve_rows(
    *,
    dataset: str,
    seed: int,
    split: str,
    treatment_grid: np.ndarray,
    ground_truth_curve: np.ndarray,
    causal_marginal_curve: np.ndarray,
    proxy_conditioned_curve: np.ndarray,
    structure_curve: np.ndarray,
    mean_outcome_curves: dict[str, np.ndarray],
    mean_outcome_curves_direct: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    references = {
        "ground_truth_oracle_conditioned": ground_truth_curve,
        "causal_marginal": causal_marginal_curve,
        "proxy_conditioned": proxy_conditioned_curve,
        "structure_term": structure_curve,
    }
    rows: list[dict[str, object]] = []
    for curve_type, values in references.items():
        value_arr = np.asarray(values, dtype=float).reshape(-1)
        for treatment, value in zip(grid, value_arr):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": split,
                    "method": "reference",
                    "predictor_type": "reference",
                    "curve_type": curve_type,
                    "treatment": float(treatment),
                    "value": float(value),
                }
            )

    for method, values in mean_outcome_curves.items():
        value_arr = np.asarray(values, dtype=float).reshape(-1)
        for treatment, value in zip(grid, value_arr):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": split,
                    "method": method,
                    "predictor_type": "mean_or_primary",
                    "curve_type": "prediction",
                    "treatment": float(treatment),
                    "value": float(value),
                }
            )

    for method, values in (mean_outcome_curves_direct or {}).items():
        value_arr = np.asarray(values, dtype=float).reshape(-1)
        for treatment, value in zip(grid, value_arr):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": split,
                    "method": method,
                    "predictor_type": "direct_bridge_average",
                    "curve_type": "prediction",
                    "treatment": float(treatment),
                    "value": float(value),
                }
            )
    return rows


def save_result_curve_tables(
    *,
    output_dir: str | Path,
    simulator: Wu2024Simulator,
    sample: FactualSample,
    dataset: str,
    seed: int,
    treatment_grid: np.ndarray,
    ground_truth_curve: np.ndarray,
    causal_marginal_curve: np.ndarray,
    proxy_conditioned_curve: np.ndarray,
    structure_curve: np.ndarray,
    mean_outcome_curves: dict[str, np.ndarray],
    mean_outcome_curves_direct: dict[str, np.ndarray] | None,
    mean_predictors: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    direct_predictors: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] | None,
    oracle_direct_predictors: dict[
        str,
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ]
    | None,
    config: ResultsExportConfig,
    verbose: int = 0,
) -> dict[str, str]:
    output_dir = ensure_dir(Path(output_dir))
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    if grid.shape[0] != int(config.treatment_grid_size):
        verbose_print(
            verbose,
            1,
            (
                f"[results-export] treatment grid has {grid.shape[0]} points; "
                f"expected {config.treatment_grid_size}"
            ),
        )

    verbose_print(verbose, 1, "[results-export] writing global curves")
    global_rows = _build_global_curve_rows(
        dataset=dataset,
        seed=seed,
        split=config.split,
        treatment_grid=grid,
        ground_truth_curve=ground_truth_curve,
        causal_marginal_curve=causal_marginal_curve,
        proxy_conditioned_curve=proxy_conditioned_curve,
        structure_curve=structure_curve,
        mean_outcome_curves=mean_outcome_curves,
        mean_outcome_curves_direct=mean_outcome_curves_direct,
    )
    global_path = output_dir / "global_curves.csv"
    _write_csv(
        global_path,
        global_rows,
        [
            "dataset",
            "seed",
            "split",
            "method",
            "predictor_type",
            "curve_type",
            "treatment",
            "value",
        ],
    )

    n_patients = min(int(config.n_surface_patients), int(sample.covariates.shape[0]))
    verbose_print(
        verbose,
        1,
        f"[results-export] building patient curves n_patients={n_patients} grid={grid.shape[0]}",
    )
    patient_rows, metadata_rows = build_patient_curve_table(
        simulator=simulator,
        sample=sample,
        mean_predictors=mean_predictors,
        direct_predictors=direct_predictors,
        oracle_direct_predictors=oracle_direct_predictors,
        treatment_grid=grid,
        n_patients=n_patients,
        dataset=dataset,
        seed=seed,
        split=config.split,
    )
    patient_path = output_dir / "patient_curves.csv"
    metadata_path = output_dir / "patient_metadata.csv"
    _write_csv(
        patient_path,
        patient_rows,
        _ordered_fieldnames(
            patient_rows,
            [
                "dataset",
                "seed",
                "split",
                "patient_rank",
                "patient_id",
                "method",
                "predictor_type",
                "curve_type",
                "treatment",
                "value",
                "associational_method",
                "associational_count",
                "associational_low_count",
                "associational_empty",
                "associational_fallback_treatment",
                "associational_fallback_outcome",
            ],
        ),
    )
    _write_csv(
        metadata_path,
        metadata_rows,
        _ordered_fieldnames(
            metadata_rows,
            [
                "dataset",
                "seed",
                "split",
                "patient_rank",
                "patient_id",
                "observed_treatment",
                "observed_outcome",
                "assignment_mean",
                "structural_optimal_treatment",
                "surface_optimal_treatment",
            ],
        ),
    )
    verbose_print(
        verbose,
        1,
        (
            f"[results-export] patient curves done rows={len(patient_rows)} "
            f"metadata_rows={len(metadata_rows)}"
        ),
    )
    return {
        "global_curves": str(global_path),
        "patient_curves": str(patient_path),
        "patient_metadata": str(metadata_path),
    }
