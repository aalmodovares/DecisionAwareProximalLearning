from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.analysis.sensitivity_utils import set_treatment_grid_size as _set_treatment_grid_size
from src.config import ExperimentConfig


def _resolve_seeds(args: argparse.Namespace, base_config: ExperimentConfig) -> list[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return [int(seed) for seed in args.seeds]
    if args.n_seeds is not None:
        if args.n_seeds <= 0:
            raise ValueError("--n-seeds must be positive.")
        return list(range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)))
    return [int(base_config.seed)]


def _prepare_config_for_seed(
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
    _set_treatment_grid_size(run_config, treatment_grid_size)
    return run_config
