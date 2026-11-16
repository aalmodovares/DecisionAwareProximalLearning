import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, all_mean_configs, all_nmmr_configs, all_weighting_configs
from src.experiment import ALL_MODEL_KEYS, VALID_MODEL_KEYS, run_experiment
from src.hparam_search import load_bundle


def _parse_nmmr_batch_size(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    parsed = int(text)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("NMMR batch size must be a positive integer or 'all'.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted proximal OOSR synthetic experiments.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "single_run")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated model keys to run, or 'all'. Valid keys: {','.join(VALID_MODEL_KEYS)}.",
    )
    parser.add_argument(
        "--dataset-version",
        choices=[
            "wu2024_original",
            "wu2024_biased_assignment",
            "wu2024_nonadditive",
            "wu2024_single_peaked_heterogeneous_nonadditive",
            "wu2024_single_peaked_heterogeneous_nonadditive_shifted",
            "wu2024_saturated_interaction",
            "wu2024_saturated_interaction_biased_assignment",
            "oracle_bridge_toy",
            "semisynthetic_tcga",
        ],
        default="wu2024_original",
    )
    parser.add_argument(
        "--structure-type",
        choices=["quardratic", "peaked", "sigmoid", "peaked_centered", "single_peaked", "single_peaked_heterogeneous", "v1"],
        default="peaked",
    )
    parser.add_argument("--mean-method", choices=["pseudo_outcomes", "flow_mc"], default="pseudo_outcomes")
    parser.add_argument("--normalization-mode", choices=["none", "paper_normalization", "exact_normalization"], default="paper_normalization")
    parser.add_argument("--stabilized", action="store_true")
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument(
        "--root-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sqrt(sample_weight) when fitting weighted KPV/PMMR/NMMR bridges.",
    )
    parser.add_argument(
        "--weight-mean-regressor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also pass OOSR sample weights to the KPV/PMMR/NMMR mean regressor.",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start weighted KPV/PMMR/NMMR rounds from the previous round when possible.",
    )
    parser.add_argument("--kpv-rounds", type=int, default=None)
    parser.add_argument("--kpv-tau", type=float, default=None)
    parser.add_argument("--kpv-lam", type=float, default=None)
    parser.add_argument("--dfpv-rounds", type=int, default=None)
    parser.add_argument("--dfpv-tau", type=float, default=None)
    parser.add_argument("--dfpv-lam", type=float, default=None)
    parser.add_argument(
        "--truncated-kernel-weights",
        "--truncated-kernel.weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="truncated_kernel_weights",
        help="Use the [-1,1] truncated and renormalized Gaussian OOSR kernel for sample weights.",
    )
    parser.add_argument("--pmmr-rounds", type=int, default=None)
    parser.add_argument("--pmmr-tau", type=float, default=None)
    parser.add_argument("--pmmr-lam", type=float, default=None)
    parser.add_argument("--nmmr-u-rounds", type=int, default=None)
    parser.add_argument("--nmmr-u-tau", type=float, default=None)
    parser.add_argument("--nmmr-u-lam", type=float, default=None)
    parser.add_argument("--nmmr-v-rounds", type=int, default=None)
    parser.add_argument("--nmmr-v-tau", type=float, default=None)
    parser.add_argument("--nmmr-v-lam", type=float, default=None)
    parser.add_argument("--naive-oosr-rounds", type=int, default=None)
    parser.add_argument("--naive-oosr-tau", type=float, default=None)
    parser.add_argument("--naive-oosr-lam", type=float, default=None)
    parser.add_argument("--oracle-naive-oosr-rounds", type=int, default=None)
    parser.add_argument("--oracle-naive-oosr-tau", type=float, default=None)
    parser.add_argument("--oracle-naive-oosr-lam", type=float, default=None)
    parser.add_argument("--nonadditive-gamma", type=float, default=None)
    parser.add_argument("--nonadditive-eta", type=float, default=None)
    parser.add_argument("--nonadditive-scale-ratio", type=float, default=None)
    parser.add_argument("--alpha-nonadditivity", type=float, default=None)
    parser.add_argument("--shifted-confounder-treatment-scale", type=float, default=None)
    parser.add_argument("--shifted-confounder-treatment-slope", type=float, default=None)
    parser.add_argument(
        "--associational-kernel",
        "--associational_kernel",
        dest="associational_kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use kernel-smoothed associational reference E[Y|X,A] in conditional patient plots (default: true).",
    )
    parser.add_argument("--tcga-data-path", type=str, default=None)
    parser.add_argument("--tcga-processed-path", type=str, default=None)
    parser.add_argument("--tcga-metadata-path", type=str, default=None)
    parser.add_argument("--tcga-preprocess-seed", type=int, default=None)
    parser.add_argument("--tcga-param-seed", type=int, default=None)
    parser.add_argument("--tcga-top-genes", type=int, default=None)
    parser.add_argument("--tcga-block-size", type=int, default=None)
    parser.add_argument("--tcga-x-pca-dim", type=int, default=None)
    parser.add_argument("--tcga-z-pca-dim", type=int, default=None)
    parser.add_argument("--tcga-w-pca-dim", type=int, default=None)
    parser.add_argument("--tcga-proxy-noise-scale", type=float, default=None)
    parser.add_argument("--tcga-treatment-phi", type=float, default=None)
    parser.add_argument("--tcga-outcome-noise-scale", type=float, default=None)
    parser.add_argument("--tcga-nonadditive-u0-treatment-x0-scale", type=float, default=None)
    parser.add_argument("--tcga-normalization-pilot-size", type=int, default=None)
    parser.add_argument("--tcga-causal-mc-samples-per-x", type=int, default=None)
    parser.add_argument("--tcga-policy-value-mc-samples", type=int, default=None)
    parser.add_argument(
        "--tcga-common-random-numbers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse the same TCGA Monte Carlo latent draws across treatment-grid points when estimating causal curves.",
    )
    parser.add_argument(
        "--tcga-sample-with-replacement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sample TCGA patient rows with replacement.",
    )
    parser.add_argument("--centered-penalty-strength", type=float, default=None)
    parser.add_argument("--additive-term-scale", type=float, default=None)
    parser.add_argument("--outcome-noise-scale", type=float, default=None)
    parser.add_argument("--pmmr-ridge", type=float, default=None)
    parser.add_argument("--pmmr-awx-scale", type=float, default=None)
    parser.add_argument("--pmmr-azx-scale", type=float, default=None)
    parser.add_argument("--kpv-lambda1", type=float, default=None)
    parser.add_argument("--kpv-lambda2", type=float, default=None)
    parser.add_argument("--kpv-lambda1-min", type=float, default=None)
    parser.add_argument("--kpv-lambda1-max", type=float, default=None)
    parser.add_argument("--kpv-n-lambda1-search", type=int, default=None)
    parser.add_argument("--kpv-lambda2-min", type=float, default=None)
    parser.add_argument("--kpv-lambda2-max", type=float, default=None)
    parser.add_argument("--kpv-n-lambda2-search", type=int, default=None)
    parser.add_argument("--kpv-kernel-variance-scale", type=float, default=None)
    parser.add_argument("--kpv-split-ratio", type=float, default=None)
    parser.add_argument("--kpv-max-train-size", type=int, default=None)
    parser.add_argument("--kpv-prediction-batch-size", type=int, default=None)
    parser.add_argument(
        "--kpv-optimize-lambda1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KPV leave-one-out selection for lambda1 unless --kpv-lambda1 is fixed.",
    )
    parser.add_argument(
        "--kpv-optimize-lambda2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KPV leave-one-out selection for lambda2 unless --kpv-lambda2 is fixed.",
    )
    parser.add_argument(
        "--kpv-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize KPV inputs and outcome before fitting the closed-form bridge.",
    )
    parser.add_argument("--dfpv-lam1", type=float, default=None)
    parser.add_argument("--dfpv-lam2", type=float, default=None)
    parser.add_argument("--dfpv-feature-dim", type=int, default=None)
    parser.add_argument("--dfpv-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--dfpv-activation", choices=["relu", "elu", "gelu", "tanh", "silu"], default=None)
    parser.add_argument("--dfpv-lr", type=float, default=None)
    parser.add_argument("--dfpv-weight-decay", type=float, default=None)
    parser.add_argument("--dfpv-prediction-batch-size", type=int, default=None)
    parser.add_argument("--dr-kpv-kpv-lambda1", type=float, default=None)
    parser.add_argument("--dr-kpv-kpv-lambda2", type=float, default=None)
    parser.add_argument("--dr-kpv-kap-lambda", type=float, default=None)
    parser.add_argument("--dr-kpv-kap-eta", type=float, default=None)
    parser.add_argument("--dr-kpv-kap-lambda2", type=float, default=None)
    parser.add_argument("--dr-kpv-kap-zeta", type=float, default=None)
    parser.add_argument("--dr-kpv-lambda-dr", type=float, default=None)
    parser.add_argument("--dr-kpv-kernel-variance-scale", type=float, default=None)
    parser.add_argument("--dr-kpv-stage1-perc", type=float, default=None)
    parser.add_argument("--dr-kpv-max-train-size", type=int, default=None)
    parser.add_argument("--dr-kpv-prediction-batch-size", type=int, default=None)
    parser.add_argument("--dr-kpv-optimize-internal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dr-kpv-standardize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dr-pmmr-pmmr-lambda", type=float, default=None)
    parser.add_argument("--dr-pmmr-kap-lambda", type=float, default=None)
    parser.add_argument("--dr-pmmr-kap-eta", type=float, default=None)
    parser.add_argument("--dr-pmmr-kap-lambda2", type=float, default=None)
    parser.add_argument("--dr-pmmr-kap-zeta", type=float, default=None)
    parser.add_argument("--dr-pmmr-lambda-dr", type=float, default=None)
    parser.add_argument("--dr-pmmr-kernel-variance-scale", type=float, default=None)
    parser.add_argument("--dr-pmmr-stage1-perc", type=float, default=None)
    parser.add_argument("--dr-pmmr-max-train-size", type=int, default=None)
    parser.add_argument("--dr-pmmr-prediction-batch-size", type=int, default=None)
    parser.add_argument("--dr-pmmr-optimize-internal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dr-pmmr-standardize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-lr", type=float, default=None)
    parser.add_argument("--nmmr-u-lr", type=float, default=None)
    parser.add_argument("--nmmr-v-lr", type=float, default=None)
    parser.add_argument("--nmmr-azx-scale", type=float, default=None)
    parser.add_argument("--nmmr-u-azx-scale", type=float, default=None)
    parser.add_argument("--nmmr-v-azx-scale", type=float, default=None)
    parser.add_argument("--nmmr-activation", choices=["relu", "elu", "gelu", "tanh", "silu"], default=None)
    parser.add_argument("--nmmr-u-activation", choices=["relu", "elu", "gelu", "tanh", "silu"], default=None)
    parser.add_argument("--nmmr-v-activation", choices=["relu", "elu", "gelu", "tanh", "silu"], default=None)
    parser.add_argument("--nmmr-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--nmmr-u-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--nmmr-v-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--nmmr-weight-decay", type=float, default=None)
    parser.add_argument("--nmmr-u-weight-decay", type=float, default=None)
    parser.add_argument("--nmmr-v-weight-decay", type=float, default=None)
    parser.add_argument("--nmmr-epochs", type=int, default=None)
    parser.add_argument("--nmmr-u-epochs", type=int, default=None)
    parser.add_argument("--nmmr-v-epochs", type=int, default=None)
    parser.add_argument("--nmmr-min-epochs", type=int, default=None)
    parser.add_argument("--nmmr-u-min-epochs", type=int, default=None)
    parser.add_argument("--nmmr-v-min-epochs", type=int, default=None)
    parser.add_argument("--nmmr-early-stop-patience", type=int, default=None)
    parser.add_argument("--nmmr-u-early-stop-patience", type=int, default=None)
    parser.add_argument("--nmmr-v-early-stop-patience", type=int, default=None)
    parser.add_argument("--nmmr-early-stop-min-delta", type=float, default=None)
    parser.add_argument("--nmmr-u-early-stop-min-delta", type=float, default=None)
    parser.add_argument("--nmmr-v-early-stop-min-delta", type=float, default=None)
    parser.add_argument("--nmmr-val-fraction", type=float, default=None)
    parser.add_argument("--nmmr-u-val-fraction", type=float, default=None)
    parser.add_argument("--nmmr-v-val-fraction", type=float, default=None)
    parser.add_argument("--nmmr-moment-weight-power", type=float, default=None)
    parser.add_argument("--nmmr-u-moment-weight-power", type=float, default=None)
    parser.add_argument("--nmmr-v-moment-weight-power", type=float, default=None)
    parser.add_argument(
        "--deactivate-rolloff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable the original off-by-one NMMR final activation rolloff.",
    )
    parser.add_argument("--nmmr-u-deactivate-rolloff", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-v-deactivate-rolloff", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--nmmr-product-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a product RBF kernel over standardized (A,Z,X) instead of an isotropic one.",
    )
    parser.add_argument("--nmmr-u-product-kernel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-v-product-kernel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--quadratic-treatment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment the NMMR bridge input with a negative quadratic treatment feature (-A^2).",
    )
    parser.add_argument("--nmmr-u-quadratic-treatment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-v-quadratic-treatment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-a-scale", type=float, default=None)
    parser.add_argument("--nmmr-u-a-scale", type=float, default=None)
    parser.add_argument("--nmmr-v-a-scale", type=float, default=None)
    parser.add_argument("--nmmr-z-scale", type=float, default=None)
    parser.add_argument("--nmmr-u-z-scale", type=float, default=None)
    parser.add_argument("--nmmr-v-z-scale", type=float, default=None)
    parser.add_argument("--nmmr-x-scale", type=float, default=None)
    parser.add_argument("--nmmr-u-x-scale", type=float, default=None)
    parser.add_argument("--nmmr-v-x-scale", type=float, default=None)
    parser.add_argument(
        "--treatment-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a treatment grid to build bridge pseudo-outcomes for PMMR/KPV/NMMR mean-regressor training.",
    )
    parser.add_argument(
        "--bridge-all-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the NMMR bridge on all training data before mean fitting.",
    )
    parser.add_argument("--nmmr-u-bridge-all-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nmmr-v-bridge-all-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--nmmr-batch-size",
        type=_parse_nmmr_batch_size,
        default=None,
        help="Global NMMR bridge batch size (positive integer or 'all').",
    )
    parser.add_argument(
        "--nmmr-u-batch-size",
        type=_parse_nmmr_batch_size,
        default=None,
        help="NMMR-U bridge batch size (positive integer or 'all'). Overrides --nmmr-batch-size.",
    )
    parser.add_argument(
        "--nmmr-v-batch-size",
        type=_parse_nmmr_batch_size,
        default=None,
        help="NMMR-V bridge batch size (positive integer or 'all'). Overrides --nmmr-batch-size.",
    )
    parser.add_argument("--nmmr-grad-clip-norm", type=float, default=None)
    parser.add_argument("--nmmr-u-grad-clip-norm", type=float, default=None)
    parser.add_argument("--nmmr-v-grad-clip-norm", type=float, default=None)
    parser.add_argument("--mean-lr", type=float, default=None)
    parser.add_argument("--mean-batch-size", type=int, default=None)
    parser.add_argument("--weight-early-stop-patience", type=int, default=None)
    parser.add_argument("--weight-min-improvement", type=float, default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument(
        "--save-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save run figures. Use --no-save-figures for faster diagnostic runs.",
    )
    parser.add_argument("--plot-losses", action="store_true")
    parser.add_argument(
        "--two-stage-slearner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable second-stage mean regressor for non-oracle s_learner.",
    )
    parser.add_argument(
        "--two-stage-oosr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable second-stage mean regressor for non-oracle naive_oosr.",
    )
    parser.add_argument("--bridge_diagnostics", action="store_true")
    parser.add_argument("--kernel_diagnostics", action="store_true")
    parser.add_argument("--kernel-diag-max-points", type=int, default=None)
    parser.add_argument("--kernel-diag-bandwidth-multipliers", type=float, nargs="+", default=None)
    parser.add_argument("--kernel-diag-max-offdiag-values", type=int, default=None)
    parser.add_argument("--kernel-diag-save-offdiag-values", action="store_true")
    parser.add_argument("--propensity-epochs", type=int, default=None)
    parser.add_argument("--propensity-min-epochs", type=int, default=None)
    parser.add_argument("--propensity-patience", type=int, default=None)
    parser.add_argument("--mean-epochs", type=int, default=None)
    parser.add_argument("--mean-min-epochs", type=int, default=None)
    parser.add_argument("--mean-patience", type=int, default=None)
    parser.add_argument(
        "--mean-early-stopping-v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tolerance-based checkpoint selection for mean regressors (mode v2).",
    )
    parser.add_argument(
        "--mean-early-stopping-v2-alpha",
        type=float,
        default=None,
        help="Tolerance alpha for mean early-stopping v2 (accept val_loss <= best + alpha).",
    )
    parser.add_argument(
        "--mean-early-stopping-v2-patience",
        type=int,
        default=None,
        help="Number of consecutive out-of-threshold epochs before stopping in mean early-stopping v2.",
    )
    parser.add_argument(
        "--disable-pmmr-mean-early-stopping",
        action="store_true",
        help="Disable early stopping only for PMMR/Weighted PMMR mean regressor.",
    )
    parser.add_argument(
        "--disable-nmmr-mean-early-stopping",
        action="store_true",
        help="Disable early stopping only for NMMR/Weighted NMMR mean regressor(s).",
    )
    parser.add_argument(
        "--disable-kpv-mean-early-stopping",
        action="store_true",
        help="Disable early stopping for the KPV mean regressor path (shared mean config).",
    )
    parser.add_argument(
        "--disable-dfpv-mean-early-stopping",
        action="store_true",
        help="Disable early stopping for the DFPV mean regressor path.",
    )
    parser.add_argument(
        "--mean-crossfit-folds",
        type=int,
        default=None,
        help="Number of folds used to cross-fit pseudo-outcomes for PMMR/KPV/NMMR mean regressors.",
    )
    parser.add_argument(
        "--no-mean-crossfit-shuffle",
        action="store_true",
        help="Disable shuffling before assigning folds for PMMR/KPV/NMMR mean-regressor cross-fitting.",
    )
    parser.add_argument("--best_model", action="store_true")
    parser.add_argument("--best-config-path", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _selected_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_MODEL_KEYS)
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(VALID_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown model keys requested: {unknown}. Valid keys: {VALID_MODEL_KEYS}")
    return selected


def _is_wu_version(version: str) -> bool:
    text = str(version).strip().lower()
    return text == "wu2024_original" or text.startswith("wu2024_")


def _wu_true_decaflow_adjacency(config: ExperimentConfig) -> tuple[tuple[int, ...], ...]:
    latent_dim = int(config.decaflow.latent_dim)
    dim_z = int(config.data.dim_z)
    dim_w = int(config.data.dim_w)
    dim_x = int(config.data.dim_x)
    observed_dim = dim_z + dim_w + 2  # Z, W, A, Y
    total_dim = latent_dim + observed_dim + dim_x

    z_start = latent_dim
    w_start = z_start + dim_z
    treatment_idx = w_start + dim_w
    outcome_idx = treatment_idx + 1
    x_start = latent_dim + observed_dim

    adj = [[0 for _ in range(total_dim)] for _ in range(total_dim)]
    for idx in range(total_dim):
        adj[idx][idx] = 1

    latent_idx = list(range(latent_dim))
    z_idx = list(range(z_start, z_start + dim_z))
    w_idx = list(range(w_start, w_start + dim_w))
    x_idx = list(range(x_start, x_start + dim_x))

    for child_idx in [*z_idx, *w_idx, treatment_idx, outcome_idx]:
        for parent_idx in latent_idx:
            adj[child_idx][parent_idx] = 1

    for zi in z_idx:
        adj[treatment_idx][zi] = 1
    for wi in w_idx:
        adj[outcome_idx][wi] = 1
    adj[outcome_idx][treatment_idx] = 1
    for xi in x_idx:
        adj[treatment_idx][xi] = 1
        adj[outcome_idx][xi] = 1

    if latent_dim >= 2:
        adj[1][0] = 1

    return tuple(tuple(int(value) for value in row) for row in adj)


def _apply_decaflow_wu_true_adjacency(config: ExperimentConfig) -> None:
    if not _is_wu_version(config.data.version):
        return
    config.decaflow.adjacency_override = _wu_true_decaflow_adjacency(config)


def _apply_runtime_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    def _apply_nmmr_cfg_overrides(
        nmmr_cfg,
        *,
        lr=None,
        azx_scale=None,
        activation=None,
        hidden_dims=None,
        weight_decay=None,
        epochs=None,
        min_epochs=None,
        early_stop_patience=None,
        early_stop_min_delta=None,
        val_fraction=None,
        moment_weight_power=None,
        deactivate_rolloff=None,
        product_kernel=None,
        quadratic_treatment=None,
        a_scale=None,
        z_scale=None,
        x_scale=None,
        bridge_all_data=None,
        batch_size=None,
        grad_clip_norm=None,
    ) -> None:
        if lr is not None:
            nmmr_cfg.lr = lr
        if azx_scale is not None:
            nmmr_cfg.azx_scale = azx_scale
        if activation is not None:
            nmmr_cfg.activation = activation
        if hidden_dims is not None:
            nmmr_cfg.hidden_dims = tuple(hidden_dims)
        if weight_decay is not None:
            nmmr_cfg.weight_decay = weight_decay
        if epochs is not None:
            nmmr_cfg.epochs = epochs
        if min_epochs is not None:
            nmmr_cfg.min_epochs = min_epochs
        if early_stop_patience is not None:
            nmmr_cfg.early_stop_patience = early_stop_patience
        if early_stop_min_delta is not None:
            nmmr_cfg.early_stop_min_delta = early_stop_min_delta
        if val_fraction is not None:
            nmmr_cfg.val_fraction = val_fraction
        if moment_weight_power is not None:
            nmmr_cfg.moment_weight_power = moment_weight_power
        if deactivate_rolloff is not None:
            nmmr_cfg.deactivate_rolloff = deactivate_rolloff
        if product_kernel is not None:
            nmmr_cfg.product_kernel = product_kernel
        if quadratic_treatment is not None:
            nmmr_cfg.quadratic_treatment = quadratic_treatment
        if a_scale is not None:
            nmmr_cfg.a_scale = a_scale
        if z_scale is not None:
            nmmr_cfg.z_scale = z_scale
        if x_scale is not None:
            nmmr_cfg.x_scale = x_scale
        if bridge_all_data is not None:
            nmmr_cfg.bridge_all_data = bridge_all_data
        if batch_size is not None:
            nmmr_cfg.batch_size = batch_size
        if grad_clip_norm is not None:
            nmmr_cfg.grad_clip_norm = grad_clip_norm

    config.seed = args.seed
    config.output_dir = args.output_dir
    config.verbose = args.verbose
    config.save_figures = bool(args.save_figures)
    config.plot_losses = args.plot_losses
    config.use_mean_slearner = args.two_stage_slearner
    config.use_mean_oosr = args.two_stage_oosr
    config.bridge_diagnostics = args.bridge_diagnostics
    config.kernel_diagnostics.enabled = args.kernel_diagnostics
    if args.kernel_diag_max_points is not None:
        config.kernel_diagnostics.max_points = args.kernel_diag_max_points
    if args.kernel_diag_bandwidth_multipliers is not None:
        config.kernel_diagnostics.bandwidth_multipliers = tuple(args.kernel_diag_bandwidth_multipliers)
    if args.kernel_diag_max_offdiag_values is not None:
        config.kernel_diagnostics.max_offdiag_values = args.kernel_diag_max_offdiag_values
    if args.kernel_diag_save_offdiag_values:
        config.kernel_diagnostics.save_offdiag_values = True
    if args.nonadditive_gamma is not None:
        config.data.nonadditive_confounding_gamma = args.nonadditive_gamma
    if args.nonadditive_eta is not None:
        config.data.nonadditive_confounding_eta = args.nonadditive_eta
    if args.nonadditive_scale_ratio is not None:
        config.data.nonadditive_confounding_scale_ratio = args.nonadditive_scale_ratio
    if args.alpha_nonadditivity is not None:
        config.data.alpha_nonadditivity = args.alpha_nonadditivity
    if args.shifted_confounder_treatment_scale is not None:
        config.data.shifted_confounder_treatment_scale = args.shifted_confounder_treatment_scale
    if args.shifted_confounder_treatment_slope is not None:
        config.data.shifted_confounder_treatment_slope = args.shifted_confounder_treatment_slope
    config.data.associational_kernel = bool(args.associational_kernel)
    if args.tcga_data_path is not None:
        config.data.tcga_data_path = args.tcga_data_path
    if args.tcga_processed_path is not None:
        config.data.tcga_processed_path = args.tcga_processed_path
    if args.tcga_metadata_path is not None:
        config.data.tcga_metadata_path = args.tcga_metadata_path
    if args.tcga_preprocess_seed is not None:
        config.data.tcga_preprocess_seed = args.tcga_preprocess_seed
    if args.tcga_param_seed is not None:
        config.data.tcga_param_seed = args.tcga_param_seed
    if args.tcga_top_genes is not None:
        config.data.tcga_top_genes = args.tcga_top_genes
    if args.tcga_block_size is not None:
        config.data.tcga_block_size = args.tcga_block_size
    if args.tcga_x_pca_dim is not None:
        config.data.tcga_x_pca_dim = args.tcga_x_pca_dim
    if args.tcga_z_pca_dim is not None:
        config.data.tcga_z_pca_dim = args.tcga_z_pca_dim
    if args.tcga_w_pca_dim is not None:
        config.data.tcga_w_pca_dim = args.tcga_w_pca_dim
    if args.tcga_proxy_noise_scale is not None:
        config.data.tcga_proxy_noise_scale = args.tcga_proxy_noise_scale
    if args.tcga_treatment_phi is not None:
        config.data.tcga_treatment_phi = args.tcga_treatment_phi
    if args.tcga_outcome_noise_scale is not None:
        config.data.tcga_outcome_noise_scale = args.tcga_outcome_noise_scale
    if args.tcga_nonadditive_u0_treatment_x0_scale is not None:
        config.data.tcga_nonadditive_u0_treatment_x0_scale = args.tcga_nonadditive_u0_treatment_x0_scale
    if args.tcga_normalization_pilot_size is not None:
        config.data.tcga_normalization_pilot_size = args.tcga_normalization_pilot_size
    if args.tcga_causal_mc_samples_per_x is not None:
        config.data.tcga_causal_mc_samples_per_x = args.tcga_causal_mc_samples_per_x
    if args.tcga_policy_value_mc_samples is not None:
        config.data.tcga_policy_value_mc_samples = args.tcga_policy_value_mc_samples
    if args.tcga_common_random_numbers is not None:
        config.data.tcga_common_random_numbers = bool(args.tcga_common_random_numbers)
    if args.tcga_sample_with_replacement is not None:
        config.data.tcga_sample_with_replacement = bool(args.tcga_sample_with_replacement)
    if args.outcome_noise_scale is not None:
        config.data.outcome_noise_scale = args.outcome_noise_scale
    if args.centered_penalty_strength is not None:
        config.data.centered_penalty_strength = args.centered_penalty_strength
    if args.additive_term_scale is not None:
        config.data.additive_term_scale = args.additive_term_scale
    if args.pmmr_ridge is not None:
        config.pmmr.ridge = args.pmmr_ridge
    if args.pmmr_awx_scale is not None:
        config.pmmr.awx_scale = args.pmmr_awx_scale
    if args.pmmr_azx_scale is not None:
        config.pmmr.azx_scale = args.pmmr_azx_scale
    if args.kpv_lambda1 is not None:
        config.kpv.lambda1 = args.kpv_lambda1
    if args.kpv_lambda2 is not None:
        config.kpv.lambda2 = args.kpv_lambda2
    config.kpv.optimize_lambda1 = bool(args.kpv_optimize_lambda1)
    config.kpv.optimize_lambda2 = bool(args.kpv_optimize_lambda2)
    config.kpv.standardize = bool(args.kpv_standardize)
    if args.kpv_lambda1_min is not None:
        config.kpv.lambda1_min = args.kpv_lambda1_min
    if args.kpv_lambda1_max is not None:
        config.kpv.lambda1_max = args.kpv_lambda1_max
    if args.kpv_n_lambda1_search is not None:
        config.kpv.n_lambda1_search = args.kpv_n_lambda1_search
    if args.kpv_lambda2_min is not None:
        config.kpv.lambda2_min = args.kpv_lambda2_min
    if args.kpv_lambda2_max is not None:
        config.kpv.lambda2_max = args.kpv_lambda2_max
    if args.kpv_n_lambda2_search is not None:
        config.kpv.n_lambda2_search = args.kpv_n_lambda2_search
    if args.kpv_kernel_variance_scale is not None:
        config.kpv.kernel_variance_scale = args.kpv_kernel_variance_scale
    if args.kpv_split_ratio is not None:
        config.kpv.split_ratio = args.kpv_split_ratio
    if args.kpv_max_train_size is not None:
        config.kpv.max_train_size = args.kpv_max_train_size
    if args.kpv_prediction_batch_size is not None:
        config.kpv.prediction_batch_size = args.kpv_prediction_batch_size
    if args.dfpv_lam1 is not None:
        config.dfpv.lam1 = args.dfpv_lam1
    if args.dfpv_lam2 is not None:
        config.dfpv.lam2 = args.dfpv_lam2
    if args.dfpv_feature_dim is not None:
        config.dfpv.feature_dim = args.dfpv_feature_dim
    if args.dfpv_hidden_dims is not None:
        config.dfpv.hidden_dims = tuple(args.dfpv_hidden_dims)
    if args.dfpv_activation is not None:
        config.dfpv.activation = args.dfpv_activation
    if args.dfpv_lr is not None:
        config.dfpv.lr = args.dfpv_lr
    if args.dfpv_weight_decay is not None:
        config.dfpv.weight_decay = args.dfpv_weight_decay
    if args.dfpv_prediction_batch_size is not None:
        config.dfpv.prediction_batch_size = args.dfpv_prediction_batch_size
    if args.dr_kpv_kpv_lambda1 is not None:
        config.dr_kpv.kpv_lambda1 = args.dr_kpv_kpv_lambda1
    if args.dr_kpv_kpv_lambda2 is not None:
        config.dr_kpv.kpv_lambda2 = args.dr_kpv_kpv_lambda2
    if args.dr_kpv_kap_lambda is not None:
        config.dr_kpv.kap_lambda = args.dr_kpv_kap_lambda
    if args.dr_kpv_kap_eta is not None:
        config.dr_kpv.kap_eta = args.dr_kpv_kap_eta
    if args.dr_kpv_kap_lambda2 is not None:
        config.dr_kpv.kap_lambda2 = args.dr_kpv_kap_lambda2
    if args.dr_kpv_kap_zeta is not None:
        config.dr_kpv.kap_zeta = args.dr_kpv_kap_zeta
    if args.dr_kpv_lambda_dr is not None:
        config.dr_kpv.lambda_dr = args.dr_kpv_lambda_dr
    if args.dr_kpv_kernel_variance_scale is not None:
        config.dr_kpv.kernel_variance_scale = args.dr_kpv_kernel_variance_scale
    if args.dr_kpv_stage1_perc is not None:
        config.dr_kpv.stage1_perc = args.dr_kpv_stage1_perc
    if args.dr_kpv_max_train_size is not None:
        config.dr_kpv.max_train_size = args.dr_kpv_max_train_size
    if args.dr_kpv_prediction_batch_size is not None:
        config.dr_kpv.prediction_batch_size = args.dr_kpv_prediction_batch_size
    if args.dr_kpv_optimize_internal is not None:
        config.dr_kpv.optimize_internal = bool(args.dr_kpv_optimize_internal)
    if args.dr_kpv_standardize is not None:
        config.dr_kpv.standardize = bool(args.dr_kpv_standardize)
    dr_pmmr_configs = (config.dr_pmmr, config.dr_pmmr_conditional)
    if args.dr_pmmr_pmmr_lambda is not None:
        for cfg in dr_pmmr_configs:
            cfg.pmmr_lambda = args.dr_pmmr_pmmr_lambda
    if args.dr_pmmr_kap_lambda is not None:
        for cfg in dr_pmmr_configs:
            cfg.kap_lambda = args.dr_pmmr_kap_lambda
    if args.dr_pmmr_kap_eta is not None:
        for cfg in dr_pmmr_configs:
            cfg.kap_eta = args.dr_pmmr_kap_eta
    if args.dr_pmmr_kap_lambda2 is not None:
        for cfg in dr_pmmr_configs:
            cfg.kap_lambda2 = args.dr_pmmr_kap_lambda2
    if args.dr_pmmr_kap_zeta is not None:
        for cfg in dr_pmmr_configs:
            cfg.kap_zeta = args.dr_pmmr_kap_zeta
    if args.dr_pmmr_lambda_dr is not None:
        for cfg in dr_pmmr_configs:
            cfg.lambda_dr = args.dr_pmmr_lambda_dr
    if args.dr_pmmr_kernel_variance_scale is not None:
        for cfg in dr_pmmr_configs:
            cfg.kernel_variance_scale = args.dr_pmmr_kernel_variance_scale
    if args.dr_pmmr_stage1_perc is not None:
        for cfg in dr_pmmr_configs:
            cfg.stage1_perc = args.dr_pmmr_stage1_perc
    if args.dr_pmmr_max_train_size is not None:
        for cfg in dr_pmmr_configs:
            cfg.max_train_size = args.dr_pmmr_max_train_size
    if args.dr_pmmr_prediction_batch_size is not None:
        for cfg in dr_pmmr_configs:
            cfg.prediction_batch_size = args.dr_pmmr_prediction_batch_size
    if args.dr_pmmr_optimize_internal is not None:
        for cfg in dr_pmmr_configs:
            cfg.optimize_internal = bool(args.dr_pmmr_optimize_internal)
    if args.dr_pmmr_standardize is not None:
        for cfg in dr_pmmr_configs:
            cfg.standardize = bool(args.dr_pmmr_standardize)
    for nmmr_cfg in all_nmmr_configs(config):
        _apply_nmmr_cfg_overrides(
            nmmr_cfg,
            lr=args.nmmr_lr,
            azx_scale=args.nmmr_azx_scale,
            activation=args.nmmr_activation,
            hidden_dims=args.nmmr_hidden_dims,
            weight_decay=args.nmmr_weight_decay,
            epochs=args.nmmr_epochs,
            min_epochs=args.nmmr_min_epochs,
            early_stop_patience=args.nmmr_early_stop_patience,
            early_stop_min_delta=args.nmmr_early_stop_min_delta,
            val_fraction=args.nmmr_val_fraction,
            moment_weight_power=args.nmmr_moment_weight_power,
            deactivate_rolloff=args.deactivate_rolloff,
            product_kernel=args.nmmr_product_kernel,
            quadratic_treatment=args.quadratic_treatment,
            a_scale=args.nmmr_a_scale,
            z_scale=args.nmmr_z_scale,
            x_scale=args.nmmr_x_scale,
            bridge_all_data=args.bridge_all_data,
            batch_size=args.nmmr_batch_size,
            grad_clip_norm=args.nmmr_grad_clip_norm,
        )
    _apply_nmmr_cfg_overrides(
        config.nmmr_u,
        lr=args.nmmr_u_lr,
        azx_scale=args.nmmr_u_azx_scale,
        activation=args.nmmr_u_activation,
        hidden_dims=args.nmmr_u_hidden_dims,
        weight_decay=args.nmmr_u_weight_decay,
        epochs=args.nmmr_u_epochs,
        min_epochs=args.nmmr_u_min_epochs,
        early_stop_patience=args.nmmr_u_early_stop_patience,
        early_stop_min_delta=args.nmmr_u_early_stop_min_delta,
        val_fraction=args.nmmr_u_val_fraction,
        moment_weight_power=args.nmmr_u_moment_weight_power,
        deactivate_rolloff=args.nmmr_u_deactivate_rolloff,
        product_kernel=args.nmmr_u_product_kernel,
        quadratic_treatment=args.nmmr_u_quadratic_treatment,
        a_scale=args.nmmr_u_a_scale,
        z_scale=args.nmmr_u_z_scale,
        x_scale=args.nmmr_u_x_scale,
        bridge_all_data=args.nmmr_u_bridge_all_data,
        batch_size=args.nmmr_u_batch_size,
        grad_clip_norm=args.nmmr_u_grad_clip_norm,
    )
    _apply_nmmr_cfg_overrides(
        config.nmmr_v,
        lr=args.nmmr_v_lr,
        azx_scale=args.nmmr_v_azx_scale,
        activation=args.nmmr_v_activation,
        hidden_dims=args.nmmr_v_hidden_dims,
        weight_decay=args.nmmr_v_weight_decay,
        epochs=args.nmmr_v_epochs,
        min_epochs=args.nmmr_v_min_epochs,
        early_stop_patience=args.nmmr_v_early_stop_patience,
        early_stop_min_delta=args.nmmr_v_early_stop_min_delta,
        val_fraction=args.nmmr_v_val_fraction,
        moment_weight_power=args.nmmr_v_moment_weight_power,
        deactivate_rolloff=args.nmmr_v_deactivate_rolloff,
        product_kernel=args.nmmr_v_product_kernel,
        quadratic_treatment=args.nmmr_v_quadratic_treatment,
        a_scale=args.nmmr_v_a_scale,
        z_scale=args.nmmr_v_z_scale,
        x_scale=args.nmmr_v_x_scale,
        bridge_all_data=args.nmmr_v_bridge_all_data,
        batch_size=args.nmmr_v_batch_size,
        grad_clip_norm=args.nmmr_v_grad_clip_norm,
    )
    if args.mean_lr is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.lr = args.mean_lr
    if args.mean_batch_size is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.batch_size = args.mean_batch_size
    for mean_cfg in all_mean_configs(config):
        mean_cfg.early_stopping_mode = "v2" if args.mean_early_stopping_v2 else "standard"
    if args.mean_early_stopping_v2_alpha is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.early_stopping_v2_alpha = args.mean_early_stopping_v2_alpha
    if args.mean_early_stopping_v2_patience is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.early_stopping_v2_patience = args.mean_early_stopping_v2_patience
    for mean_cfg in all_mean_configs(config):
        mean_cfg.use_treatment_grid = args.treatment_grid
    for weighting_cfg in all_weighting_configs(config):
        weighting_cfg.normalization_mode = args.normalization_mode
        weighting_cfg.stabilized = args.stabilized
        weighting_cfg.root_weights = args.root_weights
        weighting_cfg.weight_mean_regressor = args.weight_mean_regressor
        weighting_cfg.warm_start = args.warm_start
        weighting_cfg.truncated_kernel_weights = args.truncated_kernel_weights
    if args.weight_early_stop_patience is not None:
        for weighting_cfg in all_weighting_configs(config):
            weighting_cfg.early_stop_patience = args.weight_early_stop_patience
    if args.weight_min_improvement is not None:
        for weighting_cfg in all_weighting_configs(config):
            weighting_cfg.min_improvement = args.weight_min_improvement
    if args.tau is not None:
        for weighting_cfg in all_weighting_configs(config):
            weighting_cfg.tau = args.tau
    if args.lam is not None:
        for weighting_cfg in all_weighting_configs(config):
            weighting_cfg.lam = args.lam
    if args.n_rounds is not None:
        for weighting_cfg in all_weighting_configs(config):
            weighting_cfg.n_rounds = args.n_rounds
    if args.pmmr_rounds is not None:
        config.weighting_pmmr.n_rounds = args.pmmr_rounds
        config.weighting_local_pmmr.n_rounds = args.pmmr_rounds
    if args.pmmr_tau is not None:
        config.weighting_pmmr.tau = args.pmmr_tau
        config.weighting_local_pmmr.tau = args.pmmr_tau
    if args.pmmr_lam is not None:
        config.weighting_pmmr.lam = args.pmmr_lam
        config.weighting_local_pmmr.lam = args.pmmr_lam
    if args.nmmr_u_rounds is not None:
        config.weighting_nmmr_u.n_rounds = args.nmmr_u_rounds
    if args.nmmr_u_tau is not None:
        config.weighting_nmmr_u.tau = args.nmmr_u_tau
    if args.nmmr_u_lam is not None:
        config.weighting_nmmr_u.lam = args.nmmr_u_lam
    if args.nmmr_v_rounds is not None:
        config.weighting_nmmr_v.n_rounds = args.nmmr_v_rounds
    if args.nmmr_v_tau is not None:
        config.weighting_nmmr_v.tau = args.nmmr_v_tau
    if args.nmmr_v_lam is not None:
        config.weighting_nmmr_v.lam = args.nmmr_v_lam
    if args.kpv_rounds is not None:
        config.weighting_kpv.n_rounds = args.kpv_rounds
        config.weighting_local_kpv.n_rounds = args.kpv_rounds
    if args.kpv_tau is not None:
        config.weighting_kpv.tau = args.kpv_tau
        config.weighting_local_kpv.tau = args.kpv_tau
    if args.kpv_lam is not None:
        config.weighting_kpv.lam = args.kpv_lam
        config.weighting_local_kpv.lam = args.kpv_lam
    if args.dfpv_rounds is not None:
        config.weighting_dfpv.n_rounds = args.dfpv_rounds
        config.weighting_local_dfpv.n_rounds = args.dfpv_rounds
    if args.dfpv_tau is not None:
        config.weighting_dfpv.tau = args.dfpv_tau
        config.weighting_local_dfpv.tau = args.dfpv_tau
    if args.dfpv_lam is not None:
        config.weighting_dfpv.lam = args.dfpv_lam
        config.weighting_local_dfpv.lam = args.dfpv_lam
    if args.naive_oosr_rounds is not None:
        config.weighting_naive_oosr.n_rounds = args.naive_oosr_rounds
    if args.naive_oosr_tau is not None:
        config.weighting_naive_oosr.tau = args.naive_oosr_tau
    if args.naive_oosr_lam is not None:
        config.weighting_naive_oosr.lam = args.naive_oosr_lam
    if args.oracle_naive_oosr_rounds is not None:
        config.weighting_oracle_naive_oosr.n_rounds = args.oracle_naive_oosr_rounds
    if args.oracle_naive_oosr_tau is not None:
        config.weighting_oracle_naive_oosr.tau = args.oracle_naive_oosr_tau
    if args.oracle_naive_oosr_lam is not None:
        config.weighting_oracle_naive_oosr.lam = args.oracle_naive_oosr_lam


def _apply_smoke(config: ExperimentConfig) -> None:
    config.data.n_train = 150
    config.data.n_test = 250
    config.data.policy_grid_size = 41
    config.data.causal_mc_samples = min(int(config.data.causal_mc_samples), 128)
    config.data.associational_mc_samples = min(int(config.data.associational_mc_samples), 500)
    config.data.tcga_normalization_pilot_size = min(int(config.data.tcga_normalization_pilot_size), 500)
    config.data.tcga_causal_mc_samples_per_x = min(int(config.data.tcga_causal_mc_samples_per_x), 32)
    config.data.tcga_policy_value_mc_samples = min(int(config.data.tcga_policy_value_mc_samples), 16)
    for mean_cfg in all_mean_configs(config):
        mean_cfg.treatment_grid_size = 11
    config.propensity_flow.epochs = 5
    config.propensity_flow.min_epochs = 2
    config.propensity_flow.early_stop_patience = 2
    config.w_flow.epochs = 5
    config.w_flow.min_epochs = 2
    config.w_flow.early_stop_patience = 2
    for mean_cfg in all_mean_configs(config):
        mean_cfg.epochs = 10
        mean_cfg.min_epochs = 3
        mean_cfg.early_stop_patience = 2
        mean_cfg.crossfit_folds = min(mean_cfg.crossfit_folds, 2)
    for nmmr_cfg in all_nmmr_configs(config):
        nmmr_cfg.epochs = 15
        nmmr_cfg.min_epochs = 4
        nmmr_cfg.early_stop_patience = 3
    config.kpv.max_train_size = 3000
    config.kpv.n_lambda1_search = min(config.kpv.n_lambda1_search, 3)
    config.kpv.n_lambda2_search = min(config.kpv.n_lambda2_search, 3)
    config.dfpv.epochs = 8
    config.dfpv.min_epochs = 2
    config.dfpv.early_stop_patience = 2
    config.dfpv.hidden_dims = (8,)
    config.dfpv.feature_dim = 4
    config.dfpv.max_train_size = 100
    config.dfpv.prediction_batch_size = 64
    config.dr_kpv.max_train_size = 100
    config.dr_kpv.prediction_batch_size = 64
    config.dr_kpv.regularization_grid_points = min(config.dr_kpv.regularization_grid_points, 3)
    config.dr_pmmr.max_train_size = 100
    config.dr_pmmr.prediction_batch_size = 64
    config.dr_pmmr.regularization_grid_points = min(config.dr_pmmr.regularization_grid_points, 3)
    config.dr_pmmr_conditional.max_train_size = 100
    config.dr_pmmr_conditional.prediction_batch_size = 64
    config.dr_pmmr_conditional.regularization_grid_points = min(config.dr_pmmr_conditional.regularization_grid_points, 3)
    config.s_learner.epochs = 15
    config.s_learner.min_epochs = 4
    config.s_learner.early_stop_patience = 3
    config.decaflow.epochs = 6
    config.decaflow.min_epochs = 2
    config.decaflow.early_stop_patience = 2
    config.decaflow.encoder_hidden_features = (8,)
    config.decaflow.decoder_hidden_features = (8,)
    config.decaflow.conditioner_hidden_features = (4,)
    config.decaflow.batch_size = 64
    config.decaflow.prediction_batch_size = 64
    config.decaflow.interventional_mc_samples = 8
    for weighting_cfg in all_weighting_configs(config):
        weighting_cfg.n_rounds = min(weighting_cfg.n_rounds, 2)
    config.weighting.early_stop_patience = 1


def _apply_oracle_bridge_toy_training(config: ExperimentConfig) -> None:
    if config.data.version != "oracle_bridge_toy":
        return
    config.propensity_flow.epochs = 600
    config.propensity_flow.min_epochs = 100
    config.propensity_flow.early_stop_patience = 80
    for mean_cfg in all_mean_configs(config):
        mean_cfg.epochs = 1200
        mean_cfg.min_epochs = 150
        mean_cfg.early_stop_patience = 100


def _apply_epoch_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.propensity_epochs is not None:
        config.propensity_flow.epochs = args.propensity_epochs
    if args.propensity_min_epochs is not None:
        config.propensity_flow.min_epochs = args.propensity_min_epochs
    if args.propensity_patience is not None:
        config.propensity_flow.early_stop_patience = args.propensity_patience
    if args.mean_epochs is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.epochs = args.mean_epochs
    if args.mean_min_epochs is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.min_epochs = args.mean_min_epochs
    if args.mean_patience is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.early_stop_patience = args.mean_patience
    if args.disable_pmmr_mean_early_stopping:
        # Keep PMMR mean head active until end of schedule.
        config.mean_pmmr.early_stop_patience = max(
            int(config.mean_pmmr.epochs) + 1,
            int(config.mean_pmmr.min_epochs) + 1,
        )
    if args.disable_nmmr_mean_early_stopping:
        # Keep NMMR mean heads active until end of schedule.
        config.mean_nmmr_u.early_stop_patience = max(
            int(config.mean_nmmr_u.epochs) + 1,
            int(config.mean_nmmr_u.min_epochs) + 1,
        )
        config.mean_nmmr_v.early_stop_patience = max(
            int(config.mean_nmmr_v.epochs) + 1,
            int(config.mean_nmmr_v.min_epochs) + 1,
        )
    if args.disable_kpv_mean_early_stopping:
        # Keep KPV mean head active until end of schedule.
        config.mean_kpv.early_stop_patience = max(
            int(config.mean_kpv.epochs) + 1,
            int(config.mean_kpv.min_epochs) + 1,
        )
    if args.disable_dfpv_mean_early_stopping:
        # Keep DFPV mean head active until end of schedule.
        config.mean_dfpv.early_stop_patience = max(
            int(config.mean_dfpv.epochs) + 1,
            int(config.mean_dfpv.min_epochs) + 1,
        )
    if args.mean_crossfit_folds is not None:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.crossfit_folds = args.mean_crossfit_folds
    if args.no_mean_crossfit_shuffle:
        for mean_cfg in all_mean_configs(config):
            mean_cfg.crossfit_shuffle = False


def main() -> None:
    args = parse_args()
    selected_models = _selected_models(args.models)
    if args.best_model:
        if args.best_config_path is None or not args.best_config_path.exists():
            raise FileNotFoundError(
                "Best-model bundle not found. Run scripts/search_hparams.py first and pass --best-config-path."
            )
        config, model_configs = load_bundle(args.best_config_path)
        _apply_runtime_overrides(config, args)
        for model_config in model_configs.values():
            _apply_runtime_overrides(model_config, args)
        if args.smoke:
            _apply_smoke(config)
            for model_config in model_configs.values():
                _apply_smoke(model_config)
        else:
            _apply_oracle_bridge_toy_training(config)
            for model_config in model_configs.values():
                _apply_oracle_bridge_toy_training(model_config)
        _apply_epoch_overrides(config, args)
        for model_config in model_configs.values():
            _apply_epoch_overrides(model_config, args)
        # Comment out this line to disable the Wu-specific DeCaFlow adjacency override.
        _apply_decaflow_wu_true_adjacency(config)
        for model_config in model_configs.values():
            # Comment out this line to disable the Wu-specific DeCaFlow adjacency override.
            _apply_decaflow_wu_true_adjacency(model_config)
        available_models = list(model_configs.keys())
        if args.models.strip().lower() != "all":
            missing = sorted(set(selected_models) - set(available_models))
            if missing:
                raise ValueError(f"Requested models are not present in the best-model bundle: {missing}")
            available_models = selected_models
        run_experiment(config, selected_models=available_models, model_configs=model_configs)
        return

    config = ExperimentConfig()
    _apply_runtime_overrides(config, args)
    config.data.version = args.dataset_version
    config.data.structure_type = args.structure_type
    config.data.n_train = args.n_train
    if args.n_val is not None:
        config.data.n_val = args.n_val
    config.data.n_test = args.n_test
    for mean_cfg in all_mean_configs(config):
        mean_cfg.method = args.mean_method
    if args.smoke:
        _apply_smoke(config)
    else:
        _apply_oracle_bridge_toy_training(config)
    _apply_epoch_overrides(config, args)
    # Comment out this line to disable the Wu-specific DeCaFlow adjacency override.
    _apply_decaflow_wu_true_adjacency(config)
    run_experiment(config, selected_models=selected_models)


if __name__ == "__main__":
    main()
