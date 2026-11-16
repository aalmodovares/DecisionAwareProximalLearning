from __future__ import annotations

import numpy as np

from src.config import DataConfig
from src.data.simulator import FactualSample
from src.data.tcga_preprocessing import load_tcga_feature_cache


SEMISYNTHETIC_TCGA_VERSION = "semisynthetic_tcga"
SEMISYNTHETIC_TCGA_STRUCTURES = {"v1"}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.where(
        value >= 0.0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def _softplus(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


class SemisyntheticTCGASimulator:
    """Proximalized TCGA semi-synthetic DGP with the Wu simulator interface."""

    def __init__(self, config: DataConfig):
        if config.version != SEMISYNTHETIC_TCGA_VERSION:
            raise ValueError(f"SemisyntheticTCGASimulator requires version={SEMISYNTHETIC_TCGA_VERSION!r}.")
        if config.structure_type not in SEMISYNTHETIC_TCGA_STRUCTURES:
            raise ValueError(f"Unknown TCGA semi-synthetic structure_type: {config.structure_type!r}.")
        self.config = config
        self.config.dim_x = 5
        self.config.dim_z = 3
        self.config.dim_w = 3

        self.x_pool, self.ztilde_pool, self.wtilde_pool = load_tcga_feature_cache(config)
        self.n_pool = int(self.x_pool.shape[0])
        if self.n_pool <= 0:
            raise ValueError("TCGA feature cache is empty.")
        self.support = self._resolve_support()
        self.policy_grid = np.linspace(self.support[0], self.support[1], self.config.policy_grid_size)
        self._init_parameters()
        self._init_normalization_constants()

    def _resolve_support(self) -> tuple[float, float]:
        if self.config.support_lower is not None and self.config.support_upper is not None:
            return float(self.config.support_lower), float(self.config.support_upper)
        return 0.0, 1.0

    def _init_parameters(self) -> None:
        sqrt2 = np.sqrt(2.0)
        sqrt3 = np.sqrt(3.0)
        sqrt5 = np.sqrt(5.0)
        self.gamma_x = (0.6 / sqrt5) * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        self.gamma_u = (0.8 / sqrt2) * np.array([1.0, -1.0])
        self.gamma0 = 0.0
        self.eta_x = (0.4 / sqrt5) * np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        self.eta_u = (0.8 / sqrt2) * np.array([1.0, 1.0])
        self.eta_z = (0.6 / sqrt3) * np.array([1.0, -1.0, 1.0])
        self.eta0 = 0.0
        self.theta_x = (0.25 / sqrt5) * np.array([1.0, -1.0, 1.0, 1.0, -1.0])
        self.theta_u = (0.5 / sqrt2) * np.array([1.0, -1.0])
        self.omega_x = (0.2 / sqrt5) * np.array([1.0, 2.0, -1.0, 1.0, -2.0])
        self.omega_u = (0.3 / sqrt2) * np.array([1.0, 1.0])
        self.kappa_x = (0.3 / sqrt5) * np.array([1.0, -1.0, 0.5, 1.0, -0.5])
        self.kappa_u = (0.3 / sqrt2) * np.array([1.0, -1.0])
        self.kappa0 = 0.0
        self.lambda_w = (0.5 / sqrt3) * np.array([1.0, -1.0, 1.0])
        self.b_z = 0.8 * np.array([[1.0, 0.0], [0.0, 1.0], [1.0 / sqrt2, -1.0 / sqrt2]])
        self.b_w = 0.8 * np.array([[1.0, 0.0], [0.0, 1.0], [1.0 / sqrt2, 1.0 / sqrt2]])
        self.c_z = 0.2 * np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
        )
        self.c_w = 0.2 * np.array(
            [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.r_z = 0.5 * np.eye(3)
        self.r_w = 0.5 * np.eye(3)

    def _sample_u(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u1 = rng.normal(0.0, 1.0, size=n)
        u2 = rng.exponential(1.0, size=n) - 1.0
        return np.column_stack([u1, u2])

    def _sample_feature_indices(self, n: int, rng: np.random.Generator, *, exclude: set[int] | None = None) -> np.ndarray:
        exclude = exclude or set()
        available = np.array([idx for idx in range(self.n_pool) if idx not in exclude], dtype=int)
        if available.size == 0:
            raise ValueError("No TCGA feature rows available after exclusions.")
        replace = bool(self.config.tcga_sample_with_replacement) or n > available.size
        return rng.choice(available, size=n, replace=replace)

    def _init_normalization_constants(self) -> None:
        rng = np.random.default_rng(int(self.config.tcga_param_seed))
        n = min(max(int(self.config.tcga_normalization_pilot_size), 1), max(int(self.config.tcga_normalization_pilot_size), self.n_pool))
        idx = self._sample_feature_indices(n, rng)
        x = self.x_pool[idx]
        ztilde = self.ztilde_pool[idx]
        u = self._sample_u(n, rng)
        z, _ = self._generate_proxies(x, ztilde, self.wtilde_pool[idx], u, rng=rng, add_noise=True)
        s = self.gamma0 + x @ self.gamma_x + u @ self.gamma_u
        r = self.eta0 + x @ self.eta_x + u @ self.eta_u + z @ self.eta_z
        self.s_mean = float(np.mean(s))
        self.s_sd = float(np.std(s) + 1e-8)
        self.r_mean = float(np.mean(r))
        self.r_sd = float(np.std(r) + 1e-8)

    def _generate_proxies(
        self,
        x: np.ndarray,
        ztilde: np.ndarray,
        wtilde: np.ndarray,
        u: np.ndarray,
        *,
        rng: np.random.Generator | None,
        add_noise: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.tanh(u @ self.b_z.T + x @ self.c_z.T + ztilde @ self.r_z.T)
        w = np.tanh(u @ self.b_w.T + x @ self.c_w.T + wtilde @ self.r_w.T)
        if add_noise:
            if rng is None:
                raise ValueError("rng is required when add_noise=True.")
            noise = float(self.config.tcga_proxy_noise_scale)
            z = z + rng.normal(0.0, noise, size=z.shape)
            w = w + rng.normal(0.0, noise, size=w.shape)
        return z, w

    def _a_star(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        score = self.gamma0 + x @ self.gamma_x + u @ self.gamma_u
        score_bar = (score - self.s_mean) / self.s_sd
        return 0.15 + 0.70 * _sigmoid(score_bar)

    def _assignment_mean(self, x: np.ndarray, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        optimal = self._a_star(x, u)
        r = self.eta0 + x @ self.eta_x + u @ self.eta_u + z @ self.eta_z
        r_bar = (r - self.r_mean) / self.r_sd
        mean = 0.3 * optimal + 0.7 * _sigmoid(r_bar)
        return np.clip(mean, 0.02, 0.98)

    def _outcome_components(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = 5.0 + 5.0*_softplus(self.kappa0 + x @ self.kappa_x + u @ self.kappa_u)
        mu = x @ self.theta_x + u @ self.theta_u
        b = 0.5 * np.sin(x @ self.omega_x) + 0.5 * np.cos(u @ self.omega_u)
        return c, mu, b

    def _do_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
        latent_u: np.ndarray,
        eps_y: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        u = np.asarray(latent_u, dtype=float)
        c, mu, b = self._outcome_components(x, u)
        optimal = self._a_star(x, u)
        nonadditive = float(self.config.tcga_nonadditive_gamma) * u[:, 0] * treatment * x[:, 0]
        eps = np.asarray(eps_y, dtype=float)
        return (mu + b + w @ self.lambda_w + nonadditive - c * np.square(treatment - optimal) + eps) / 4

    def latent_outcome_confounding_term(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        latent_uz: np.ndarray,
        latent_uw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Outcome contribution induced by latent U, for diagnostic plots.

        The TCGA DGP has a two-dimensional latent confounder and its outcome
        effect is not purely additive: U also changes the curvature and the
        individual optimum. For diagnostics we report the contrast between the
        noiseless outcome at the supplied U and the noiseless outcome at U=0,
        keeping A and X fixed and setting the W contribution to zero.
        """
        u1 = np.asarray(latent_uz, dtype=float).reshape(-1, 1)
        if latent_uw is None:
            u2 = np.zeros_like(u1)
        else:
            u2 = np.asarray(latent_uw, dtype=float).reshape(-1, 1)
        u = np.concatenate([u1, u2], axis=1)
        x = np.asarray(covariates, dtype=float)
        treatment_arr = np.asarray(treatment, dtype=float).reshape(-1)
        zero_w = np.zeros((x.shape[0], self.config.dim_w), dtype=float)
        zero_u = np.zeros_like(u)
        with_u = self._do_outcome(treatment_arr, x, zero_w, u, eps_y=0.0)
        without_u = self._do_outcome(treatment_arr, x, zero_w, zero_u, eps_y=0.0)
        return with_u - without_u

    def true_bridge(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray | None:
        """Closed-form bridge target is not available for the TCGA DGP."""
        _ = treatment, outcome_proxy, covariates
        return None

    def _sample_rows_by_indices(self, idx: np.ndarray, rng: np.random.Generator) -> FactualSample:
        x = self.x_pool[idx]
        ztilde = self.ztilde_pool[idx]
        wtilde = self.wtilde_pool[idx]
        n = int(idx.shape[0])
        u = self._sample_u(n, rng)
        z, w = self._generate_proxies(x, ztilde, wtilde, u, rng=rng, add_noise=True)
        optimal = self._a_star(x, u)
        assignment_mean = self._assignment_mean(x, u, z)
        phi = float(self.config.tcga_treatment_phi)
        treatment = rng.beta(phi * assignment_mean, phi * (1.0 - assignment_mean))
        eps_y = rng.normal(0.0, float(self.config.tcga_outcome_noise_scale), size=n)
        y = self._do_outcome(treatment, x, w, u, eps_y=eps_y)
        return FactualSample(
            treatment=treatment[:, None],
            treatment_proxy=z,
            outcome_proxy=w,
            covariates=x,
            outcome=y[:, None],
            optimal_treatment=optimal[:, None],
            assignment_mean=assignment_mean[:, None],
            latent_uz=u[:, [0]],
            latent_uw=u[:, [1]],
        )

    def sample_factual(self, n: int, seed: int) -> FactualSample:
        rng = np.random.default_rng(seed)
        idx = self._sample_feature_indices(int(n), rng)
        return self._sample_rows_by_indices(idx, rng)

    def sample_train_test(self, seed: int) -> tuple[FactualSample, FactualSample]:
        rng = np.random.default_rng(seed)
        n_train = int(self.config.n_train)
        n_test = int(self.config.n_test)
        total = n_train + n_test
        if bool(self.config.tcga_sample_with_replacement) or total > self.n_pool:
            train_idx = self._sample_feature_indices(n_train, rng)
            test_idx = self._sample_feature_indices(n_test, rng)
        else:
            idx = rng.choice(np.arange(self.n_pool, dtype=int), size=total, replace=False)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
        return self._sample_rows_by_indices(train_idx, rng), self._sample_rows_by_indices(test_idx, rng)

    def proxy_conditioned_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        rng = np.random.default_rng(int(self.config.causal_mc_seed) + 123)
        n = x.shape[0]
        n_mc = max(1, int(self.config.tcga_policy_value_mc_samples))
        total = np.zeros(n, dtype=float)
        for _ in range(n_mc):
            u = self._sample_u(n, rng)
            total += self._do_outcome(treatment, x, w, u, eps_y=0.0)
        return total / float(n_mc)

    def oracle_conditioned_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
        latent_uz: np.ndarray,
        latent_uw: np.ndarray | None = None,
    ) -> np.ndarray:
        u1 = np.asarray(latent_uz, dtype=float).reshape(-1, 1)
        if latent_uw is None:
            u2 = np.zeros_like(u1)
        else:
            u2 = np.asarray(latent_uw, dtype=float).reshape(-1, 1)
        u = np.concatenate([u1, u2], axis=1)
        return self._do_outcome(treatment, covariates, outcome_proxy, u, eps_y=0.0)

    def _causal_mean_for_treatment_x(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        *,
        n_mc_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        n = x.shape[0]
        total = np.zeros(n, dtype=float)
        bsz = max(1, int(self.config.causal_mc_batch_size))
        used = 0
        while used < n_mc_samples:
            batch = min(bsz, n_mc_samples - used)
            x_rep = np.repeat(x, batch, axis=0)
            treatment_rep = np.repeat(treatment, batch)
            idx = self._sample_feature_indices(n * batch, rng)
            u = self._sample_u(n * batch, rng)
            _, w = self._generate_proxies(
                x_rep,
                self.ztilde_pool[idx],
                self.wtilde_pool[idx],
                u,
                rng=rng,
                add_noise=True,
            )
            y = self._do_outcome(treatment_rep, x_rep, w, u, eps_y=0.0).reshape(n, batch)
            total += np.sum(y, axis=1)
            used += batch
        return total / float(n_mc_samples)

    def _sample_causal_latent_batch(
        self,
        covariates: np.ndarray,
        mc_batch: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(covariates, dtype=float)
        n = x.shape[0]
        x_rep = np.repeat(x, int(mc_batch), axis=0)
        idx = self._sample_feature_indices(n * int(mc_batch), rng)
        u = self._sample_u(n * int(mc_batch), rng)
        _, w = self._generate_proxies(
            x_rep,
            self.ztilde_pool[idx],
            self.wtilde_pool[idx],
            u,
            rng=rng,
            add_noise=True,
        )
        return x_rep, w, u

    def _causal_curve_given_x_common_random_numbers(
        self,
        x_row: np.ndarray,
        treatment_grid: np.ndarray,
        *,
        n_mc_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        curve_sum = np.zeros(treatment_grid.shape[0], dtype=float)
        bsz = max(1, int(self.config.causal_mc_batch_size))
        used = 0
        while used < n_mc_samples:
            batch = min(bsz, n_mc_samples - used)
            x_rep, w, u = self._sample_causal_latent_batch(x, batch, rng)
            for idx, a_val in enumerate(treatment_grid):
                treatment_rep = np.full(batch, float(a_val), dtype=float)
                curve_sum[idx] += float(np.sum(self._do_outcome(treatment_rep, x_rep, w, u, eps_y=0.0)))
            used += batch
        return curve_sum / float(n_mc_samples)

    def _average_treatment_effect_curve_common_random_numbers(
        self,
        covariates: np.ndarray,
        treatment_grid: np.ndarray,
        *,
        n_mc_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(covariates, dtype=float)
        n = x.shape[0]
        curve_sum = np.zeros(treatment_grid.shape[0], dtype=float)
        bsz = max(1, int(self.config.causal_mc_batch_size))
        used = 0
        while used < n_mc_samples:
            batch = min(bsz, n_mc_samples - used)
            x_rep, w, u = self._sample_causal_latent_batch(x, batch, rng)
            for idx, a_val in enumerate(treatment_grid):
                treatment_rep = np.full(n * batch, float(a_val), dtype=float)
                curve_sum[idx] += float(np.sum(self._do_outcome(treatment_rep, x_rep, w, u, eps_y=0.0)))
            used += batch
        return curve_sum / float(n * n_mc_samples)

    def causal_curve_given_x_mc(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = batch_size
        x_prime = np.asarray(x_row, dtype=float).reshape(-1)
        if x_prime.size != self.config.dim_x:
            raise ValueError(f"Expected x_row with dim {self.config.dim_x}, got {x_prime.size}.")
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        n_total = int(self.config.tcga_causal_mc_samples_per_x if n_mc_samples is None else n_mc_samples)
        rng = np.random.default_rng(self.config.causal_mc_seed if seed is None else seed)
        x = x_prime.reshape(1, -1)
        if bool(self.config.tcga_common_random_numbers):
            return treatment_grid, self._causal_curve_given_x_common_random_numbers(
                x_prime,
                treatment_grid,
                n_mc_samples=n_total,
                rng=rng,
            )
        curve = np.empty(treatment_grid.shape[0], dtype=float)
        for idx, a_val in enumerate(treatment_grid):
            curve[idx] = self._causal_mean_for_treatment_x(
                np.array([float(a_val)]),
                x,
                n_mc_samples=n_total,
                rng=rng,
            )[0]
        return treatment_grid, curve

    def mean_outcome(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(int(self.config.causal_mc_seed) + 211)
        return self._causal_mean_for_treatment_x(
            np.asarray(treatment, dtype=float).reshape(-1),
            np.asarray(covariates, dtype=float),
            n_mc_samples=max(1, int(self.config.tcga_policy_value_mc_samples)),
            rng=rng,
        )

    def average_treatment_effect_curve(
        self,
        covariates: np.ndarray,
        grid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        rng = np.random.default_rng(int(self.config.causal_mc_seed) + 307)
        curve = np.empty(treatment_grid.shape[0], dtype=float)
        n_mc = max(1, int(self.config.tcga_causal_mc_samples_per_x))
        if bool(self.config.tcga_common_random_numbers):
            curve = self._average_treatment_effect_curve_common_random_numbers(
                x,
                treatment_grid,
                n_mc_samples=n_mc,
                rng=rng,
            )
            structure_curve = np.full_like(curve, np.nan, dtype=float)
            return treatment_grid, curve, structure_curve
        for idx, a_val in enumerate(treatment_grid):
            curve[idx] = float(
                np.mean(
                    self._causal_mean_for_treatment_x(
                        np.full(x.shape[0], float(a_val), dtype=float),
                        x,
                        n_mc_samples=n_mc,
                        rng=rng,
                    )
                )
            )
        structure_curve = np.full_like(curve, np.nan, dtype=float)
        return treatment_grid, curve, structure_curve

    def optimal_treatment(self, covariates: np.ndarray) -> np.ndarray:
        x = np.asarray(covariates, dtype=float)
        out = np.empty(x.shape[0], dtype=float)
        for idx, row in enumerate(x):
            grid, curve = self.causal_curve_given_x_mc(
                row,
                grid=self.policy_grid,
                n_mc_samples=max(1, int(self.config.tcga_policy_value_mc_samples)),
                seed=int(self.config.causal_mc_seed) + 1009 + idx,
            )
            out[idx] = grid[int(np.argmax(curve))]
        return out

    def policy_value(self, policy_treatment: np.ndarray, covariates: np.ndarray) -> float:
        rng = np.random.default_rng(int(self.config.causal_mc_seed) + 401)
        values = self._causal_mean_for_treatment_x(
            np.asarray(policy_treatment, dtype=float).reshape(-1),
            np.asarray(covariates, dtype=float),
            n_mc_samples=max(1, int(self.config.tcga_policy_value_mc_samples)),
            rng=rng,
        )
        return float(np.mean(values))

    def _sample_observational_given_x(
        self,
        x_prime: np.ndarray,
        n_total: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.repeat(np.asarray(x_prime, dtype=float).reshape(1, -1), int(n_total), axis=0)
        idx = self._sample_feature_indices(int(n_total), rng)
        u = self._sample_u(int(n_total), rng)
        z, w = self._generate_proxies(x, self.ztilde_pool[idx], self.wtilde_pool[idx], u, rng=rng, add_noise=True)
        mean = self._assignment_mean(x, u, z)
        phi = float(self.config.tcga_treatment_phi)
        treatment = rng.beta(phi * mean, phi * (1.0 - mean))
        eps_y = rng.normal(0.0, float(self.config.tcga_outcome_noise_scale), size=int(n_total))
        outcome = self._do_outcome(treatment, x, w, u, eps_y=eps_y)
        return treatment, outcome

    def associational_curve_given_x_binned(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        min_bin_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        n_total = int(self.config.associational_mc_samples if n_mc_samples is None else n_mc_samples)
        min_count = int(self.config.associational_min_bin_count if min_bin_count is None else min_bin_count)
        step = float(np.median(np.diff(np.sort(treatment_grid)))) if treatment_grid.size > 1 else 0.05
        epsilon = 0.5 * max(step, 1e-8)
        rng = np.random.default_rng((self.config.causal_mc_seed + 911) if seed is None else seed)
        treatment_obs, outcome_obs = self._sample_observational_given_x(np.asarray(x_row), n_total, rng)
        curve = np.empty(treatment_grid.shape[0], dtype=float)
        counts = np.zeros(treatment_grid.shape[0], dtype=int)
        fallback_t = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        fallback_y = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        for idx, a_val in enumerate(treatment_grid):
            mask = np.abs(treatment_obs - float(a_val)) <= epsilon
            counts[idx] = int(np.sum(mask))
            if counts[idx] > 0:
                curve[idx] = float(np.mean(outcome_obs[mask]))
            else:
                nearest = int(np.argmin(np.abs(treatment_obs - float(a_val))))
                curve[idx] = float(outcome_obs[nearest])
                fallback_t[idx] = float(treatment_obs[nearest])
                fallback_y[idx] = float(outcome_obs[nearest])
        empty = counts == 0
        low = (counts > 0) & (counts < min_count)
        return treatment_grid, curve, counts, low, empty, fallback_t, fallback_y

    def associational_curve_given_x_kernel(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        n_total = int(self.config.associational_mc_samples if n_mc_samples is None else n_mc_samples)
        min_count = int(self.config.associational_min_bin_count)
        step = float(np.median(np.diff(np.sort(treatment_grid)))) if treatment_grid.size > 1 else 0.05
        bandwidth = max(step, 1e-8)
        rng = np.random.default_rng((self.config.causal_mc_seed + 911) if seed is None else seed)
        treatment_obs, outcome_obs = self._sample_observational_given_x(np.asarray(x_row), n_total, rng)
        curve = np.empty(treatment_grid.shape[0], dtype=float)
        counts = np.full(treatment_grid.shape[0], n_total, dtype=int)
        low = np.zeros(treatment_grid.shape[0], dtype=bool)
        empty = np.zeros(treatment_grid.shape[0], dtype=bool)
        fallback_t = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        fallback_y = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        for idx, a_val in enumerate(treatment_grid):
            z = (treatment_obs - float(a_val)) / bandwidth
            weights = np.exp(-0.5 * np.square(z))
            sum_w = float(np.sum(weights))
            if sum_w <= 1e-12:
                nearest = int(np.argmin(np.abs(treatment_obs - float(a_val))))
                curve[idx] = float(outcome_obs[nearest])
                empty[idx] = True
                fallback_t[idx] = float(treatment_obs[nearest])
                fallback_y[idx] = float(outcome_obs[nearest])
            else:
                curve[idx] = float(np.sum(weights * outcome_obs) / sum_w)
                ess = (sum_w * sum_w) / max(float(np.sum(np.square(weights))), 1e-12)
                counts[idx] = int(max(1.0, round(ess)))
                low[idx] = bool(ess < float(min_count))
        return treatment_grid, curve, counts, low, empty, fallback_t, fallback_y

    def associational_curve_given_x(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        method: str | None = None,
        n_mc_samples: int | None = None,
        min_bin_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        chosen = "kernel" if (self.config.associational_kernel if method is None else str(method).lower() == "kernel") else "binned"
        if chosen == "kernel":
            result = self.associational_curve_given_x_kernel(x_row, grid=grid, n_mc_samples=n_mc_samples, seed=seed)
        else:
            result = self.associational_curve_given_x_binned(
                x_row,
                grid=grid,
                n_mc_samples=n_mc_samples,
                min_bin_count=min_bin_count,
                seed=seed,
            )
        return (*result, chosen)

    def overlap_rate(self, sample: FactualSample) -> float:
        return float(
            np.mean(
                np.abs(sample.treatment.reshape(-1) - sample.optimal_treatment.reshape(-1)) <= self.config.local_eval_eps
            )
        )
