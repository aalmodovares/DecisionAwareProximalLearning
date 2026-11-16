from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.sparse import diags

from src.config import DataConfig


NONADDITIVE_WU_VERSIONS = {
    "wu2024_nonadditive",
    "wu2024_single_peaked_heterogeneous_nonadditive",
    "wu2024_single_peaked_heterogeneous_nonadditive_shifted",
}
ORIGINAL_WU_VERSIONS = {"wu2024_original", "wu2024_biased_assignment"} | NONADDITIVE_WU_VERSIONS
SATURATED_INTERACTION_VERSIONS = {
    "wu2024_saturated_interaction",
    "wu2024_saturated_interaction_biased_assignment",
}
ORACLE_BRIDGE_VERSIONS = {"oracle_bridge_toy"}
ALL_WU_VERSIONS = ORIGINAL_WU_VERSIONS | SATURATED_INTERACTION_VERSIONS | ORACLE_BRIDGE_VERSIONS
BIAS_VERSIONS = {"wu2024_biased_assignment", "wu2024_saturated_interaction_biased_assignment"}


def _lambda_fn(t: np.ndarray) -> np.ndarray:
    return (0.9 - 0.1) * np.exp(t) / (1.0 + np.exp(t)) + 0.1


def _theta(dim: int) -> np.ndarray:
    return np.array([1.0 / (k * k) for k in range(1, dim + 1)], dtype=float)


def _base_structure(a: np.ndarray, kind: str) -> np.ndarray:
    if kind == "quardratic":
        return 1.2 * a + np.square(a)
    if kind == "peaked":
        return 2.0 * (np.power(a, 4) / 600.0 + np.exp(-4.0 * np.square(a)) + a / 10.0 - 2.0) + 1.2 * a
    if kind == "sigmoid":
        return np.log(np.abs(16.0 * a - 8.0) + 1.0) * np.sign(a - 0.5) + 1.2 * a
    raise ValueError(f"Unknown structure type: {kind}")


@dataclass
class FactualSample:
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray
    optimal_treatment: np.ndarray
    assignment_mean: np.ndarray
    latent_uz: np.ndarray
    latent_uw: np.ndarray


class Wu2024Simulator:
    """Wu et al. (2024) high-dimensional DGP with bounded-treatment variants.

    The original public Wu generator defines
      Z = v_z + 0.25 * U_z
      W = v_w + 0.25 * U_w
      A_raw = Lambda(3 X theta_x + 3 Z theta_z) + 0.25 * U_w
      Y = structure(A) + 1.2(X theta_x + W theta_w) + A * X_0 + 0.25 * U_z

    OOSR assumes a bounded continuous treatment domain, so this simulator first
    estimates a sensible support from the original Wu raw-treatment process using
    pilot quantiles, then clips observational treatments and assignment means
    into that support. The policy grid, optimal treatment, and patient-surface
    plots all live on this same bounded region. For the saturated-interaction
    family, the support is estimated from that family's softened assignment
    mechanism, not from the original Wu one.

    Available treatment-response shapes:
      - quardratic: original Wu low-order shape
      - peaked: original Wu peaked shape
      - sigmoid: original Wu sigmoid shape
      - peaked_centered: modified version of Wu peaked that adds a mild
        quadratic center-seeking penalty so optimal doses occur in the
        intermediate region more often.
      - single_peaked: interior single-peaked baseline.
      - single_peaked_heterogeneous: interior single-peaked family with
        heterogeneous peak location/shape.

    Available dataset versions:
      - wu2024_original: original bounded Wu treatment assignment with the
        original A * X0 heterogeneity term.
      - wu2024_biased_assignment: same as wu2024_original, but the assignment
        mean is shifted away from the optimal policy.
      - wu2024_nonadditive: same treatment assignment and structural
        interaction as wu2024_original, but replaces the additive 0.25 * U_z
        outcome-confounding term by a treatment- and X0-dependent latent term.
      - wu2024_single_peaked_heterogeneous_nonadditive: keeps the original Wu
        assignment/proxy mechanism but pairs it with the
        single_peaked_heterogeneous response family and a stronger
        treatment-dependent non-additive latent outcome term.
      - wu2024_single_peaked_heterogeneous_nonadditive_shifted: same as the
        previous variant, but with a latent-confounder-dependent horizontal
        shift in the treatment argument of the real outcome surface.
      - wu2024_saturated_interaction: keeps the Wu proxy/confounding structure
        and baseline treatment-response shape, but replaces A * X0 by a centered
        saturating interaction and softens the assignment score to improve
        overlap in (X0, A).
      - wu2024_saturated_interaction_biased_assignment: the saturated-
        interaction version with assignment bias away from the optimal policy.
      - oracle_bridge_toy: a debugging DGP with a known outcome bridge
        h(a,w,x), intended to isolate bridge-estimation failures from
        downstream policy-learning failures.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.theta_x = _theta(config.dim_x)
        self.theta_z = _theta(config.dim_z)
        self.theta_w = _theta(config.dim_w)
        self.sigma_x = self._make_sigma_x()
        self.support = self._resolve_support()
        self.policy_grid = np.linspace(self.support[0], self.support[1], self.config.policy_grid_size)

    def _make_sigma_x(self) -> np.ndarray:
        rho = self.config.x_cov_rho
        k = [rho * np.ones(self.config.dim_x - 1), np.ones(self.config.dim_x), rho * np.ones(self.config.dim_x - 1)]
        return diags(k, [-1, 0, 1]).toarray()

    def _resolve_support(self) -> tuple[float, float]:
        if self.config.support_lower is not None and self.config.support_upper is not None:
            return float(self.config.support_lower), float(self.config.support_upper)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            return 0.0, 1.0

        rng = np.random.default_rng(self.config.support_seed)
        uz, uw, vz, _, x = self._sample_latents(self.config.support_pilot_size, rng)
        z = vz + self.config.proxy_confounding_scale * np.repeat(uz[:, None], self.config.dim_z, axis=1)
        raw_treatment = self._assignment_mean_base(x, z) + self.config.factual_noise_scale * uw
        lower = float(np.quantile(raw_treatment, self.config.support_quantile_low))
        upper = float(np.quantile(raw_treatment, self.config.support_quantile_high))
        if not lower < upper:
            raise ValueError("Estimated treatment support is invalid.")
        return lower, upper

    def _clip_to_support(self, value: np.ndarray) -> np.ndarray:
        return np.clip(value, self.support[0], self.support[1])

    def _sample_latents(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        e1 = rng.normal(0.0, 1.0, n)
        e2 = rng.normal(0.0, 1.0, n)
        e3 = rng.normal(0.0, 1.0, n)
        uz = e1 + e3
        uw = e2 + e3
        vz = rng.uniform(-1.0, 1.0, size=(n, self.config.dim_z))
        vw = rng.uniform(-1.0, 1.0, size=(n, self.config.dim_w))
        x = rng.multivariate_normal(np.zeros(self.config.dim_x), self.sigma_x, size=n)
        return uz, uw, vz, vw, x

    @staticmethod
    def _rescale_like(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        ref = np.asarray(reference, dtype=float)
        cand = np.asarray(candidate, dtype=float)
        ref_mean = np.mean(ref, axis=0, keepdims=True)
        cand_mean = np.mean(cand, axis=0, keepdims=True)
        ref_std = np.std(ref, axis=0, keepdims=True)
        cand_std = np.std(cand, axis=0, keepdims=True)
        scaled = (cand - cand_mean) * (ref_std / np.maximum(cand_std, 1e-8))
        return scaled + ref_mean

    def _proxy_latent_signal(self, latent: np.ndarray, dim: int) -> np.ndarray:
        return self.config.proxy_confounding_scale * np.repeat(latent[:, None], dim, axis=1)

    @staticmethod
    def _proxy_column_scales(dim: int) -> np.ndarray:
        if dim <= 1:
            return np.ones(1, dtype=float)
        return np.linspace(0.85, 1.15, dim, dtype=float)

    def _shared_proxy_spurious_signal(self, shared_latent: np.ndarray, dim: int, *, kind: str) -> np.ndarray:
        s = np.asarray(shared_latent, dtype=float).reshape(-1)
        if kind == "z":
            base = np.tanh(s)
        elif kind == "w":
            base = s * np.tanh(s)
        else:
            raise ValueError(f"Unknown spurious proxy signal kind: {kind!r}")
        scales = self._proxy_column_scales(dim)[None, :]
        return self.config.proxy_confounding_scale * base[:, None] * scales

    def _completeness_threshold(self, max_threshold: float, strength: float) -> float:
        bounded = float(np.clip(strength, 0.0, 1.0))
        return bounded * float(max_threshold)

    def _apply_completeness_violation_to_signal(
        self,
        latent: np.ndarray,
        dim: int,
        *,
        keep_negative_side: bool,
        max_threshold: float,
    ) -> np.ndarray:
        base_signal = self._proxy_latent_signal(latent, dim)
        strength = float(np.clip(self.config.completeness_violation_strength, 0.0, 1.0))
        if strength <= 0.0:
            return base_signal
        if self.config.completeness_violation_mode == "moving_relu_thresholds":
            threshold = self._completeness_threshold(max_threshold, strength)
            if keep_negative_side:
                coarse_latent = -np.maximum(-latent - threshold, 0.0)
            else:
                coarse_latent = np.maximum(latent - threshold, 0.0)
            return self._proxy_latent_signal(coarse_latent, dim)
        if self.config.completeness_violation_mode != "central_band_dropout_noise":
            raise ValueError(
                f"Unknown completeness_violation_mode: {self.config.completeness_violation_mode!r}"
            )
        threshold = self._completeness_threshold(max_threshold, strength)
        coarse_latent = np.asarray(latent, dtype=float) * (np.abs(np.asarray(latent, dtype=float)) > threshold)
        return self._proxy_latent_signal(coarse_latent, dim)

    def _completeness_noise_scale(self, strength: float) -> float:
        bounded = float(np.clip(strength, 0.0, 1.0))
        return bounded * float(self.config.completeness_noise_scale_max)

    def _apply_completeness_noise(
        self,
        signal: np.ndarray,
        *,
        rng: np.random.Generator,
        strength: float,
    ) -> np.ndarray:
        noise_scale = self._completeness_noise_scale(strength)
        if noise_scale <= 0.0:
            return signal
        noise = rng.normal(0.0, noise_scale, size=np.asarray(signal, dtype=float).shape)
        return np.asarray(signal, dtype=float) + noise

    @staticmethod
    def _align_columns(values: np.ndarray, target_dim: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape[1] == target_dim:
            return arr
        repeats = int(np.ceil(target_dim / max(arr.shape[1], 1)))
        return np.tile(arr, (1, repeats))[:, :target_dim]

    def _gaussian_copula_correlated_uniform_noise(
        self,
        vz: np.ndarray,
        vw: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Correlate aligned Uniform(-1, 1) disturbances via a Gaussian copula.

        ``rho`` is the latent Gaussian correlation, not the raw Pearson
        correlation between the returned uniform disturbances. Z columns are
        repeated or truncated with :meth:`_align_columns` to match W.
        """
        bounded_rho = float(np.clip(rho, 0.0, 1.0))
        if bounded_rho == 0.0:
            return vw

        vz_aligned = self._align_columns(vz, np.asarray(vw).shape[1])
        probability_floor = 1e-7
        uz_noise = np.clip((vz_aligned + 1.0) / 2.0, probability_floor, 1.0 - probability_floor)
        uw_noise = np.clip((vw + 1.0) / 2.0, probability_floor, 1.0 - probability_floor)
        gz = ndtri(uz_noise)
        gw = ndtri(uw_noise)
        independent_scale = np.sqrt(max(0.0, 1.0 - bounded_rho * bounded_rho))
        gw_rho = bounded_rho * gz + independent_scale * gw
        return 2.0 * ndtr(gw_rho) - 1.0

    @staticmethod
    def _spawn_proxy_rng(rng: np.random.Generator) -> np.random.Generator:
        return np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max)))

    def _generate_proxies(
        self,
        uz: np.ndarray,
        uw: np.ndarray,
        vz: np.ndarray,
        vw: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        completeness_strength = float(np.clip(self.config.completeness_violation_strength, 0.0, 1.0))
        z_signal = self._apply_completeness_violation_to_signal(
            uz,
            self.config.dim_z,
            keep_negative_side=False,
            max_threshold=self.config.completeness_relu_max_threshold_z,
        )
        w_signal = self._apply_completeness_violation_to_signal(
            uw,
            self.config.dim_w,
            keep_negative_side=True,
            max_threshold=self.config.completeness_relu_max_threshold_w,
        )
        if self.config.completeness_violation_mode == "central_band_dropout_noise" and completeness_strength > 0.0:
            z_signal = self._apply_completeness_noise(z_signal, rng=rng, strength=completeness_strength)
            w_signal = self._apply_completeness_noise(w_signal, rng=rng, strength=completeness_strength)
        structural_z = vz + z_signal
        structural_w = vw + w_signal
        strength = float(np.clip(self.config.proxy_independence_violation_strength, 0.0, 1.0))
        if strength <= 0.0:
            return structural_z, structural_w, structural_z, structural_w

        mode = self.config.proxy_independence_violation_mode
        if mode == "gaussian_copula_correlated_disturbances_structural":
            vw_rho = self._gaussian_copula_correlated_uniform_noise(vz, vw, strength)
            structural_w = vw_rho + w_signal
            return structural_z, structural_w, structural_z, structural_w

        if mode == "shared_latent_nonlinear_substitution":
            shared_latent = rng.normal(0.0, 1.0, size=z_signal.shape[0])
            spurious_z = self._shared_proxy_spurious_signal(shared_latent, z_signal.shape[1], kind="z")
            spurious_w = self._shared_proxy_spurious_signal(shared_latent, w_signal.shape[1], kind="w")
            observed_z = vz + (1.0 - strength) * z_signal + strength * spurious_z
            observed_w = vw + (1.0 - strength) * w_signal + strength * spurious_w
            return structural_z, structural_w, observed_z, observed_w

        raise ValueError(f"Unknown proxy_independence_violation_mode: {mode!r}")

    def _assignment_mean_original(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return _lambda_fn(3.0 * x @ self.theta_x + 3.0 * z @ self.theta_z)

    def _assignment_mean_softened(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        base = _lambda_fn(
            self.config.softened_assignment_scale * x @ self.theta_x
            + self.config.softened_assignment_scale * z @ self.theta_z
        )
        return (1.0 - self.config.softened_assignment_center_mix) * base + self.config.softened_assignment_center_mix * 0.5

    def _assignment_mean_base(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.config.version in ORIGINAL_WU_VERSIONS:
            return self._assignment_mean_original(x, z)
        if self.config.version in SATURATED_INTERACTION_VERSIONS:
            return self._assignment_mean_softened(x, z)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            score = 0.55 * x[:, 0] + 0.35 * z[:, 0]
            if x.shape[1] > 1:
                score = score - 0.25 * x[:, 1]
            return 0.15 + 0.70 / (1.0 + np.exp(-score))
        raise ValueError(f"Unknown data version: {self.config.version}")

    def _uses_saturated_interaction(self) -> bool:
        return self.config.version in SATURATED_INTERACTION_VERSIONS

    def _uses_biased_assignment(self) -> bool:
        return self.config.version in BIAS_VERSIONS

    def _uses_nonadditive_confounding(self) -> bool:
        return self.config.version in NONADDITIVE_WU_VERSIONS

    def _treatment_interaction(self, treatment: np.ndarray, x0: np.ndarray) -> np.ndarray:
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            return 0.5 * x0 * (treatment - 0.5)
        if self.config.structure_type == "single_peaked_heterogeneous":
            return np.zeros_like(np.asarray(treatment, dtype=float))
        if not self._uses_saturated_interaction():
            return treatment * x0
        support_width = max(self.support[1] - self.support[0], 1e-6)
        center = 0.5 * (self.support[0] + self.support[1])
        width = max(self.config.saturated_interaction_width_ratio * support_width, 1e-6)
        width = 0.45 * width
        scale = self.config.saturated_interaction_slope_ratio * width
        return scale * x0 * np.tanh((treatment - center) / width)

    def _nonadditive_confounding_scale(self) -> float:
        support_width = max(self.support[1] - self.support[0], 1e-8)
        return max(self.config.nonadditive_confounding_scale_ratio * support_width, 1e-8)

    def _shifted_treatment_by_latent(self, treatment: np.ndarray, latent_uz: np.ndarray) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        uz = np.asarray(latent_uz, dtype=float).reshape(-1)
        support_width = max(self.support[1] - self.support[0], 1e-8)
        latent_shift = (
            self.config.shifted_confounder_treatment_scale
            * support_width
            * np.tanh(self.config.shifted_confounder_treatment_slope * uz)
        )
        return treatment - latent_shift

    def latent_outcome_confounding_term(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        latent_uz: np.ndarray,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        uz = np.asarray(latent_uz, dtype=float).reshape(-1)
        if not self._uses_nonadditive_confounding():
            return self.config.outcome_confounding_scale * uz
        center = 0.5 * (self.support[0] + self.support[1])
        scale = self._nonadditive_confounding_scale()
        x0_factor = 1.0 + self.config.nonadditive_confounding_eta * x[:, 0]
        treatment_factor = 1 + self.config.nonadditive_treatment_term * np.square((treatment - center) / scale)
        base = self.config.nonadditive_confounding_gamma * treatment_factor * x0_factor * uz
        if self.config.version in {
            "wu2024_single_peaked_heterogeneous_nonadditive",
            "wu2024_single_peaked_heterogeneous_nonadditive_shifted",
        }:
            # Keep non-additivity hard but bounded so the structural peak is not washed out.
            shifted_center = center
            phase = (treatment - shifted_center) / scale
            envelope = np.exp(-0.5 * np.square(phase))
            x_gate = 1.0 + 0.35 * np.tanh(1.1 * x[:, 0])
            base = self.config.nonadditive_confounding_gamma * x0_factor * uz * x_gate * envelope
        return base + self.config.outcome_confounding_scale * uz

    def structure(self, treatment: np.ndarray) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float)
        if self.config.structure_type in {"single_peaked", "single_peaked_heterogeneous"}:
            center = 0.5 * (self.support[0] + self.support[1])
            support_width = max(self.support[1] - self.support[0], 1e-8)
            width = 0.22 * support_width
            z = (treatment - center) / max(width, 1e-8)
            return (
                2.4 * np.exp(-0.5 * z**2) * (1.0 + 0.25 * np.tanh(0.8 * z))
                - 0.18 * z**2
            )
        if self.config.structure_type == "peaked_centered":
            center = 0.5 * (self.support[0] + self.support[1])
            base = _base_structure(treatment, "peaked")
            penalty = self.config.centered_penalty_strength * np.square(treatment - center)
            return base - penalty
        return _base_structure(treatment, self.config.structure_type)

    def _single_peaked_heterogeneous(self, treatment: np.ndarray, x0: np.ndarray) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        support_width = max(self.support[1] - self.support[0], 1e-8)
        base_center = 0.5 * (self.support[0] + self.support[1])

        shift_scale = self.config.single_peak_shift_scale
        width_ratio = self.config.single_peak_width_ratio
        amplitude_scale = self.config.single_peak_amplitude_scale
        tail_penalty = self.config.single_peak_tail_penalty
        if self._uses_nonadditive_confounding() and self.config.structure_type == "single_peaked_heterogeneous":
            # Sharpen the heterogeneous single-peak only for non-additive versions.
            shift_scale = shift_scale * self.config.single_peak_nonadd_shift_scale_multiplier
            width_ratio = width_ratio * self.config.single_peak_nonadd_width_multiplier
            amplitude_scale = amplitude_scale + self.config.single_peak_nonadd_amplitude_boost
            tail_penalty = tail_penalty + self.config.single_peak_nonadd_tail_penalty_boost

        center = base_center + shift_scale * np.tanh(
            self.config.single_peak_shift_slope * x0
        )

        width = width_ratio * support_width * (
            1.0 + 0.10 * np.tanh(0.8 * x0)
        )
        width = np.maximum(width, 1e-8)

        amplitude = 1.0 + amplitude_scale * np.tanh(x0)

        z = (treatment - center) / width
        asym = self.config.single_peak_asymmetry_scale * np.tanh(x0)

        return (
            amplitude
            * np.exp(-0.5 * z**2)
            * (1.0 + asym * np.tanh(1.5 * z))
            - tail_penalty * z**2
        )

    def mean_outcome(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            x1 = x[:, 1] if x.shape[1] > 1 else np.zeros(x.shape[0])
            return (
                self.structure(treatment)
                + 0.8 * x[:, 0]
                + 0.4 * x1
                + self._treatment_interaction(treatment, x[:, 0])
            )
        if self.config.structure_type == "single_peaked_heterogeneous":
            return self._single_peaked_heterogeneous(treatment, x[:, 0]) + 1.2 * (x @ self.theta_x)
        return self.structure(treatment) + 1.2 * (x @ self.theta_x) + self._treatment_interaction(treatment, x[:, 0])

    def proxy_conditioned_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
    ) -> np.ndarray:
        """Noiseless proxy-conditioned target used for naive observed-confounder baselines.

        This is not the proximal causal estimand. It keeps the observed outcome
        proxy term in the structural outcome equation so S-learner and naive
        OOSR surfaces can be compared against their direct target up to the
        latent outcome-confounding component.
        """
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            truth = self.true_bridge(treatment, w, x)
            if truth is None:
                raise RuntimeError("Oracle bridge target was unexpectedly unavailable.")
            return truth.reshape(-1)
        if self.config.structure_type == "single_peaked_heterogeneous":
            return self._single_peaked_heterogeneous(treatment, x[:, 0]) + 1.2 * (x @ self.theta_x + w @ self.theta_w)
        return self.mean_outcome(treatment, x) + 1.2 * (w @ self.theta_w)

    def oracle_conditioned_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
        latent_uz: np.ndarray,
        latent_uw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Noiseless target for oracle direct baselines that observe latent confounders.

        The outcome equation depends on ``U_z`` but not directly on ``U_w``.
        ``latent_uw`` is accepted for interface symmetry with oracle predictors
        that condition on both latent variables.
        """
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        uz = np.asarray(latent_uz, dtype=float).reshape(-1)
        _ = latent_uw
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            truth = self.true_bridge(treatment, w, x)
            if truth is None:
                raise RuntimeError("Oracle bridge target was unexpectedly unavailable.")
            return truth.reshape(-1)
        if self.config.structure_type == "single_peaked_heterogeneous":
            treatment_for_structure = treatment
            if self.config.version == "wu2024_single_peaked_heterogeneous_nonadditive_shifted":
                treatment_for_structure = self._shifted_treatment_by_latent(treatment, uz)
            return (
                    self._single_peaked_heterogeneous(treatment_for_structure, x[:, 0])
                    + 1.2 * (x @ self.theta_x + w @ self.theta_w)
                    + self.latent_outcome_confounding_term(treatment, x, uz)
            )

        return (
                self.structure(treatment)
                + 1.2 * (x @ self.theta_x + w @ self.theta_w)
                + self._treatment_interaction(treatment, x[:, 0])
                + self.latent_outcome_confounding_term(treatment, x, uz)
        )

    def true_bridge(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray | None:
        if self.config.version not in ORACLE_BRIDGE_VERSIONS:
            return None
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        x1 = x[:, 1] if x.shape[1] > 1 else np.zeros(x.shape[0])
        return (
            self.structure(treatment)
            + 0.8 * x[:, 0]
            + 0.4 * x1
            + self._treatment_interaction(treatment, x[:, 0])
            + 0.7 * w[:, 0]
        )

    def _do_outcome(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        outcome_proxy: np.ndarray,
        latent_uz: np.ndarray,
        eps_y: np.ndarray,
    ) -> np.ndarray:
        """Evaluates the factual outcome equation under do(A=a)."""
        treatment = np.asarray(treatment, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        uz = np.asarray(latent_uz, dtype=float).reshape(-1)
        eps = np.asarray(eps_y, dtype=float).reshape(-1)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            truth = self.true_bridge(treatment, w, x)
            if truth is None:
                raise RuntimeError("Oracle bridge target was unexpectedly unavailable.")
            return truth.reshape(-1) + eps

        if self.config.structure_type == "single_peaked_heterogeneous":
            structural_treatment = treatment
            if (
                self.config.version == "wu2024_single_peaked_heterogeneous_nonadditive_shifted"
                and self.config.structure_type == "single_peaked_heterogeneous"
            ):
                structural_treatment = self._shifted_treatment_by_latent(treatment, uz)
            observed_mean = self._single_peaked_heterogeneous(structural_treatment, x[:, 0]) + 1.2 * (
                x @ self.theta_x + w @ self.theta_w
            )
        else:
            observed_mean = self.mean_outcome(treatment, x) + 1.2 * (w @ self.theta_w)

        return observed_mean + self.latent_outcome_confounding_term(treatment, x, uz) + eps

    def causal_curve_given_x_mc(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo approximation of E[Y(a) | X=x] across a treatment grid."""
        x_prime = np.asarray(x_row, dtype=float).reshape(-1)
        if x_prime.size != self.config.dim_x:
            raise ValueError(f"Expected x_row with dim {self.config.dim_x}, got {x_prime.size}.")

        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        n_total = int(self.config.causal_mc_samples if n_mc_samples is None else n_mc_samples)
        bsz = int(self.config.causal_mc_batch_size if batch_size is None else batch_size)
        if n_total <= 0:
            raise ValueError("n_mc_samples must be positive.")
        if bsz <= 0:
            raise ValueError("batch_size must be positive.")

        rng = np.random.default_rng(self.config.causal_mc_seed if seed is None else seed)
        curve = np.empty(treatment_grid.shape[0], dtype=float)

        for idx, a_val in enumerate(treatment_grid):
            total = 0.0
            used = 0
            while used < n_total:
                n_batch = min(bsz, n_total - used)
                uz, uw, vz, vw, _ = self._sample_latents(n_batch, rng)
                proxy_rng = self._spawn_proxy_rng(rng)
                _structural_z, structural_w, _observed_z, _observed_w = self._generate_proxies(
                    uz, uw, vz, vw, proxy_rng
                )
                x_batch = np.repeat(x_prime[None, :], n_batch, axis=0)
                treatment = np.full(n_batch, float(a_val), dtype=float)
                eps_scale = 0.10 if self.config.version in ORACLE_BRIDGE_VERSIONS else self.config.outcome_noise_scale
                eps_y = rng.normal(0.0, eps_scale, size=n_batch)
                y_do = self._do_outcome(
                    treatment=treatment,
                    covariates=x_batch,
                    outcome_proxy=structural_w,
                    latent_uz=uz,
                    eps_y=eps_y,
                )
                total += float(np.sum(y_do))
                used += n_batch
            curve[idx] = total / float(n_total)

        return treatment_grid, curve

    def associational_curve_given_x_binned(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        min_bin_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Approximation of E[Y | X=x, A=a] via hard bins over observational A.

        Returns:
            treatment_grid,
            associational_curve,
            bin_counts,
            low_count_mask,
            empty_bin_mask,
            empty_fallback_treatment,
            empty_fallback_outcome.
        """
        x_prime = np.asarray(x_row, dtype=float).reshape(-1)
        if x_prime.size != self.config.dim_x:
            raise ValueError(f"Expected x_row with dim {self.config.dim_x}, got {x_prime.size}.")

        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        if treatment_grid.size == 0:
            raise ValueError("grid must contain at least one point.")
        n_total = int(self.config.associational_mc_samples if n_mc_samples is None else n_mc_samples)
        min_count = int(self.config.associational_min_bin_count if min_bin_count is None else min_bin_count)
        if n_total <= 0:
            raise ValueError("n_mc_samples must be positive.")
        if min_count <= 0:
            raise ValueError("min_bin_count must be positive.")

        if treatment_grid.size > 1:
            sorted_grid = np.sort(treatment_grid)
            step = float(np.median(np.diff(sorted_grid)))
            epsilon = 0.5 * max(step, 1e-8)
        else:
            epsilon = 0.5 * max(self.support[1] - self.support[0], 1e-8)

        rng = np.random.default_rng((self.config.causal_mc_seed + 911) if seed is None else seed)
        treatment_obs, outcome_obs = self._sample_observational_given_x(x_prime, n_total, rng)

        curve = np.empty(treatment_grid.shape[0], dtype=float)
        counts = np.zeros(treatment_grid.shape[0], dtype=int)
        empty_fallback_treatment = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        empty_fallback_outcome = np.full(treatment_grid.shape[0], np.nan, dtype=float)

        for idx, a_val in enumerate(treatment_grid):
            mask = np.abs(treatment_obs - float(a_val)) <= epsilon
            n_bin = int(np.sum(mask))
            counts[idx] = n_bin
            if n_bin > 0:
                curve[idx] = float(np.mean(outcome_obs[mask]))
            else:
                nearest = int(np.argmin(np.abs(treatment_obs - float(a_val))))
                curve[idx] = float(outcome_obs[nearest])
                empty_fallback_treatment[idx] = float(treatment_obs[nearest])
                empty_fallback_outcome[idx] = float(outcome_obs[nearest])

        empty_bin_mask = counts == 0
        low_count_mask = (counts > 0) & (counts < min_count)
        return (
            treatment_grid,
            curve,
            counts,
            low_count_mask,
            empty_bin_mask,
            empty_fallback_treatment,
            empty_fallback_outcome,
        )

    def _sample_observational_given_x(
        self,
        x_prime: np.ndarray,
        n_total: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        uz, uw, vz, vw, _ = self._sample_latents(n_total, rng)
        proxy_rng = self._spawn_proxy_rng(rng)
        structural_z, structural_w, _observed_z, _observed_w = self._generate_proxies(uz, uw, vz, vw, proxy_rng)
        x_batch = np.repeat(np.asarray(x_prime, dtype=float).reshape(1, -1), n_total, axis=0)
        raw_assignment_mean_base = self._assignment_mean_base(x_batch, structural_z)
        if self._uses_biased_assignment():
            optimal = self.optimal_treatment(x_batch)
            reflected = self.support[0] + self.support[1] - optimal
            raw_assignment_mean = (1.0 - self.config.bias_strength) * raw_assignment_mean_base + self.config.bias_strength * reflected
        else:
            raw_assignment_mean = raw_assignment_mean_base
        raw_treatment = raw_assignment_mean + self.config.factual_noise_scale * uw
        treatment_obs = self._clip_to_support(raw_treatment)
        eps_scale = 0.10 if self.config.version in ORACLE_BRIDGE_VERSIONS else self.config.outcome_noise_scale
        eps_y = rng.normal(0.0, eps_scale, size=n_total)
        outcome_obs = self._do_outcome(
            treatment=treatment_obs,
            covariates=x_batch,
            outcome_proxy=structural_w,
            latent_uz=uz,
            eps_y=eps_y,
        )
        return treatment_obs, outcome_obs

    def associational_curve_given_x_kernel(
        self,
        x_row: np.ndarray,
        grid: np.ndarray | None = None,
        *,
        n_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Approximation of E[Y | X=x, A=a] via kernel smoothing over observational A."""
        x_prime = np.asarray(x_row, dtype=float).reshape(-1)
        if x_prime.size != self.config.dim_x:
            raise ValueError(f"Expected x_row with dim {self.config.dim_x}, got {x_prime.size}.")
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        if treatment_grid.size == 0:
            raise ValueError("grid must contain at least one point.")
        n_total = int(self.config.associational_mc_samples if n_mc_samples is None else n_mc_samples)
        min_count = int(self.config.associational_min_bin_count)
        if n_total <= 0:
            raise ValueError("n_mc_samples must be positive.")

        if treatment_grid.size > 1:
            sorted_grid = np.sort(treatment_grid)
            step = float(np.median(np.diff(sorted_grid)))
            bandwidth = max(step, 1e-8)
        else:
            bandwidth = max(0.10 * (self.support[1] - self.support[0]), 1e-8)

        rng = np.random.default_rng((self.config.causal_mc_seed + 911) if seed is None else seed)
        treatment_obs, outcome_obs = self._sample_observational_given_x(x_prime, n_total, rng)

        curve = np.empty(treatment_grid.shape[0], dtype=float)
        counts = np.full(treatment_grid.shape[0], n_total, dtype=int)
        low_count_mask = np.zeros(treatment_grid.shape[0], dtype=bool)
        empty_bin_mask = np.zeros(treatment_grid.shape[0], dtype=bool)
        empty_fallback_treatment = np.full(treatment_grid.shape[0], np.nan, dtype=float)
        empty_fallback_outcome = np.full(treatment_grid.shape[0], np.nan, dtype=float)

        for idx, a_val in enumerate(treatment_grid):
            z = (treatment_obs - float(a_val)) / bandwidth
            weights = np.exp(-0.5 * np.square(z))
            sum_w = float(np.sum(weights))
            if sum_w <= 1e-12:
                nearest = int(np.argmin(np.abs(treatment_obs - float(a_val))))
                curve[idx] = float(outcome_obs[nearest])
                empty_bin_mask[idx] = True
                empty_fallback_treatment[idx] = float(treatment_obs[nearest])
                empty_fallback_outcome[idx] = float(outcome_obs[nearest])
            else:
                curve[idx] = float(np.sum(weights * outcome_obs) / sum_w)
                # Effective sample size of normalized kernel weights at this grid point.
                sum_w2 = float(np.sum(np.square(weights)))
                ess = (sum_w * sum_w) / max(sum_w2, 1e-12)
                counts[idx] = int(max(1.0, round(ess)))
                low_count_mask[idx] = bool(ess < float(min_count))

        return (
            treatment_grid,
            curve,
            counts,
            low_count_mask,
            empty_bin_mask,
            empty_fallback_treatment,
            empty_fallback_outcome,
        )

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
            result = self.associational_curve_given_x_kernel(
                x_row,
                grid=grid,
                n_mc_samples=n_mc_samples,
                seed=seed,
            )
        else:
            result = self.associational_curve_given_x_binned(
                x_row,
                grid=grid,
                n_mc_samples=n_mc_samples,
                min_bin_count=min_bin_count,
                seed=seed,
            )
        return (*result, chosen)

    def average_treatment_effect_curve(
        self,
        covariates: np.ndarray,
        grid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        treatment_grid = self.policy_grid if grid is None else np.asarray(grid, dtype=float).reshape(-1)
        x = np.asarray(covariates, dtype=float)
        ate_curve = np.array(
            [float(np.mean(self.mean_outcome(np.full(x.shape[0], a_val), x))) for a_val in treatment_grid],
            dtype=float,
        )
        structure_curve = self.structure(treatment_grid)
        return treatment_grid, ate_curve, structure_curve

    def optimal_treatment(self, covariates: np.ndarray) -> np.ndarray:
        x = np.asarray(covariates, dtype=float)
        values = np.stack(
            [self.mean_outcome(np.full(x.shape[0], a_val), x) for a_val in self.policy_grid],
            axis=0,
        )
        return self.policy_grid[np.argmax(values, axis=0)]

    def policy_value(self, policy_treatment: np.ndarray, covariates: np.ndarray) -> float:
        return float(np.mean(self.mean_outcome(policy_treatment, covariates)))

    def sample_factual(self, n: int, seed: int) -> FactualSample:
        rng = np.random.default_rng(seed)
        uz, uw, vz, vw, x = self._sample_latents(n, rng)
        proxy_rng = self._spawn_proxy_rng(rng)
        structural_z, structural_w, observed_z, observed_w = self._generate_proxies(uz, uw, vz, vw, proxy_rng)
        optimal = self.optimal_treatment(x)

        raw_assignment_mean_base = self._assignment_mean_base(x, structural_z)
        if self.config.version not in ALL_WU_VERSIONS:
            raise ValueError(f"Unknown data version: {self.config.version}")
        if self._uses_biased_assignment():
            reflected = self.support[0] + self.support[1] - optimal
            raw_assignment_mean = (1.0 - self.config.bias_strength) * raw_assignment_mean_base + self.config.bias_strength * reflected
        else:
            raw_assignment_mean = raw_assignment_mean_base

        raw_treatment = raw_assignment_mean + self.config.factual_noise_scale * uw
        assignment_mean = self._clip_to_support(raw_assignment_mean)
        treatment = self._clip_to_support(raw_treatment)
        if self.config.version in ORACLE_BRIDGE_VERSIONS:
            y = self.true_bridge(treatment, structural_w, x) + rng.normal(0.0, 0.10, size=n)
        else:
            structural_treatment = treatment
            if (
                self.config.version == "wu2024_single_peaked_heterogeneous_nonadditive_shifted"
                and self.config.structure_type == "single_peaked_heterogeneous"
            ):
                structural_treatment = self._shifted_treatment_by_latent(treatment, uz)
            if self.config.structure_type == "single_peaked_heterogeneous":
                observed_mean = self._single_peaked_heterogeneous(structural_treatment, x[:, 0]) + 1.2 * (
                    x @ self.theta_x + structural_w @ self.theta_w
                )
            else:
                observed_mean = self.mean_outcome(treatment, x) + 1.2 * (structural_w @ self.theta_w)
            y = (
                observed_mean
                + self.latent_outcome_confounding_term(treatment, x, uz)
                + rng.normal(0.0, self.config.outcome_noise_scale, size=n)
            )
        return FactualSample(
            treatment=treatment[:, None],
            treatment_proxy=observed_z,
            outcome_proxy=observed_w,
            covariates=x,
            outcome=y[:, None],
            optimal_treatment=optimal[:, None],
            assignment_mean=assignment_mean[:, None],
            latent_uz=uz[:, None],
            latent_uw=uw[:, None],
        )

    def sample_train_test(self, seed: int) -> tuple[FactualSample, FactualSample]:
        return self.sample_factual(self.config.n_train, seed), self.sample_factual(self.config.n_test, seed + 1)

    def overlap_rate(self, sample: FactualSample) -> float:
        return float(
            np.mean(
                np.abs(sample.treatment.reshape(-1) - sample.optimal_treatment.reshape(-1)) <= self.config.local_eval_eps
            )
        )
