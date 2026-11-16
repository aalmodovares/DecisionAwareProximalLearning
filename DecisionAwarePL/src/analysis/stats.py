from __future__ import annotations

import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest


def wilcoxon_paired_by_label(
    paired_values,
    *,
    alpha=0.05,
    correction="uncorrected",
    alternative="two-sided",
    zero_method="wilcox",
):
    """Run paired Wilcoxon tests for multiple labels.

    Parameters
    ----------
    paired_values:
        Dict-like mapping label -> (x_values, y_values). Arrays are paired by index.
    alpha:
        Significance level.
    correction:
        "uncorrected" or "holm".
    alternative:
        SciPy Wilcoxon alternative hypothesis.
    zero_method:
        SciPy Wilcoxon zero_method.
    """

    valid_corrections = {"uncorrected", "holm"}
    if correction not in valid_corrections:
        raise ValueError(f"Unknown correction {correction!r}. Expected one of {sorted(valid_corrections)}.")

    labels = []
    rows = []
    p_values = []
    for label, pair in dict(paired_values).items():
        if pair is None or len(pair) != 2:
            raise ValueError(f"Each paired_values entry must be a (x, y) tuple. Invalid entry for {label!r}.")
        x_raw, y_raw = pair
        x = np.asarray(x_raw, dtype=float).reshape(-1)
        y = np.asarray(y_raw, dtype=float).reshape(-1)
        n_pairs = int(min(x.size, y.size))
        if n_pairs == 0:
            statistic = np.nan
            p_value = 1.0
        else:
            x = x[:n_pairs]
            y = y[:n_pairs]
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            n_pairs = int(x.size)
            if n_pairs == 0:
                statistic = np.nan
                p_value = 1.0
            else:
                try:
                    # Access named fields to be explicit across SciPy versions and IDE type checkers.
                    wilcoxon_result = stats.wilcoxon(
                        x,
                        y,
                        alternative=alternative,
                        zero_method=zero_method,
                    )
                    statistic = float(wilcoxon_result.statistic)
                    p_value = float(wilcoxon_result.pvalue)
                except ValueError:
                    statistic, p_value = np.nan, 1.0
        labels.append(label)
        p_values.append(float(p_value))
        rows.append(
            {
                "label": label,
                "n_pairs": n_pairs,
                "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value_uncorrected": float(p_value),
            }
        )

    if correction == "holm" and p_values:
        reject, p_corr, _, _ = multitest.multipletests(p_values, alpha=alpha, method="holm")
    else:
        reject = np.asarray([p < alpha for p in p_values], dtype=bool)
        p_corr = np.asarray(p_values, dtype=float)

    out = {}
    for idx, label in enumerate(labels):
        row = rows[idx]
        row["p_value"] = float(p_corr[idx])
        row["rejected"] = bool(reject[idx])
        row["correction"] = correction
        row["alternative"] = alternative
        out[label] = row
    return out


