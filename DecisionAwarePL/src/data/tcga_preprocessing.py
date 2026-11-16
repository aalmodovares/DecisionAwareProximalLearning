from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DataConfig
from src.utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / candidate
    return PROJECT_ROOT / candidate


def _as_numpy_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        arr = value.to_numpy()
    elif hasattr(value, "values"):
        arr = value.values
    else:
        arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"Expected TCGA rnaseq matrix with two dimensions, got shape {arr.shape}.")
    return np.asarray(arr, dtype=float)


def _pca_scores(block: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = np.asarray(block, dtype=float) - np.mean(block, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    components = vt[:n_components]
    explained_variance = np.square(s[:n_components]) / max(centered.shape[0] - 1, 1)
    return scores, components, explained_variance


def build_tcga_feature_cache(config: DataConfig, *, force: bool = False) -> Path:
    """Build the cached TCGA PCA feature table used by the semi-synthetic DGP.

    This function is the only code path that reads ``tcga.p``. If the processed
    CSV already exists and ``force`` is false, it returns without opening the
    pickle file.
    """
    processed_path = resolve_project_path(config.tcga_processed_path)
    metadata_path = resolve_project_path(config.tcga_metadata_path)
    if processed_path.exists() and not force:
        return processed_path

    data_path = resolve_project_path(config.tcga_data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"TCGA raw data file not found: {data_path}. Download the dataset via "
            "https://github.com/ioanabica/SCIGAN and place tcga.p at this path "
            "or configure tcga_data_path."
        )

    with data_path.open("rb") as handle:
        tcga_data = pickle.load(handle)
    if "rnaseq" not in tcga_data:
        raise KeyError("TCGA pickle does not contain key 'rnaseq'.")
    raw = _as_numpy_matrix(tcga_data["rnaseq"])

    p0 = int(config.tcga_top_genes)
    block_size = int(config.tcga_block_size)
    if p0 < 3 * block_size:
        raise ValueError("tcga_top_genes must be at least 3 * tcga_block_size.")
    if raw.shape[1] < p0:
        raise ValueError(f"TCGA rnaseq has only {raw.shape[1]} genes, cannot select top {p0}.")

    variances = np.var(raw, axis=0)
    selected_idx = np.argsort(variances)[-p0:][::-1]
    selected_raw = raw[:, selected_idx]
    standardized = (selected_raw - np.mean(selected_raw, axis=0, keepdims=True)) / (
        np.std(selected_raw, axis=0, ddof=0, keepdims=True) + 1e-8
    )

    gx = standardized[:, :block_size]
    gz = standardized[:, block_size : 2 * block_size]
    gw = standardized[:, 2 * block_size : 3 * block_size]

    x_scores, x_components, x_var = _pca_scores(gx, int(config.tcga_x_pca_dim))
    z_scores, z_components, z_var = _pca_scores(gz, int(config.tcga_z_pca_dim))
    w_scores, w_components, w_var = _pca_scores(gw, int(config.tcga_w_pca_dim))

    columns = ["patient_id"]
    arrays = [np.arange(raw.shape[0], dtype=float)[:, None]]
    for prefix, scores in (("x", x_scores), ("ztilde", z_scores), ("wtilde", w_scores)):
        for idx in range(scores.shape[1]):
            columns.append(f"{prefix}{idx}")
        arrays.append(scores)
    table = np.concatenate(arrays, axis=1)

    ensure_dir(processed_path.parent)
    header = ",".join(columns)
    np.savetxt(processed_path, table, delimiter=",", header=header, comments="", fmt="%.12g")

    metadata = {
        "raw_data_path": str(data_path),
        "processed_path": str(processed_path),
        "n_patients": int(raw.shape[0]),
        "n_genes": int(raw.shape[1]),
        "selected_gene_indices": [int(idx) for idx in selected_idx.tolist()],
        "top_genes": p0,
        "block_size": block_size,
        "x_pca_dim": int(config.tcga_x_pca_dim),
        "z_pca_dim": int(config.tcga_z_pca_dim),
        "w_pca_dim": int(config.tcga_w_pca_dim),
        "x_explained_variance": [float(v) for v in x_var.tolist()],
        "z_explained_variance": [float(v) for v in z_var.tolist()],
        "w_explained_variance": [float(v) for v in w_var.tolist()],
        "x_components": x_components.tolist(),
        "z_components": z_components.tolist(),
        "w_components": w_components.tolist(),
    }
    ensure_dir(metadata_path.parent)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return processed_path


def load_tcga_feature_cache(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processed_path = build_tcga_feature_cache(config, force=False)
    data = np.genfromtxt(processed_path, delimiter=",", names=True, dtype=float)
    names = list(data.dtype.names or [])
    x_cols = [name for name in names if name.startswith("x")]
    z_cols = [name for name in names if name.startswith("ztilde")]
    w_cols = [name for name in names if name.startswith("wtilde")]
    if len(x_cols) != int(config.tcga_x_pca_dim):
        raise ValueError(f"Expected {config.tcga_x_pca_dim} cached X PCA columns, found {len(x_cols)}.")
    if len(z_cols) != int(config.tcga_z_pca_dim):
        raise ValueError(f"Expected {config.tcga_z_pca_dim} cached Z PCA columns, found {len(z_cols)}.")
    if len(w_cols) != int(config.tcga_w_pca_dim):
        raise ValueError(f"Expected {config.tcga_w_pca_dim} cached W PCA columns, found {len(w_cols)}.")
    x = np.column_stack([data[name] for name in x_cols]).astype(float)
    ztilde = np.column_stack([data[name] for name in z_cols]).astype(float)
    wtilde = np.column_stack([data[name] for name in w_cols]).astype(float)
    return x, ztilde, wtilde
