import csv

from src.analysis.sensitivity_utils import read_metric_csv_if_exists, run_outputs_complete


def test_run_outputs_complete_requires_full_export_set(tmp_path):
    run_dir = tmp_path / "seed_0001"
    run_dir.mkdir()

    required = [
        "metrics.csv",
        "validation_metrics.csv",
        "global_curves.csv",
        "patient_curves.csv",
        "patient_metadata.csv",
    ]
    for name in required[:-1]:
        (run_dir / name).write_text("placeholder", encoding="utf-8")

    assert not run_outputs_complete(run_dir)

    (run_dir / required[-1]).write_text("placeholder", encoding="utf-8")
    assert run_outputs_complete(run_dir)


def test_read_metric_csv_if_exists_returns_rows_or_empty(tmp_path):
    path = tmp_path / "metrics.csv"

    assert read_metric_csv_if_exists(path) == []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "regret"])
        writer.writeheader()
        writer.writerow({"method": "PMMR", "regret": "0.25"})

    assert read_metric_csv_if_exists(path) == [{"method": "PMMR", "regret": "0.25"}]
