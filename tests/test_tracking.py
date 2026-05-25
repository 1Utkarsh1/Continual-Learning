from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cl_bench.config import load_config
from cl_bench.experiments import run_experiment

mlflow = pytest.importorskip("mlflow")


def test_mlflow_tracking_logs_params_metrics_and_artifacts(tmp_path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = replace(
        load_config("smoke"),
        output_dir=str(tmp_path / "runs"),
        method="baseline",
        tracking="both",
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment="unit-cl-bench",
    )

    result = run_experiment(config)

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("unit-cl-bench")
    assert experiment is not None
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["method"] == "baseline"
    assert "average_final_accuracy" in run.data.metrics
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="metrics.json",
    )
    assert Path(artifact_path).exists()
    assert result.metrics_path.exists()
