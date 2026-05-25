from __future__ import annotations

from dataclasses import replace

from cl_bench.config import load_config
from cl_bench.experiments import run_experiment


def test_cpu_smoke_benchmark_writes_reproducibility_artifacts(tmp_path) -> None:
    config = replace(load_config("smoke"), output_dir=str(tmp_path), method="derpp")

    result = run_experiment(config)

    assert result.run_dir.exists()
    assert result.config_path.exists()
    assert result.metrics_path.exists()
    assert (result.run_dir / "metrics.jsonl").exists()
    assert (result.run_dir / "accuracy_matrix.csv").exists()
    assert (result.run_dir / "forgetting_matrix.csv").exists()
    assert (result.run_dir / "transfer_baselines.json").exists()
    assert result.summary["num_tasks"] == 2
    assert "forward_transfer" in result.summary


def test_synthetic_suite_runs_core_memory_methods(tmp_path) -> None:
    for method in ["baseline", "replay", "derpp", "agem", "er_ace", "gdumb", "car", "joint"]:
        config = replace(load_config("smoke"), output_dir=str(tmp_path), method=method)
        result = run_experiment(config)

        assert result.method == method
        assert result.metrics_path.exists()
        assert result.summary["num_tasks"] == 2
