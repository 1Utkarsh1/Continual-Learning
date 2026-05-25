from __future__ import annotations

import json

from cl_bench.reporting import aggregate_records, collect_runs, write_export, write_report


def _write_metrics(run_dir, method: str, seed: int, final_accuracy: float) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        "benchmark": "unit",
        "method": method,
        "task_names": ["a", "b"],
        "summary": {
            "average_final_accuracy": final_accuracy,
            "average_learning_accuracy": 80.0,
            "average_forgetting": 5.0,
            "backward_transfer": -5.0,
            "runtime_seconds": 1.5,
            "seed": seed,
            "replay_buffer_size": 500,
            "model": "linear",
        },
        "accuracy_matrix": [[80.0, None], [70.0, final_accuracy]],
        "forgetting_matrix": [[0.0, None], [10.0, 0.0]],
        "runtime_seconds": 1.5,
        "seed": seed,
        "git_commit": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_and_aggregate_runs(tmp_path) -> None:
    _write_metrics(tmp_path / "baseline_seed_1", "baseline", 1, 60.0)
    _write_metrics(tmp_path / "baseline_seed_2", "baseline", 2, 80.0)
    _write_metrics(tmp_path / "replay_seed_1", "replay", 1, 90.0)

    records = collect_runs([tmp_path])
    leaderboard = aggregate_records(records)

    assert [row["method"] for row in leaderboard] == ["replay", "baseline"]
    baseline = next(row for row in leaderboard if row["method"] == "baseline")
    assert baseline["runs"] == 2
    assert baseline["seeds"] == "1,2"
    assert baseline["average_final_accuracy_mean"] == 70.0


def test_write_report_without_plots(tmp_path) -> None:
    _write_metrics(tmp_path / "replay_seed_1", "replay", 1, 90.0)
    records = collect_runs([tmp_path])

    report = write_report(records, tmp_path / "report", "Unit report", make_plots=False)

    assert report.leaderboard_csv.exists()
    assert report.summary_json.exists()
    assert report.markdown.exists()
    assert report.plots == []


def test_collect_runs_from_mlflow_artifact_export_shape(tmp_path) -> None:
    export_dir = tmp_path / "mlruns" / "0" / "run-id" / "artifacts" / "run"
    _write_metrics(export_dir, "derpp", 13, 88.0)

    records = collect_runs([tmp_path / "mlruns"])

    assert len(records) == 1
    assert records[0].method == "derpp"
    assert records[0].summary["average_final_accuracy"] == 88.0


def test_write_paper_report_and_export_without_plots(tmp_path) -> None:
    _write_metrics(tmp_path / "car_seed_1", "car", 1, 90.0)
    records = collect_runs([tmp_path])

    report = write_report(records, tmp_path / "paper", "Paper report", make_plots=False, paper=True)
    exports = write_export(records, tmp_path / "export", "mammoth")

    assert report.markdown.exists()
    assert (tmp_path / "paper" / "leaderboard_table.tex").exists()
    assert (tmp_path / "paper" / "per_seed_results.csv").exists()
    assert (tmp_path / "paper" / "claims_table.md").exists()
    assert all(path.exists() for path in exports)
