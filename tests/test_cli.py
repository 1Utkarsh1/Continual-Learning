from __future__ import annotations

from cl_bench.cli import main


def test_cli_run_report_and_export(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    report_dir = tmp_path / "report"
    export_dir = tmp_path / "export"

    assert (
        main(
            [
                "run",
                "--config-name",
                "smoke",
                "--method",
                "baseline",
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--output-dir",
                str(runs_dir),
                "--tracking",
                "json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                "--runs",
                str(runs_dir),
                "--output-dir",
                str(report_dir),
                "--title",
                "CLI report",
                "--paper",
                "--no-plots",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "export",
                "--runs",
                str(runs_dir),
                "--output-dir",
                str(export_dir),
                "--format",
                "csv",
            ]
        )
        == 0
    )

    assert (report_dir / "leaderboard.csv").exists()
    assert (report_dir / "leaderboard_table.tex").exists()
    assert (export_dir / "csv_runs.csv").exists()
