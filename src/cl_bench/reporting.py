from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunRecord:
    """Parsed metrics for one completed benchmark run."""

    run_dir: Path
    benchmark: str
    method: str
    seed: int
    task_names: list[str]
    summary: dict[str, float | int | str | None]
    accuracy_matrix: np.ndarray
    forgetting_matrix: np.ndarray
    runtime_seconds: float
    git_commit: str | None


@dataclass(frozen=True)
class ReportArtifacts:
    """Files written by the reporting pipeline."""

    report_dir: Path
    leaderboard_csv: Path
    summary_json: Path
    markdown: Path
    plots: list[Path]


def collect_runs(sources: Sequence[str | Path]) -> list[RunRecord]:
    """Load metrics from run directories, metrics files, or parent directories."""

    metric_paths = discover_metrics(sources)
    if not metric_paths:
        raise FileNotFoundError("No metrics.json files were found in the supplied run paths.")
    return [load_run(path) for path in metric_paths]


def discover_metrics(sources: Sequence[str | Path]) -> list[Path]:
    """Return sorted metrics.json paths discovered from files or directories."""

    discovered: set[Path] = set()
    for source in sources:
        path = Path(source)
        if path.is_file():
            if path.name != "metrics.json":
                raise ValueError(f"Expected metrics.json file, got: {path}")
            discovered.add(path)
            continue

        if not path.exists():
            raise FileNotFoundError(path)

        direct_metrics = path / "metrics.json"
        if direct_metrics.exists():
            discovered.add(direct_metrics)
            continue

        for metrics_path in path.rglob("metrics.json"):
            discovered.add(metrics_path)

    return sorted(discovered)


def load_run(metrics_path: str | Path) -> RunRecord:
    """Parse one run metrics artifact."""

    path = Path(metrics_path)
    if path.is_dir():
        path = path / "metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = dict(payload["summary"])
    runtime_seconds = float(payload.get("runtime_seconds", summary.get("runtime_seconds", 0.0)))
    seed = int(payload.get("seed", summary.get("seed", 0)))
    return RunRecord(
        run_dir=path.parent,
        benchmark=str(payload["benchmark"]),
        method=str(payload["method"]),
        seed=seed,
        task_names=[str(name) for name in payload["task_names"]],
        summary=summary,
        accuracy_matrix=_matrix_from_json(payload["accuracy_matrix"]),
        forgetting_matrix=_matrix_from_json(payload["forgetting_matrix"]),
        runtime_seconds=runtime_seconds,
        git_commit=payload.get("git_commit"),
    )


def aggregate_records(records: Sequence[RunRecord]) -> list[dict[str, float | int | str]]:
    """Aggregate run summaries by method for leaderboard-style reporting."""

    by_method: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_method[record.method].append(record)

    rows: list[dict[str, float | int | str]] = []
    for method, method_records in sorted(by_method.items()):
        final_accuracy = [_metric(record, "average_final_accuracy") for record in method_records]
        learning_accuracy = [
            _metric(record, "average_learning_accuracy") for record in method_records
        ]
        forgetting = [_metric(record, "average_forgetting") for record in method_records]
        backward_transfer = [_metric(record, "backward_transfer") for record in method_records]
        runtimes = [record.runtime_seconds for record in method_records]
        memory_budgets = [_metric(record, "replay_buffer_size") for record in method_records]
        seeds = ",".join(
            str(record.seed) for record in sorted(method_records, key=lambda item: item.seed)
        )

        rows.append(
            {
                "method": method,
                "runs": len(method_records),
                "seeds": seeds,
                "average_final_accuracy_mean": _mean(final_accuracy),
                "average_final_accuracy_std": _std(final_accuracy),
                "average_learning_accuracy_mean": _mean(learning_accuracy),
                "average_forgetting_mean": _mean(forgetting),
                "average_forgetting_std": _std(forgetting),
                "backward_transfer_mean": _mean(backward_transfer),
                "runtime_seconds_mean": _mean(runtimes),
                "memory_budget_mean": _mean(memory_budgets),
            }
        )

    return sorted(rows, key=lambda row: float(row["average_final_accuracy_mean"]), reverse=True)


def write_report(
    records: Sequence[RunRecord],
    output_dir: str | Path,
    title: str,
    make_plots: bool = True,
) -> ReportArtifacts:
    """Write CSV, JSON, Markdown, and optional plot artifacts for a benchmark suite."""

    if not records:
        raise ValueError("At least one run record is required to write a report.")

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = aggregate_records(records)
    leaderboard_csv = _write_leaderboard_csv(report_dir / "leaderboard.csv", leaderboard)
    summary_json = _write_summary_json(report_dir / "summary.json", title, records, leaderboard)
    plots: list[Path] = []
    if make_plots:
        plots = _write_plots(records, leaderboard, report_dir, title)
    markdown = _write_markdown(report_dir / "README.md", title, records, leaderboard, plots)
    return ReportArtifacts(
        report_dir=report_dir,
        leaderboard_csv=leaderboard_csv,
        summary_json=summary_json,
        markdown=markdown,
        plots=plots,
    )


def _write_leaderboard_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> Path:
    fieldnames = [
        "method",
        "runs",
        "seeds",
        "average_final_accuracy_mean",
        "average_final_accuracy_std",
        "average_learning_accuracy_mean",
        "average_forgetting_mean",
        "average_forgetting_std",
        "backward_transfer_mean",
        "runtime_seconds_mean",
        "memory_budget_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_summary_json(
    path: Path,
    title: str,
    records: Sequence[RunRecord],
    leaderboard: Sequence[dict[str, float | int | str]],
) -> Path:
    payload = {
        "title": title,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": sorted({record.benchmark for record in records}),
        "num_runs": len(records),
        "leaderboard": list(leaderboard),
        "runs": [
            {
                "run_dir": str(record.run_dir),
                "benchmark": record.benchmark,
                "method": record.method,
                "seed": record.seed,
                "task_names": record.task_names,
                "summary": record.summary,
                "runtime_seconds": record.runtime_seconds,
                "git_commit": record.git_commit,
            }
            for record in records
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _write_markdown(
    path: Path,
    title: str,
    records: Sequence[RunRecord],
    leaderboard: Sequence[dict[str, float | int | str]],
    plots: Sequence[Path],
) -> Path:
    benchmark_names = ", ".join(sorted({record.benchmark for record in records}))
    task_count = max(len(record.task_names) for record in records)
    lines = [
        f"# {title}",
        "",
        f"Benchmarks: `{benchmark_names}`",
        f"Runs: `{len(records)}`",
        f"Tasks per run: `{task_count}`",
        "",
        "## Leaderboard",
        "",
        "| Method | Runs | Seeds | Memory | Final accuracy | Forgetting | Backward transfer | Mean runtime |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in leaderboard:
        lines.append(
            "| {method} | {runs} | {seeds} | {memory:.0f} | {accuracy} | {forgetting} | {bwt} | {runtime} |".format(
                method=row["method"],
                runs=row["runs"],
                seeds=row["seeds"],
                memory=float(row["memory_budget_mean"]),
                accuracy=_format_with_std(
                    float(row["average_final_accuracy_mean"]),
                    float(row["average_final_accuracy_std"]),
                    suffix="%",
                ),
                forgetting=_format_with_std(
                    float(row["average_forgetting_mean"]),
                    float(row["average_forgetting_std"]),
                    suffix="%",
                ),
                bwt=f"{float(row['backward_transfer_mean']):.2f}%",
                runtime=f"{float(row['runtime_seconds_mean']):.1f}s",
            )
        )

    if plots:
        lines.extend(["", "## Plots", ""])
        for plot in plots:
            lines.append(f"![{plot.stem}]({plot.name})")
            lines.append("")

    lines.extend(
        [
            "## Source Runs",
            "",
            "| Method | Seed | Run directory |",
            "| --- | ---: | --- |",
        ]
    )
    for record in sorted(records, key=lambda item: (item.method, item.seed, str(item.run_dir))):
        lines.append(f"| {record.method} | {record.seed} | `{record.run_dir}` |")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_plots(
    records: Sequence[RunRecord],
    leaderboard: Sequence[dict[str, float | int | str]],
    report_dir: Path,
    title: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [
        _plot_leaderboard(plt, leaderboard, report_dir / "leaderboard.png", title),
        _plot_retention_curves(plt, records, report_dir / "retention_curves.png"),
        _plot_accuracy_matrices(plt, records, report_dir / "accuracy_matrices.png"),
    ]
    plt.close("all")
    return paths


def _plot_leaderboard(
    plt: Any, leaderboard: Sequence[dict[str, float | int | str]], path: Path, title: str
) -> Path:
    methods = [str(row["method"]) for row in leaderboard]
    final_means = [float(row["average_final_accuracy_mean"]) for row in leaderboard]
    final_stds = [float(row["average_final_accuracy_std"]) for row in leaderboard]
    forgetting_means = [float(row["average_forgetting_mean"]) for row in leaderboard]
    forgetting_stds = [float(row["average_forgetting_std"]) for row in leaderboard]
    colors = [_method_color(method) for method in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    _bar_chart(
        axes[0],
        methods,
        final_means,
        final_stds,
        colors,
        title="Average final accuracy",
        ylabel="Accuracy (%)",
        higher_is_better=True,
    )
    _bar_chart(
        axes[1],
        methods,
        forgetting_means,
        forgetting_stds,
        colors,
        title="Average forgetting",
        ylabel="Forgetting (%)",
        higher_is_better=False,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _bar_chart(
    axis: Any,
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    colors: Sequence[str],
    title: str,
    ylabel: str,
    higher_is_better: bool,
) -> None:
    positions = np.arange(len(labels))
    axis.bar(
        positions, means, yerr=stds, color=colors, capsize=4, edgecolor="#111827", linewidth=0.8
    )
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, max(100.0 if higher_is_better else 5.0, max(means, default=0.0) * 1.2 + 2.0))
    axis.grid(axis="y", alpha=0.22)
    for position, value in zip(positions, means, strict=True):
        axis.text(position, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)


def _plot_retention_curves(plt: Any, records: Sequence[RunRecord], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for method, method_records in _records_by_method(records).items():
        curves = []
        for record in method_records:
            curve = []
            for step in range(record.accuracy_matrix.shape[0]):
                seen = record.accuracy_matrix[step, : step + 1]
                curve.append(float(np.nanmean(seen)))
            curves.append(curve)
        matrix = np.asarray(curves, dtype=float)
        x_values = np.arange(1, matrix.shape[1] + 1)
        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0)
        color = _method_color(method)
        axis.plot(x_values, mean_curve, marker="o", linewidth=2.2, label=method, color=color)
        if matrix.shape[0] > 1:
            axis.fill_between(
                x_values, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.16
            )

    axis.set_title("Retention across the task stream", fontsize=13, fontweight="bold")
    axis.set_xlabel("After training task")
    axis.set_ylabel("Mean accuracy on seen tasks (%)")
    axis.set_ylim(0, 100)
    axis.set_xticks(np.arange(1, max(record.accuracy_matrix.shape[0] for record in records) + 1))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _plot_accuracy_matrices(plt: Any, records: Sequence[RunRecord], path: Path) -> Path:
    representative_records = [
        max(method_records, key=lambda record: _metric(record, "average_final_accuracy"))
        for method_records in _records_by_method(records).values()
    ]
    columns = min(2, len(representative_records))
    rows = int(np.ceil(len(representative_records) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.4 * columns, 5.4 * rows), squeeze=False, constrained_layout=True
    )

    for axis in axes.ravel()[len(representative_records) :]:
        axis.axis("off")

    image = None
    for axis, record in zip(axes.ravel(), representative_records, strict=False):
        matrix = np.ma.masked_invalid(record.accuracy_matrix)
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        axis.set_title(f"{record.method} accuracy matrix", fontsize=12, fontweight="bold")
        axis.set_xlabel("Evaluated task")
        axis.set_ylabel("After training task")
        axis.set_xticks(range(len(record.task_names)))
        axis.set_yticks(range(len(record.task_names)))
        axis.set_xticklabels(
            [_short_task_name(name) for name in record.task_names], rotation=35, ha="right"
        )
        axis.set_yticklabels([str(index + 1) for index in range(len(record.task_names))])
        for row in range(record.accuracy_matrix.shape[0]):
            for column in range(record.accuracy_matrix.shape[1]):
                value = record.accuracy_matrix[row, column]
                if np.isnan(value):
                    continue
                axis.text(
                    column, row, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8
                )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Accuracy (%)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _records_by_method(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    by_method: dict[str, list[RunRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.method, item.seed)):
        by_method[record.method].append(record)
    return dict(by_method)


def _matrix_from_json(values: Sequence[Sequence[float | None]]) -> np.ndarray:
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in values]
    )


def _metric(record: RunRecord, key: str) -> float:
    value = record.summary.get(key, 0.0)
    return 0.0 if value is None else float(value)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values, ddof=0)) if len(values) > 1 else 0.0


def _format_with_std(mean: float, std: float, suffix: str) -> str:
    if std == 0.0:
        return f"{mean:.2f}{suffix}"
    return f"{mean:.2f} +- {std:.2f}{suffix}"


def _method_color(method: str) -> str:
    return {
        "baseline": "#475569",
        "ewc": "#2563eb",
        "replay": "#16a34a",
        "lwf": "#c2410c",
        "derpp": "#7c3aed",
        "agem": "#0f766e",
        "er_ace": "#0891b2",
        "gdumb": "#db2777",
    }.get(method, "#7c3aed")


def _short_task_name(name: str) -> str:
    if "_" not in name:
        return name
    return name.split("_", 1)[1]
