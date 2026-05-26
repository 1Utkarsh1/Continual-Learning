from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cl_bench.config import load_config
from cl_bench.experiments import run_experiment
from cl_bench.reporting import collect_runs, write_report

METHODS = [
    "joint",
    "replay",
    "derpp",
    "er_ace",
    "gdumb",
    "car",
    "bic",
    "icarl",
    "x_der_lite",
]
MEMORY_BUDGETS = [200, 500, 1000, 2000, 5000]
STAGES = {
    "cifar10": ("paper/split_cifar10_full", [13, 21, 34, 55, 89]),
    "cifar100": ("paper/split_cifar100_full", [13, 21, 34, 55, 89]),
    "tinyimagenet": ("paper/split_tinyimagenet", [13, 21, 34]),
}


class RunSpec(NamedTuple):
    stage: str
    config_name: str
    method: str
    seed: int
    memory_budget: int


def build_plan(
    stages: list[str],
    methods: list[str],
    memory_budgets: list[int],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for stage in stages:
        config_name, seeds = STAGES[stage]
        for seed in seeds:
            for memory_budget in memory_budgets:
                for method in methods:
                    specs.append(
                        RunSpec(
                            stage=stage,
                            config_name=config_name,
                            method=method,
                            seed=seed,
                            memory_budget=memory_budget,
                        )
                    )
    return specs


def spec_key(spec: RunSpec) -> str:
    return "|".join(
        [
            spec.stage,
            spec.method,
            str(spec.seed),
            str(spec.memory_budget),
        ]
    )


def completed_specs(output_dir: Path) -> dict[str, Path]:
    completed: dict[str, Path] = {}
    for metrics_path in sorted(output_dir.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        benchmark = str(payload.get("benchmark", ""))
        stage = _stage_from_benchmark(benchmark)
        if stage is None:
            continue
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            continue
        key = "|".join(
            [
                stage,
                str(payload.get("method")),
                str(payload.get("seed", summary.get("seed"))),
                str(int(float(summary.get("replay_buffer_size", 0)))),
            ]
        )
        completed[key] = metrics_path.parent
    return completed


def _stage_from_benchmark(benchmark: str) -> str | None:
    return {
        "split_cifar10_full": "cifar10",
        "split_cifar100_full": "cifar100",
        "split_tinyimagenet": "tinyimagenet",
    }.get(benchmark)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full matched-memory paper experiment matrix."
    )
    parser.add_argument(
        "--stage",
        choices=["all", *STAGES.keys()],
        default="all",
        help="Dataset stage to run.",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--memory-budgets", nargs="+", type=int, default=MEMORY_BUDGETS)
    parser.add_argument("--output-dir", default="runs/paper")
    parser.add_argument("--report-root", default="docs/paper/assets")
    parser.add_argument("--tracking", choices=["json", "mlflow", "both"], default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="Refresh paper reports after this many newly executed runs. Use 0 to disable.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    plan = build_plan(stages, args.methods, args.memory_budgets)
    output_dir = Path(args.output_dir)
    report_root = Path(args.report_root)
    already_completed = {} if args.no_resume else completed_specs(output_dir)
    print(f"Planned runs: {len(plan)}")
    print(f"Already completed runs: {len(already_completed)}")
    for spec in plan:
        status = "SKIP" if spec_key(spec) in already_completed else "RUN"
        print(
            " ".join(
                [
                    f"[{status}]",
                    "cl-bench run",
                    f"--config-name {spec.config_name}",
                    f"--method {spec.method}",
                    f"--seed {spec.seed}",
                    f"--output-dir {args.output_dir}",
                    f"--tracking {args.tracking}",
                    f"strategy.replay_buffer_size={spec.memory_budget}",
                ]
            )
        )

    if args.dry_run:
        return 0

    completed_by_stage: dict[str, list[Path]] = {stage: [] for stage in stages}
    for spec in plan:
        existing = already_completed.get(spec_key(spec))
        if existing is not None:
            completed_by_stage[spec.stage].append(existing)

    executed = 0
    for spec in plan:
        if spec_key(spec) in already_completed:
            continue
        if args.max_runs is not None and executed >= args.max_runs:
            break
        config = replace(
            load_config(spec.config_name),
            method=spec.method,
            seed=spec.seed,
            output_dir=args.output_dir,
            tracking=args.tracking,
            replay_buffer_size=spec.memory_budget,
        )
        if args.device is not None:
            config = replace(config, device=args.device)
        result = run_experiment(config)
        executed += 1
        completed_by_stage[spec.stage].append(result.run_dir)
        already_completed[spec_key(spec)] = result.run_dir
        print(
            f"completed stage={spec.stage} method={spec.method} seed={spec.seed} "
            f"memory={spec.memory_budget} run_dir={result.run_dir}"
        )
        _write_progress(output_dir, plan, already_completed)
        if args.report_every > 0 and executed % args.report_every == 0:
            _write_reports(
                completed_by_stage=completed_by_stage,
                report_root=report_root,
                methods=args.methods,
                memory_budgets=args.memory_budgets,
                plan=plan,
            )

    _write_reports(
        completed_by_stage=completed_by_stage,
        report_root=report_root,
        methods=args.methods,
        memory_budgets=args.memory_budgets,
        plan=plan,
    )
    return 0


def _write_progress(output_dir: Path, plan: list[RunSpec], completed: dict[str, Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "planned_runs": len(plan),
        "completed_runs": len(completed),
        "remaining_runs": len(plan) - len(completed),
        "completed": {key: str(path) for key, path in sorted(completed.items())},
    }
    (output_dir / "paper_run_progress.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_reports(
    completed_by_stage: dict[str, list[Path]],
    report_root: Path,
    methods: list[str],
    memory_budgets: list[int],
    plan: list[RunSpec],
) -> None:
    for stage, run_dirs in completed_by_stage.items():
        if not run_dirs:
            continue
        records = collect_runs(run_dirs)
        report_dir = report_root / stage
        report = write_report(
            records,
            output_dir=report_dir,
            title=f"{stage} full-data paper protocol",
            make_plots=True,
            paper=True,
        )
        expected_runs = sum(1 for spec in plan if spec.stage == stage)
        manifest = {
            "stage": stage,
            "config": STAGES[stage][0],
            "methods": methods,
            "memory_budgets": memory_budgets,
            "completed_runs": len(run_dirs),
            "expected_runs": expected_runs,
            "is_complete": len(run_dirs) >= expected_runs,
            "run_dirs": [str(path) for path in run_dirs],
            "report_dir": str(report.report_dir),
        }
        (report_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_confidence_intervals(report_dir)


def _write_confidence_intervals(report_dir: Path) -> None:
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("leaderboard", []):
        runs = int(row.get("runs", 0))
        std = float(row.get("average_final_accuracy_std", 0.0))
        ci95 = 0.0 if runs <= 1 else 1.96 * std / float(np.sqrt(runs))
        enriched = dict(row)
        enriched["average_final_accuracy_ci95"] = ci95
        rows.append(enriched)
    (report_dir / "confidence_intervals.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
