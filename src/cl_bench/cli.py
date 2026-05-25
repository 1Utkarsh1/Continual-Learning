from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from cl_bench.config import ExperimentConfig, load_config_with_overrides
from cl_bench.experiments import run_experiment
from cl_bench.reporting import collect_runs, write_report
from cl_bench.tracking import MLflowRunLogger

METHODS = ("baseline", "ewc", "replay", "lwf", "derpp", "agem", "er_ace", "gdumb")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-configs":
        for path in sorted((Path.cwd() / "configs").glob("*.yaml")):
            print(path)
        return 0

    if args.command == "run":
        config = apply_cli_overrides(load_cli_config(args), args)
        result = run_experiment(config)
        print(f"Run directory: {result.run_dir}")
        print(f"Metrics: {result.metrics_path}")
        print(f"Average final accuracy: {result.summary['average_final_accuracy']:.2f}%")
        print(f"Average forgetting: {result.summary['average_forgetting']:.2f}%")
        return 0

    if args.command == "suite":
        base_config = apply_cli_overrides(load_cli_config(args), args)
        seeds = args.seeds or [base_config.seed]
        run_dirs: list[Path] = []

        for seed in seeds:
            for method in args.methods:
                config = replace(base_config, method=method, seed=seed)
                result = run_experiment(config)
                run_dirs.append(result.run_dir)
                print(
                    f"{method} seed={seed}: "
                    f"{result.summary['average_final_accuracy']:.2f}% final accuracy, "
                    f"{result.summary['average_forgetting']:.2f}% forgetting "
                    f"({result.run_dir})"
                )

        if args.report_dir:
            records = collect_runs(run_dirs)
            report = write_report(
                records,
                output_dir=args.report_dir,
                title=args.title or f"{base_config.name} benchmark report",
                make_plots=not args.no_plots,
            )
            log_suite_report_to_mlflow(base_config, report.report_dir, len(records))
            print(f"Report directory: {report.report_dir}")
            print(f"Leaderboard: {report.leaderboard_csv}")
        return 0

    if args.command == "report":
        records = collect_runs(args.runs)
        report = write_report(
            records,
            output_dir=args.output_dir,
            title=args.title,
            make_plots=not args.no_plots,
        )
        print(f"Report directory: {report.report_dir}")
        print(f"Leaderboard: {report.leaderboard_csv}")
        if report.plots:
            print("Plots:")
            for plot in report.plots:
                print(f"  {plot}")
        return 0

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cl-bench",
        description="Run reproducible continual-learning benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a benchmark from a YAML config.")
    add_config_arguments(run_parser)
    run_parser.add_argument("--method", choices=METHODS)
    add_runtime_overrides(run_parser)
    run_parser.add_argument("overrides", nargs="*", help="Hydra/OmegaConf key=value overrides.")

    suite_parser = subparsers.add_parser(
        "suite",
        help="Run multiple methods/seeds from one config and optionally build a report.",
    )
    add_config_arguments(suite_parser)
    suite_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    suite_parser.add_argument("--seeds", nargs="+", type=int)
    suite_parser.add_argument("--report-dir")
    suite_parser.add_argument("--title")
    suite_parser.add_argument("--no-plots", action="store_true")
    add_runtime_overrides(suite_parser)
    suite_parser.add_argument("overrides", nargs="*", help="Hydra/OmegaConf key=value overrides.")

    report_parser = subparsers.add_parser(
        "report",
        help="Aggregate existing run directories or metrics.json files into a report.",
    )
    report_parser.add_argument("--runs", nargs="+", required=True)
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--title", default="Continual-learning benchmark report")
    report_parser.add_argument("--no-plots", action="store_true")

    subparsers.add_parser("list-configs", help="List YAML configs in ./configs.")
    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Config path or config name.")
    config_group.add_argument("--config-name", help="Hydra-style config name from ./configs.")


def add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=["linear", "mlp", "small_cnn", "cnn", "cifar_convnet", "resnet18_cifar"],
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--tracking", choices=["json", "mlflow", "both"])
    parser.add_argument("--save-checkpoint", action="store_true")


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates = {}
    for arg_name, field_name in [
        ("method", "method"),
        ("model", "model"),
        ("epochs", "epochs"),
        ("seed", "seed"),
        ("device", "device"),
        ("output_dir", "output_dir"),
        ("data_dir", "data_dir"),
        ("batch_size", "batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("learning_rate", "learning_rate"),
        ("tracking", "tracking"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value

    if getattr(args, "save_checkpoint", False):
        updates["save_checkpoint"] = True

    return replace(config, **updates)


def load_cli_config(args: argparse.Namespace) -> ExperimentConfig:
    source = args.config_name or args.config
    overrides = getattr(args, "overrides", None) or []
    return load_config_with_overrides(source, overrides)


def log_suite_report_to_mlflow(config: ExperimentConfig, report_dir: Path, run_count: int) -> None:
    if config.tracking.lower() not in {"mlflow", "both"}:
        return
    with MLflowRunLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment,
        run_name=f"{config.name}_suite_report",
        enabled=True,
    ) as logger:
        logger.log_params(
            {
                "benchmark": config.name,
                "report_dir": str(report_dir),
                "run_count": run_count,
                "artifact_type": "suite_report",
            }
        )
        logger.log_artifacts(report_dir)
