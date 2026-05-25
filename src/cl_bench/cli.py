from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from cl_bench.config import ExperimentConfig, load_config_with_overrides
from cl_bench.experiments import run_experiment
from cl_bench.reporting import collect_runs, write_export, write_report
from cl_bench.tracking import MLflowRunLogger

METHODS = (
    "baseline",
    "ewc",
    "replay",
    "lwf",
    "derpp",
    "agem",
    "er_ace",
    "gdumb",
    "car",
    "bic",
    "icarl",
    "x_der_lite",
    "joint",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-configs":
        for path in sorted((Path.cwd() / "configs").rglob("*.yaml")):
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
        memory_budgets = args.memory_budgets or [base_config.replay_buffer_size]
        run_dirs: list[Path] = []

        for memory_budget in memory_budgets:
            for seed in seeds:
                for method in args.methods:
                    config = replace(
                        base_config,
                        method=method,
                        seed=seed,
                        replay_buffer_size=memory_budget,
                    )
                    result = run_experiment(config)
                    run_dirs.append(result.run_dir)
                    print(
                        f"{method} seed={seed} memory={memory_budget}: "
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
                paper=args.paper,
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
            paper=args.paper,
        )
        print(f"Report directory: {report.report_dir}")
        print(f"Leaderboard: {report.leaderboard_csv}")
        if report.plots:
            print("Plots:")
            for plot in report.plots:
                print(f"  {plot}")
        return 0

    if args.command == "export":
        records = collect_runs(args.runs)
        paths = write_export(records, args.output_dir, args.format)
        print("Exported:")
        for path in paths:
            print(f"  {path}")
        return 0

    if args.command == "sweep":
        base_config = apply_cli_overrides(load_cli_config(args), args)
        run_sweep(base_config, args)
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
    suite_parser.add_argument("--memory-budgets", nargs="+", type=int)
    suite_parser.add_argument("--report-dir")
    suite_parser.add_argument("--title")
    suite_parser.add_argument("--no-plots", action="store_true")
    suite_parser.add_argument("--paper", action="store_true")
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
    report_parser.add_argument("--paper", action="store_true")

    export_parser = subparsers.add_parser(
        "export",
        help="Export run summaries in comparison-friendly CSV/JSON formats.",
    )
    export_parser.add_argument("--runs", nargs="+", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--format", choices=["csv", "mammoth", "avalanche"], required=True)

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run an Optuna hyperparameter sweep for a method/config.",
    )
    add_config_arguments(sweep_parser)
    sweep_parser.add_argument("--method", choices=METHODS, default="car")
    sweep_parser.add_argument("--study-name", required=True)
    sweep_parser.add_argument("--n-trials", type=int, default=20)
    sweep_parser.add_argument("--storage", default="sqlite:///mlruns/optuna.db")
    sweep_parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    add_runtime_overrides(sweep_parser)
    sweep_parser.add_argument("overrides", nargs="*", help="Hydra/OmegaConf key=value overrides.")

    subparsers.add_parser("list-configs", help="List YAML configs in ./configs.")
    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Config path or config name.")
    config_group.add_argument("--config-name", help="Hydra-style config name from ./configs.")


def add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=[
            "linear",
            "mlp",
            "small_cnn",
            "cnn",
            "cifar_convnet",
            "resnet18_cifar",
            "cifar_resnet18",
        ],
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


def run_sweep(config: ExperimentConfig, args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "The sweep command requires Optuna. Install with: "
            'python -m pip install -e ".[experiment]"'
        ) from exc

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_config = replace(
            config,
            method=args.method,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
            replay_buffer_size=trial.suggest_categorical(
                "replay_buffer_size",
                [200, 500, 1000, 2000, 5000],
            ),
            replay_batch_size=trial.suggest_categorical("replay_batch_size", [32, 64, 128, 256]),
            car_logit_anchor_weight=trial.suggest_float("car_logit_anchor_weight", 0.0, 1.0),
            car_replay_ce_weight=trial.suggest_float("car_replay_ce_weight", 0.25, 4.0),
            car_feature_anchor_weight=trial.suggest_float("car_feature_anchor_weight", 0.0, 0.5),
            car_prototype_anchor_weight=trial.suggest_float(
                "car_prototype_anchor_weight",
                0.0,
                0.5,
            ),
        )
        result = run_experiment(trial_config)
        return float(result.summary["average_final_accuracy"]) - 0.2 * float(
            result.summary["average_forgetting"]
        )

    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
