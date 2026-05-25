from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskSpec:
    """Declarative description of one task in a continual-learning benchmark."""

    name: str
    dataset: str
    classes: list[int] | str
    samples_per_class: int | None = None
    test_samples_per_class: int | None = None
    train_limit: int | None = None
    test_limit: int | None = None
    train_feature_cache: str | None = None
    test_feature_cache: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskSpec:
        classes = raw["classes"]
        if classes != "all":
            classes = [int(label) for label in classes]
        return cls(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            classes=classes,
            samples_per_class=_optional_int(raw.get("samples_per_class")),
            test_samples_per_class=_optional_int(raw.get("test_samples_per_class")),
            train_limit=_optional_int(raw.get("train_limit")),
            test_limit=_optional_int(raw.get("test_limit")),
            train_feature_cache=_optional_str(raw.get("train_feature_cache")),
            test_feature_cache=_optional_str(raw.get("test_feature_cache")),
        )


@dataclass
class ExperimentConfig:
    """Runtime configuration for a benchmark run."""

    name: str
    method: str
    tasks: list[TaskSpec]
    seed: int = 42
    device: str = "auto"
    model: str = "mlp"
    data_dir: str = "data"
    output_dir: str = "runs"
    tracking: str = "json"
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "continual-learning-bench"
    epochs: int = 1
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: str = "none"
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    val_fraction: float = 0.1
    num_workers: int = 0
    augment: bool = True
    ewc_lambda: float = 50.0
    fisher_samples: int = 128
    replay_buffer_size: int = 512
    replay_batch_size: int = 32
    replay_loss_weight: float = 1.0
    lwf_alpha: float = 0.5
    lwf_temperature: float = 2.0
    derpp_alpha: float = 0.5
    derpp_beta: float = 1.0
    agem_memory_batch_size: int = 64
    gdumb_epochs: int = 20
    car_logit_anchor_weight: float = 0.25
    car_replay_ce_weight: float = 1.0
    car_feature_anchor_weight: float = 0.05
    car_prototype_anchor_weight: float = 0.05
    car_calibration_epochs: int = 10
    car_calibration_lr: float = 0.01
    car_calibration_weight_decay: float = 0.0
    car_replay_augment: bool = True
    car_use_current_task_mask: bool = True
    save_checkpoint: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        training = raw.get("training", {})
        strategy = raw.get("strategy", {})
        tracking = raw.get("tracking", {})
        if not isinstance(tracking, dict):
            tracking = {"mode": tracking}
        tasks = [TaskSpec.from_dict(task) for task in raw["tasks"]]
        return cls(
            name=str(raw.get("name", "continual_learning")),
            method=str(raw.get("method", "baseline")),
            tasks=tasks,
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            model=str(raw.get("model", "mlp")),
            data_dir=str(raw.get("data_dir", "data")),
            output_dir=str(raw.get("output_dir", "runs")),
            tracking=str(tracking.get("mode", raw.get("tracking", "json"))),
            mlflow_tracking_uri=str(
                tracking.get(
                    "mlflow_tracking_uri",
                    raw.get("mlflow_tracking_uri", "sqlite:///mlruns/mlflow.db"),
                )
            ),
            mlflow_experiment=str(
                tracking.get(
                    "mlflow_experiment",
                    raw.get("mlflow_experiment", "continual-learning-bench"),
                )
            ),
            epochs=int(training.get("epochs", raw.get("epochs", 1))),
            batch_size=int(training.get("batch_size", raw.get("batch_size", 64))),
            eval_batch_size=int(training.get("eval_batch_size", raw.get("eval_batch_size", 256))),
            learning_rate=float(training.get("learning_rate", raw.get("learning_rate", 1e-3))),
            optimizer=str(training.get("optimizer", raw.get("optimizer", "adam"))),
            momentum=float(training.get("momentum", raw.get("momentum", 0.9))),
            weight_decay=float(training.get("weight_decay", raw.get("weight_decay", 0.0))),
            scheduler=str(training.get("scheduler", raw.get("scheduler", "none"))),
            warmup_epochs=int(training.get("warmup_epochs", raw.get("warmup_epochs", 0))),
            label_smoothing=float(training.get("label_smoothing", raw.get("label_smoothing", 0.0))),
            val_fraction=float(training.get("val_fraction", raw.get("val_fraction", 0.1))),
            num_workers=int(training.get("num_workers", raw.get("num_workers", 0))),
            augment=bool(training.get("augment", raw.get("augment", True))),
            ewc_lambda=float(strategy.get("ewc_lambda", raw.get("ewc_lambda", 50.0))),
            fisher_samples=int(strategy.get("fisher_samples", raw.get("fisher_samples", 128))),
            replay_buffer_size=int(
                strategy.get("replay_buffer_size", raw.get("replay_buffer_size", 512))
            ),
            replay_batch_size=int(
                strategy.get("replay_batch_size", raw.get("replay_batch_size", 32))
            ),
            replay_loss_weight=float(
                strategy.get("replay_loss_weight", raw.get("replay_loss_weight", 1.0))
            ),
            lwf_alpha=float(strategy.get("lwf_alpha", raw.get("lwf_alpha", 0.5))),
            lwf_temperature=float(strategy.get("lwf_temperature", raw.get("lwf_temperature", 2.0))),
            derpp_alpha=float(strategy.get("derpp_alpha", raw.get("derpp_alpha", 0.5))),
            derpp_beta=float(strategy.get("derpp_beta", raw.get("derpp_beta", 1.0))),
            agem_memory_batch_size=int(
                strategy.get("agem_memory_batch_size", raw.get("agem_memory_batch_size", 64))
            ),
            gdumb_epochs=int(strategy.get("gdumb_epochs", raw.get("gdumb_epochs", 20))),
            car_logit_anchor_weight=float(
                strategy.get(
                    "car_logit_anchor_weight",
                    raw.get("car_logit_anchor_weight", 0.25),
                )
            ),
            car_replay_ce_weight=float(
                strategy.get("car_replay_ce_weight", raw.get("car_replay_ce_weight", 1.0))
            ),
            car_feature_anchor_weight=float(
                strategy.get(
                    "car_feature_anchor_weight",
                    raw.get("car_feature_anchor_weight", 0.05),
                )
            ),
            car_prototype_anchor_weight=float(
                strategy.get(
                    "car_prototype_anchor_weight",
                    raw.get("car_prototype_anchor_weight", 0.05),
                )
            ),
            car_calibration_epochs=int(
                strategy.get("car_calibration_epochs", raw.get("car_calibration_epochs", 10))
            ),
            car_calibration_lr=float(
                strategy.get("car_calibration_lr", raw.get("car_calibration_lr", 0.01))
            ),
            car_calibration_weight_decay=float(
                strategy.get(
                    "car_calibration_weight_decay",
                    raw.get("car_calibration_weight_decay", 0.0),
                )
            ),
            car_replay_augment=bool(
                strategy.get("car_replay_augment", raw.get("car_replay_augment", True))
            ),
            car_use_current_task_mask=bool(
                strategy.get(
                    "car_use_current_task_mask",
                    raw.get("car_use_current_task_mask", True),
                )
            ),
            save_checkpoint=bool(raw.get("save_checkpoint", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Summary returned by an experiment run."""

    run_dir: Path
    method: str
    task_names: list[str]
    accuracy_matrix: list[list[float | None]]
    forgetting_matrix: list[list[float | None]]
    summary: dict[str, float | int | str | None]
    metrics_path: Path
    config_path: Path
    runtime_seconds: float
    git_commit: str | None = None


def load_config(source: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML path or a known config name."""

    path = resolve_config_path(source)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return ExperimentConfig.from_dict(raw)


def load_config_with_overrides(
    source: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load config and apply Hydra/OmegaConf dot-list overrides."""

    overrides = overrides or []
    if not overrides:
        return load_config(source)

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "Hydra-style overrides require the experiment extra: "
            'python -m pip install -e ".[experiment]"'
        ) from exc

    path = resolve_config_path(source)
    base = OmegaConf.load(path)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base, override_conf)
    raw = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return ExperimentConfig.from_dict(raw)


def resolve_config_path(source: str | Path) -> Path:
    candidate = Path(source)
    if candidate.exists():
        return candidate

    name = str(source)
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"

    search_roots = [
        Path.cwd() / "configs",
        Path.cwd() / "configs" / "experiments",
        Path(__file__).resolve().parents[2] / "configs",
        Path(__file__).resolve().parents[2] / "configs" / "experiments",
    ]
    for root in search_roots:
        path = root / name
        if path.exists():
            return path

    for root in [Path.cwd() / "configs", Path(__file__).resolve().parents[2] / "configs"]:
        matches = sorted(root.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            joined = ", ".join(str(match) for match in matches)
            raise FileExistsError(f"Config name '{source}' is ambiguous: {joined}")

    searched = ", ".join(str(root / name) for root in search_roots)
    raise FileNotFoundError(f"Could not find config '{source}'. Searched: {searched}")


def dump_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
